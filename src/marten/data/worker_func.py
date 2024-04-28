import math
import time
import pandas as pd
import numpy as np

import exchange_calendars as xcals

from tenacity import (
    stop_after_attempt,
    wait_exponential,
    Retrying,
)

# import exchange_calendars as xcals
from datetime import datetime, timedelta
from sqlalchemy import (
    text,
)
from sqlalchemy.dialects.postgresql import insert

# module_path = os.getenv("LOCAL_AKSHARE_DEV_MODULE")
# if module_path is not None and module_path not in sys.path:
# sys.path.insert(0, module_path)
import akshare as ak  # noqa: E402

from marten.utils.worker import await_futures
from marten.data.tabledef import (
    table_def_index_daily_em,
    table_def_hk_index_daily_em,
    table_def_us_index_daily_sina,
    table_def_hk_index_spot_em,
    table_def_fund_etf_spot_em,
    table_def_index_spot_em,
    table_def_fund_etf_perf_em,
    table_def_fund_etf_list_sina,
    table_def_fund_etf_daily_em,
    table_def_bond_metrics_em,
    bond_zh_hs_spot,
    bond_zh_hs_daily,
    stock_zh_a_spot_em,
    stock_zh_a_hist_em,
    currency_boc_safe,
    spot_symbol_table_sge,
    spot_hist_sge,
    cn_bond_index_period,
    cn_bond_indices,
)

from marten.data.api.snowball import SnowballAPI

from dask.distributed import worker_client, get_worker, Variable

from functools import lru_cache


@lru_cache()
def last_trade_date():
    xshg = xcals.get_calendar("XSHG")
    current_date = datetime.now().date()
    # Iterate backwards from current_date until a valid session is found
    last_session = current_date
    while not xshg.is_session(last_session):
        last_session -= timedelta(days=1)
    return last_session


def update_on_conflict(table_def, conn, df: pd.DataFrame, primary_keys):
    """
    Insert new records, update existing records without nullifying columns not included in the dataframe
    """
    # Load the table metadata
    # table = sqlalchemy.Table(table, sqlalchemy.MetaData(), autoload_with=conn)
    # Create an insert statement from the DataFrame records
    insert_stmt = insert(table_def).values(df.to_dict(orient="records"))

    if hasattr(table_def, "__table__"):
        table_columns = table_def.__table__.columns
    else:
        table_columns = table_def.columns

    # Build a dictionary of column values to be updated, excluding primary keys and non-existent columns
    update_dict = {
        c.name: insert_stmt.excluded[c.name]
        for c in table_columns
        if c.name in df.columns and c.name not in primary_keys
    }
    # Construct the on_conflict_do_update statement
    on_conflict_stmt = insert_stmt.on_conflict_do_update(
        index_elements=primary_keys, set_=update_dict
    )
    # Execute the on_conflict_do_update statement
    conn.execute(on_conflict_stmt)


def ignore_on_conflict(table_def, conn, df, primary_keys):
    """
    Insert new records, ignore existing records
    """
    # table = sqlalchemy.Table(table, sqlalchemy.MetaData(), autoload_with=conn)
    insert_stmt = insert(table_def).values(df.to_dict(orient="records"))
    on_conflict_stmt = insert_stmt.on_conflict_do_nothing(index_elements=primary_keys)
    conn.execute(on_conflict_stmt)


def get_latest_date(conn, symbol, table, col=None):
    query = f"SELECT max(date) FROM {table} where 1=1"
    if col is not None:
        query += f" and {col} is not null"
    if symbol is not None:
        query += " and symbol = :symbol"
        result = conn.execute(text(query), {"symbol": symbol})
    else:
        result = conn.execute(text(query))
    return result.fetchone()[0]


def saveAsCsv(file_name_main: str, df):
    """
    Save dataframe to CSV file
    """
    # save to file
    # Get the current timestamp to append to the filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save the dataframe to a csv file with timestamp as suffix. Need to properly encode and display Chinese characters.
    df.to_csv(f"{file_name_main}_{current_time}.csv", encoding="utf_8_sig", index=False)


def hk_index_daily(future_hk_index_list):
    precursor_task_completed = future_hk_index_list

    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    ##query index list from table and submit tasks to client for each
    with alchemyEngine.begin() as conn:
        result = conn.execute(text("select symbol from hk_index_spot_em"))
        result_set = result.fetchall()
    index_list = [row[0] for row in result_set]

    futures = []
    with worker_client() as client:
        logger.info("starting tasks on function update_hk_indices()...")
        for symbol in index_list:
            futures.append(client.submit(update_hk_indices, symbol, priority=1))
            await_futures(futures, False)

    await_futures(futures)
    return len(index_list)


def update_hk_indices(symbol):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    try:
        shide = ak.stock_hk_index_daily_em(symbol=symbol)

        # if shide is empty, return immediately
        if shide.empty:
            return None

        shide.loc[:, "symbol"] = symbol
        shide.rename(
            columns={
                "latest": "close",
            },
            inplace=True,
        )
        # Convert the 'date' column to datetime
        shide.loc[:, "date"] = pd.to_datetime(shide["date"]).dt.date
        with alchemyEngine.begin() as conn:
            latest_date = get_latest_date(conn, symbol, "hk_index_daily_em")

            if latest_date is not None:
                ## keep rows only with `date` later than the latest record in database.
                shide = shide[shide["date"] > (latest_date - timedelta(days=10))]
            update_on_conflict(
                table_def_hk_index_daily_em(), conn, shide, ["symbol", "date"]
            )

        return len(shide)
    except Exception as e:
        logger.error(f"failed to update hk_index_daily_em for {symbol}", exc_info=True)
        raise e


def get_us_indices(us_index_list):
    worker = get_worker()
    logger = worker.logger

    futures = []
    with worker_client() as client:
        logger.info("starting task on function update_us_indices()...")
        for symbol in us_index_list:
            futures.append(client.submit(update_us_indices, symbol))
            await_futures(futures, False)

    await_futures(futures)
    return len(us_index_list)


def update_us_indices(symbol):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    try:
        iuss = ak.index_us_stock_sina(symbol=symbol)
        iuss.loc[:, "symbol"] = symbol
        # Convert iuss["date"] to datetime and normalize to date only
        iuss.loc[:, "date"] = pd.to_datetime(iuss["date"]).dt.date
        with alchemyEngine.begin() as conn:
            latest_date = get_latest_date(conn, symbol, "us_index_daily_sina")
            if latest_date is not None:
                iuss = iuss[iuss["date"] > (latest_date - timedelta(days=10))]
            update_on_conflict(
                table_def_us_index_daily_sina(), conn, iuss, ["symbol", "date"]
            )
        return len(iuss)
    except Exception as e:
        logger.error(
            f"failed to update us_index_daily_sina for {symbol}", exc_info=True
        )
        raise e


def bond_spot():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    logger.info("running bond_spot()...")

    bzhs = None
    for attempt in Retrying(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=5)
    ):
        with attempt:
            bzhs = ak.bond_zh_hs_spot()

    bzhs.rename(
        columns={
            "代码": "symbol",
            "名称": "name",
            "最新价": "close",
            "涨跌额": "change_amount",
            "涨跌幅": "change_rate",
            "买入": "bid_price",
            "卖出": "ask_price",
            "今开": "open",
            "最高": "high",
            "最低": "low",
            "昨收": "prev_close",
            "成交量": "volume",
            "成交额": "turnover",
        },
        inplace=True,
    )

    with alchemyEngine.begin() as conn:
        update_on_conflict(bond_zh_hs_spot, conn, bzhs, ["symbol"])
    return len(bzhs)


def get_bond_zh_hs_daily(symbol, shared_dict):
    st_dict = shared_dict.get()
    st_dict["start_time"] = datetime.now()
    shared_dict.set(st_dict)

    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    with alchemyEngine.begin() as conn:
        update_query = text("""
            UPDATE bond_zh_hs_spot 
            SET last_checked = CURRENT_TIMESTAMP
            WHERE symbol = :symbol 
        """)
        params = {"symbol": symbol}
        conn.execute(update_query, params)

    try:
        bzhd = ak.bond_zh_hs_daily(symbol)

        # if shide is empty, return immediately
        if bzhd.empty:
            logger.warning("bond daily history data is empty: %s", symbol)
            return None

        with alchemyEngine.begin() as conn:
            latest_date = get_latest_date(conn, symbol, "bond_zh_hs_daily")
            if latest_date is not None:
                ## keep rows only with `date` later than the latest record in database.
                bzhd = bzhd[bzhd["date"] > (latest_date - timedelta(days=10))]

            bzhd.loc[:, "symbol"] = symbol

            ignore_on_conflict(bond_zh_hs_daily, conn, bzhd, ["symbol", "date"])
        return len(bzhd)
    except KeyError as e:
        if "'date'" in str(e):
            logger.warning("ak.bond_zh_hs_daily(%s) could be empty: %s", symbol, str(e))
        else:
            logger.error(
                f"failed to update bond_zh_hs_daily for {symbol}", exc_info=True
            )
            raise e
    except Exception as e:
        logger.error(f"failed to update bond_zh_hs_daily for {symbol}", exc_info=True)
        raise e


def bond_daily_hs(future_bond_spot, n_threads):
    precursor_task_completed = future_bond_spot

    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    bond_list = pd.read_sql(
        """
        select symbol from (
            (SELECT symbol, turnover, last_checked FROM bond_zh_hs_spot WHERE turnover != 0)
            UNION
            (SELECT symbol, turnover, last_checked FROM bond_zh_hs_spot WHERE turnover = 0 and last_checked IS NULL LIMIT 2000) -- Adjust the limit as needed
            UNION
            (SELECT symbol, turnover, last_checked FROM bond_zh_hs_spot 
                WHERE turnover = 0 
                AND last_checked IS NOT NULL
                AND last_checked < NOW() - INTERVAL '48 hours'
                AND random() < (
                    CASE
                        WHEN last_checked < NOW() - INTERVAL '96 hours' THEN 0.4
                        WHEN last_checked < NOW() - INTERVAL '72 hours' THEN 0.3
                        ELSE 0.2
                    END
                ) order by last_checked asc limit 1000
            )
        ) order by turnover desc, last_checked asc
        """,
        alchemyEngine,
    )
    logger.info(
        "starting tasks on function bond_daily_hs(). #symbols: %s", len(bond_list)
    )

    futures = {}
    shared_vars = {}
    task_timeout = 280  # seconds

    with worker_client() as client:
        for symbol in bond_list["symbol"]:
            var_st = Variable()
            var_st.set({"symbol": symbol})
            shared_vars[symbol] = var_st

            futures[symbol] = client.submit(get_bond_zh_hs_daily, symbol, var_st)

            await_futures(futures, False, task_timeout, shared_vars, n_threads)

    await_futures(futures, True, task_timeout, shared_vars)

    return len(bond_list)


def hk_index_spot():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    logger.info("running stock_hk_index_spot_em()...")
    hk_index_list_df = ak.stock_hk_index_spot_em()
    hk_index_list_df.rename(
        columns={
            "序号": "seq",
            "内部编号": "internal_code",
            "代码": "symbol",
            "名称": "name",
            "最新价": "close",
            "涨跌额": "change_amount",
            "涨跌幅": "change_rate",
            "今开": "open",
            "最高": "high",
            "最低": "low",
            "昨收": "prev_close",
            "成交量": "volume",
            "成交额": "amount",
        },
        inplace=True,
    )

    with alchemyEngine.begin() as conn:
        update_on_conflict(
            table_def_hk_index_spot_em(), conn, hk_index_list_df, ["symbol"]
        )
    return len(hk_index_list_df)


def cn_index_daily(future_cn_index_list):
    precursor_task_completed = future_cn_index_list

    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    cn_index_fulllist = pd.read_sql(
        "SELECT symbol, src FROM index_spot_em", alchemyEngine
    )
    logger.info("starting tasks on function stock_zh_index_daily_em()...")

    futures = []
    with worker_client() as client:
        for symbol, src in zip(cn_index_fulllist["symbol"], cn_index_fulllist["src"]):
            futures.append(client.submit(stock_zh_index_daily_em, symbol, src, priority=1))
            await_futures(futures, False)

    await_futures(futures)
    return len(cn_index_fulllist)


def stock_zh_index_daily_em(symbol, src):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    try:
        with alchemyEngine.begin() as conn:
            latest_date = get_latest_date(conn, symbol, "index_daily_em")

            start_date = "19900101"  # For entire history.
            if latest_date is not None:
                start_date = (latest_date - timedelta(days=10)).strftime("%Y%m%d")

            end_date = datetime.now().strftime("%Y%m%d")

            szide = ak.stock_zh_index_daily_em(f"{src}{symbol}", start_date, end_date)

            # if shide is empty, return immediately
            if szide.empty:
                logger.warning("index data is empty: %s", symbol)
                return None

            szide.loc[:, "symbol"] = symbol

            ignore_on_conflict(
                table_def_index_daily_em(), conn, szide, ["symbol", "date"]
            )
        return len(szide)
    except Exception as e:
        logger.error(f"failed to update index_daily_em for {symbol}", exc_info=True)
        raise e


def stock_zh_index_spot_em(symbol, src):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    try:
        szise = ak.stock_zh_index_spot_em(symbol)
        szise = szise.rename(
            columns={
                "序号": "seq",
                "代码": "symbol",
                "名称": "name",
                "最新价": "close",
                "涨跌幅": "change_rate",
                "涨跌额": "change_amount",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "最高": "high",
                "最低": "low",
                "今开": "open",
                "昨收": "prev_close",
                "量比": "volume_ratio",
            }
        )
        szise.loc[:, "src"] = src
        with alchemyEngine.begin() as conn:
            update_on_conflict(table_def_index_spot_em(), conn, szise, ["symbol"])
        return len(szise)
    except Exception as e:
        logger.error(f"failed to update index_spot_em for {symbol}", exc_info=True)
        raise e


def get_cn_index_list(cn_index_types):
    worker = get_worker()
    logger = worker.logger
    logger.info("starting task on function stock_zh_index_spot_em()...")
    ##loop thru cn_index_types and send off further tasks to client
    futures = []
    with worker_client() as client:
        for symbol, src in cn_index_types:
            futures.append(client.submit(stock_zh_index_spot_em, symbol, src))
            await_futures(futures, False)

    await_futures(futures)
    return True


def calc_etf_metrics(symbol, end_date):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    interval = 250  # assume 250 trading days annualy
    try:
        with alchemyEngine.begin() as conn:
            # load the latest (top) `interval` records of historical market data records from `fund_etf_daily_em` table for `symbol`, order by `date`.
            # select columns: date, change_rate
            query = """SELECT date, change_rate FROM fund_etf_daily_em WHERE symbol = '{}' ORDER BY date DESC LIMIT {}""".format(
                symbol, interval
            )
            df = pd.read_sql(query, conn, parse_dates=["date"])

            # get oldest df['date'] as state_date
            start_date = df["date"].iloc[-1]
            # get 2-years CN bond IR as risk-free IR from bond_metrics_em table. 1-year series (natural dates).
            # select date, china_yield_2y from table `bond_metrics_em`, where date is between start_date and end_date (inclusive). Load into a dataframe.
            query = """SELECT date, china_yield_2y FROM bond_metrics_em WHERE date BETWEEN '{}' AND '{}' and china_yield_2y <> 'nan'""".format(
                start_date, end_date
            )
            bme_df = pd.read_sql(query, conn, parse_dates=["date"])
            # Convert annualized rate to a daily rate
            bme_df.loc[:, "china_yield_2y_daily"] = bme_df["china_yield_2y"] / 365.25

            # merge df with bme_df by matching dates.
            df = pd.merge_asof(
                df.sort_values("date"),
                bme_df.sort_values("date"),
                on="date",
                direction="backward",
            ).dropna(subset=["change_rate"])

            # calculate the Sharpe ratio, Sortino ratio, and max drawdown with the time series data inside df.
            df.loc[:, "excess_return"] = df["change_rate"] - df["china_yield_2y_daily"]
            # Annualize the excess return
            annualized_excess_return = np.mean(df["excess_return"])

            # Calculate the standard deviation of the excess returns
            std_dev = df["excess_return"].std()

            # Sharpe ratio
            sharpe_ratio = annualized_excess_return / std_dev

            # Calculate the downside deviation (Sortino ratio denominator)
            downside_dev = df[df["excess_return"] < 0]["excess_return"].std()

            # Sortino ratio
            sortino_ratio = (
                annualized_excess_return / downside_dev if downside_dev > 0 else None
            )

            # To calculate max drawdown, get the cummulative_returns
            df.loc[:, "cumulative_returns"] = (
                np.cumprod(1 + df["change_rate"] / 100.0) - 1
            )
            # Calculate the maximum cumulative return up to each point
            peak = np.maximum.accumulate(df["cumulative_returns"])
            # Calculate drawdown as the difference between the current value and the peak
            drawdown = (df["cumulative_returns"] - peak) / (1 + peak) * 100
            # Calculate max drawdown
            max_drawdown = np.min(drawdown)  # This is a negative number

            # update the `sharperatio, sortinoratio, maxdrawdown` columns for `symbol` in the table `fund_etf_perf_em` using the calculated metrics.
            update_query = text(
                "UPDATE fund_etf_perf_em SET sharperatio = :sharperatio, sortinoratio = :sortinoratio, maxdrawdown = :maxdrawdown WHERE fundcode = :fundcode"
            )
            params = {
                "sharperatio": (
                    round(sharpe_ratio, 2)
                    if sharpe_ratio is not None and math.isfinite(sharpe_ratio)
                    else None
                ),
                "sortinoratio": (
                    round(sortino_ratio, 2)
                    if sortino_ratio is not None and math.isfinite(sortino_ratio)
                    else None
                ),
                "maxdrawdown": (
                    round(max_drawdown, 2) if math.isfinite(max_drawdown) else None
                ),
                "fundcode": symbol,
            }
            conn.execute(update_query, params)

        return len(df)
    except Exception as e:
        logger.error(f"failed to update ETF metrics for {symbol}", exc_info=True)
        raise e


# load historical data from daily table and calc metrics, then update perf table
def update_etf_metrics(future_etf_list, future_bond_ir):
    precursor_task_completed = future_etf_list
    precursor_task_completed = future_bond_ir

    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    end_date = last_trade_date()
    logger.info(
        "starting task on function update_etf_metrics(), last trade date: %s",
        end_date,
    )

    with alchemyEngine.begin() as conn:
        result = conn.execute(text("select symbol from fund_etf_list_sina"))
        result_set = result.fetchall()
    etf_list = [row[0] for row in result_set]

    ##submit tasks to calculate metrics for each symbol
    futures = []
    with worker_client() as client:
        logger.info(f"starting tasks on function calc_etf_metrics()...")
        for symbol in etf_list:
            futures.append(client.submit(calc_etf_metrics, symbol, end_date, priority=1))
            await_futures(futures, False)

    await_futures(futures)

    return len(etf_list)


def bond_ir():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    logger.info(f"running bond_zh_us_rate()...")
    try:
        start_date = None  # For entire history.
        with alchemyEngine.begin() as conn:
            latest_date = get_latest_date(conn, None, "bond_metrics_em")
            if latest_date is not None:
                start_date = (latest_date - timedelta(days=20)).strftime("%Y%m%d")
            bzur = ak.bond_zh_us_rate(start_date)
            bzur = bzur.rename(
                columns={
                    "日期": "date",
                    "中国国债收益率2年": "china_yield_2y",
                    "中国国债收益率5年": "china_yield_5y",
                    "中国国债收益率10年": "china_yield_10y",
                    "中国国债收益率30年": "china_yield_30y",
                    "中国国债收益率10年-2年": "china_yield_spread_10y_2y",
                    "中国GDP年增率": "china_gdp_growth",
                    "美国国债收益率2年": "us_yield_2y",
                    "美国国债收益率5年": "us_yield_5y",
                    "美国国债收益率10年": "us_yield_10y",
                    "美国国债收益率30年": "us_yield_30y",
                    "美国国债收益率10年-2年": "us_yield_spread_10y_2y",
                    "美国GDP年增率": "us_gdp_growth",
                }
            )
            update_on_conflict(table_def_bond_metrics_em(), conn, bzur, ["date"])
        return len(bzur)
    except Exception as e:
        logger.exception("failed to get bond interest rate")


# Function to fetch and process ETF data
def get_etf_daily(symbol):

    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    try:
        logger.debug(f"running fund_etf_hist_em({symbol})...")
        with alchemyEngine.begin() as conn:
            # check latest date on fund_etf_daily_em
            latest_date = get_latest_date(conn, symbol, "fund_etf_daily_em")

            start_date = "19700101"  # For entire history.
            if latest_date is not None:
                start_date = (latest_date - timedelta(days=10)).strftime("%Y%m%d")

            end_date = datetime.now().strftime("%Y%m%d")

            df = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )

            # if df contains no row at all, return immediately
            if df.empty:
                return None

            df.loc[:, "symbol"] = symbol
            df = df.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "turnover",
                    "振幅": "amplitude",
                    "涨跌幅": "change_rate",
                    "涨跌额": "change_amount",
                    "换手率": "turnover_rate",
                }
            )
            df = df[
                [
                    "symbol",
                    "date",
                    "open",
                    "close",
                    "high",
                    "low",
                    "volume",
                    "turnover",
                    "amplitude",
                    "change_rate",
                    "change_amount",
                    "turnover_rate",
                ]
            ]

            ignore_on_conflict(
                table_def_fund_etf_daily_em(), conn, df, ["symbol", "date"]
            )

            return len(df)
    except Exception as e:
        logger.error(
            f"failed to get daily trade history data for {symbol}", exc_info=True
        )
        raise e


def etf_list():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    try:
        logger.info("running fund_etf_category_sina()...")
        fund_etf_category_sina_df = ak.fund_etf_category_sina(symbol="ETF基金")

        # keep only 2 columns from `fund_etf_category_sina_df`: 代码, 名称.
        # split `代码` values by `exchange code` and `symbol` and store into 2 columns. No need to keep the `代码` column.
        # for example: 代码=sz159998, split into `exch=sz`, `symbol=159998`.
        df = fund_etf_category_sina_df[["代码", "名称"]].copy()
        df.columns = ["code", "name"]
        # df[["exch", "symbol"]] = df["code"].str.extract(r"([a-z]+)(\d+)")
        split_codes = df["code"].str.extract(r"([a-z]+)(\d+)")
        df.loc[:, "exch"] = split_codes[0]
        df.loc[:, "symbol"] = split_codes[1]
        df.drop(columns=["code"], inplace=True)

        # Now, use the update_on_conflict function to insert or update the data
        with alchemyEngine.begin() as conn:
            update_on_conflict(
                table_def_fund_etf_list_sina(), conn, df, ["exch", "symbol"]
            )

        ##submit new task (get_etf_daily) for each symbol to get historical data
        futures = []
        with worker_client() as client:
            logger.info(f"starting task on function get_etf_daily()...")
            for symbol in df["symbol"]:
                futures.append(client.submit(get_etf_daily, symbol, priority=1))
                await_futures(futures, False)

        await_futures(futures)

        return len(df)
    except Exception as e:
        logger.exception("failed to get ETF list")


def etf_perf():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    try:
        logger.info("running fund_exchange_rank_em()...")
        fund_exchange_rank_em_df = ak.fund_exchange_rank_em()
        saveAsCsv("fund_exchange_rank_em", fund_exchange_rank_em_df)
        column_mapping = {
            "序号": "id",
            "基金代码": "fundcode",
            "基金简称": "fundname",
            "类型": "type",
            "日期": "date",
            "单位净值": "unitnav",
            "累计净值": "accumulatednav",
            "近1周": "pastweek",
            "近1月": "pastmonth",
            "近3月": "past3months",
            "近6月": "past6months",
            "近1年": "pastyear",
            "近2年": "past2years",
            "近3年": "past3years",
            "今年来": "ytd",
            "成立来": "sinceinception",
            "成立日期": "inceptiondate",
        }
        fund_exchange_rank_em_df.rename(columns=column_mapping, inplace=True)
        fund_exchange_rank_em_df.dropna(subset=["date"], inplace=True)
        with alchemyEngine.begin() as conn:
            update_on_conflict(
                table_def_fund_etf_perf_em(),
                conn,
                fund_exchange_rank_em_df,
                ["fundcode"],
            )
        return len(fund_exchange_rank_em_df)
    except Exception as e:
        logger.exception("failed to get ETF performance data")


def etf_spot():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    logger.info("running fund_etf_spot_em()...")
    try:
        df = ak.fund_etf_spot_em()
        df = df[
            [
                "代码",
                "名称",
                "最新价",
                "IOPV实时估值",
                "基金折价率",
                "涨跌额",
                "涨跌幅",
                "成交量",
                "成交额",
                "开盘价",
                "最高价",
                "最低价",
                "昨收",
                "换手率",
                "量比",
                "委比",
                "外盘",
                "内盘",
                "主力净流入-净额",
                "主力净流入-净占比",
                "超大单净流入-净额",
                "超大单净流入-净占比",
                "大单净流入-净额",
                "大单净流入-净占比",
                "中单净流入-净额",
                "中单净流入-净占比",
                "小单净流入-净额",
                "小单净流入-净占比",
                "流通市值",
                "总市值",
                "最新份额",
                "数据日期",
                "更新时间",
            ]
        ]
        saveAsCsv("fund_etf_spot_em", df)
        # Rename the columns of df to match the table's column names
        df.rename(
            columns={
                "数据日期": "date",
                "更新时间": "update_time",
                "代码": "code",
                "名称": "name",
                "最新价": "latest_price",
                "IOPV实时估值": "iopv",
                "基金折价率": "fund_discount_rate",
                "涨跌额": "change_amount",
                "涨跌幅": "change_rate",
                "成交量": "volume",
                "成交额": "turnover",
                "开盘价": "opening_price",
                "最高价": "highest_price",
                "最低价": "lowest_price",
                "昨收": "previous_close",
                "换手率": "turnover_rate",
                "量比": "volume_ratio",
                "委比": "order_ratio",
                "外盘": "external_disc",
                "内盘": "internal_disc",
                "主力净流入-净额": "main_force_net_inflow_amount",
                "主力净流入-净占比": "main_force_net_inflow_ratio",
                "超大单净流入-净额": "super_large_net_inflow_amount",
                "超大单净流入-净占比": "super_large_net_inflow_ratio",
                "大单净流入-净额": "large_net_inflow_amount",
                "大单净流入-净占比": "large_net_inflow_ratio",
                "中单净流入-净额": "medium_net_inflow_amount",
                "中单净流入-净占比": "medium_net_inflow_ratio",
                "小单净流入-净额": "small_net_inflow_amount",
                "小单净流入-净占比": "small_net_inflow_ratio",
                "流通市值": "circulating_market_value",
                "总市值": "total_market_value",
                "最新份额": "latest_shares",
            }, inplace=True
        )
        with alchemyEngine.begin() as conn:
            update_on_conflict(table_def_fund_etf_spot_em(), conn, df, ["code", "date"])

        return len(df)
    except Exception as e:
        logger.exception("failed to get ETF spot data")


def stock_zh_spot():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    logger.info("running stock_zh_spot()...")

    retry_attempts = 3
    retry_delay = 5  # seconds

    for attempt in range(retry_attempts):
        try:
            stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()
            break
        except Exception as e:
            logger.warn(f"Attempt {attempt+1} failed with error: {e}")
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
            else:
                raise

    stock_zh_a_spot_em_df.rename(
        columns={
            "序号": "serial_no",
            "代码": "symbol",
            "名称": "name",
            "最新价": "latest_price",
            "涨跌幅": "price_change_pct",
            "涨跌额": "price_change_amt",
            "成交量": "volume",
            "成交额": "turnover",
            "振幅": "amplitude",
            "最高": "highest",
            "最低": "lowest",
            "今开": "open_today",
            "昨收": "close_yesterday",
            "量比": "volume_ratio",
            "换手率": "turnover_rate",
            "市盈率-动态": "pe_ratio_dynamic",
            "市净率": "pb_ratio",
            "总市值": "total_market_value",
            "流通市值": "circulating_market_value",
            "涨速": "rise_speed",
            "5分钟涨跌": "five_min_change",
            "60日涨跌幅": "sixty_day_change_pct",
            "年初至今涨跌幅": "ytd_change_pct",
        },
        inplace=True,
    )

    with alchemyEngine.begin() as conn:
        update_on_conflict(stock_zh_a_spot_em, conn, stock_zh_a_spot_em_df, ["symbol"])

    return stock_zh_a_spot_em_df[["symbol", "name"]]


def get_stock_daily(symbol):
    worker = get_worker()
    alchemyEngine = worker.alchemyEngine

    with alchemyEngine.begin() as conn:
        latest_date = get_latest_date(conn, symbol, "stock_zh_a_hist_em")

        start_date = "19700101"  # For entire history.
        if latest_date is not None:
            start_date = (latest_date - timedelta(days=30)).strftime("%Y%m%d")

        end_date = datetime.now().strftime("%Y%m%d")
        adjust = "hfq"

        stock_zh_a_hist_df = ak.stock_zh_a_hist(
            symbol, "daily", start_date, end_date, adjust
        )

        if stock_zh_a_hist_df.empty:
            return None

        stock_zh_a_hist_df.loc[:, "symbol"] = symbol
        stock_zh_a_hist_df.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "turnover",
                "振幅": "amplitude",
                "涨跌幅": "change_rate",
                "涨跌额": "change_amt",
                "换手率": "turnover_rate",
            },
            inplace=True,
        )

        ignore_on_conflict(
            stock_zh_a_hist_em, conn, stock_zh_a_hist_df, ["symbol", "date"]
        )

    return len(stock_zh_a_hist_df)


def stock_zh_daily_hist(stock_list, threads):
    worker = get_worker()
    logger = worker.logger

    logger.info("running stock_zh_daily_hist() for %s stocks", len(stock_list))

    futures = []
    with worker_client() as client:
        for symbol in stock_list["symbol"]:
            futures.append(client.submit(get_stock_daily, symbol, priority=1))
            await_futures(futures, False, multiplier=threads)

    await_futures(futures)

    return len(stock_list)


def get_sge_spot_daily(symbol):
    worker = get_worker()
    alchemyEngine = worker.alchemyEngine

    with alchemyEngine.begin() as conn:
        spot_hist_sge_df = ak.spot_hist_sge(symbol=symbol)

        if spot_hist_sge_df.empty:
            return None

        latest_date = get_latest_date(conn, symbol, "spot_hist_sge")

        if latest_date is not None:
            start_date = latest_date - timedelta(days=20)
            spot_hist_sge_df = spot_hist_sge_df[spot_hist_sge_df["date"] >= start_date]

        spot_hist_sge_df.insert(0, "symbol", symbol)

        ignore_on_conflict(spot_hist_sge, conn, spot_hist_sge_df, ["symbol", "date"])

    return len(spot_hist_sge_df)


def sge_spot_daily_hist(spot_list):
    worker = get_worker()
    logger = worker.logger

    logger.info("running sge_spot_daily_hist() for %s spot", len(spot_list))

    futures = []
    with worker_client() as client:
        for symbol in spot_list["product"]:
            futures.append(client.submit(get_sge_spot_daily, symbol, priority=1))
            await_futures(futures, False)

    await_futures(futures)

    return len(spot_list)


def sge_spot():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    logger.info("running sge_spot()...")

    ssts = ak.spot_symbol_table_sge()

    ssts.rename(columns={"序号": "serial", "品种": "product"}, inplace=True)

    with alchemyEngine.begin() as conn:
        update_on_conflict(spot_symbol_table_sge, conn, ssts, ["product"])

    return ssts


def rmb_exchange_rates():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    logger.info("running rmb_exchange_rates()...")

    currency_boc_safe_df = ak.currency_boc_safe()

    currency_boc_safe_df.rename(
        columns={
            "日期": "date",
            "美元": "usd",
            "欧元": "eur",
            "日元": "jpy",
            "港元": "hkd",
            "英镑": "gbp",
            "澳元": "aud",
            "新西兰元": "nzd",
            "新加坡元": "sgd",
            "瑞士法郎": "chf",
            "加元": "cad",
            "林吉特": "myr",
            "卢布": "rub",
            "兰特": "zar",
            "韩元": "krw",
            "迪拉姆": "aed",
            "里亚尔": "qar",
            "福林": "huf",
            "兹罗提": "pln",
            "丹麦克朗": "dkk",
            "瑞典克朗": "sek",
            "挪威克朗": "nok",
            "里拉": "try",
            "比索": "php",
            "泰铢": "thb",
            "澳门元": "mop",
        },
        inplace=True,
    )

    with alchemyEngine.begin() as conn:
        update_on_conflict(currency_boc_safe, conn, currency_boc_safe_df, ["date"])

    return len(currency_boc_safe_df)


def get_cn_bond_index_metrics(symbol, symbol_cn):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    column_mapping = {
        '全价': 'fullprice',
        '净价': 'cleanprice',
        '财富': 'wealth',
        '平均市值法久期': 'avgmv_duration',
        '平均现金流法久期': 'avgcf_duration',
        '平均市值法凸性': 'avgmv_convexity',
        '平均现金流法凸性': 'avgcf_convexity',
        '平均现金流法到期收益率': 'avgcf_ytm',
        '平均市值法到期收益率': 'avgmv_ytm',
        '平均基点价值': 'avgbpv',
        '平均待偿期': 'avgmaturity',
        '平均派息率': 'avgcouponrate',
        '指数上日总市值': 'indexprevdaymv',
        '财富指数涨跌幅': 'wealthindex_change',
        '全价指数涨跌幅': 'fullpriceindex_change',
        '净价指数涨跌幅': 'cleanpriceindex_change',
        '现券结算量': 'spotsettlementvolume'
    }

    for indicator in list(column_mapping.keys()):
        try:
            df = ak.bond_new_composite_index_cbond(
                indicator=indicator, period=symbol_cn
            )
        except KeyError as e:
            logger.warning("%s - %s - %s could be empty: %s", symbol, symbol_cn, indicator, str(e))
            continue
        except Exception as e:
            logger.exception(
                "%s - %s - %s encountered error. skipping", symbol, symbol_cn, indicator
            )
            continue

        if df.empty:
            continue

        df = df.dropna(axis=1, how="all")
        if df.empty:
            continue

        value_col_name = column_mapping[indicator]
        df.rename(columns={"value": value_col_name}, inplace=True)

        start_date = None
        with alchemyEngine.begin() as conn:
            latest_date = get_latest_date(
                conn, symbol, "cn_bond_indices", value_col_name
            )
            if latest_date is not None:
                start_date = latest_date - timedelta(days=20)
                df = df[df["date"] >= start_date]
            df.insert(0, "symbol", symbol)
            update_on_conflict(cn_bond_indices, conn, df, ["symbol", "date"])

    return True


def cn_bond_index():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    logger.info("running cn_bond_index_periods()...")

    # Define the data as a list of tuples
    data = [
        ("totalvalue", "总值"),
        ("below1yr", "1年以下"),
        ("yr1to3", "1-3年"),
        ("yr3to5", "3-5年"),
        ("yr5to7", "5-7年"),
        ("yr7to10", "7-10年"),
        ("over10yr", "10年以上"),
        ("mo0to3", "0-3个月"),
        ("mo3to6", "3-6个月"),
        ("mo6to9", "6-9个月"),
        ("mo9to12", "9-12个月"),
        ("mo0to6", "0-6个月"),
        ("mo6to12", "6-12个月")
    ]

    # Create the DataFrame
    df = pd.DataFrame(data, columns=["symbol", "symbol_cn"])

    with alchemyEngine.begin() as conn:
        ignore_on_conflict(cn_bond_index_period, conn, df, ["symbol"])

    # submit tasks to get bond metrics for each period
    futures = []
    with worker_client() as client:
        logger.info("starting tasks on function get_cn_bond_index_metrics()...")
        for symbol, symbol_cn in data:
            futures.append(client.submit(get_cn_bond_index_metrics, symbol, symbol_cn, priority=1))
            await_futures(futures, False)

    await_futures(futures)

    return len(df)


def get_stock_bond_ratio_index():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    logger.info("running get_stock_bond_ratio_index()...")

    df = SnowballAPI.stock_bond_ratio_index()

    start_date = None
    with alchemyEngine.begin() as conn:
        latest_date = get_latest_date(
            conn, table="stock_bond_ratio_index")
        if latest_date is not None:
            start_date = latest_date - timedelta(days=20)
            df = df[df["date"] >= start_date]
        update_on_conflict(table_def_bond_metrics_em(), conn, df, ["date"])

    return len(df)
