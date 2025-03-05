import math
import time
import pandas as pd
import numpy as np
import warnings
import logging
import requests

logging.getLogger("NP.plotly").setLevel(logging.CRITICAL)
logging.getLogger("prophet.plot").disabled = True

import exchange_calendars as xcals

from tenacity import (
    stop_after_attempt,
    wait_exponential,
    Retrying,
)

# import exchange_calendars as xcals
from datetime import datetime, timedelta, date
from sqlalchemy import (
    text,
)


from sklearn.preprocessing import StandardScaler

# module_path = os.getenv("LOCAL_AKSHARE_DEV_MODULE")
# if module_path is not None and module_path not in sys.path:
# sys.path.insert(0, module_path)
import akshare as ak  # noqa: E402

from neuralprophet import (
    set_random_seed as np_random_seed,
    set_log_level,
    NeuralProphet,
)

from marten.utils.worker import await_futures
from marten.utils.trainer import select_device
from marten.data.tabledef import (
    table_def_index_daily_em,
    table_def_hk_index_daily_em,
    table_def_us_index_daily_sina,
    table_def_hk_index_spot_em,
    table_def_fund_etf_spot_em,
    table_def_index_spot_em,
    table_def_index_spot_sina,
    table_def_fund_etf_perf_em,
    table_def_fund_etf_list_sina,
    table_def_fund_etf_daily_em,
    table_def_bond_metrics_em,
    table_def_option_qvix,
    bond_zh_hs_spot,
    bond_zh_hs_daily,
    stock_zh_a_spot_em,
    stock_zh_a_hist_em,
    currency_boc_safe,
    spot_symbol_table_sge,
    spot_hist_sge,
    cn_bond_index_period,
    cn_bond_indices,
    fund_dividend_events,
    fund_portfolio_holdings,
    interbank_rate_list,
    interbank_rate_hist,
    etf_cash_inflow,
)
from marten.data.db import update_on_conflict, ignore_on_conflict, get_max_for_column
from marten.data.api.snowball import SnowballAPI
from marten.data.api.em import EastMoneyAPI
from marten.data.api.sina import SinaAPI

from dask.distributed import worker_client, get_worker, Variable

from functools import lru_cache

original_http_get = requests.get

@lru_cache()
def last_trade_date():
    xshg = xcals.get_calendar("XSHG")
    current_date = datetime.now().date()
    # Iterate backwards from current_date until a valid session is found
    last_session = current_date
    while not xshg.is_session(last_session):
        last_session -= timedelta(days=1)
    return last_session

class TimeoutHTTPAdapter(requests.adapters.HTTPAdapter):
    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.pop("timeout", None)
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        timeout = kwargs.get("timeout")
        if timeout is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)

def patch_requests_get():
    session = requests.Session()
    adapter = TimeoutHTTPAdapter(timeout=(15, 60))
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    def timeout_get(*args, **kwargs):
        return session.get(*args, **kwargs)

    # Monkey-patch requests.get in akshare
    requests.get = timeout_get

def restore_requests_get():
    global original_get
    requests.get = original_get

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
    with alchemyEngine.connect() as conn:
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

        shide.rename(
            columns={
                "latest": "close",
            },
            inplace=True,
        )
        # Convert the 'date' column to datetime
        shide.loc[:, "date"] = pd.to_datetime(shide["date"]).dt.date

        with alchemyEngine.connect() as conn:
            # latest_date = get_max_for_column(conn, symbol, "hk_index_daily_em")
            latest_dates = [
                get_max_for_column(conn, symbol, "hk_index_daily_em", non_null_col=c)
                for c in [
                    "change_rate",
                    "open_preclose_rate",
                    "high_preclose_rate",
                    "low_preclose_rate",
                ]
            ]

        latest_date = None if None in latest_dates else min(latest_dates)

        if latest_date is not None:
            ## keep rows only with `date` later than the latest record in database.
            shide = shide[shide["date"] > (latest_date - timedelta(days=10))]

        # calculate all change rates
        if len(shide) > 1:
            shide.sort_values(["date"], inplace=True)
            shide["lag_close"] = shide["close"].shift(1)
            shide["change_rate"] = (
                (shide["close"] - shide["lag_close"]) / shide["lag_close"] * 100
            ).round(5)
            shide["open_preclose_rate"] = (
                (shide["open"] - shide["lag_close"]) / shide["lag_close"] * 100
            ).round(5)
            shide["high_preclose_rate"] = (
                (shide["high"] - shide["lag_close"]) / shide["lag_close"] * 100
            ).round(5)
            shide["low_preclose_rate"] = (
                (shide["low"] - shide["lag_close"]) / shide["lag_close"] * 100
            ).round(5)

            shide.drop(["lag_close"], axis=1, inplace=True)

            # if latest_date is not None, drop the first row
            if latest_date is not None:
                shide.drop(shide.index[0], inplace=True)

        shide.replace([np.inf, -np.inf, np.nan], None, inplace=True)

        shide.insert(0, "symbol", symbol)

        with alchemyEngine.begin() as conn:
            update_on_conflict(
                table_def_hk_index_daily_em(), conn, shide, ["symbol", "date"]
            )

        return len(shide)
    except KeyError as e:
        logger.warning(
            "ak.stock_hk_index_daily_em(symbol=%s) could be empty: %s", symbol, str(e)
        )
        return 0
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

        # Convert iuss["date"] to datetime and normalize to date only
        iuss.loc[:, "date"] = pd.to_datetime(iuss["date"]).dt.date
        with alchemyEngine.connect() as conn:
            # latest_date = get_max_for_column(conn, symbol, "us_index_daily_sina")
            latest_dates = [
                get_max_for_column(conn, symbol, "us_index_daily_sina", non_null_col=c)
                for c in [
                    "change_rate",
                    "open_preclose_rate",
                    "high_preclose_rate",
                    "low_preclose_rate",
                    "vol_change_rate",
                    "amt_change_rate",
                ]
            ]

        latest_date = None if None in latest_dates else min(latest_dates)

        if latest_date is not None:
            iuss = iuss[iuss["date"] > (latest_date - timedelta(days=10))]

        # calculate all change rates
        if len(iuss) > 1:
            iuss.sort_values(["date"], inplace=True)
            iuss["lag_close"] = iuss["close"].shift(1)
            iuss["lag_volume"] = iuss["volume"].shift(1)
            iuss["lag_amount"] = iuss["amount"].shift(1)
            iuss["change_rate"] = (
                (iuss["close"] - iuss["lag_close"]) / iuss["lag_close"] * 100
            ).round(5)
            iuss["open_preclose_rate"] = (
                (iuss["open"] - iuss["lag_close"]) / iuss["lag_close"] * 100
            ).round(5)
            iuss["high_preclose_rate"] = (
                (iuss["high"] - iuss["lag_close"]) / iuss["lag_close"] * 100
            ).round(5)
            iuss["low_preclose_rate"] = (
                (iuss["low"] - iuss["lag_close"]) / iuss["lag_close"] * 100
            ).round(5)
            iuss["vol_change_rate"] = (
                (iuss["volume"] - iuss["lag_volume"]) / iuss["lag_volume"] * 100
            ).round(5)
            iuss["amt_change_rate"] = (
                (iuss["amount"] - iuss["lag_amount"]) / iuss["lag_amount"] * 100
            ).round(5)

            iuss.drop(
                ["lag_close", "lag_volume", "lag_amount"],
                axis=1,
                inplace=True,
            )

            # if latest_date is not None, drop the first row
            if latest_date is not None:
                iuss.drop(iuss.index[0], inplace=True)

        iuss.replace([np.inf, -np.inf, np.nan], None, inplace=True)

        iuss.insert(0, "symbol", symbol)

        with alchemyEngine.begin() as conn:
            update_on_conflict(
                table_def_us_index_daily_sina(), conn, iuss, ["symbol", "date"]
            )
        return len(iuss)
    except Exception as e:
        logger.error(
            f"failed to update us_index_daily_sina for {symbol}", exc_info=True
        )
        raise e


def option_qvix():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    def get_qvix(qvix_func):
        nonlocal logger
        try:
            return qvix_func()
        except Exception as e:
            logger.warning(
                "failed to get qvix %s: %s", qvix_func.__name__, e, exc_info=True
            )
            return pd.DataFrame()

    try:
        qvix50 = get_qvix(ak.index_option_50etf_qvix)
        if not qvix50.empty:
            with alchemyEngine.connect() as conn:
                latest_date = get_max_for_column(conn, "50etf", "option_qvix")
            if latest_date:
                ## keep rows only with `date` later than the latest record in database.
                qvix50 = qvix50[qvix50["date"] > (latest_date - timedelta(days=10))]
            qvix50.insert(0, "symbol", "50etf")

        qvix300 = get_qvix(ak.index_option_300etf_qvix)
        if not qvix300.empty:
            with alchemyEngine.connect() as conn:
                latest_date = get_max_for_column(conn, "300etf", "option_qvix")
            if latest_date:
                qvix300 = qvix300[qvix300["date"] > (latest_date - timedelta(days=10))]
            qvix300.insert(0, "symbol", "300etf")

        qvix1000 = get_qvix(ak.index_option_1000index_qvix)
        if not qvix1000.empty:
            with alchemyEngine.connect() as conn:
                latest_date = get_max_for_column(conn, "1000index", "option_qvix")
            if latest_date:
                qvix1000 = qvix1000[
                    qvix1000["date"] > (latest_date - timedelta(days=10))
                ]
            qvix1000.insert(0, "symbol", "1000index")

        qvix100 = get_qvix(ak.index_option_100etf_qvix)
        if not qvix100.empty:
            with alchemyEngine.connect() as conn:
                latest_date = get_max_for_column(conn, "100etf", "option_qvix")
            if latest_date:
                qvix100 = qvix100[qvix100["date"] > (latest_date - timedelta(days=10))]
            qvix100.insert(0, "symbol", "100etf")

        qvix300index = get_qvix(ak.index_option_300index_qvix)
        if not qvix300index.empty:
            with alchemyEngine.connect() as conn:
                latest_date = get_max_for_column(conn, "300index", "option_qvix")
            if latest_date:
                qvix300index = qvix300index[
                    qvix300index["date"] > (latest_date - timedelta(days=10))
                ]
            qvix300index.insert(0, "symbol", "300index")

        qvix500 = get_qvix(ak.index_option_500etf_qvix)
        if not qvix500.empty:
            with alchemyEngine.connect() as conn:
                latest_date = get_max_for_column(conn, "500etf", "option_qvix")
            if latest_date:
                qvix500 = qvix500[qvix500["date"] > (latest_date - timedelta(days=10))]
            qvix500.insert(0, "symbol", "500etf")

        qvix50index = get_qvix(ak.index_option_50index_qvix)
        if not qvix50index.empty:
            with alchemyEngine.connect() as conn:
                latest_date = get_max_for_column(conn, "50index", "option_qvix")
            if latest_date:
                qvix50index = qvix50index[
                    qvix50index["date"] > (latest_date - timedelta(days=10))
                ]
            qvix50index.insert(0, "symbol", "50index")

        cyb = get_qvix(ak.index_option_cyb_qvix)
        if not cyb.empty:
            with alchemyEngine.connect() as conn:
                latest_date = get_max_for_column(conn, "cyb", "option_qvix")
            if latest_date:
                cyb = cyb[cyb["date"] > (latest_date - timedelta(days=10))]
            cyb.insert(0, "symbol", "cyb")

        kcb = get_qvix(ak.index_option_kcb_qvix)
        if not kcb.empty:
            with alchemyEngine.connect() as conn:
                latest_date = get_max_for_column(conn, "kcb", "option_qvix")
            if latest_date:
                kcb = kcb[kcb["date"] > (latest_date - timedelta(days=10))]
            kcb.insert(0, "symbol", "kcb")

        qvix = pd.concat(
            [
                qvix50,
                qvix300,
                qvix1000,
                qvix100,
                qvix300index,
                qvix500,
                qvix50index,
                cyb,
                kcb,
            ],
            ignore_index=True,
        )
        qvix.replace({np.nan: None}, inplace=True)

        with alchemyEngine.begin() as conn:
            update_on_conflict(table_def_option_qvix, conn, qvix, ["symbol", "date"])
        return len(qvix)
    except Exception as e:
        logger.error("failed to get option qvix: %s", e, exc_info=True)
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
        update_query = text(
            """
            UPDATE bond_zh_hs_spot 
            SET last_checked = CURRENT_TIMESTAMP
            WHERE symbol = :symbol 
        """
        )
        params = {"symbol": symbol}
        conn.execute(update_query, params)

    try:
        bzhd = SinaAPI.bond_zh_hs_daily(symbol)

        # if shide is empty, return immediately
        if bzhd.empty:
            logger.warning("bond daily history data is empty: %s", symbol)
            return None

        with alchemyEngine.connect() as conn:
            latest_dates = [
                get_max_for_column(conn, symbol, "bond_zh_hs_daily", non_null_col=c)
                for c in [
                    "change_rate",
                    "open_preclose_rate",
                    "high_preclose_rate",
                    "low_preclose_rate",
                    "vol_change_rate",
                ]
            ]

        latest_date = None if None in latest_dates else min(latest_dates)

        if latest_date is not None:
            ## keep rows only with `date` later than the latest record in database.
            bzhd = bzhd[bzhd["date"] > (latest_date - timedelta(days=10))]

        # calculate all change rates
        if len(bzhd) > 1:
            bzhd.sort_values(["date"], inplace=True)
            bzhd["lag_close"] = bzhd["close"].shift(1)
            bzhd["lag_volume"] = bzhd["volume"].shift(1)
            bzhd["change_rate"] = (
                (bzhd["close"] - bzhd["lag_close"]) / bzhd["lag_close"] * 100
            ).round(5)
            bzhd["open_preclose_rate"] = (
                (bzhd["open"] - bzhd["lag_close"]) / bzhd["lag_close"] * 100
            ).round(5)
            bzhd["high_preclose_rate"] = (
                (bzhd["high"] - bzhd["lag_close"]) / bzhd["lag_close"] * 100
            ).round(5)
            bzhd["low_preclose_rate"] = (
                (bzhd["low"] - bzhd["lag_close"]) / bzhd["lag_close"] * 100
            ).round(5)
            bzhd["vol_change_rate"] = (
                (bzhd["volume"] - bzhd["lag_volume"]) / bzhd["lag_volume"] * 100
            ).round(5)

            bzhd.drop(["lag_close", "lag_volume"], axis=1, inplace=True)

            # if latest_date is not None, drop the first row
            if latest_date is not None:
                bzhd.drop(bzhd.index[0], inplace=True)

        bzhd.replace([np.inf, -np.inf, np.nan], None, inplace=True)

        bzhd.insert(0, "symbol", symbol)

        with alchemyEngine.begin() as conn:
            update_on_conflict(bond_zh_hs_daily, conn, bzhd, ["symbol", "date"])
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
            (SELECT symbol, turnover, last_checked FROM bond_zh_hs_spot WHERE turnover = 0 and last_checked IS NULL LIMIT 500)
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
                ) order by last_checked asc limit 500
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
        """
            select
                symbol, src
            from
                index_spot_em ise
            union
            select
                right(symbol, length(symbol)-2) symbol,
                left(symbol, 2) src
            from
                index_spot_sina iss
        """,
        alchemyEngine,
    )
    logger.info(
        "starting tasks on function stock_zh_index_daily_em(), length: %s",
        len(cn_index_fulllist),
    )

    futures = []
    with worker_client() as client:
        for symbol, src in zip(cn_index_fulllist["symbol"], cn_index_fulllist["src"]):
            futures.append(
                client.submit(stock_zh_index_daily_em, symbol, src, priority=1)
            )
            await_futures(futures, False)

        logger.debug("cn_index_daily futures before final wait: %s", len(futures))
        await_futures(futures)
        logger.debug("cn_index_daily futures after final wait: %s", len(futures))

    return len(cn_index_fulllist)


def stock_zh_index_daily_em(symbol, src):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    try:
        with alchemyEngine.connect() as conn:
            # latest_date = get_max_for_column(conn, symbol, "index_daily_em")
            latest_dates = [
                get_max_for_column(conn, symbol, "index_daily_em", non_null_col=c)
                for c in [
                    "open_preclose_rate",
                    "high_preclose_rate",
                    "low_preclose_rate",
                    "vol_change_rate",
                    "change_rate",
                    "amt_change_rate",
                ]
            ]

        latest_date = None if None in latest_dates else min(latest_dates)

        start_date = "19900101"  # For entire history.
        if latest_date is not None:
            start_date = (latest_date - timedelta(days=10)).strftime("%Y%m%d")

        end_date = datetime.now().strftime("%Y%m%d")

        logger.debug(
            f"calling ak.stock_zh_index_daily_em({src}{symbol}, {start_date}, {end_date})"
        )
        szide = ak.stock_zh_index_daily_em(f"{src}{symbol}", start_date, end_date)

        # if shide is empty, return immediately
        if szide.empty:
            logger.warning("index data is empty: %s", symbol)
            return None

        # calculate all change rates
        if len(szide) > 1:
            szide.sort_values(["date"], inplace=True)
            szide["lag_amount"] = szide["amount"].shift(1)
            szide["lag_close"] = szide["close"].shift(1)
            szide["lag_volume"] = szide["volume"].shift(1)

            szide["amt_change_rate"] = (
                (szide["amount"] - szide["lag_amount"]) / szide["lag_amount"] * 100
            ).round(5)
            szide["open_preclose_rate"] = (
                (szide["open"] - szide["lag_close"]) / szide["lag_close"] * 100
            ).round(5)
            szide["high_preclose_rate"] = (
                (szide["high"] - szide["lag_close"]) / szide["lag_close"] * 100
            ).round(5)
            szide["low_preclose_rate"] = (
                (szide["low"] - szide["lag_close"]) / szide["lag_close"] * 100
            ).round(5)
            szide["vol_change_rate"] = (
                (szide["volume"] - szide["lag_volume"]) / szide["lag_volume"] * 100
            ).round(5)
            szide["change_rate"] = (
                (szide["close"] - szide["lag_close"]) / szide["lag_close"] * 100
            ).round(5)

            szide.drop(["lag_amount", "lag_close", "lag_volume"], axis=1, inplace=True)

            # if latest_date is not None, drop the first row
            if latest_date is not None:
                szide.drop(szide.index[0], inplace=True)

        szide.replace([np.inf, -np.inf, np.nan], None, inplace=True)

        szide.insert(0, "symbol", symbol)

        with alchemyEngine.begin() as conn:
            update_on_conflict(
                table_def_index_daily_em(), conn, szide, ["symbol", "date"]
            )
        return len(szide)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            logger.warning(
                "ak.stock_zh_index_daily_em(%s%s, %s, %s) - data source could be empty: %s",
                src,
                symbol,
                start_date,
                end_date,
                str(e),
            )
            return 0
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
    logger.info(
        "starting task on function stock_zh_index_spot_em() and stock_zh_index_spot_sina()..."
    )
    ##loop thru cn_index_types and send off further tasks to client
    futures = []
    with worker_client() as client:
        for symbol, src in cn_index_types:
            futures.append(client.submit(stock_zh_index_spot_em, symbol, src))
            await_futures(futures, False)
        futures.append(client.submit(stock_zh_index_spot_sina))
        await_futures(futures)
    return True


def stock_zh_index_spot_sina():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    try:
        sziss = ak.stock_zh_index_spot_sina()
        sziss = sziss.rename(
            columns={
                "代码": "symbol",
                "名称": "name",
                "最新价": "close",
                "涨跌幅": "change_rate",
                "涨跌额": "change_amount",
                "成交量": "volume",
                "成交额": "amount",
                "最高": "high",
                "最低": "low",
                "今开": "open",
                "昨收": "prev_close",
            }
        )
        with alchemyEngine.begin() as conn:
            update_on_conflict(table_def_index_spot_sina(), conn, sziss, ["symbol"])
        return len(sziss)
    except Exception as e:
        logger.error(f"failed to update index_spot_sina", exc_info=True)
        raise e


def calc_etf_metrics(symbol, end_date):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    interval = 250  # assume 250 trading days annualy
    try:
        with alchemyEngine.connect() as conn:
            # load the latest (top) `interval` records of historical market data records from `fund_etf_daily_em` table for `symbol`, order by `date`.
            # select columns: date, change_rate
            query = """SELECT date, change_rate FROM fund_etf_daily_em WHERE symbol = '{}' ORDER BY date DESC LIMIT {}""".format(
                symbol, interval
            )
            df = pd.read_sql(query, conn, parse_dates=["date"])

            if df.empty:
                return 0

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
        df.loc[:, "cumulative_returns"] = np.cumprod(1 + df["change_rate"] / 100.0) - 1
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

        with alchemyEngine.begin() as conn:
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

    with alchemyEngine.connect() as conn:
        result = conn.execute(text("select symbol from fund_etf_list_sina"))
        result_set = result.fetchall()
        etf_list = [row[0] for row in result_set]

    ##submit tasks to calculate metrics for each symbol
    futures = []
    with worker_client() as client:
        logger.info(f"starting tasks on function calc_etf_metrics()...")
        for symbol in etf_list:
            futures.append(
                client.submit(calc_etf_metrics, symbol, end_date, priority=1)
            )
            await_futures(futures, False)

        await_futures(futures)

    return len(etf_list)


def bond_ir():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    logger.info(f"running bond_zh_us_rate()...")
    try:
        start_date = None  # For entire history.
        with alchemyEngine.connect() as conn:
            # latest_date = get_max_for_column(conn, None, "bond_metrics_em")
            latest_dates = [
                get_max_for_column(conn, None, "bond_metrics_em", non_null_col=c)
                for c in [
                    "china_yield_2y_change_rate",
                    "china_yield_5y_change_rate",
                    "china_yield_10y_change_rate",
                    "china_yield_30y_change_rate",
                    "china_yield_spread_10y_2y_change_rate",
                    "us_yield_2y_change_rate",
                    "us_yield_5y_change_rate",
                    "us_yield_10y_change_rate",
                    "us_yield_30y_change_rate",
                    "us_yield_spread_10y_2y_change_rate",
                ]
            ]

        latest_date = None if None in latest_dates else min(latest_dates)

        if latest_date is not None:
            start_date = (latest_date - timedelta(days=20)).strftime("%Y%m%d")

        bzur = ak.bond_zh_us_rate(start_date)
        bzur.rename(
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
            },
            inplace=True,
        )

        # calculate all change rates
        if len(bzur) > 1:
            bzur.sort_values(["date"], inplace=True)

            bzur["lag_china_yield_2y"] = bzur["china_yield_2y"].shift(1)
            bzur["lag_china_yield_5y"] = bzur["china_yield_5y"].shift(1)
            bzur["lag_china_yield_10y"] = bzur["china_yield_10y"].shift(1)
            bzur["lag_china_yield_30y"] = bzur["china_yield_30y"].shift(1)
            bzur["lag_china_yield_spread_10y_2y"] = bzur[
                "china_yield_spread_10y_2y"
            ].shift(1)
            bzur["lag_us_yield_2y"] = bzur["us_yield_2y"].shift(1)
            bzur["lag_us_yield_5y"] = bzur["us_yield_5y"].shift(1)
            bzur["lag_us_yield_10y"] = bzur["us_yield_10y"].shift(1)
            bzur["lag_us_yield_30y"] = bzur["us_yield_30y"].shift(1)
            bzur["lag_us_yield_spread_10y_2y"] = bzur["us_yield_spread_10y_2y"].shift(1)

            bzur["china_yield_2y_change_rate"] = (
                (bzur["china_yield_2y"] - bzur["lag_china_yield_2y"])
                / bzur["lag_china_yield_2y"]
                * 100
            ).round(5)
            bzur["china_yield_5y_change_rate"] = (
                (bzur["china_yield_5y"] - bzur["lag_china_yield_5y"])
                / bzur["lag_china_yield_5y"]
                * 100
            ).round(5)
            bzur["china_yield_10y_change_rate"] = (
                (bzur["china_yield_10y"] - bzur["lag_china_yield_10y"])
                / bzur["lag_china_yield_10y"]
                * 100
            ).round(5)
            bzur["china_yield_30y_change_rate"] = (
                (bzur["china_yield_30y"] - bzur["lag_china_yield_30y"])
                / bzur["lag_china_yield_30y"]
                * 100
            ).round(5)
            bzur["china_yield_spread_10y_2y_change_rate"] = (
                (
                    bzur["china_yield_spread_10y_2y"]
                    - bzur["lag_china_yield_spread_10y_2y"]
                )
                / bzur["lag_china_yield_spread_10y_2y"]
                * 100
            ).round(5)
            bzur["us_yield_2y_change_rate"] = (
                (bzur["us_yield_2y"] - bzur["lag_us_yield_2y"])
                / bzur["lag_us_yield_2y"]
                * 100
            ).round(5)
            bzur["us_yield_5y_change_rate"] = (
                (bzur["us_yield_5y"] - bzur["lag_us_yield_5y"])
                / bzur["lag_us_yield_5y"]
                * 100
            ).round(5)
            bzur["us_yield_10y_change_rate"] = (
                (bzur["us_yield_10y"] - bzur["lag_us_yield_10y"])
                / bzur["lag_us_yield_10y"]
                * 100
            ).round(5)
            bzur["us_yield_30y_change_rate"] = (
                (bzur["us_yield_30y"] - bzur["lag_us_yield_30y"])
                / bzur["lag_us_yield_30y"]
                * 100
            ).round(5)
            bzur["us_yield_spread_10y_2y_change_rate"] = (
                (bzur["us_yield_spread_10y_2y"] - bzur["lag_us_yield_spread_10y_2y"])
                / bzur["lag_us_yield_spread_10y_2y"]
                * 100
            ).round(5)

            bzur.drop(
                [
                    "lag_china_yield_2y",
                    "lag_china_yield_5y",
                    "lag_china_yield_10y",
                    "lag_china_yield_30y",
                    "lag_china_yield_spread_10y_2y",
                    "lag_us_yield_2y",
                    "lag_us_yield_5y",
                    "lag_us_yield_10y",
                    "lag_us_yield_30y",
                    "lag_us_yield_spread_10y_2y",
                ],
                axis=1,
                inplace=True,
            )

            # if latest_date is not None, drop the first row
            if latest_date is not None:
                bzur.drop(bzur.index[0], inplace=True)

        bzur.replace([np.inf, -np.inf, np.nan], None, inplace=True)

        with alchemyEngine.begin() as conn:
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
        with alchemyEngine.connect() as conn:
            # check latest date on fund_etf_daily_em
            # latest_date = get_max_for_column(conn, symbol, "fund_etf_daily_em")
            latest_dates = [
                get_max_for_column(conn, symbol, "fund_etf_daily_em", non_null_col=c)
                for c in [
                    "turnover_change_rate",
                    "open_preclose_rate",
                    "high_preclose_rate",
                    "low_preclose_rate",
                    "vol_change_rate",
                ]
            ]
        latest_date = None if None in latest_dates else min(latest_dates)

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

        df.insert(0, "symbol", symbol)
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

        # calculate all change rates
        if len(df) > 1:
            df.sort_values(["date"], inplace=True)
            df["lag_turnover"] = df["turnover"].shift(1)
            df["lag_close"] = df["close"].shift(1)
            df["lag_volume"] = df["volume"].shift(1)
            df["turnover_change_rate"] = (
                (df["turnover"] - df["lag_turnover"]) / df["lag_turnover"] * 100
            ).round(5)
            df["open_preclose_rate"] = (
                (df["open"] - df["lag_close"]) / df["lag_close"] * 100
            ).round(5)
            df["high_preclose_rate"] = (
                (df["high"] - df["lag_close"]) / df["lag_close"] * 100
            ).round(5)
            df["low_preclose_rate"] = (
                (df["low"] - df["lag_close"]) / df["lag_close"] * 100
            ).round(5)
            df["vol_change_rate"] = (
                (df["volume"] - df["lag_volume"]) / df["lag_volume"] * 100
            ).round(5)

            df.drop(["lag_turnover", "lag_close", "lag_volume"], axis=1, inplace=True)

            # if latest_date is not None, drop the first row
            if latest_date is not None:
                df.drop(df.index[0], inplace=True)

        df.replace([np.inf, -np.inf, np.nan], None, inplace=True)

        with alchemyEngine.begin() as conn:
            update_on_conflict(
                table_def_fund_etf_daily_em(), conn, df, ["symbol", "date"]
            )

        return len(df)
    except Exception as e:
        logger.error(
            f"failed to get daily trade history data for {symbol}", exc_info=True
        )
        raise e


def etf_list(etf_spot_df):
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

        etf_spot_df.rename(columns={"code": "symbol"}, inplace=True)
        etf_spot_df.loc[etf_spot_df["symbol"].str.startswith("5"), "exch"] = "sh"
        etf_spot_df.loc[etf_spot_df["symbol"].str.startswith("1"), "exch"] = "sz"

        df = df.merge(etf_spot_df, on=["symbol"], how="outer")
        # Using combine_first() to merge columns
        df.loc[:, "exch"] = df["exch_x"].combine_first(df["exch_y"])
        df.drop(["exch_x", "exch_y"], axis=1, inplace=True)

        ## get historical data and holdings for each ETF
        futures = []
        with worker_client() as client:
            logger.info("starting sub-tasks for each ETF. Length: %s", len(df))
            for symbol, exch in zip(df["symbol"], df["exch"]):
                futures.append(client.submit(get_etf_daily, symbol, priority=1))
                futures.append(client.submit(fund_holding, symbol, priority=1))
                futures.append(client.submit(cash_inflow, symbol, exch, priority=1))
                await_futures(futures, False, multiplier=3)

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
            },
            inplace=True,
        )
        with alchemyEngine.begin() as conn:
            update_on_conflict(table_def_fund_etf_spot_em(), conn, df, ["code", "date"])

        return df[["code", "name", "date"]]
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
            logger.warning(f"Attempt {attempt+1} failed with error: {e}")
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

    with alchemyEngine.connect() as conn:
        # latest_date = get_max_for_column(conn, symbol, "stock_zh_a_hist_em")
        latest_dates = [
            get_max_for_column(conn, symbol, "stock_zh_a_hist_em", non_null_col=c)
            for c in [
                "turnover_change_rate",
                "open_preclose_rate",
                "high_preclose_rate",
                "low_preclose_rate",
                "vol_change_rate",
            ]
        ]

    latest_date = None if None in latest_dates else min(latest_dates)

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

    # stock_zh_a_hist_df.insert(0, "symbol", symbol)
    stock_zh_a_hist_df.rename(
        columns={
            "股票代码": "symbol",
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

    # calculate all change rates
    if len(stock_zh_a_hist_df) > 1:
        stock_zh_a_hist_df.sort_values(["symbol", "date"], inplace=True)
        stock_zh_a_hist_df["lag_turnover"] = stock_zh_a_hist_df["turnover"].shift(1)
        stock_zh_a_hist_df["lag_close"] = stock_zh_a_hist_df["close"].shift(1)
        stock_zh_a_hist_df["lag_volume"] = stock_zh_a_hist_df["volume"].shift(1)
        stock_zh_a_hist_df["turnover_change_rate"] = (
            (stock_zh_a_hist_df["turnover"] - stock_zh_a_hist_df["lag_turnover"])
            / stock_zh_a_hist_df["lag_turnover"]
            * 100
        ).round(5)
        stock_zh_a_hist_df["open_preclose_rate"] = (
            (stock_zh_a_hist_df["open"] - stock_zh_a_hist_df["lag_close"])
            / stock_zh_a_hist_df["lag_close"]
            * 100
        ).round(5)
        stock_zh_a_hist_df["high_preclose_rate"] = (
            (stock_zh_a_hist_df["high"] - stock_zh_a_hist_df["lag_close"])
            / stock_zh_a_hist_df["lag_close"]
            * 100
        ).round(5)
        stock_zh_a_hist_df["low_preclose_rate"] = (
            (stock_zh_a_hist_df["low"] - stock_zh_a_hist_df["lag_close"])
            / stock_zh_a_hist_df["lag_close"]
            * 100
        ).round(5)
        stock_zh_a_hist_df["vol_change_rate"] = (
            (stock_zh_a_hist_df["volume"] - stock_zh_a_hist_df["lag_volume"])
            / stock_zh_a_hist_df["lag_volume"]
            * 100
        ).round(5)

        stock_zh_a_hist_df.drop(
            ["lag_turnover", "lag_close", "lag_volume"], axis=1, inplace=True
        )

        # if latest_date is not None, drop the first row
        if latest_date is not None:
            stock_zh_a_hist_df.drop(stock_zh_a_hist_df.index[0], inplace=True)

    stock_zh_a_hist_df.replace([np.inf, -np.inf, np.nan], None, inplace=True)

    with alchemyEngine.begin() as conn:
        update_on_conflict(
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
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    with alchemyEngine.connect() as conn:
        # latest_date = get_max_for_column(conn, symbol, "spot_hist_sge")
        latest_dates = [
            get_max_for_column(conn, symbol, "spot_hist_sge", non_null_col=c)
            for c in [
                "change_rate",
                "open_preclose_rate",
                "high_preclose_rate",
                "low_preclose_rate",
            ]
        ]

    latest_date = None if None in latest_dates else min(latest_dates)

    spot_hist_sge_df = ak.spot_hist_sge(symbol=symbol)
    if spot_hist_sge_df.empty:
        return None

    if latest_date is not None:
        start_date = latest_date - timedelta(days=20)
        spot_hist_sge_df = spot_hist_sge_df[spot_hist_sge_df["date"] >= start_date]

    # calculate all change rates
    if len(spot_hist_sge_df) > 1:
        spot_hist_sge_df.sort_values(["date"], inplace=True)
        spot_hist_sge_df["lag_close"] = spot_hist_sge_df["close"].shift(1)
        spot_hist_sge_df["change_rate"] = (
            (spot_hist_sge_df["close"] - spot_hist_sge_df["lag_close"])
            / spot_hist_sge_df["lag_close"]
            * 100
        ).round(5)
        spot_hist_sge_df["open_preclose_rate"] = (
            (spot_hist_sge_df["open"] - spot_hist_sge_df["lag_close"])
            / spot_hist_sge_df["lag_close"]
            * 100
        ).round(5)
        spot_hist_sge_df["high_preclose_rate"] = (
            (spot_hist_sge_df["high"] - spot_hist_sge_df["lag_close"])
            / spot_hist_sge_df["lag_close"]
            * 100
        ).round(5)
        spot_hist_sge_df["low_preclose_rate"] = (
            (spot_hist_sge_df["low"] - spot_hist_sge_df["lag_close"])
            / spot_hist_sge_df["lag_close"]
            * 100
        ).round(5)

        spot_hist_sge_df.drop(["lag_close"], axis=1, inplace=True)
        # if latest_date is not None, drop the first row
        if latest_date is not None:
            spot_hist_sge_df.drop(spot_hist_sge_df.index[0], inplace=True)

    spot_hist_sge_df.replace([np.inf, -np.inf, np.nan], None, inplace=True)

    spot_hist_sge_df.insert(0, "symbol", symbol)

    try:
        with alchemyEngine.begin() as conn:
            update_on_conflict(
                spot_hist_sge, conn, spot_hist_sge_df, ["symbol", "date"]
            )
    except Exception as e:
        logger.error(
            "failed to save spot_hist_sge for symbol: %s, %s", symbol, e, exc_info=True
        )
        raise e

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

    with alchemyEngine.connect() as conn:
        latest_dates = [
            get_max_for_column(conn, None, "currency_boc_safe", non_null_col=c)
            for c in [
                "usd_change_rate",
                "eur_change_rate",
                "jpy_change_rate",
                "hkd_change_rate",
                "gbp_change_rate",
                "aud_change_rate",
                "nzd_change_rate",
                "sgd_change_rate",
                "chf_change_rate",
                "cad_change_rate",
                "myr_change_rate",
                "rub_change_rate",
                "zar_change_rate",
                "krw_change_rate",
                "aed_change_rate",
                "qar_change_rate",
                "huf_change_rate",
                "pln_change_rate",
                "dkk_change_rate",
                "sek_change_rate",
                "nok_change_rate",
                "try_change_rate",
                "php_change_rate",
                "thb_change_rate",
                "mop_change_rate",
            ]
        ]

    latest_date = None if None in latest_dates else min(latest_dates)

    if latest_date is not None:
        start_date = latest_date - timedelta(days=20)
        currency_boc_safe_df = currency_boc_safe_df[
            currency_boc_safe_df["date"] >= start_date
        ]

    cols = [
        "usd",
        "eur",
        "jpy",
        "hkd",
        "gbp",
        "aud",
        "nzd",
        "sgd",
        "chf",
        "cad",
        "myr",
        "rub",
        "zar",
        "krw",
        "aed",
        "qar",
        "huf",
        "pln",
        "dkk",
        "sek",
        "nok",
        "try",
        "php",
        "thb",
        "mop",
    ]

    # calculate all change rates
    if len(currency_boc_safe_df) > 1:
        currency_boc_safe_df.sort_values(["date"], inplace=True)
        for col in cols:
            lag_col = f"lag_{currency_boc_safe_df}"
            currency_boc_safe_df[lag_col] = currency_boc_safe_df[col].shift(1)
            currency_boc_safe_df[f"{col}_change_rate"] = (
                (currency_boc_safe_df[col] - currency_boc_safe_df[lag_col])
                / currency_boc_safe_df[lag_col]
                * 100
            ).round(5)
            currency_boc_safe_df.drop([lag_col], axis=1, inplace=True)
        # if latest_date is not None, drop the first row
        if latest_date is not None:
            currency_boc_safe_df.drop(currency_boc_safe_df.index[0], inplace=True)

    currency_boc_safe_df.replace([np.inf, -np.inf, np.nan], None, inplace=True)

    with alchemyEngine.begin() as conn:
        update_on_conflict(currency_boc_safe, conn, currency_boc_safe_df, ["date"])

    return len(currency_boc_safe_df)


def get_cn_bond_index_metrics(symbol, symbol_cn):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    column_mapping = {
        "全价": "fullprice",
        "净价": "cleanprice",
        "财富": "wealth",
        "平均市值法久期": "avgmv_duration",
        "平均现金流法久期": "avgcf_duration",
        "平均市值法凸性": "avgmv_convexity",
        "平均现金流法凸性": "avgcf_convexity",
        "平均现金流法到期收益率": "avgcf_ytm",
        "平均市值法到期收益率": "avgmv_ytm",
        "平均基点价值": "avgbpv",
        "平均待偿期": "avgmaturity",
        "平均派息率": "avgcouponrate",
        "指数上日总市值": "indexprevdaymv",
        "财富指数涨跌幅": "wealthindex_change",
        "全价指数涨跌幅": "fullpriceindex_change",
        "净价指数涨跌幅": "cleanpriceindex_change",
        "现券结算量": "spotsettlementvolume",
    }
    change_rates = [
        "avgmv_duration_change_rate",
        "avgcf_duration_change_rate",
        "avgmv_convexity_change_rate",
        "avgcf_convexity_change_rate",
        "avgcf_ytm_change_rate",
        "avgmv_ytm_change_rate",
        "avgbpv_change_rate",
        "avgmaturity_change_rate",
        "avgcouponrate_change_rate",
        "indexprevdaymv_change_rate",
        "spotsettlementvolume_change_rate",
    ]

    for indicator in list(column_mapping.keys()):
        try:
            df = ak.bond_new_composite_index_cbond(
                indicator=indicator, period=symbol_cn
            )
        except KeyError as e:
            logger.warning(
                "%s - %s - %s could be empty: %s", symbol, symbol_cn, indicator, str(e)
            )
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
        change_rate_col = next(
            (s for s in change_rates if s.startswith(value_col_name)), None
        )
        with alchemyEngine.connect() as conn:
            latest_date = get_max_for_column(
                conn,
                symbol,
                "cn_bond_indices",
                non_null_col=(change_rate_col if change_rate_col else value_col_name),
            )

        if latest_date is not None:
            start_date = latest_date - timedelta(days=20)
            df = df[df["date"] >= start_date]

        # calculate change rate
        if change_rate_col:
            lag_col = f"lag_{value_col_name}"
            df.sort_values(["date"], inplace=True)
            df[lag_col] = df[value_col_name].shift(1)
            df[change_rate_col] = (
                (df[value_col_name] - df[lag_col]) / df[lag_col] * 100
            ).round(5)
            df.drop([lag_col], axis=1, inplace=True)
            df.replace([np.inf, -np.inf, np.nan], None, inplace=True)
            # if latest_date is not None, drop the first row
            if latest_date is not None:
                df.drop(df.index[0], inplace=True)

        df.insert(0, "symbol", symbol)

        with alchemyEngine.begin() as conn:
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
        ("mo6to12", "6-12个月"),
    ]

    # Create the DataFrame
    df = pd.DataFrame(data, columns=["symbol", "symbol_cn"])

    with alchemyEngine.begin() as conn:
        update_on_conflict(cn_bond_index_period, conn, df, ["symbol"])

    # submit tasks to get bond metrics for each period
    futures = []
    with worker_client() as client:
        logger.info("starting tasks on function get_cn_bond_index_metrics()...")
        for symbol, symbol_cn in data:
            futures.append(
                client.submit(get_cn_bond_index_metrics, symbol, symbol_cn, priority=1)
            )
            await_futures(futures, False)

        await_futures(futures, task_timeout=180)

    return len(df)


def get_interbank_rate(symbol, market, symbol_type, indicator):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    try:
        df = ak.rate_interbank(market=market, symbol=symbol_type, indicator=indicator)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            logger.warning(
                "ak.rate_interbank(%s, %s, %s) - data source could be empty: %s",
                market,
                symbol_type,
                indicator,
                str(e),
            )
            return 0

    if df.empty:
        return 0

    df = df.dropna(axis=1, how="all")
    if df.empty:
        return 0

    df.rename(
        columns={"报告日": "date", "利率": "interest_rate", "涨跌": "change_rate"},
        inplace=True,
    )

    with alchemyEngine.connect() as conn:
        latest_date = get_max_for_column(
            conn, symbol, "interbank_rate_hist", non_null_col="change_rate"
        )

    if latest_date is not None:
        start_date = latest_date - timedelta(days=20)
        df = df[df["date"] >= start_date]

    df.insert(0, "symbol", symbol)

    with alchemyEngine.begin() as conn:
        update_on_conflict(interbank_rate_hist, conn, df, ["symbol", "date"])

    return len(df)


def interbank_rate():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    logger.info("running interbank_rate()...")

    # Define the data as a list of tuples
    data = [
        ("shibor_rmb_ovn", "上海银行同业拆借市场", "Shibor人民币", "隔夜"),
        ("shibor_rmb_1wk", "上海银行同业拆借市场", "Shibor人民币", "1周"),
        ("shibor_rmb_2wk", "上海银行同业拆借市场", "Shibor人民币", "2周"),
        ("shibor_rmb_1mo", "上海银行同业拆借市场", "Shibor人民币", "1月"),
        ("shibor_rmb_3mo", "上海银行同业拆借市场", "Shibor人民币", "3月"),
        ("shibor_rmb_6mo", "上海银行同业拆借市场", "Shibor人民币", "6月"),
        ("shibor_rmb_9mo", "上海银行同业拆借市场", "Shibor人民币", "9月"),
        ("shibor_rmb_1yr", "上海银行同业拆借市场", "Shibor人民币", "1年"),
        ("chibor_rmb_ovn", "中国银行同业拆借市场", "Chibor人民币", "隔夜"),
        ("chibor_rmb_1w", "中国银行同业拆借市场", "Chibor人民币", "1周"),
        ("chibor_rmb_2w", "中国银行同业拆借市场", "Chibor人民币", "2周"),
        ("chibor_rmb_3w", "中国银行同业拆借市场", "Chibor人民币", "3周"),
        ("chibor_rmb_1mo", "中国银行同业拆借市场", "Chibor人民币", "1月"),
        ("chibor_rmb_2mo", "中国银行同业拆借市场", "Chibor人民币", "2月"),
        ("chibor_rmb_3mo", "中国银行同业拆借市场", "Chibor人民币", "3月"),
        ("chibor_rmb_4mo", "中国银行同业拆借市场", "Chibor人民币", "4月"),
        ("chibor_rmb_6mo", "中国银行同业拆借市场", "Chibor人民币", "6月"),
        ("chibor_rmb_9mo", "中国银行同业拆借市场", "Chibor人民币", "9月"),
        ("chibor_rmb_1yr", "中国银行同业拆借市场", "Chibor人民币", "1年"),
        ("libor_gbp_ovn", "伦敦银行同业拆借市场", "Libor英镑", "隔夜"),
        ("libor_gbp_1w", "伦敦银行同业拆借市场", "Libor英镑", "1周"),
        ("libor_gbp_1mo", "伦敦银行同业拆借市场", "Libor英镑", "1月"),
        ("libor_gbp_2mo", "伦敦银行同业拆借市场", "Libor英镑", "2月"),
        ("libor_gbp_3mo", "伦敦银行同业拆借市场", "Libor英镑", "3月"),
        ("libor_gbp_8mo", "伦敦银行同业拆借市场", "Libor英镑", "8月"),
        ("libor_usd_ovn", "伦敦银行同业拆借市场", "Libor美元", "隔夜"),
        ("libor_usd_1w", "伦敦银行同业拆借市场", "Libor美元", "1周"),
        ("libor_usd_1mo", "伦敦银行同业拆借市场", "Libor美元", "1月"),
        ("libor_usd_2mo", "伦敦银行同业拆借市场", "Libor美元", "2月"),
        ("libor_usd_3mo", "伦敦银行同业拆借市场", "Libor美元", "3月"),
        ("libor_usd_8mo", "伦敦银行同业拆借市场", "Libor美元", "8月"),
        ("libor_eur_ovn", "伦敦银行同业拆借市场", "Libor欧元", "隔夜"),
        ("libor_eur_1w", "伦敦银行同业拆借市场", "Libor欧元", "1周"),
        ("libor_eur_1mo", "伦敦银行同业拆借市场", "Libor欧元", "1月"),
        ("libor_eur_2mo", "伦敦银行同业拆借市场", "Libor欧元", "2月"),
        ("libor_eur_3mo", "伦敦银行同业拆借市场", "Libor欧元", "3月"),
        ("libor_eur_8mo", "伦敦银行同业拆借市场", "Libor欧元", "8月"),
        ("libor_jpy_ovn", "伦敦银行同业拆借市场", "Libor日元", "隔夜"),
        ("libor_jpy_1w", "伦敦银行同业拆借市场", "Libor日元", "1周"),
        ("libor_jpy_1mo", "伦敦银行同业拆借市场", "Libor日元", "1月"),
        ("libor_jpy_2mo", "伦敦银行同业拆借市场", "Libor日元", "2月"),
        ("libor_jpy_3mo", "伦敦银行同业拆借市场", "Libor日元", "3月"),
        ("libor_jpy_8mo", "伦敦银行同业拆借市场", "Libor日元", "8月"),
        ("euribor_eur_1w", "欧洲银行同业拆借市场", "Euribor欧元", "1周"),
        ("euribor_eur_2w", "欧洲银行同业拆借市场", "Euribor欧元", "2周"),
        ("euribor_eur_3w", "欧洲银行同业拆借市场", "Euribor欧元", "3周"),
        ("euribor_eur_1mo", "欧洲银行同业拆借市场", "Euribor欧元", "1月"),
        ("euribor_eur_2mo", "欧洲银行同业拆借市场", "Euribor欧元", "2月"),
        ("euribor_eur_3mo", "欧洲银行同业拆借市场", "Euribor欧元", "3月"),
        ("euribor_eur_4mo", "欧洲银行同业拆借市场", "Euribor欧元", "4月"),
        ("euribor_eur_5mo", "欧洲银行同业拆借市场", "Euribor欧元", "5月"),
        ("euribor_eur_6mo", "欧洲银行同业拆借市场", "Euribor欧元", "6月"),
        ("euribor_eur_7mo", "欧洲银行同业拆借市场", "Euribor欧元", "7月"),
        ("euribor_eur_8mo", "欧洲银行同业拆借市场", "Euribor欧元", "8月"),
        ("euribor_eur_9mo", "欧洲银行同业拆借市场", "Euribor欧元", "9月"),
        ("euribor_eur_10mo", "欧洲银行同业拆借市场", "Euribor欧元", "10月"),
        ("euribor_eur_11mo", "欧洲银行同业拆借市场", "Euribor欧元", "11月"),
        ("euribor_eur_1yr", "欧洲银行同业拆借市场", "Euribor欧元", "1年"),
        ("hibor_hkd_ovn", "香港银行同业拆借市场", "Hibor港币", "隔夜"),
        ("hibor_hkd_1w", "香港银行同业拆借市场", "Hibor港币", "1周"),
        ("hibor_hkd_2w", "香港银行同业拆借市场", "Hibor港币", "2周"),
        ("hibor_hkd_1mo", "香港银行同业拆借市场", "Hibor港币", "1月"),
        ("hibor_hkd_2mo", "香港银行同业拆借市场", "Hibor港币", "2月"),
        ("hibor_hkd_3mo", "香港银行同业拆借市场", "Hibor港币", "3月"),
        ("hibor_hkd_4mo", "香港银行同业拆借市场", "Hibor港币", "4月"),
        ("hibor_hkd_5mo", "香港银行同业拆借市场", "Hibor港币", "5月"),
        ("hibor_hkd_6mo", "香港银行同业拆借市场", "Hibor港币", "6月"),
        ("hibor_hkd_7mo", "香港银行同业拆借市场", "Hibor港币", "7月"),
        ("hibor_hkd_8mo", "香港银行同业拆借市场", "Hibor港币", "8月"),
        ("hibor_hkd_9mo", "香港银行同业拆借市场", "Hibor港币", "9月"),
        ("hibor_hkd_10mo", "香港银行同业拆借市场", "Hibor港币", "10月"),
        ("hibor_hkd_11mo", "香港银行同业拆借市场", "Hibor港币", "11月"),
        ("hibor_hkd_1yr", "香港银行同业拆借市场", "Hibor港币", "1年"),
        ("hibor_usd_ovn", "香港银行同业拆借市场", "Hibor美元", "隔夜"),
        ("hibor_usd_1w", "香港银行同业拆借市场", "Hibor美元", "1周"),
        ("hibor_usd_2w", "香港银行同业拆借市场", "Hibor美元", "2周"),
        ("hibor_usd_1mo", "香港银行同业拆借市场", "Hibor美元", "1月"),
        ("hibor_usd_2mo", "香港银行同业拆借市场", "Hibor美元", "2月"),
        ("hibor_usd_3mo", "香港银行同业拆借市场", "Hibor美元", "3月"),
        ("hibor_usd_4mo", "香港银行同业拆借市场", "Hibor美元", "4月"),
        ("hibor_usd_5mo", "香港银行同业拆借市场", "Hibor美元", "5月"),
        ("hibor_usd_6mo", "香港银行同业拆借市场", "Hibor美元", "6月"),
        ("hibor_usd_7mo", "香港银行同业拆借市场", "Hibor美元", "7月"),
        ("hibor_usd_8mo", "香港银行同业拆借市场", "Hibor美元", "8月"),
        ("hibor_usd_9mo", "香港银行同业拆借市场", "Hibor美元", "9月"),
        ("hibor_usd_10mo", "香港银行同业拆借市场", "Hibor美元", "10月"),
        ("hibor_usd_11mo", "香港银行同业拆借市场", "Hibor美元", "11月"),
        ("hibor_usd_1yr", "香港银行同业拆借市场", "Hibor美元", "1年"),
        ("hibor_rmb_ovn", "香港银行同业拆借市场", "Hibor人民币", "隔夜"),
        ("hibor_rmb_1w", "香港银行同业拆借市场", "Hibor人民币", "1周"),
        ("hibor_rmb_2w", "香港银行同业拆借市场", "Hibor人民币", "2周"),
        ("hibor_rmb_1mo", "香港银行同业拆借市场", "Hibor人民币", "1月"),
        ("hibor_rmb_2mo", "香港银行同业拆借市场", "Hibor人民币", "2月"),
        ("hibor_rmb_3mo", "香港银行同业拆借市场", "Hibor人民币", "3月"),
        ("hibor_rmb_6mo", "香港银行同业拆借市场", "Hibor人民币", "6月"),
        ("hibor_rmb_1yr", "香港银行同业拆借市场", "Hibor人民币", "1年"),
        ("sibor_sgd_1mo", "新加坡银行同业拆借市场", "Sibor星元", "1月"),
        ("sibor_sgd_2mo", "新加坡银行同业拆借市场", "Sibor星元", "2月"),
        ("sibor_sgd_3mo", "新加坡银行同业拆借市场", "Sibor星元", "3月"),
        ("sibor_sgd_6mo", "新加坡银行同业拆借市场", "Sibor星元", "6月"),
        ("sibor_sgd_9mo", "新加坡银行同业拆借市场", "Sibor星元", "9月"),
        ("sibor_sgd_1yr", "新加坡银行同业拆借市场", "Sibor星元", "1年"),
        ("sibor_usd_1mo", "新加坡银行同业拆借市场", "Sibor美元", "1月"),
        ("sibor_usd_2mo", "新加坡银行同业拆借市场", "Sibor美元", "2月"),
        ("sibor_usd_3mo", "新加坡银行同业拆借市场", "Sibor美元", "3月"),
        ("sibor_usd_6mo", "新加坡银行同业拆借市场", "Sibor美元", "6月"),
        ("sibor_usd_9mo", "新加坡银行同业拆借市场", "Sibor美元", "9月"),
        ("sibor_usd_1yr", "新加坡银行同业拆借市场", "Sibor美元", "1年"),
    ]

    # Create the DataFrame
    df = pd.DataFrame(data, columns=["symbol", "market", "symbol_type", "indicator"])

    with alchemyEngine.begin() as conn:
        update_on_conflict(interbank_rate_list, conn, df, ["symbol"])

    # submit tasks to get interbank rate for each symbol
    futures = []
    with worker_client() as client:
        logger.info("starting tasks on function get_interbank_rate()...")
        for symbol, market, symbol_type, indicator in data:
            futures.append(
                client.submit(
                    get_interbank_rate,
                    symbol,
                    market,
                    symbol_type,
                    indicator,
                    priority=1,
                )
            )
            await_futures(futures, False)

        await_futures(futures)

    return len(df)


def get_fund_dividend_events():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    logger.info("running fund_dividend_events()...")

    df = ak.fund_fh_em()
    # with worker_client() as client:
    #     df = EastMoneyAPI.fund_fh_em(client, priority=1)

    df.drop(columns=["序号"], inplace=True)
    df.rename(
        columns={
            "基金代码": "symbol",
            "基金简称": "short_name",
            "权益登记日": "rights_registration_date",
            "除息日期": "ex_dividend_date",
            "分红": "dividend",
            "分红发放日": "dividend_payment_date",
        },
        inplace=True,
    )
    # remove column `id` from df

    # convert `NaT` values in `ex_dividend_date` or `dividend_payment_date` columns of `df` to None
    df[["ex_dividend_date", "dividend_payment_date"]] = df[
        ["ex_dividend_date", "dividend_payment_date"]
    ].where(df[["ex_dividend_date", "dividend_payment_date"]].notnull(), None)

    with alchemyEngine.connect() as conn:
        max_reg_date = get_max_for_column(
            conn,
            symbol=None,
            table="fund_dividend_events",
            col_for_max="rights_registration_date",
        )

    if max_reg_date is not None:
        df = df[df["rights_registration_date"] >= (max_reg_date - timedelta(days=30))]

    df = df.drop_duplicates(subset=["symbol", "rights_registration_date"])
    df.reset_index(drop=True, inplace=True)

    with alchemyEngine.begin() as conn:
        update_on_conflict(
            fund_dividend_events, conn, df, ["symbol", "rights_registration_date"]
        )

    return len(df)


def get_stock_bond_ratio_index():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger
    logger.info("running get_stock_bond_ratio_index()...")

    df = SnowballAPI.stock_bond_ratio_index()

    with alchemyEngine.connect() as conn:
        latest_date = get_max_for_column(
            conn,
            symbol=None,
            table="bond_metrics_em",
            non_null_col="performance_benchmark_change_rate",
        )

    if latest_date is not None:
        start_date = latest_date - timedelta(days=20)
        df = df[df["date"] >= start_date]

    # calculate all change rates
    if len(df) > 1:
        df.sort_values(["date"], inplace=True)
        df["lag_performance_benchmark"] = df["performance_benchmark"].shift(1)
        df["performance_benchmark_change_rate"] = (
            (df["performance_benchmark"] - df["lag_performance_benchmark"])
            / df["lag_performance_benchmark"]
            * 100
        ).round(5)

        df.drop(["lag_performance_benchmark"], axis=1, inplace=True)

        # if latest_date is not None, drop the first row
        if latest_date is not None:
            df.drop(df.index[0], inplace=True)

    df.replace([np.inf, -np.inf, np.nan], None, inplace=True)

    with alchemyEngine.begin() as conn:
        update_on_conflict(table_def_bond_metrics_em(), conn, df, ["date"])

    return len(df)


def fund_holding(symbol):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    # get current year in yyyy format
    current_year = datetime.now().strftime("%Y")
    last_year = str(int(current_year) - 1)
    year = current_year

    fund_type = None
    with alchemyEngine.connect() as conn:
        result = conn.execute(
            text("select type from fund_etf_perf_em where fundcode=:symbol"),
            {"symbol": symbol},
        )
        row = result.fetchone()
        # check if result is empty
        if row is not None:
            fund_type = row[0]

    df = None
    while year >= last_year:
        try:
            if fund_type == "指数型-固收":  ## Bonds
                df = EastMoneyAPI.fund_portfolio_bond_hold_em(symbol=symbol, date=year)
                # df = ak.fund_portfolio_bond_hold_em(symbol=symbol, date=year)
                df.rename(
                    columns={
                        "序号": "serial_number",
                        "债券代码": "stock_code",
                        "债券名称": "stock_name",
                        "占净值比例": "proportion_of_net_value",
                        "持仓市值": "market_value_of_holdings",
                        "季度": "quarter",
                    },
                    inplace=True,
                )
            else:
                df = ak.fund_portfolio_hold_em(symbol=symbol, date=year)
                df.rename(
                    columns={
                        "序号": "serial_number",
                        "股票代码": "stock_code",
                        "股票名称": "stock_name",
                        "占净值比例": "proportion_of_net_value",
                        "持股数": "number_of_shares",
                        "持仓市值": "market_value_of_holdings",
                        "季度": "quarter",
                    },
                    inplace=True,
                )

            df.insert(0, "symbol", symbol)
            break
        except KeyError as e:
            # try last year. and if it still fails, could be a fund which composition it not available, such as bond
            if year != last_year:
                year = last_year
                continue
            else:
                logger.warning(
                    "fund_portfolio_hold_em(%s, %s) data could be  unavailable.",
                    symbol,
                    year,
                )
                return 0

    if df is None or df.empty:
        return 0

    with alchemyEngine.begin() as conn:
        update_on_conflict(
            fund_portfolio_holdings, conn, df, ["symbol", "serial_number"]
        )

    return len(df)


def _get_market(alchemyEngine, symbol, asset_type):

    with alchemyEngine.connect() as conn:
        if asset_type.lower() == "etf":
            results = conn.execute(
                text(
                    """
                select exch 
                from fund_etf_list_sina 
                where symbol = :symbol
            """
                ),
                {
                    "symbol": symbol,
                },
            )
            row = results.fetchone()

    return row[0] if row is not None else None


def cash_inflow(symbol, exch):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    try:
        # the API can support other asset types such as ETF, stock.
        df = ak.stock_individual_fund_flow(stock=symbol, market=exch)
    except TypeError as e:
        if "'NoneType' object is not subscriptable" in str(e):
            logger.warning(
                "ak.stock_individual_fund_flow(%s, %s) - data source could be empty: %s",
                symbol,
                exch,
                str(e),
            )
            return 0

    if df is None or df.empty:
        return 0

    column_name_mapping = {
        "日期": "date",
        "收盘价": "close_price",
        "涨跌幅": "change_pct",
        "主力净流入-净额": "main_net_inflow",
        "主力净流入-净占比": "main_net_inflow_pct",
        "超大单净流入-净额": "ultra_large_net_inflow",
        "超大单净流入-净占比": "ultra_large_net_inflow_pct",
        "大单净流入-净额": "large_net_inflow",
        "大单净流入-净占比": "large_net_inflow_pct",
        "中单净流入-净额": "medium_net_inflow",
        "中单净流入-净占比": "medium_net_inflow_pct",
        "小单净流入-净额": "small_net_inflow",
        "小单净流入-净占比": "small_net_inflow_pct",
    }

    # Rename the columns
    df.rename(columns=column_name_mapping, inplace=True)

    df.insert(0, "symbol", symbol)

    # check the data type of df["date"]. If its not a date, convert it to date
    if not df["date"].apply(lambda x: isinstance(x, date)).all():
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    with alchemyEngine.connect() as conn:
        latest_date = get_max_for_column(conn, symbol=symbol, table="etf_cash_inflow")

    if latest_date is not None:
        start_date = latest_date - timedelta(days=20)
        df = df[df["date"] >= start_date]

    with alchemyEngine.begin() as conn:
        update_on_conflict(etf_cash_inflow, conn, df, ["symbol", "date"])

    return len(df)


def impute(df, random_seed, client=None):
    df_na = df.iloc[:, 1:].isna()

    if not df_na.any().any():
        return df, None

    na_counts = df_na.sum()
    na_cols = na_counts[na_counts > 0].index.tolist()
    na_row_indices = df[df.iloc[:, 1:].isna().any(axis=1)].index

    def _func(client):
        futures = []
        for na_col in na_cols:
            df_na = df[["ds", na_col]]
            futures.append(
                client.submit(_neural_impute, df_na, random_seed, priority=100)
            )
        return client.gather(futures)

    if client is not None:
        results = _func(client)
    elif len(na_cols) > 1:
        with worker_client() as client:
            results = _func(client)
    else:
        results = [_neural_impute(df[["ds", na_cols[0]]].copy(), random_seed)]
    imputed_df = results[0]
    for result in results[1:]:
        imputed_df = imputed_df.merge(result, on="ds", how="left")

    for na_col in na_cols:
        df[na_col].fillna(imputed_df[na_col], inplace=True)

    # Select imputed rows only
    imputed_df = imputed_df.loc[na_row_indices].copy()

    return df, imputed_df


def _neural_impute(df, random_seed):
    na_col = df.columns[1]
    df.rename(columns={na_col: "y"}, inplace=True)

    seed_logger = logging.getLogger("lightning_fabric.utilities.seed")
    orig_seed_log_level = seed_logger.getEffectiveLevel()
    seed_logger.setLevel(logging.FATAL)

    np_random_seed(random_seed)
    set_log_level("ERROR")
    # optimize_torch()

    na_positions = df.isna()
    df_nona = df.dropna()
    scaler = StandardScaler()
    scaler.fit(df_nona.iloc[:, 1:])
    df_filled = df.ffill().bfill()
    df.iloc[:, 1:] = scaler.transform(df_filled.iloc[:, 1:])
    df[na_positions] = np.nan

    try:
        m = NeuralProphet(
            accelerator=select_device(True),
            # changepoints_range=1.0,
        )
        m.fit(
            df,
            progress=None,
            #   early_stopping=True,
            checkpointing=False,
        )
    except Exception as e:
        m = NeuralProphet(
            # changepoints_range=1.0,
        )
        m.fit(
            df,
            progress=None,
            #   early_stopping=True,
            checkpointing=False,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        forecast = m.predict(df)

    seed_logger.setLevel(orig_seed_log_level)

    forecast = forecast[["ds", "yhat1"]]
    forecast["ds"] = forecast["ds"].dt.date
    forecast["yhat1"] = forecast["yhat1"].astype(float)
    forecast.rename(columns={"yhat1": na_col}, inplace=True)
    forecast.iloc[:, 1:] = scaler.inverse_transform(forecast.iloc[:, 1:])

    return forecast
