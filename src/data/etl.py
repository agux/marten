# %% [markdown]
# # Import

# %%
import os
import sys
import logging
import time
import math
import multiprocessing
import pandas as pd
import numpy as np
import sqlalchemy
import exchange_calendars as xcals
from dotenv import load_dotenv

# import exchange_calendars as xcals
from datetime import datetime, timedelta

# import pytz
# import pandas as pd
# from IPython.display import display, HTML
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# %% [markdown]
# # Init

# %%
t_start = time.time()

load_dotenv()  # take environment variables from .env.

# module_path = os.getenv("LOCAL_AKSHARE_DEV_MODULE")
# if module_path is not None and module_path not in sys.path:
    # sys.path.insert(0, module_path)
import akshare as ak  # noqa: E402

print(ak.__version__)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Create an engine instance
alchemyEngine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_recycle=3600,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

file_handler = logging.FileHandler("etl.log")
console_handler = logging.StreamHandler()

# Step 4: Create a formatter
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

# Step 5: Attach the formatter to the handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Step 6: Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

xshg = xcals.get_calendar("XSHG")

# %% [markdown]
# # Helper functions

# %%
def update_on_conflict(table, conn, df: pd.DataFrame, primary_keys):
    """
    Insert new records, update existing records without nullifying columns not included in the dataframe
    """
    # Load the table metadata
    table = sqlalchemy.Table(table, sqlalchemy.MetaData(), autoload_with=conn)
    # Create an insert statement from the DataFrame records
    insert_stmt = insert(table).values(df.to_dict(orient="records"))
    # Build a dictionary of column values to be updated, excluding primary keys and non-existent columns
    update_dict = {
        c.name: insert_stmt.excluded[c.name]
        for c in table.columns
        if c.name in df.columns and c.name not in primary_keys
    }
    # Construct the on_conflict_do_update statement
    on_conflict_stmt = insert_stmt.on_conflict_do_update(
        index_elements=primary_keys, set_=update_dict
    )
    # Execute the on_conflict_do_update statement
    conn.execute(on_conflict_stmt)


def ignore_on_conflict(table, conn, df, primary_keys):
    """
    Insert new records, ignore existing records
    """
    table = sqlalchemy.Table(table, sqlalchemy.MetaData(), autoload_with=conn)
    insert_stmt = insert(table).values(df.to_dict(orient="records"))
    on_conflict_stmt = insert_stmt.on_conflict_do_nothing(index_elements=primary_keys)
    conn.execute(on_conflict_stmt)


def saveAsCsv(file_name_main: str, df):
    """
    Save dataframe to CSV file
    """
    # save to file
    # Get the current timestamp to append to the filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save the dataframe to a csv file with timestamp as suffix. Need to properly encode and display Chinese characters.
    df.to_csv(f"{file_name_main}_{current_time}.csv", encoding="utf_8_sig", index=False)


@lru_cache()
def last_trade_date():
    current_date = datetime.now().date()
    # Iterate backwards from current_date until a valid session is found
    last_session = current_date
    while not xshg.is_session(last_session):
        last_session -= timedelta(days=1)
    return last_session

# %% [markdown]
# # fund_etf_spot_em

# %%
# Get laste fund / ETF data set for today (or latest trading date), and persists into database.

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
df = df.rename(
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
    }
)

with alchemyEngine.begin() as conn:
    update_on_conflict("fund_etf_spot_em", conn, df, ["code", "date"])

# %% [markdown]
# # fund_etf_perf_em

# %%
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

with alchemyEngine.begin() as conn:
    update_on_conflict("fund_etf_perf_em", conn, fund_exchange_rank_em_df, ["fundcode"])

# %% [markdown]
# # Get a full list of ETF fund

# %%
# retrieve list from Sina
fund_etf_category_sina_df = ak.fund_etf_category_sina(symbol="ETF基金")

# keep only 2 columns from `fund_etf_category_sina_df`: 代码, 名称.
# split `代码` values by `exchange code` and `symbol` and store into 2 columns. No need to keep the `代码` column.
# for example: 代码=sz159998, split into `exch=sz`, `symbol=159998`.
df = fund_etf_category_sina_df[["代码", "名称"]].copy()
df.columns = ["code", "name"]
df[["exch", "symbol"]] = df["code"].str.extract(r"([a-z]+)(\d+)")
df.drop(columns=["code"], inplace=True)

# Now, use the update_on_conflict function to insert or update the data
with alchemyEngine.begin() as conn:
    update_on_conflict("fund_etf_list_sina", conn, df, ["exch", "symbol"])

# %% [markdown]
# # Get historical trades

# %%
end_date = datetime.now().strftime("%Y%m%d")
start_date = (datetime.now() - timedelta(days=20)).strftime("%Y%m%d")
# start_date = '19700101' # For entire history.


# Function to fetch and process ETF data
def fetch_and_process_etf(symbol):
    try:
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
        
        df["symbol"] = symbol
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
        with alchemyEngine.begin() as conn:
            ignore_on_conflict("fund_etf_daily_em", conn, df, ["symbol", "date"])
    except Exception:
        logging.error(
            f"failed to get daily trade history data for {symbol}", exc_info=True
        )
        return None
    return df


# Fetch the ETF list
etf_list_df = pd.read_sql("SELECT symbol FROM fund_etf_list_sina", alchemyEngine)

# get the number of CPU cores
num_cores = multiprocessing.cpu_count()

# Use ThreadPoolExecutor to fetch data in parallel
with ThreadPoolExecutor(max_workers=num_cores) as executor:
    futures = [
        executor.submit(fetch_and_process_etf, symbol)
        for symbol in etf_list_df["symbol"]
    ]
    results = [future.result() for future in futures]

# %% [markdown]
# # Calculate ETF Performance Metrics

# %% [markdown]
# ## Get historical bond rate (risk-free interest rate)

# %%
# start_date = (datetime.now() - timedelta(days=20)).strftime('%Y%m%d')
start_date = None  # For entire history.

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
with alchemyEngine.begin() as conn:
    ignore_on_conflict("bond_metrics_em", conn, bzur, ["date"])

# %% [markdown]
# ## Calc / Update metrics in fund_etf_perf_em table

# %%
interval = 250  # assume 250 trading days annualy
end_date = last_trade_date()
# start_date = (end_date - timedelta(days=interval)).strftime('%Y%m%d')
# start_date = '19700101' # For entire history.

# load historical data from daily table and calc metrics, then update perf table
def update_etf_metrics(symbol):
    try:
        with alchemyEngine.begin() as conn:
            # load the latest (top) `interval` records of historical market data records from `fund_etf_daily_em` table for `symbol`, order by `date`.
            # select columns: date, change_rate
            query = """SELECT date, change_rate FROM fund_etf_daily_em WHERE symbol = '{}' ORDER BY date DESC LIMIT {}""".format(
                symbol, interval
            )
            df = pd.read_sql(query, conn, parse_dates=["date"])
    
            # get oldest df['date'] as state_date
            start_date = df['date'].iloc[-1]
            # get 2-years CN bond IR as risk-free IR from bond_metrics_em table. 1-year series (natural dates).
            # select date, china_yield_2y from table `bond_metrics_em`, where date is between start_date and end_date (inclusive). Load into a dataframe.
            query = """SELECT date, china_yield_2y FROM bond_metrics_em WHERE date BETWEEN '{}' AND '{}' and china_yield_2y <> 'nan'""".format(
                start_date, end_date
            )
            bme_df = pd.read_sql(query, conn, parse_dates=["date"])
            # Convert annualized rate to a daily rate
            bme_df["china_yield_2y_daily"] = bme_df["china_yield_2y"] / 365.25

            # merge df with bme_df by matching dates.
            df = pd.merge_asof(
                df.sort_values("date"),
                bme_df.sort_values("date"),
                on="date",
                direction="backward",
            ).dropna(subset=["change_rate"])

            # calculate the Sharpe ratio, Sortino ratio, and max drawdown with the time series data inside df.
            df["excess_return"] = df["change_rate"] - df["china_yield_2y_daily"]
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
            df["cumulative_returns"] = np.cumprod(1 + df["change_rate"]/100.) - 1
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
                "sharperatio": round(sharpe_ratio, 2)
                if sharpe_ratio is not None and math.isfinite(sharpe_ratio)
                else None,
                "sortinoratio": round(sortino_ratio, 2)
                if sortino_ratio is not None and math.isfinite(sortino_ratio)
                else None,
                "maxdrawdown": round(max_drawdown, 2)
                if math.isfinite(max_drawdown)
                else None,
                "fundcode": symbol,
            }
            conn.execute(update_query, params)

    except Exception:
        logging.error(f"failed to update ETF metrics for {symbol}", exc_info=True)
        return None
    return df


# Fetch the ETF list
etf_list_df = pd.read_sql("SELECT symbol FROM fund_etf_list_sina", alchemyEngine)

# get the number of CPU cores
num_proc = int((multiprocessing.cpu_count() + 1) / 2.0)

# Use ThreadPoolExecutor to calculate metrics in parallel
with ThreadPoolExecutor(max_workers=num_proc) as executor:
    futures = [
        executor.submit(update_etf_metrics, symbol) for symbol in etf_list_df["symbol"]
    ]
    results = [future.result() for future in futures]

# %% [markdown]
# # China Market Indices

# %%
cn_index_list = [
    ("上证系列指数", "sh"),
    ("深证系列指数", "sz"),
    # ("指数成份", ""),
    ("中证系列指数", "csi"),
]

def update_cn_indices_em(symbol, src):
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
        szise["src"] = src
        with alchemyEngine.begin() as conn:
            update_on_conflict("index_spot_em", conn, szise, ["symbol"])

    except Exception:
        logging.error(f"failed to update index_spot_em for {symbol}", exc_info=True)
        return None
    return szise


# get the number of CPU cores
num_proc = int((multiprocessing.cpu_count() + 1) / 2.0)

# Use ThreadPoolExecutor to calculate metrics in parallel
with ThreadPoolExecutor(max_workers=num_proc) as executor:
    futures = [
        executor.submit(update_cn_indices_em, symbol, src) for symbol, src in cn_index_list
    ]
    results = [future.result() for future in futures]

# %%
# get daily historical data
def update_cn_indices(symbol, src):
    try:
        szide = ak.stock_zh_index_daily_em(f"{src}{symbol}")

        # if shide is empty, return immediately
        if szide.empty:
            return None

        szide["symbol"] = symbol
        with alchemyEngine.begin() as conn:
            ignore_on_conflict("index_daily_em", conn, szide, ["symbol", "date"])

    except Exception:
        logging.error(f"failed to update index_daily_em for {symbol}", exc_info=True)
        return None
    return szide


conn = alchemyEngine.connect()
cn_index_fulllist = pd.read_sql("SELECT src, symbol FROM index_spot_em", conn)
conn.close()

# get the number of CPU cores
num_proc = int((multiprocessing.cpu_count() + 1) / 2.0)

# Use ThreadPoolExecutor to calculate metrics in parallel
with ThreadPoolExecutor(max_workers=num_proc) as executor:
    futures = [
        executor.submit(update_cn_indices, symbol, src)
        for symbol, src in zip(cn_index_fulllist["symbol"], cn_index_fulllist["src"])
    ]
    results = [future.result() for future in futures]

# %% [markdown]
# # Get HK Market Indices

# %%
# refresh the list

hk_index_list_df = ak.stock_hk_index_spot_em()
hk_index_list_df = hk_index_list_df.rename(
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
    }
)

# saveAsCsv("hk_index_spot_em", df)

with alchemyEngine.begin() as conn:
    update_on_conflict("hk_index_spot_em", conn, hk_index_list_df, ["symbol"])

# %%
# get daily historical data
def update_hk_indices(symbol):
    try:
        shide = ak.stock_hk_index_daily_em(symbol=symbol)

        # if shide is empty, return immediately
        if shide.empty:
            return None
        
        shide["symbol"] = symbol
        shide = shide.rename(
            columns={
                "latest":"close",
            }
        )
        with alchemyEngine.begin() as conn:
            ignore_on_conflict("hk_index_daily_em", conn, shide, ["symbol", "date"])

    except Exception:
        logging.error(f"failed to update hk_index_daily_em for {symbol}", exc_info=True)
        return None
    return shide

# get the number of CPU cores
num_proc = int((multiprocessing.cpu_count() + 1) / 2.0)

# Use ThreadPoolExecutor to calculate metrics in parallel
with ThreadPoolExecutor(max_workers=num_proc) as executor:
    futures = [
        executor.submit(update_hk_indices, symbol) for symbol in hk_index_list_df["symbol"]
    ]
    results = [future.result() for future in futures]

# %% [markdown]
# # Get US market indices

# %%
idx_symbol_list = [".IXIC", ".DJI", ".INX", ".NDX"]


def update_us_indices(symbol):
    try:
        iuss = ak.index_us_stock_sina(symbol=symbol)
        iuss['symbol'] = symbol
        with alchemyEngine.begin() as conn:
            update_on_conflict("us_index_daily_sina", conn, iuss, ["symbol", "date"])

    except Exception:
        logging.error(
            f"failed to update us_index_daily_sina for {symbol}", exc_info=True
        )
        return None
    return iuss


# get the number of CPU cores
num_proc = int((multiprocessing.cpu_count() + 1) / 2.0)

# Use ThreadPoolExecutor to calculate metrics in parallel
with ThreadPoolExecutor(max_workers=num_proc) as executor:
    futures = [executor.submit(update_us_indices, symbol) for symbol in idx_symbol_list]
    results = [future.result() for future in futures]

# %% [markdown]
# # Finally

# %%
# calculate and print outthe time taken to execute all the codes above
logger.info("Time taken: %s seconds", time.time() - t_start)
