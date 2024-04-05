# %% [markdown]
# # Import

# %%
import os
import sys
import logging
import time
import math
import asyncio
import multiprocessing
import pandas as pd
import numpy as np
import sqlalchemy
import exchange_calendars as xcals
from dotenv import load_dotenv

import cProfile
import pstats

# import exchange_calendars as xcals
from datetime import datetime, timedelta

# import pytz
# import pandas as pd
# from IPython.display import display, HTML
from sqlalchemy import (
    create_engine,
    text,
    Table,
    Column,
    MetaData,
    Text,
    String,
    Date,
    Numeric,
    Integer,
    DateTime,
    PrimaryKeyConstraint,
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import insert

# from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
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

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("etl.log")
console_handler = logging.StreamHandler()

# Step 4: Create a formatter
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Step 5: Attach the formatter to the handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Step 6: Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

xshg = xcals.get_calendar("XSHG")


def table_def_index_daily_em():
    return Table(
        "index_daily_em",
        MetaData(),
        Column("symbol", Text, primary_key=True),
        Column("date", Date, primary_key=True),
        Column("open", Numeric),
        Column("close", Numeric),
        Column("high", Numeric),
        Column("low", Numeric),
        Column("volume", Numeric),
        Column("last_modified", DateTime, default=datetime.utcnow),
        Column("amount", Numeric),
    )


def table_def_hk_index_daily_em():
    return Table(
        "hk_index_daily_em",
        MetaData(),
        Column("symbol", Text, nullable=False),
        Column("date", Date, nullable=False),
        Column("open", Numeric),
        Column("close", Numeric),
        Column("high", Numeric),
        Column("low", Numeric),
        Column(
            "last_modified", DateTime, default=func.current_timestamp(), nullable=False
        ),
        PrimaryKeyConstraint("symbol", "date", name="hk_index_daily_em_pkey"),
    )


def table_def_us_index_daily_sina():
    return Table(
        "us_index_daily_sina",
        MetaData(),
        Column("symbol", Text, nullable=False),
        Column("date", Date, nullable=False),
        Column("open", Numeric),
        Column("close", Numeric),
        Column("high", Numeric),
        Column("low", Numeric),
        Column("volume", Numeric),
        Column("amount", Numeric),
        Column(
            "last_modified", DateTime, default=func.current_timestamp(), nullable=False
        ),
        PrimaryKeyConstraint("symbol", "date", name="us_index_daily_sina_pkey"),
    )


def table_def_hk_index_spot_em():
    return Table(
        "hk_index_spot_em",
        MetaData(),
        Column("seq", Integer),
        Column("internal_code", Text),
        Column("symbol", Text, nullable=False),
        Column("name", Text, nullable=False),
        Column("open", Numeric),
        Column("close", Numeric),
        Column("prev_close", Numeric),
        Column("high", Numeric),
        Column("low", Numeric),
        Column("volume", Numeric),
        Column("amount", Numeric),
        Column("change_amount", Numeric),
        Column("change_rate", Numeric),
        Column(
            "last_modified", DateTime, default=func.current_timestamp(), nullable=False
        ),
        PrimaryKeyConstraint("symbol", name="hk_index_spot_em_pkey"),
    )


def table_def_fund_etf_spot_em():
    return Table(
        "fund_etf_spot_em",
        MetaData(),
        Column("date", Date, nullable=False, comment="Date of the record"),
        Column("code", String(10), nullable=False, comment="Stock code"),
        Column("name", String(100), comment="Stock name"),
        Column("latest_price", Numeric, comment="Latest trading price"),
        Column("change_amount", Numeric, comment="Price change amount (涨跌额)"),
        Column(
            "change_rate", Numeric, comment="Price change rate in percentage (涨跌幅)"
        ),
        Column("volume", Numeric, comment="Trading volume (成交量)"),
        Column("turnover", Numeric, comment="Trading turnover (成交额)"),
        Column("opening_price", Numeric, comment="Opening price"),
        Column("highest_price", Numeric, comment="Highest price of the day"),
        Column("lowest_price", Numeric, comment="Lowest price of the day"),
        Column("previous_close", Numeric, comment="Previous closing price"),
        Column("turnover_rate", Numeric, comment="Turnover rate (换手率)"),
        Column("volume_ratio", Numeric, comment="Volume ratio (量比)"),
        Column("order_ratio", Numeric, comment="Order ratio (委比)"),
        Column("external_disc", Numeric, comment="External market volume (外盘)"),
        Column("internal_disc", Numeric, comment="Internal market volume (内盘)"),
        Column(
            "circulating_market_value",
            Numeric,
            comment="Circulating market value (流通市值)",
        ),
        Column("total_market_value", Numeric, comment="Total market value (总市值)"),
        Column("latest_shares", Numeric, comment="Latest number of shares (最新份额)"),
        Column(
            "main_force_net_inflow_amount",
            Numeric,
            comment="Net amount of main force inflow (主力净流入-净额)",
        ),
        Column(
            "main_force_net_inflow_ratio",
            Numeric,
            comment="Net ratio of main force inflow (主力净流入-净占比)",
        ),
        Column(
            "super_large_net_inflow_amount",
            Numeric,
            comment="Net amount of super large orders inflow (超大单净流入-净额)",
        ),
        Column(
            "super_large_net_inflow_ratio",
            Numeric,
            comment="Net ratio of super large orders inflow (超大单净流入-净占比)",
        ),
        Column(
            "large_net_inflow_amount",
            Numeric,
            comment="Net amount of large orders inflow (大单净流入-净额)",
        ),
        Column(
            "large_net_inflow_ratio",
            Numeric,
            comment="Net ratio of large orders inflow (大单净流入-净占比)",
        ),
        Column(
            "medium_net_inflow_amount",
            Numeric,
            comment="Net amount of medium orders inflow (中单净流入-净额)",
        ),
        Column(
            "medium_net_inflow_ratio",
            Numeric,
            comment="Net ratio of medium orders inflow (中单净流入-净占比)",
        ),
        Column(
            "small_net_inflow_amount",
            Numeric,
            comment="Net amount of small orders inflow (小单净流入-净额)",
        ),
        Column(
            "small_net_inflow_ratio",
            Numeric,
            comment="Net ratio of small orders inflow (小单净流入-净占比)",
        ),
        Column(
            "iopv",
            Numeric,
            comment="Indicative Optimized Portfolio Value (aka iNAV, 实时估值)",
        ),
        Column(
            "fund_discount_rate", Numeric, comment="Fund discount rate (基金折价率)"
        ),
        Column(
            "update_time",
            DateTime,
            comment="Data timestamp provided by the data source (更新时间)",
        ),
        PrimaryKeyConstraint("code", "date", name="fund_etf_spot_em_pk"),
    )


def table_def_index_spot_em():
    return Table(
        "index_spot_em",
        MetaData(),
        Column("seq", Integer),
        Column("symbol", Text, nullable=False),
        Column("name", Text, nullable=False),
        Column("open", Numeric),
        Column("close", Numeric),
        Column("prev_close", Numeric),
        Column("high", Numeric),
        Column("low", Numeric),
        Column("amplitude", Numeric),
        Column("volume_ratio", Numeric),
        Column("change_rate", Numeric),
        Column("change_amount", Numeric),
        Column("volume", Numeric),
        Column("amount", Numeric),
        Column(
            "last_modified", DateTime, default=func.current_timestamp(), nullable=False
        ),
        Column("src", Text),
        PrimaryKeyConstraint("symbol", name="index_spot_em_pkey"),
    )


def table_def_fund_etf_perf_em():
    return Table(
        "fund_etf_perf_em",
        MetaData(),
        Column(
            "id",
            Integer,
            primary_key=True,
            autoincrement=True,
            comment="Identifier (序号)",
        ),
        Column(
            "fundcode",
            String(10),
            nullable=False,
            unique=True,
            comment="Fund Code (基金代码)",
        ),
        Column("fundname", String(255), nullable=False, comment="Fund Name (基金简称)"),
        Column("type", String(50), nullable=False, comment="Type (类型)"),
        Column("date", Date, nullable=False, comment="Date (日期)"),
        Column(
            "unitnav",
            Numeric(10, 4),
            nullable=False,
            comment="Unit Net Asset Value (单位净值)",
        ),
        Column(
            "accumulatednav",
            Numeric(10, 4),
            nullable=False,
            comment="Accumulated Net Asset Value (累计净值)",
        ),
        Column(
            "pastweek", Numeric(5, 2), comment="Performance over the past week (近1周)"
        ),
        Column(
            "pastmonth",
            Numeric(5, 2),
            comment="Performance over the past month (近1月)",
        ),
        Column(
            "past3months",
            Numeric(5, 2),
            comment="Performance over the past three months (近3月)",
        ),
        Column(
            "past6months",
            Numeric(5, 2),
            comment="Performance over the past six months (近6月)",
        ),
        Column(
            "pastyear", Numeric(5, 2), comment="Performance over the past year (近1年)"
        ),
        Column(
            "past2years",
            Numeric(5, 2),
            comment="Performance over the past two years (近2年)",
        ),
        Column(
            "past3years",
            Numeric(5, 2),
            comment="Performance over the past three years (近3年)",
        ),
        Column("ytd", Numeric(5, 2), comment="Year To Date performance (今年来)"),
        Column(
            "sinceinception",
            Numeric(10, 2),
            comment="Performance since inception (成立来)",
        ),
        Column(
            "inceptiondate", Date, nullable=False, comment="Inception Date (成立日期)"
        ),
        Column("sharperatio", Numeric, comment="Sharpe Ratio (夏普比率)"),
        Column("sortinoratio", Numeric, comment="Sortino Ratio (索提诺比率)"),
        Column("maxdrawdown", Numeric, comment="Maximum Drawdown (最大回撤)"),
        Column(
            "last_modified", DateTime, default=func.current_timestamp(), nullable=False
        ),
        PrimaryKeyConstraint("id", name="etf_perf_em_pkey"),
    )


def table_def_fund_etf_list_sina():
    return Table(
        "fund_etf_list_sina",
        MetaData(),
        Column("exch", String, nullable=False),
        Column("symbol", String, nullable=False),
        Column("name", String, nullable=False),
        Column(
            "last_modified", DateTime, default=func.current_timestamp(), nullable=False
        ),
        PrimaryKeyConstraint("exch", "symbol", name="fund_etf_list_sina_pk"),
    )


def table_def_fund_etf_daily_em():
    return Table(
        "fund_etf_daily_em",
        MetaData(),
        Column("symbol", Text, nullable=False, comment="ETF symbol (股票代码)"),
        Column("date", Date, nullable=False, comment="Trade date (交易日期)"),
        Column("open", Numeric, comment="Opening price (开盘价)"),
        Column("close", Numeric, comment="Closing price (收盘价)"),
        Column("high", Numeric, comment="Highest price (最高价)"),
        Column("low", Numeric, comment="Lowest price (最低价)"),
        Column("volume", Numeric, comment="Trade volume (成交量)"),
        Column("turnover", Numeric, comment="Turnover (成交额)"),
        Column("amplitude", Numeric, comment="Amplitude (振幅)"),
        Column("change_rate", Numeric, comment="Change rate (涨跌幅)"),
        Column("change_amount", Numeric, comment="Change amount (涨跌额)"),
        Column("turnover_rate", Numeric, comment="Turnover rate (换手率)"),
        Column(
            "last_modified",
            DateTime,
            default=func.current_timestamp(),
            nullable=False,
            comment="Last modified timestamp (最后修改时间)",
        ),
        PrimaryKeyConstraint("symbol", "date", name="fund_etf_daily_em_pkey"),
    )


def table_def_bond_metrics_em():
    return Table(
        "bond_metrics_em",
        MetaData(),
        Column("date", Date, nullable=False, comment="Date of the metrics (数据日期)"),
        Column(
            "china_yield_2y",
            Numeric,
            comment="China 2-year government bond yield (中国国债收益率2年)",
        ),
        Column(
            "china_yield_5y",
            Numeric,
            comment="China 5-year government bond yield (中国国债收益率5年)",
        ),
        Column(
            "china_yield_10y",
            Numeric,
            comment="China 10-year government bond yield (中国国债收益率10年)",
        ),
        Column(
            "china_yield_30y",
            Numeric,
            comment="China 30-year government bond yield (中国国债收益率30年)",
        ),
        Column(
            "china_yield_spread_10y_2y",
            Numeric,
            comment="China 10-year to 2-year government bond yield spread (中国国债收益率10年-2年)",
        ),
        Column(
            "china_gdp_growth",
            Numeric,
            comment="China annual GDP growth rate (中国GDP年增率)",
        ),
        Column(
            "us_yield_2y",
            Numeric,
            comment="US 2-year government bond yield (美国国债收益率2年)",
        ),
        Column(
            "us_yield_5y",
            Numeric,
            comment="US 5-year government bond yield (美国国债收益率5年)",
        ),
        Column(
            "us_yield_10y",
            Numeric,
            comment="US 10-year government bond yield (美国国债收益率10年)",
        ),
        Column(
            "us_yield_30y",
            Numeric,
            comment="US 30-year government bond yield (美国国债收益率30年)",
        ),
        Column(
            "us_yield_spread_10y_2y",
            Numeric,
            comment="US 10-year to 2-year government bond yield spread (美国国债收益率10年-2年)",
        ),
        Column(
            "us_gdp_growth",
            Numeric,
            comment="US annual GDP growth rate (美国GDP年增率)",
        ),
        Column(
            "last_modified",
            DateTime,
            default=func.current_timestamp(),
            nullable=False,
            comment="Last modified timestamp (最后修改时间)",
        ),
        PrimaryKeyConstraint("date", name="bond_metrics_em_pk"),
    )


# %% [markdown]
# # Helper functions


# %%
def update_on_conflict(table_def, conn, df: pd.DataFrame, primary_keys):
    """
    Insert new records, update existing records without nullifying columns not included in the dataframe
    """
    start = time.time()
    # Load the table metadata
    # table = sqlalchemy.Table(table, sqlalchemy.MetaData(), autoload_with=conn)
    # Create an insert statement from the DataFrame records
    insert_stmt = insert(table_def).values(df.to_dict(orient="records"))
    # Build a dictionary of column values to be updated, excluding primary keys and non-existent columns
    update_dict = {
        c.name: insert_stmt.excluded[c.name]
        for c in table_def.columns
        if c.name in df.columns and c.name not in primary_keys
    }
    # Construct the on_conflict_do_update statement
    on_conflict_stmt = insert_stmt.on_conflict_do_update(
        index_elements=primary_keys, set_=update_dict
    )
    # Execute the on_conflict_do_update statement
    conn.execute(on_conflict_stmt)
    print(f"{time.time()-start} update_on_conflict({table_def.name})")


def ignore_on_conflict(table_def, conn, df, primary_keys):
    """
    Insert new records, ignore existing records
    """
    start = time.time()
    # table = sqlalchemy.Table(table, sqlalchemy.MetaData(), autoload_with=conn)
    insert_stmt = insert(table_def).values(df.to_dict(orient="records"))
    on_conflict_stmt = insert_stmt.on_conflict_do_nothing(index_elements=primary_keys)
    conn.execute(on_conflict_stmt)
    print(f"{time.time()-start} ignore_on_conflict({table_def.name})")


def get_latest_date(conn, symbol, table):
    query = f"SELECT max(date) FROM {table}"
    if symbol is not None:
        query += " WHERE symbol = :symbol"
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


@lru_cache()
def last_trade_date():
    current_date = datetime.now().date()
    # Iterate backwards from current_date until a valid session is found
    last_session = current_date
    while not xshg.is_session(last_session):
        last_session -= timedelta(days=1)
    return last_session


def main():
    logger.info("Using akshare version: %s", ak.__version__)
    # Create an engine instance
    # Define your database engine outside of the parallel function
    # Using NullPool disables the connection pooling
    alchemyEngine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_recycle=3600,
        # pool_size=1,
        poolclass=NullPool,
    )
    SessionLocal = sessionmaker(bind=alchemyEngine)

    # %%
    # Get latest fund / ETF data set for today (or latest trading date), and persists into database.
    logger.info("running fund_etf_spot_em()...")
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
        update_on_conflict(table_def_fund_etf_spot_em(), conn, df, ["code", "date"])

    # %% [markdown]
    # # fund_etf_perf_em

    # %%
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

    with alchemyEngine.begin() as conn:
        update_on_conflict(
            table_def_fund_etf_perf_em(), conn, fund_exchange_rank_em_df, ["fundcode"]
        )

    # %% [markdown]
    # # Get a full list of ETF fund

    # %%
    # retrieve list from Sina
    logger.info("running fund_etf_category_sina()...")
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
        update_on_conflict(table_def_fund_etf_list_sina(), conn, df, ["exch", "symbol"])

    # %% [markdown]
    # # Get historical trades

    # %%

    # Function to fetch and process ETF data
    def fetch_and_process_etf(symbol, url):
        try:
            logger.info(f"running fund_etf_hist_em({symbol})...")
            alchemyEngine = create_engine(url, poolclass=NullPool)
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

                ignore_on_conflict(
                    table_def_fund_etf_daily_em(), conn, df, ["symbol", "date"]
                )
        except Exception:
            logging.error(
                f"failed to get daily trade history data for {symbol}", exc_info=True
            )
            return None
        return df

    # Fetch the ETF list
    etf_list_df = pd.read_sql("SELECT symbol FROM fund_etf_list_sina", alchemyEngine)
    logger.info(f"starting joblib on function fetch_and_process_etf()...")
    Parallel(n_jobs=-1)(
        delayed(fetch_and_process_etf)(symbol, alchemyEngine.url)
        for symbol in etf_list_df["symbol"]
    )

    # %% [markdown]
    # # Calculate ETF Performance Metrics

    # %% [markdown]
    # ## Get historical bond rate (risk-free interest rate)

    # %%
    # start_date = (datetime.now() - timedelta(days=20)).strftime('%Y%m%d')
    start_date = None  # For entire history.
    with alchemyEngine.begin() as conn:
        latest_date = get_latest_date(conn, None, "bond_metrics_em")
        if latest_date is not None:
            start_date = latest_date.strftime("%Y%m%d")
        logger.info(f"running bond_zh_us_rate()...")
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

        ignore_on_conflict(table_def_bond_metrics_em(), conn, bzur, ["date"])

    # %% [markdown]
    # ## Calc / Update metrics in fund_etf_perf_em table

    # %%
    end_date = last_trade_date()
    # start_date = (end_date - timedelta(days=interval)).strftime('%Y%m%d')
    # start_date = '19700101' # For entire history.

    # load historical data from daily table and calc metrics, then update perf table
    def update_etf_metrics(symbol, url, end_date):
        interval = 250  # assume 250 trading days annualy
        alchemyEngine = create_engine(url, poolclass=NullPool)
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
                    annualized_excess_return / downside_dev
                    if downside_dev > 0
                    else None
                )

                # To calculate max drawdown, get the cummulative_returns
                df["cumulative_returns"] = np.cumprod(1 + df["change_rate"] / 100.0) - 1
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

        except Exception:
            logging.error(f"failed to update ETF metrics for {symbol}", exc_info=True)
            return None
        return df

    # Fetch the ETF list
    etf_list_df = pd.read_sql("SELECT symbol FROM fund_etf_list_sina", alchemyEngine)

    # get the number of CPU cores
    num_proc = int((multiprocessing.cpu_count() + 1) / 2.0)
    logger.info(f"starting joblib on function update_etf_metrics()...")
    Parallel(n_jobs=num_proc)(
        delayed(update_etf_metrics)(symbol, alchemyEngine.url, end_date)
        for symbol in etf_list_df["symbol"]
    )


    # %% [markdown]
    # # China Market Indices

    # %%
    cn_index_list = [
        ("上证系列指数", "sh"),
        ("深证系列指数", "sz"),
        # ("指数成份", ""),
        ("中证系列指数", "csi"),
    ]

    def stock_zh_index_spot_em(symbol, src, url):
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
            alchemyEngine = create_engine(url, poolclass=NullPool)
            with alchemyEngine.begin() as conn:
                update_on_conflict(table_def_index_spot_em(), conn, szise, ["symbol"])

        except Exception:
            logging.error(f"failed to update index_spot_em for {symbol}", exc_info=True)
            return None
        return szise

    # get the number of CPU cores
    num_proc = int((multiprocessing.cpu_count() + 1) / 2.0)
    logger.info("starting joblib on function stock_zh_index_spot_em()...")
    Parallel(n_jobs=num_proc)(
        delayed(stock_zh_index_spot_em)(
            symbol,
            src,
            alchemyEngine.url,
        )
        for symbol, src in cn_index_list
    )

    # %%
    # get daily historical data
    def stock_zh_index_daily_em(symbol, src, url):
        try:
            alchemyEngine = create_engine(url, poolclass=NullPool)
            with alchemyEngine.begin() as conn:
                latest_date = get_latest_date(conn, symbol, "index_daily_em")

                start_date = "19900101"  # For entire history.
                if latest_date is not None:
                    start_date = (latest_date - timedelta(days=10)).strftime("%Y%m%d")

                end_date = datetime.now().strftime("%Y%m%d")

                szide = ak.stock_zh_index_daily_em(
                    f"{src}{symbol}", start_date, end_date
                )

                # if shide is empty, return immediately
                if szide.empty:
                    logger.warning("index data is empty: %s", symbol)
                    return None

                szide["symbol"] = symbol

                ignore_on_conflict(
                    table_def_index_daily_em(), conn, szide, ["symbol", "date"]
                )

        except Exception:
            logging.error(
                f"failed to update index_daily_em for {symbol}", exc_info=True
            )
            return None
        return szide

    conn = alchemyEngine.connect()
    cn_index_fulllist = pd.read_sql("SELECT src, symbol FROM index_spot_em", conn)
    conn.close()

    # get the number of CPU cores
    # num_proc = int((multiprocessing.cpu_count() + 1) / 2.0)
    logger.info("starting joblib on function stock_zh_index_daily_em()...")
    Parallel(n_jobs=-1)(
        delayed(stock_zh_index_daily_em)(symbol, src, alchemyEngine.url)
        for symbol, src in zip(cn_index_fulllist["symbol"], cn_index_fulllist["src"])
    )

    # %% [markdown]
    # # Get HK Market Indices

    # %%
    # refresh the list
    logger.info("running stock_hk_index_spot_em()...")
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
        update_on_conflict(
            table_def_hk_index_spot_em(), conn, hk_index_list_df, ["symbol"]
        )

    # %%
    # get daily historical data
    def update_hk_indices(symbol, url):
        try:
            shide = ak.stock_hk_index_daily_em(symbol=symbol)

            # if shide is empty, return immediately
            if shide.empty:
                return None

            shide["symbol"] = symbol
            shide = shide.rename(
                columns={
                    "latest": "close",
                }
            )
            # Convert the 'date' column to datetime
            shide["date"] = pd.to_datetime(shide["date"]).dt.date
            alchemyEngine = create_engine(url, poolclass=NullPool)
            with alchemyEngine.begin() as conn:
                latest_date = get_latest_date(conn, symbol, "hk_index_daily_em")

                if latest_date is not None:
                    ## keep rows only with `date` later than the latest record in database.
                    shide = shide[shide["date"] > (latest_date - timedelta(days=10))]
                update_on_conflict(
                    table_def_hk_index_daily_em(), conn, shide, ["symbol", "date"]
                )

        except Exception:
            logging.error(
                f"failed to update hk_index_daily_em for {symbol}", exc_info=True
            )
            return None
        return shide

    # get the number of CPU cores
    # num_proc = int((multiprocessing.cpu_count() + 1) / 2.0)
    logger.info("starting joblib on function update_hk_indices()...")
    Parallel(n_jobs=-1)(
        delayed(update_hk_indices)(
            symbol,
            alchemyEngine.url,
        )
        for symbol in hk_index_list_df["symbol"]
    )

    # %% [markdown]
    # # Get US market indices

    # %%
    idx_symbol_list = [".IXIC", ".DJI", ".INX", ".NDX"]

    def update_us_indices(symbol, url):
        try:
            iuss = ak.index_us_stock_sina(symbol=symbol)
            iuss["symbol"] = symbol
            # Convert iuss["date"] to datetime and normalize to date only
            iuss["date"] = pd.to_datetime(iuss["date"]).dt.date
            alchemyEngine = create_engine(url, poolclass=NullPool)
            with alchemyEngine.begin() as conn:
                latest_date = get_latest_date(conn, symbol, "us_index_daily_sina")
                if latest_date is not None:
                    iuss = iuss[iuss["date"] > (latest_date - timedelta(days=10))]
                update_on_conflict(
                    table_def_us_index_daily_sina(), conn, iuss, ["symbol", "date"]
                )
        except Exception:
            logging.error(
                f"failed to update us_index_daily_sina for {symbol}", exc_info=True
            )
            return None
        return iuss

    # get the number of CPU cores
    num_proc = int((multiprocessing.cpu_count() + 1) / 2.0)
    logger.info("starting joblib on function update_us_indices()...")
    Parallel(n_jobs=num_proc)(
        delayed(update_us_indices)(
            symbol,
            alchemyEngine.url,
        )
        for symbol in idx_symbol_list
    )

    # %% [markdown]
    # # Finally

    # %%
    # calculate and print outthe time taken to execute all the codes above
    logger.info("Time taken: %s seconds", time.time() - t_start)


if __name__ == "__main__":
    try:
        # profiler = cProfile.Profile()
        # profiler.enable()

        main()

        # profiler.disable()
        # stats = pstats.Stats(profiler).sort_stats("cumtime")
        # stats.print_stats()
    except Exception as e:
        logger.exception("main process terminated")
