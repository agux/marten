from datetime import datetime
from sqlalchemy.sql import func
from sqlalchemy import (
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