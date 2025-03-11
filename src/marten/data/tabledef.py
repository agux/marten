from sqlalchemy.sql import func, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Table,
    Index,
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

Base = declarative_base()

## Important Node:
## All column names must align with SQL DDL (with lower-case characters)


class etf_cash_inflow(Base):
    __tablename__ = "etf_cash_inflow"
    __table_args__ = (PrimaryKeyConstraint("symbol", "date"),)

    symbol = Column(
        Text,
        nullable=False,
        comment="ETF symbol, the unique identifier also known as the ticker symbol. ('symbol')",
    )
    date = Column(Date, nullable=False, comment="The date of the data record. ('日期')")
    close_price = Column(
        Numeric,
        nullable=False,
        comment="Closing price of the ETF on the given date. ('收盘价')",
    )
    change_pct = Column(
        Numeric,
        nullable=False,
        comment="Percentage change in the closing price compared to the previous trading day. ('涨跌幅')",
    )
    main_net_inflow = Column(
        Numeric,
        nullable=False,
        comment="Net cash inflow from main investors (net amount). ('主力净流入-净额')",
    )
    main_net_inflow_pct = Column(
        Numeric,
        nullable=False,
        comment="Net cash inflow from main investors as a percentage of total volume. ('主力净流入-净占比')",
    )
    ultra_large_net_inflow = Column(
        Numeric,
        nullable=False,
        comment="Net cash inflow from ultra-large transactions (net amount). ('超大单净流入-净额')",
    )
    ultra_large_net_inflow_pct = Column(
        Numeric,
        nullable=False,
        comment="Net cash inflow from ultra-large transactions as a percentage of total volume. ('超大单净流入-净占比')",
    )
    large_net_inflow = Column(
        Numeric,
        nullable=False,
        comment="Net cash inflow from large transactions (net amount). ('大单净流入-净额')",
    )
    large_net_inflow_pct = Column(
        Numeric,
        nullable=False,
        comment="Net cash inflow from large transactions as a percentage of total volume. ('大单净流入-净占比')",
    )
    medium_net_inflow = Column(
        Numeric,
        nullable=False,
        comment="Net cash inflow from medium transactions (net amount). ('中单净流入-净额')",
    )
    medium_net_inflow_pct = Column(
        Numeric,
        nullable=False,
        comment="Net cash inflow from medium transactions as a percentage of total volume. ('中单净流入-净占比')",
    )
    small_net_inflow = Column(
        Numeric,
        nullable=False,
        comment="Net cash inflow from small transactions (net amount). ('小单净流入-净额')",
    )
    small_net_inflow_pct = Column(
        Numeric,
        nullable=False,
        comment="Net cash inflow from small transactions as a percentage of total volume. ('小单净流入-净占比')",
    )
    last_modified = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.current_timestamp(),
        comment="Timestamp of the last modification to the record. Automatically updated on each row change. ('last_modified')",
    )


class interbank_rate_list(Base):
    __tablename__ = "interbank_rate_list"

    symbol = Column(Text, primary_key=True, nullable=False)
    market = Column(Text, nullable=False)
    symbol_type = Column(Text, nullable=False)
    indicator = Column(Text, nullable=False)
    last_modified = Column(
        DateTime(timezone=True), nullable=False, default=func.current_timestamp()
    )


class interbank_rate_hist(Base):
    __tablename__ = "interbank_rate_hist"
    __table_args__ = (PrimaryKeyConstraint("symbol", "date"),)

    symbol = Column(Text, nullable=False)
    date = Column(Date, nullable=False)
    interest_rate = Column(Numeric, nullable=False)
    change_rate = Column(Numeric, nullable=False)
    last_modified = Column(
        DateTime(timezone=True), nullable=False, default=func.current_timestamp()
    )


class fund_portfolio_holdings(Base):
    __tablename__ = "fund_portfolio_holdings"
    __table_args__ = (PrimaryKeyConstraint("symbol", "serial_number"),)

    symbol = Column(
        Text, nullable=False, comment="基金符号 - Unique symbol for the fund"
    )
    serial_number = Column(Integer, nullable=False, comment="序号 - Serial number")
    stock_code = Column(String(20), nullable=False, comment="股票代码 - Stock code")
    stock_name = Column(Text, comment="股票名称 - Name of the stock")
    proportion_of_net_value = Column(
        Numeric, comment="占净值比例 - Proportion of net value"
    )
    number_of_shares = Column(Numeric, comment="持股数 - Number of shares held")
    market_value_of_holdings = Column(
        Numeric, comment="持仓市值 - Market value of the holdings"
    )
    quarter = Column(Text, comment="季度 - Quarter for which the data is reported")
    last_modified = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.current_timestamp(),
        comment="最后修改时间 - Timestamp of the last modification of the record",
    )


class fund_dividend_events(Base):
    __tablename__ = "fund_dividend_events"
    __table_args__ = (PrimaryKeyConstraint("symbol", "rights_registration_date"),)

    symbol = Column(String, nullable=False, comment="基金代码 - 基金的唯一代码")
    short_name = Column(String, nullable=False, comment="基金简称 - 基金的简短名称")
    rights_registration_date = Column(
        Date, nullable=False, comment="权益登记日 - 权益登记的日期"
    )
    ex_dividend_date = Column(Date, comment="除息日期 - 股票或基金除息的日期")
    dividend = Column(Numeric, nullable=False, comment="分红 - 基金分红的金额")
    dividend_payment_date = Column(Date, comment="分红发放日 - 分红款项发放的日期")
    last_modified = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.current_timestamp(),
        comment="最后修改时间",
    )


class cn_bond_index_period(Base):
    __tablename__ = "cn_bond_index_period"

    symbol = Column(Text, primary_key=True, nullable=False, comment="Period Symbol")
    symbol_cn = Column(Text, nullable=False, comment="Period Symbol in Chinese")
    last_modified = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.current_timestamp(),
        comment="Last Modified Timestamp",
    )


class cn_bond_indices(Base):
    __tablename__ = "cn_bond_indices"
    __table_args__ = (PrimaryKeyConstraint("symbol", "date"),)

    symbol = Column(Text, nullable=False, comment="Symbol")
    date = Column(Date, nullable=False, comment="Date")
    fullprice = Column(Numeric, comment="全价")
    cleanprice = Column(Numeric, comment="净价")
    wealth = Column(Numeric, comment="财富")
    avgmv_duration = Column(Numeric, comment="平均市值法久期")
    avgcf_duration = Column(Numeric, comment="平均现金流法久期")
    avgmv_convexity = Column(Numeric, comment="平均市值法凸性")
    avgcf_convexity = Column(Numeric, comment="平均现金流法凸性")
    avgcf_ytm = Column(Numeric, comment="平均现金流法到期收益率")
    avgmv_ytm = Column(Numeric, comment="平均市值法到期收益率")
    avgbpv = Column(Numeric, comment="平均基点价值")
    avgmaturity = Column(Numeric, comment="平均待偿期")
    avgcouponrate = Column(Numeric, comment="平均派息率")
    indexprevdaymv = Column(Numeric, comment="指数上日总市值")
    wealthindex_change = Column(Numeric, comment="财富指数涨跌幅")
    fullpriceindex_change = Column(Numeric, comment="全价指数涨跌幅")
    cleanpriceindex_change = Column(Numeric, comment="净价指数涨跌幅")
    spotsettlementvolume = Column(Numeric, comment="现券结算量")
    last_modified = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.current_timestamp(),
        comment="Last Modified Timestamp",
    )
    avgmv_duration_change_rate = Column(Numeric)
    avgcf_duration_change_rate = Column(Numeric)
    avgmv_convexity_change_rate = Column(Numeric)
    avgcf_convexity_change_rate = Column(Numeric)
    avgcf_ytm_change_rate = Column(Numeric)
    avgmv_ytm_change_rate = Column(Numeric)
    avgbpv_change_rate = Column(Numeric)
    avgmaturity_change_rate = Column(Numeric)
    avgcouponrate_change_rate = Column(Numeric)
    indexprevdaymv_change_rate = Column(Numeric)
    spotsettlementvolume_change_rate = Column(Numeric)


class spot_symbol_table_sge(Base):
    __tablename__ = "spot_symbol_table_sge"
    __table_args__ = (
        Index(
            "spot_symbol_table_sge_product_idx",
            "product",
            unique=True,
            postgresql_using="btree",
        ),
    )

    serial = Column(Integer, nullable=True)
    product = Column(Text, primary_key=True)


class table_def_option_qvix(Base):
    __tablename__ = "option_qvix"
    __table_args__ = (
        PrimaryKeyConstraint("symbol", "date"),
        Index(
            "option_qvix_date_idx",
            desc("date"),
            postgresql_using="btree",
        ),
    )

    symbol = Column(Text)
    date = Column(Date)
    open = Column(Numeric, nullable=True)
    close = Column(Numeric, nullable=True)
    high = Column(Numeric, nullable=True)
    low = Column(Numeric, nullable=True)
    last_modified = Column(
        DateTime(timezone=True), default=func.current_timestamp(), nullable=True
    )


class spot_hist_sge(Base):
    __tablename__ = "spot_hist_sge"
    __table_args__ = (
        PrimaryKeyConstraint("symbol", "date"),
        Index(
            "spot_hist_sge_date_idx",
            desc("date"),
            postgresql_using="btree",
        ),
    )

    symbol = Column(Text)
    date = Column(Date)
    open = Column(Numeric, nullable=True)
    close = Column(Numeric, nullable=True)
    high = Column(Numeric, nullable=True)
    low = Column(Numeric, nullable=True)
    change_rate = Column(Numeric, nullable=True)
    open_preclose_rate = Column(Numeric, nullable=True)
    high_preclose_rate = Column(Numeric, nullable=True)
    low_preclose_rate = Column(Numeric, nullable=True)
    last_modified = Column(
        DateTime(timezone=True), default=func.current_timestamp(), nullable=True
    )


class currency_boc_safe(Base):
    __tablename__ = "currency_boc_safe"
    __table_args__ = (
        PrimaryKeyConstraint("date"),
        {"comment": "public.currency_boc_safe definition"},
    )

    date = Column(Date, nullable=False, comment="Date")
    usd = Column(Numeric, comment="USD")
    eur = Column(Numeric, comment="EUR")
    jpy = Column(Numeric, comment="JPY")
    hkd = Column(Numeric, comment="HKD")
    gbp = Column(Numeric, comment="GBP")
    aud = Column(Numeric, comment="AUD")
    nzd = Column(Numeric, comment="NZD")
    sgd = Column(Numeric, comment="SGD")
    chf = Column(Numeric, comment="CHF")
    cad = Column(Numeric, comment="CAD")
    myr = Column(Numeric, comment="MYR")
    rub = Column(Numeric, comment="RUB")
    zar = Column(Numeric, comment="ZAR")
    krw = Column(Numeric, comment="KRW")
    aed = Column(Numeric, comment="AED")
    qar = Column(Numeric, comment="QAR")
    huf = Column(Numeric, comment="HUF")
    pln = Column(Numeric, comment="PLN")
    dkk = Column(Numeric, comment="DKK")
    sek = Column(Numeric, comment="SEK")
    nok = Column(Numeric, comment="NOK")
    try_ = Column("try", Numeric, comment="TRY")
    php = Column(Numeric, comment="PHP")
    thb = Column(Numeric, comment="THB")
    mop = Column(Numeric, comment="MOP")
    usd_change_rate = Column(Numeric)
    eur_change_rate = Column(Numeric)
    jpy_change_rate = Column(Numeric)
    hkd_change_rate = Column(Numeric)
    gbp_change_rate = Column(Numeric)
    aud_change_rate = Column(Numeric)
    nzd_change_rate = Column(Numeric)
    sgd_change_rate = Column(Numeric)
    chf_change_rate = Column(Numeric)
    cad_change_rate = Column(Numeric)
    myr_change_rate = Column(Numeric)
    rub_change_rate = Column(Numeric)
    zar_change_rate = Column(Numeric)
    krw_change_rate = Column(Numeric)
    aed_change_rate = Column(Numeric)
    qar_change_rate = Column(Numeric)
    huf_change_rate = Column(Numeric)
    pln_change_rate = Column(Numeric)
    dkk_change_rate = Column(Numeric)
    sek_change_rate = Column(Numeric)
    nok_change_rate = Column(Numeric)
    try_change_rate = Column(Numeric)
    php_change_rate = Column(Numeric)
    thb_change_rate = Column(Numeric)
    mop_change_rate = Column(Numeric)
    last_modified = Column(
        DateTime(timezone=True),
        default=func.current_timestamp(),
        nullable=False,
        comment="Last Modified Timestamp",
    )


class stock_zh_a_hist_em(Base):
    __tablename__ = "stock_zh_a_hist_em"
    __table_args__ = (
        PrimaryKeyConstraint("symbol", "date"),
        {"comment": "public.stock_zh_a_hist_em definition"},
    )

    symbol = Column(String, comment="代码 (Code)")
    date = Column(Date, comment="日期 (Date)")
    open = Column(Numeric, comment="开盘 (Opening Price)")
    close = Column(Numeric, comment="收盘 (Closing Price)")
    high = Column(Numeric, comment="最高 (Highest Price)")
    low = Column(Numeric, comment="最低 (Lowest Price)")
    volume = Column(Numeric, comment="成交量 (Volume)")
    turnover = Column(Numeric, comment="成交额 (Turnover)")
    amplitude = Column(Numeric, comment="振幅 (Amplitude)")
    change_rate = Column(Numeric, comment="涨跌幅 (Price Change Percentage)")
    change_amt = Column(Numeric, comment="涨跌额 (Price Change Amount)")
    turnover_rate = Column(Numeric, comment="换手率 (Turnover Rate)")
    turnover_change_rate = Column(Numeric)
    open_preclose_rate = Column(Numeric)
    high_preclose_rate = Column(Numeric)
    low_preclose_rate = Column(Numeric)
    vol_change_rate = Column(Numeric)
    last_modified = Column(
        DateTime(timezone=True),
        default=func.current_timestamp(),
        nullable=False,
        comment="Last Modified Timestamp",
    )


class stock_zh_a_spot_em(Base):
    __tablename__ = "stock_zh_a_spot_em"
    __table_args__ = {"comment": "public.stock_zh_a_spot_em definition"}

    serial_no = Column(Integer, comment="序号 (Serial Number)")
    symbol = Column(String, primary_key=True, comment="代码 (Code)")
    name = Column(String, comment="名称 (Name)")
    latest_price = Column(Numeric, comment="最新价 (Latest Price)")
    price_change_pct = Column(Numeric, comment="涨跌幅 (Price Change Percentage)")
    price_change_amt = Column(Numeric, comment="涨跌额 (Price Change Amount)")
    volume = Column(Numeric, comment="成交量 (Volume)")
    turnover = Column(Numeric, comment="成交额 (Turnover)")
    amplitude = Column(Numeric, comment="振幅 (Amplitude)")
    highest = Column(Numeric, comment="最高 (Highest)")
    lowest = Column(Numeric, comment="最低 (Lowest)")
    open_today = Column(Numeric, comment="今开 (Open Today)")
    close_yesterday = Column(Numeric, comment="昨收 (Close Yesterday)")
    volume_ratio = Column(Numeric, comment="量比 (Volume Ratio)")
    turnover_rate = Column(Numeric, comment="换手率 (Turnover Rate)")
    pe_ratio_dynamic = Column(Numeric, comment="市盈率-动态 (Dynamic P/E Ratio)")
    pb_ratio = Column(Numeric, comment="市净率 (P/B Ratio)")
    total_market_value = Column(Numeric, comment="总市值 (Total Market Value)")
    circulating_market_value = Column(
        Numeric, comment="流通市值 (Circulating Market Value)"
    )
    rise_speed = Column(Numeric, comment="涨速 (Rise Speed)")
    five_min_change = Column(Numeric, comment="5分钟涨跌 (5-minute Price Change)")
    sixty_day_change_pct = Column(
        Numeric, comment="60日涨跌幅 (60-day Price Change Percentage)"
    )
    ytd_change_pct = Column(
        Numeric, comment="年初至今涨跌幅 (Year-to-date Price Change Percentage)"
    )
    last_modified = Column(
        DateTime(timezone=True),
        default=func.current_timestamp(),
        nullable=False,
        comment="Last Modified Timestamp",
    )


class bond_zh_hs_daily(Base):
    __tablename__ = "bond_zh_hs_daily"
    __table_args__ = {"comment": "Daily market bond data"}

    symbol = Column(String, primary_key=True, comment="Symbol")
    date = Column(Date, primary_key=True, comment="Date")
    open = Column(Numeric, comment="Open Price")
    high = Column(Numeric, comment="High Price")
    low = Column(Numeric, comment="Low Price")
    close = Column(Numeric, comment="Close Price")
    volume = Column(Numeric, comment="Trade volume")
    last_modified = Column(
        DateTime(timezone=True),
        default=func.current_timestamp(),
        comment="Last Modified Timestamp",
    )
    open_preclose_rate = Column(Numeric)
    high_preclose_rate = Column(Numeric)
    low_preclose_rate = Column(Numeric)
    change_rate = Column(Numeric)
    vol_change_rate = Column(Numeric)


class bond_zh_hs_spot(Base):
    __tablename__ = "bond_zh_hs_spot"
    __table_args__ = {"comment": "Spot market bond data"}

    symbol = Column(String, primary_key=True, comment="Code")
    name = Column(String, comment="Name")
    close = Column(Numeric, comment="Latest Price")
    change_amount = Column(Numeric, comment="Change Amount")
    change_rate = Column(Numeric, comment="Change Percent")
    bid_price = Column(Numeric, comment="Bid Price")
    ask_price = Column(Numeric, comment="Ask Price")
    prev_close = Column(Numeric, comment="Previous Close")
    open = Column(Numeric, comment="Open Price")
    high = Column(Numeric, comment="High Price")
    low = Column(Numeric, comment="Low Price")
    volume = Column(Numeric, comment="Volume")
    turnover = Column(Numeric, comment="Turnover")
    last_modified = Column(
        DateTime(timezone=True),
        default=func.current_timestamp(),
        nullable=False,
        comment="Last Modified Timestamp",
    )
    last_checked = Column(DateTime(timezone=True))


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
        Column(
            "last_modified",
            DateTime(timezone=True),
            default=func.current_timestamp(),
        ),
        Column("amount", Numeric),
        Column("open_preclose_rate", Numeric),
        Column("high_preclose_rate", Numeric),
        Column("low_preclose_rate", Numeric),
        Column("vol_change_rate", Numeric),
        Column("change_rate", Numeric),
        Column("amt_change_rate", Numeric),
    )


def table_def_ts_features():
    return Table(
        "ts_features",
        MetaData(),
        Column("symbol_table", Text, primary_key=True),
        Column("symbol", Text, primary_key=True),
        Column("cov_table", Text, primary_key=True),
        Column("cov_symbol", Text, primary_key=True),
        Column("feature", Text, primary_key=True),
        Column("date", Date, primary_key=True),
        Column("value", Numeric),
        Column(
            "last_modified",
            DateTime(timezone=True),
            default=func.current_timestamp(),
        ),
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
        Column("open_preclose_rate", Numeric),
        Column("high_preclose_rate", Numeric),
        Column("low_preclose_rate", Numeric),
        Column("change_rate", Numeric),
        Column(
            "last_modified",
            DateTime(timezone=True),
            default=func.current_timestamp(),
            nullable=False,
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
        Column("change_rate", Numeric),
        Column("open_preclose_rate", Numeric),
        Column("high_preclose_rate", Numeric),
        Column("low_preclose_rate", Numeric),
        Column("vol_change_rate", Numeric),
        Column("amt_change_rate", Numeric),
        Column(
            "last_modified",
            DateTime(timezone=True),
            default=func.current_timestamp(),
            nullable=False,
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
            "last_modified",
            DateTime(timezone=True),
            default=func.current_timestamp(),
            nullable=False,
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
            "last_modified",
            DateTime(timezone=True),
            default=func.current_timestamp(),
            nullable=False,
        ),
        Column("src", Text),
        PrimaryKeyConstraint("symbol", name="index_spot_em_pkey"),
    )


def table_def_index_spot_sina():
    return Table(
        "index_spot_sina",
        MetaData(),
        Column("symbol", Text, nullable=False),
        Column("name", Text, nullable=False),
        Column("open", Numeric),
        Column("close", Numeric),
        Column("prev_close", Numeric),
        Column("high", Numeric),
        Column("low", Numeric),
        Column("change_rate", Numeric),
        Column("change_amount", Numeric),
        Column("volume", Numeric),
        Column("amount", Numeric),
        Column(
            "last_modified",
            DateTime(timezone=True),
            default=func.current_timestamp(),
            nullable=False,
        ),
        PrimaryKeyConstraint("symbol", name="index_spot_sina_pkey"),
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
            "last_modified",
            DateTime(timezone=True),
            default=func.current_timestamp(),
            nullable=False,
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
            "last_modified",
            DateTime(timezone=True),
            default=func.current_timestamp(),
            nullable=False,
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
            DateTime(timezone=True),
            default=func.current_timestamp(),
            nullable=False,
            comment="Last modified timestamp (最后修改时间)",
        ),
        Column("turnover_change_rate", Numeric),
        Column("open_preclose_rate", Numeric),
        Column("high_preclose_rate", Numeric),
        Column("low_preclose_rate", Numeric),
        Column("vol_change_rate", Numeric),
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
            DateTime(timezone=True),
            default=func.current_timestamp(),
            nullable=False,
            comment="Last modified timestamp (最后修改时间)",
        ),
        Column("quantile", Numeric, comment="雪球-股债性价比指数-百分位"),
        Column("performance_benchmark", Numeric, comment="雪球-股债性价比指数-分值"),
        Column("china_yield_2y_change_rate", Numeric),
        Column("china_yield_5y_change_rate", Numeric),
        Column("china_yield_10y_change_rate", Numeric),
        Column("china_yield_30y_change_rate", Numeric),
        Column("china_yield_spread_10y_2y_change_rate", Numeric),
        Column("us_yield_2y_change_rate", Numeric),
        Column("us_yield_5y_change_rate", Numeric),
        Column("us_yield_10y_change_rate", Numeric),
        Column("us_yield_30y_change_rate", Numeric),
        Column("us_yield_spread_10y_2y_change_rate", Numeric),
        Column("performance_benchmark_change_rate", Numeric),
        PrimaryKeyConstraint("date", name="bond_metrics_em_pk"),
    )


class ta_price_trends(Base):
    __tablename__ = "ta_price_trends"

    symbol = Column(Text, primary_key=True, nullable=False)
    date = Column(Date, primary_key=True, nullable=False)
    atr_stop = Column(Numeric, nullable=True)
    aroon_up = Column(Numeric, nullable=True)
    aroon_down = Column(Numeric, nullable=True)
    aroon_oscillator = Column(Numeric, nullable=True)
    adx_pdi = Column(Numeric, nullable=True)
    adx_mdi = Column(Numeric, nullable=True)
    adx_adx = Column(Numeric, nullable=True)
    adx_adxr = Column(Numeric, nullable=True)
    elder_ray_ema = Column(Numeric, nullable=True)
    elder_ray_bull_power = Column(Numeric, nullable=True)
    elder_ray_bear_power = Column(Numeric, nullable=True)
    gator_upper = Column(Numeric, nullable=True)
    gator_lower = Column(Numeric, nullable=True)
    gator_is_upper_expanding = Column(Numeric, nullable=True)
    gator_is_lower_expanding = Column(Numeric, nullable=True)
    hurst_exponent = Column(Numeric, nullable=True)
    ichimoku_tenkan_sen = Column(Numeric, nullable=True)
    ichimoku_kijun_sen = Column(Numeric, nullable=True)
    ichimoku_senkou_span_a = Column(Numeric, nullable=True)
    ichimoku_senkou_span_b = Column(Numeric, nullable=True)
    ichimoku_chikou_span = Column(Numeric, nullable=True)
    macd_macd = Column(Numeric, nullable=True)
    macd_signal = Column(Numeric, nullable=True)
    macd_histogram = Column(Numeric, nullable=True)
    macd_fast_ema = Column(Numeric, nullable=True)
    macd_slow_ema = Column(Numeric, nullable=True)
    super_trend_super_trend = Column(Numeric, nullable=True)
    super_trend_upper_band = Column(Numeric, nullable=True)
    super_trend_lower_band = Column(Numeric, nullable=True)
    vortex_pvi = Column(Numeric, nullable=True)
    vortex_nvi = Column(Numeric, nullable=True)
    alligator_jaw = Column(Numeric, nullable=True)
    alligator_teeth = Column(Numeric, nullable=True)
    alligator_lips = Column(Numeric, nullable=True)
    last_modified = Column(
        DateTime(timezone=True), nullable=False, default=func.current_timestamp()
    )
    table = Column(Text, primary_key=True, nullable=False)


class ta_ma(Base):
    __tablename__ = "ta_ma"

    symbol = Column(Text, primary_key=True, nullable=False)
    date = Column(Date, primary_key=True, nullable=False)
    alma_alma = Column(Numeric, nullable=True)
    dema_dema9 = Column(Numeric, nullable=True)
    dema_dema20 = Column(Numeric, nullable=True)
    dema_dema50 = Column(Numeric, nullable=True)
    epma_epma5 = Column(Numeric, nullable=True)
    epma_epma20 = Column(Numeric, nullable=True)
    epma_epma100 = Column(Numeric, nullable=True)
    ema_ema5 = Column(Numeric, nullable=True)
    ema_ema20 = Column(Numeric, nullable=True)
    ema_ema50 = Column(Numeric, nullable=True)
    ht_trendline_trendline = Column(Numeric, nullable=True)
    ht_trendline_dc_periods = Column(Numeric, nullable=True)
    ht_trendline_smooth_price = Column(Numeric, nullable=True)
    hma_hma9 = Column(Numeric, nullable=True)
    hma_hma20 = Column(Numeric, nullable=True)
    hma_hma50 = Column(Numeric, nullable=True)
    kama_efficiency_ratio = Column(Numeric, nullable=True)
    kama_kama = Column(Numeric, nullable=True)
    mama_mama = Column(Numeric, nullable=True)
    mama_fama = Column(Numeric, nullable=True)
    dynamic_dynamic10 = Column(Numeric, nullable=True)
    dynamic_dynamic30 = Column(Numeric, nullable=True)
    dynamic_dynamic100 = Column(Numeric, nullable=True)
    smma_smma5 = Column(Numeric, nullable=True)
    smma_smma20 = Column(Numeric, nullable=True)
    smma_smma100 = Column(Numeric, nullable=True)
    sma_sma5 = Column(Numeric, nullable=True)
    sma_sma20 = Column(Numeric, nullable=True)
    sma_sma100 = Column(Numeric, nullable=True)
    t3_t3 = Column(Numeric, nullable=True)
    tema_tema5 = Column(Numeric, nullable=True)
    tema_tema20 = Column(Numeric, nullable=True)
    tema_tema100 = Column(Numeric, nullable=True)
    vwap_vwap = Column(Numeric, nullable=True)
    vwma_vwma5 = Column(Numeric, nullable=True)
    vwma_vwma20 = Column(Numeric, nullable=True)
    vwma_vwma100 = Column(Numeric, nullable=True)
    wma_wma5 = Column(Numeric, nullable=True)
    wma_wma20 = Column(Numeric, nullable=True)
    wma_wma100 = Column(Numeric, nullable=True)
    last_modified = Column(
        DateTime(timezone=True), nullable=False, default=func.current_timestamp()
    )
    table = Column(Text, primary_key=True, nullable=False)


class ta_numerical_analysis(Base):
    __tablename__ = "ta_numerical_analysis"

    symbol = Column(Text, primary_key=True, nullable=False)
    date = Column(Date, primary_key=True, nullable=False)
    slope_slope10 = Column(Numeric, nullable=True)
    slope_slope30 = Column(Numeric, nullable=True)
    slope_slope100 = Column(Numeric, nullable=True)
    slope_intercept10 = Column(Numeric, nullable=True)
    slope_intercept30 = Column(Numeric, nullable=True)
    slope_intercept100 = Column(Numeric, nullable=True)
    slope_stdev10 = Column(Numeric, nullable=True)
    slope_stdev30 = Column(Numeric, nullable=True)
    slope_stdev100 = Column(Numeric, nullable=True)
    slope_r_squared10 = Column(Numeric, nullable=True)
    slope_r_squared30 = Column(Numeric, nullable=True)
    slope_r_squared100 = Column(Numeric, nullable=True)
    slope_line10 = Column(Numeric, nullable=True)
    slope_line30 = Column(Numeric, nullable=True)
    slope_line100 = Column(Numeric, nullable=True)
    last_modified = Column(
        DateTime(timezone=True), nullable=False, default=func.current_timestamp()
    )
    table = Column(Text, primary_key=True, nullable=False)


class ta_oscillators(Base):
    __tablename__ = "ta_oscillators"

    symbol = Column(Text, primary_key=True, nullable=False)
    date = Column(Date, primary_key=True, nullable=False)
    ao_oscillator = Column(Numeric, nullable=True)
    ao_normalized = Column(Numeric, nullable=True)
    cmo_cmo = Column(Numeric, nullable=True)
    cci_cci = Column(Numeric, nullable=True)
    connors_rsi_rsi_close = Column(Numeric, nullable=True)
    connors_rsi_rsi_streak = Column(Numeric, nullable=True)
    connors_rsi_percent_rank = Column(Numeric, nullable=True)
    connors_rsi_connors_rsi = Column(Numeric, nullable=True)
    dpo_sma = Column(Numeric, nullable=True)
    dpo_dpo = Column(Numeric, nullable=True)
    stoch_oscillator = Column(Numeric, nullable=True)
    stoch_signal = Column(Numeric, nullable=True)
    stoch_percent_j = Column(Numeric, nullable=True)
    rsi_rsi = Column(Numeric, nullable=True)
    stc_stc = Column(Numeric, nullable=True)
    smi_smi = Column(Numeric, nullable=True)
    smi_signal = Column(Numeric, nullable=True)
    stoch_rsi_stoch_rsi = Column(Numeric, nullable=True)
    stoch_rsi_signal = Column(Numeric, nullable=True)
    trix_ema3 = Column(Numeric, nullable=True)
    trix_trix = Column(Numeric, nullable=True)
    trix_signal = Column(Numeric, nullable=True)
    ultimate_ultimate = Column(Numeric, nullable=True)
    williams_r_williams_r = Column(Numeric, nullable=True)
    last_modified = Column(
        DateTime(timezone=True), nullable=False, default=func.current_timestamp()
    )
    table = Column(Text, primary_key=True, nullable=False)


class ta_other_price_patterns(Base):
    __tablename__ = "ta_other_price_patterns"

    symbol = Column(Text, primary_key=True, nullable=False)
    date = Column(Date, primary_key=True, nullable=False)
    pivots_high_point = Column(Numeric, nullable=True)
    pivots_low_point = Column(Numeric, nullable=True)
    pivots_high_line = Column(Numeric, nullable=True)
    pivots_low_line = Column(Numeric, nullable=True)
    pivots_high_trend = Column(Numeric, nullable=True)
    pivots_low_trend = Column(Numeric, nullable=True)
    fractal_bear = Column(Numeric, nullable=True)
    fractal_bull = Column(Numeric, nullable=True)
    last_modified = Column(
        DateTime(timezone=True), nullable=False, default=func.current_timestamp()
    )
    table = Column(Text, primary_key=True, nullable=False)


class ta_price_channel(Base):
    __tablename__ = "ta_price_channel"

    symbol = Column(Text, primary_key=True, nullable=False)
    date = Column(Date, primary_key=True, nullable=False)
    bollinger_sma = Column(Numeric, nullable=True)
    bollinger_upper_band = Column(Numeric, nullable=True)
    bollinger_lower_band = Column(Numeric, nullable=True)
    bollinger_percent_b = Column(Numeric, nullable=True)
    bollinger_z_score = Column(Numeric, nullable=True)
    bollinger_width = Column(Numeric, nullable=True)
    donchian_upper_band = Column(Numeric, nullable=True)
    donchian_center_line = Column(Numeric, nullable=True)
    donchian_lower_band = Column(Numeric, nullable=True)
    donchian_width = Column(Numeric, nullable=True)
    fcb_upper_band = Column(Numeric, nullable=True)
    fcb_lower_band = Column(Numeric, nullable=True)
    keltner_upper_band = Column(Numeric, nullable=True)
    keltner_center_line = Column(Numeric, nullable=True)
    keltner_lower_band = Column(Numeric, nullable=True)
    keltner_width = Column(Numeric, nullable=True)
    ma_envelopes_center_line = Column(Numeric, nullable=True)
    ma_envelopes_upper_envelope = Column(Numeric, nullable=True)
    ma_envelopes_lower_envelope = Column(Numeric, nullable=True)
    pivot_points_r3 = Column(Numeric, nullable=True)
    pivot_points_r2 = Column(Numeric, nullable=True)
    pivot_points_r1 = Column(Numeric, nullable=True)
    pivot_points_pp = Column(Numeric, nullable=True)
    pivot_points_s1 = Column(Numeric, nullable=True)
    pivot_points_s2 = Column(Numeric, nullable=True)
    pivot_points_s3 = Column(Numeric, nullable=True)
    rolling_pivots_r3 = Column(Numeric, nullable=True)
    rolling_pivots_r2 = Column(Numeric, nullable=True)
    rolling_pivots_r1 = Column(Numeric, nullable=True)
    rolling_pivots_pp = Column(Numeric, nullable=True)
    rolling_pivots_s1 = Column(Numeric, nullable=True)
    rolling_pivots_s2 = Column(Numeric, nullable=True)
    rolling_pivots_s3 = Column(Numeric, nullable=True)
    starc_bands_upper_band = Column(Numeric, nullable=True)
    starc_bands_center_line = Column(Numeric, nullable=True)
    starc_bands_lower_band = Column(Numeric, nullable=True)
    stdev_channels_center_line = Column(Numeric, nullable=True)
    stdev_channels_upper_channel = Column(Numeric, nullable=True)
    stdev_channels_lower_channel = Column(Numeric, nullable=True)
    stdev_channels_break_point = Column(Numeric, nullable=True)
    last_modified = Column(
        DateTime(timezone=True), nullable=False, default=func.current_timestamp()
    )
    table = Column(Text, primary_key=True, nullable=False)


class ta_price_characteristics(Base):
    __tablename__ = "ta_price_characteristics"

    symbol = Column(Text, primary_key=True, nullable=False)
    date = Column(Date, primary_key=True, nullable=False)
    atr_tr = Column(Numeric, nullable=True)
    atr_atr = Column(Numeric, nullable=True)
    atr_atrp = Column(Numeric, nullable=True)
    bop_bop = Column(Numeric, nullable=True)
    chop_chop = Column(Numeric, nullable=True)
    stdev_mean10 = Column(Numeric, nullable=True)
    stdev_mean30 = Column(Numeric, nullable=True)
    stdev_mean100 = Column(Numeric, nullable=True)
    stdev_stdev10 = Column(Numeric, nullable=True)
    stdev_stdev30 = Column(Numeric, nullable=True)
    stdev_stdev100 = Column(Numeric, nullable=True)
    stdev_stdev_sma10 = Column(Numeric, nullable=True)
    stdev_stdev_sma30 = Column(Numeric, nullable=True)
    stdev_stdev_sma100 = Column(Numeric, nullable=True)
    stdev_z_score10 = Column(Numeric, nullable=True)
    stdev_z_score30 = Column(Numeric, nullable=True)
    stdev_z_score100 = Column(Numeric, nullable=True)
    roc_momentum5 = Column(Numeric, nullable=True)
    roc_momentum20 = Column(Numeric, nullable=True)
    roc_momentum50 = Column(Numeric, nullable=True)
    roc_roc5 = Column(Numeric, nullable=True)
    roc_roc20 = Column(Numeric, nullable=True)
    roc_roc50 = Column(Numeric, nullable=True)
    roc_roc_sma5 = Column(Numeric, nullable=True)
    roc_roc_sma20 = Column(Numeric, nullable=True)
    roc_roc_sma50 = Column(Numeric, nullable=True)
    roc2_with_band_roc5 = Column(Numeric, nullable=True)
    roc2_with_band_roc20 = Column(Numeric, nullable=True)
    roc2_with_band_roc50 = Column(Numeric, nullable=True)
    roc2_with_band_roc_ema5 = Column(Numeric, nullable=True)
    roc2_with_band_roc_ema20 = Column(Numeric, nullable=True)
    roc2_with_band_roc_ema50 = Column(Numeric, nullable=True)
    roc2_with_band_upper_band5 = Column(Numeric, nullable=True)
    roc2_with_band_upper_band20 = Column(Numeric, nullable=True)
    roc2_with_band_upper_band50 = Column(Numeric, nullable=True)
    roc2_with_band_lower_band5 = Column(Numeric, nullable=True)
    roc2_with_band_lower_band20 = Column(Numeric, nullable=True)
    roc2_with_band_lower_band50 = Column(Numeric, nullable=True)
    pmo_pmo = Column(Numeric, nullable=True)
    pmo_signal = Column(Numeric, nullable=True)
    tsi_tsi = Column(Numeric, nullable=True)
    tsi_signal = Column(Numeric, nullable=True)
    ulcer_index_ui = Column(Numeric, nullable=True)
    last_modified = Column(
        DateTime(timezone=True), nullable=False, default=func.current_timestamp()
    )
    table = Column(Text, primary_key=True, nullable=False)


class ta_price_transforms(Base):
    __tablename__ = "ta_price_transforms"

    symbol = Column(Text, primary_key=True, nullable=False)
    date = Column(Date, primary_key=True, nullable=False)
    fisher_transform_fisher = Column(Numeric, nullable=True)
    fisher_transform_trigger = Column(Numeric, nullable=True)
    heikin_ashi_open = Column(Numeric, nullable=True)
    heikin_ashi_high = Column(Numeric, nullable=True)
    heikin_ashi_low = Column(Numeric, nullable=True)
    heikin_ashi_close = Column(Numeric, nullable=True)
    heikin_ashi_volume = Column(Numeric, nullable=True)
    renko_open = Column(Numeric, nullable=True)
    renko_high = Column(Numeric, nullable=True)
    renko_low = Column(Numeric, nullable=True)
    renko_close = Column(Numeric, nullable=True)
    renko_volume = Column(Numeric, nullable=True)
    renko_is_up = Column(Numeric, nullable=True)
    renko_atr_open = Column(Numeric, nullable=True)
    renko_atr_high = Column(Numeric, nullable=True)
    renko_atr_low = Column(Numeric, nullable=True)
    renko_atr_close = Column(Numeric, nullable=True)
    renko_atr_volume = Column(Numeric, nullable=True)
    renko_atr_is_up = Column(Numeric, nullable=True)
    zig_zag_zig_zag = Column(Numeric, nullable=True)
    zig_zag_point_type = Column(Numeric, nullable=True)
    zig_zag_retrace_high = Column(Numeric, nullable=True)
    zig_zag_retrace_low = Column(Numeric, nullable=True)
    last_modified = Column(
        DateTime(timezone=True), nullable=False, default=func.current_timestamp()
    )
    table = Column(Text, primary_key=True, nullable=False)


class ta_stop_reverse(Base):
    __tablename__ = "ta_stop_reverse"

    symbol = Column(Text, primary_key=True, nullable=False)
    date = Column(Date, primary_key=True, nullable=False)
    chandelier_chandelier_exit = Column(Numeric, nullable=True)
    parabolic_sar_sar = Column(Numeric, nullable=True)
    parabolic_sar_is_reversal = Column(Numeric, nullable=True)
    volatility_stop_sar = Column(Numeric, nullable=True)
    volatility_stop_is_stop = Column(Numeric, nullable=True)
    volatility_stop_upper_band = Column(Numeric, nullable=True)
    volatility_stop_lower_band = Column(Numeric, nullable=True)
    last_modified = Column(
        DateTime(timezone=True), nullable=False, default=func.current_timestamp()
    )
    table = Column(Text, primary_key=True, nullable=False)


class ta_volume_based(Base):
    __tablename__ = "ta_volume_based"

    symbol = Column(Text, primary_key=True, nullable=False)
    date = Column(Date, primary_key=True, nullable=False)
    adl_money_flow_multiplier = Column(Numeric, nullable=True)
    adl_money_flow_volume = Column(Numeric, nullable=True)
    adl_adl = Column(Numeric, nullable=True)
    adl_adl_sma = Column(Numeric, nullable=True)
    cmf_money_flow_multiplier = Column(Numeric, nullable=True)
    cmf_money_flow_volume = Column(Numeric, nullable=True)
    cmf_cmf = Column(Numeric, nullable=True)
    chaikin_osc_money_flow_multiplier = Column(Numeric, nullable=True)
    chaikin_osc_money_flow_volume = Column(Numeric, nullable=True)
    chaikin_osc_adl = Column(Numeric, nullable=True)
    chaikin_osc_oscillator = Column(Numeric, nullable=True)
    force_index_force_index = Column(Numeric, nullable=True)
    kvo_oscillator = Column(Numeric, nullable=True)
    kvo_signal = Column(Numeric, nullable=True)
    mfi_mfi = Column(Numeric, nullable=True)
    obv_obv = Column(Numeric, nullable=True)
    obv_obv_sma = Column(Numeric, nullable=True)
    pvo_pvo = Column(Numeric, nullable=True)
    pvo_signal = Column(Numeric, nullable=True)
    pvo_histogram = Column(Numeric, nullable=True)
    last_modified = Column(
        DateTime(timezone=True), nullable=False, default=func.current_timestamp()
    )
    table = Column(Text, primary_key=True, nullable=False)
