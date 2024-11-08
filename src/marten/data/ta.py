import pandas as pd
from sqlalchemy import text
from dask.distributed import worker_client, get_worker

from stock_indicators import indicators
from stock_indicators.indicators.common.quote import Quote
from stock_indicators.indicators.common.enums import PeriodSize

from marten.utils.worker import await_futures
from marten.data.db import update_on_conflict
from marten.data.tabledef import (
    ta_ma,
    ta_numerical_analysis,
    ta_oscillators,
    ta_other_price_patterns,
    ta_price_channel,
    ta_price_characteristics,
    ta_price_transforms,
    ta_price_trends,
    ta_stop_reverse,
    ta_volume_based,
)

def tofl(value, mapping=None):
    if value is None:
        return value
    else:
        return float(value) if mapping is None else mapping[value]

def calc_ta():
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    total = 0
    futures = []
    # load symbol list from the asset tables
    etf_list = pd.read_sql(
        "select distinct symbol from fund_etf_daily_em", alchemyEngine
    )
    logger.info("%s ETF", len(etf_list))
    total += len(etf_list)
    cn_index_list = pd.read_sql(
        "SELECT distinct symbol FROM index_daily_em", alchemyEngine
    )
    logger.info("%s CN indices", len(cn_index_list))
    total += len(cn_index_list)
    bond_list = pd.read_sql(
        "select distinct symbol from bond_zh_hs_daily", alchemyEngine
    )
    logger.info("%s bonds", len(bond_list))
    total += len(bond_list)
    stock_list = pd.read_sql(
        "select distinct symbol from stock_zh_a_hist_em", alchemyEngine
    )
    logger.info("%s stocks", len(stock_list))
    total += len(stock_list)
    us_index_list = pd.read_sql(
        "select distinct symbol from us_index_daily_sina", alchemyEngine
    )
    logger.info("%s US indices", len(us_index_list))
    total += len(us_index_list)
    with worker_client() as client:
        for symbol in etf_list["symbol"]:
            futures.append(client.submit(calc_ta_for, symbol, "fund_etf_daily_em", 
                                         key=f"{calc_ta_for.__name__}_ETF--{symbol.lower()}"))
            await_futures(futures, False, multiplier=1.5)
        for symbol in cn_index_list["symbol"]:
            futures.append(
                client.submit(
                    calc_ta_for,
                    symbol,
                    "index_daily_em",
                    key=f"{calc_ta_for.__name__}_INDEX--{symbol.lower()}",
                )
            )
            await_futures(futures, False, multiplier=1.5)
        for symbol in us_index_list:
            futures.append(
                client.submit(
                    calc_ta_for,
                    symbol,
                    "us_index_daily_sina",
                    key=f"{calc_ta_for.__name__}_US_INDEX--{symbol.lower()}",
                )
            )
            await_futures(futures, False, multiplier=1.5)
        for symbol in bond_list["symbol"]:
            futures.append(
                client.submit(
                    calc_ta_for,
                    symbol,
                    "bond_zh_hs_daily",
                    key=f"{calc_ta_for.__name__}_BOND--{symbol.lower()}",
                )
            )
            await_futures(futures, False, multiplier=1.5)
        for symbol in stock_list["symbol"]:
            futures.append(
                client.submit(calc_ta_for, symbol, "stock_zh_a_hist_em",
                              key=f"{calc_ta_for.__name__}_STOCK--{symbol.lower()}")
            )
            await_futures(futures, False, multiplier=1.5)

        await_futures(futures)

    return total

def calc_ta_for(symbol, table):
    worker = get_worker()
    alchemyEngine, logger = worker.alchemyEngine, worker.logger

    # load historical data
    df = load_historical(symbol, alchemyEngine, table)
    quotes_list = [
        Quote(d, o, h, l, c, v)
        for d, o, h, l, c, v in zip(
            df["ds"], df["open"], df["high"], df["low"], df["close"], df["volume"]
        )
    ]
    # calculate all the TAs using common function, organized by TA genre
    price_trends(quotes_list, symbol, table)
    price_channel(quotes_list, symbol, table)
    oscillators(quotes_list, symbol, table)
    stop_reverse(quotes_list, symbol, table)
    other_price_patterns(quotes_list, symbol, table)
    volume_based(quotes_list, symbol, table)
    ma(quotes_list, symbol, table)
    price_transforms(quotes_list, symbol, table)
    price_characteristics(quotes_list, symbol, table)
    numerical_analysis(quotes_list, symbol, table)


def save_ta(ta_table, df):
    worker = get_worker()
    alchemyEngine = worker.alchemyEngine
    with alchemyEngine.begin() as conn:
        update_on_conflict(
            ta_table, conn, df, primary_keys=["table", "symbol", "date"]
        )

def count_ta(ta_table, table, symbol):
    worker = get_worker()
    alchemyEngine = worker.alchemyEngine
    with alchemyEngine.connect() as conn:
        result = conn.execute(
            text(
                f"""
                    select count(*)
                    from {ta_table}
                    where 
                        "table" = :table
                        and symbol = :symbol
                """
            ),
            {
                "table": table,
                "symbol": symbol,
            },
        )
        return result.fetchone()[0]

def numerical_analysis(quotes_list, symbol, table):
    c = count_ta("ta_numerical_analysis", table, symbol)
    if c == len(quotes_list):
        return
    slope10 = indicators.get_slope(quotes_list, lookback_periods=10)
    slope30 = indicators.get_slope(quotes_list, lookback_periods=30)
    slope100 = indicators.get_slope(quotes_list, lookback_periods=100)
    columns = [
        "table",
        "symbol",
        "date",
        "slope_slope10",
        "slope_slope30",
        "slope_slope100",
        "slope_intercept10",
        "slope_intercept30",
        "slope_intercept100",
        "slope_stdev10",
        "slope_stdev30",
        "slope_stdev100",
        "slope_r_squared10",
        "slope_r_squared30",
        "slope_r_squared100",
        "slope_line10",
        "slope_line30",
        "slope_line100",
    ]
    data = []
    for i, q in enumerate(quotes_list):
        data.append(
            [
                table,
                symbol,
                q.date.strftime("%Y-%m-%d"),
                tofl(slope10[i].slope),
                tofl(slope30[i].slope),
                tofl(slope100[i].slope),
                tofl(slope10[i].intercept),
                tofl(slope30[i].intercept),
                tofl(slope100[i].intercept),
                tofl(slope10[i].stdev),
                tofl(slope30[i].stdev),
                tofl(slope100[i].stdev),
                tofl(slope10[i].r_squared),
                tofl(slope30[i].r_squared),
                tofl(slope100[i].r_squared),
                tofl(slope10[i].line),
                tofl(slope30[i].line),
                tofl(slope100[i].line),
            ]
        )
    df = pd.DataFrame(data, columns=columns)
    save_ta(ta_numerical_analysis, df)

def price_characteristics(quotes_list, symbol, table):
    c = count_ta("ta_price_characteristics", table, symbol)
    if c == len(quotes_list):
        return
    atr = indicators.get_atr(quotes_list)
    bop = indicators.get_bop(quotes_list)
    chop = indicators.get_chop(quotes_list)
    stdev10 = indicators.get_stdev(quotes_list, lookback_periods=10, sma_periods=5)
    stdev30 = indicators.get_stdev(quotes_list, lookback_periods=30, sma_periods=5)
    stdev100 = indicators.get_stdev(quotes_list, lookback_periods=100, sma_periods=5)
    roc5 = indicators.get_roc(quotes_list, lookback_periods=5, sma_periods=5)
    roc20 = indicators.get_roc(quotes_list, lookback_periods=20, sma_periods=5)
    roc50 = indicators.get_roc(quotes_list, lookback_periods=50, sma_periods=5)
    roc_with_band5 = indicators.get_roc_with_band(
        quotes_list, lookback_periods=5, ema_periods=3, std_dev_periods=5
    )
    roc_with_band20 = indicators.get_roc_with_band(
        quotes_list, lookback_periods=20, ema_periods=3, std_dev_periods=5
    )
    roc_with_band50 = indicators.get_roc_with_band(
        quotes_list, lookback_periods=50, ema_periods=3, std_dev_periods=5
    )
    pmo = indicators.get_pmo(quotes_list)
    tsi = indicators.get_tsi(quotes_list)
    ulcer_index = indicators.get_ulcer_index(quotes_list)
    columns = [
        "table",
        "symbol",
        "date",
        "atr_tr",
        "atr_atr",
        "atr_atrp",
        "bop_bop",
        "chop_chop",
        "stdev_mean10",
        "stdev_mean30",
        "stdev_mean100",
        "stdev_stdev10",
        "stdev_stdev30",
        "stdev_stdev100",
        "stdev_stdev_sma10",
        "stdev_stdev_sma30",
        "stdev_stdev_sma100",
        "stdev_z_score10",
        "stdev_z_score30",
        "stdev_z_score100",
        "roc_momentum5",
        "roc_momentum20",
        "roc_momentum50",
        "roc_roc5",
        "roc_roc20",
        "roc_roc50",
        "roc_roc_sma5",
        "roc_roc_sma20",
        "roc_roc_sma50",
        "roc2_with_band_roc5",
        "roc2_with_band_roc20",
        "roc2_with_band_roc50",
        "roc2_with_band_roc_ema5",
        "roc2_with_band_roc_ema20",
        "roc2_with_band_roc_ema50",
        "roc2_with_band_upper_band5",
        "roc2_with_band_upper_band20",
        "roc2_with_band_upper_band50",
        "roc2_with_band_lower_band5",
        "roc2_with_band_lower_band20",
        "roc2_with_band_lower_band50",
        "pmo_pmo",
        "pmo_signal",
        "tsi_tsi",
        "tsi_signal",
        "ulcer_index_ui",
    ]
    data = []
    for i, q in enumerate(quotes_list):
        data.append(
            [
                table,
                symbol,
                q.date.strftime("%Y-%m-%d"),
                tofl(atr[i].tr),
                tofl(atr[i].atr),
                tofl(atr[i].atrp),
                tofl(bop[i].bop),
                tofl(chop[i].chop),
                tofl(stdev10[i].mean),
                tofl(stdev30[i].mean),
                tofl(stdev100[i].mean),
                tofl(stdev10[i].stdev),
                tofl(stdev30[i].stdev),
                tofl(stdev100[i].stdev),
                tofl(stdev10[i].stdev_sma),
                tofl(stdev30[i].stdev_sma),
                tofl(stdev100[i].stdev_sma),
                tofl(stdev10[i].z_score),
                tofl(stdev30[i].z_score),
                tofl(stdev100[i].z_score),
                tofl(roc5[i].momentum),
                tofl(roc20[i].momentum),
                tofl(roc50[i].momentum),
                tofl(roc5[i].roc),
                tofl(roc20[i].roc),
                tofl(roc50[i].roc),
                tofl(roc5[i].roc_sma),
                tofl(roc20[i].roc_sma),
                tofl(roc50[i].roc_sma),
                tofl(roc_with_band5[i].roc),
                tofl(roc_with_band20[i].roc),
                tofl(roc_with_band50[i].roc),
                tofl(roc_with_band5[i].roc_ema),
                tofl(roc_with_band20[i].roc_ema),
                tofl(roc_with_band50[i].roc_ema),
                tofl(roc_with_band5[i].upper_band),
                tofl(roc_with_band20[i].upper_band),
                tofl(roc_with_band50[i].upper_band),
                tofl(roc_with_band5[i].lower_band),
                tofl(roc_with_band20[i].lower_band),
                tofl(roc_with_band50[i].lower_band),
                tofl(pmo[i].pmo),
                tofl(pmo[i].signal),
                tofl(tsi[i].tsi),
                tofl(tsi[i].signal),
                tofl(ulcer_index[i].ui),
            ]
        )
    df = pd.DataFrame(data, columns=columns)
    save_ta(ta_price_characteristics, df)

def price_transforms(quotes_list, symbol, table):
    c = count_ta("ta_price_transforms", table, symbol)
    if c == len(quotes_list):
        return
    fisher_transform = indicators.get_fisher_transform(quotes_list)
    heikin_ashi = indicators.get_heikin_ashi(quotes_list)
    # NOTE Unlike most indicators in this library, this indicator
    # DOES NOT return the same number of elements as there are in the historical quotes
    # renko = indicators.get_renko(quotes_list, brick_size=2.5)
    # renko_atr = indicators.get_renko_atr(quotes_list, atr_periods=14)
    zig_zag = indicators.get_zig_zag(quotes_list)
    mapping = {
        "H": 1.0,
        "L": 0.0,
    }
    columns = [
        "table",
        "symbol",
        "date",
        "fisher_transform_fisher",
        "fisher_transform_trigger",
        "heikin_ashi_open",
        "heikin_ashi_high",
        "heikin_ashi_low",
        "heikin_ashi_close",
        "heikin_ashi_volume",
        # "renko_open",
        # "renko_high",
        # "renko_low",
        # "renko_close",
        # "renko_volume",
        # "renko_is_up",
        # "renko_atr_open",
        # "renko_atr_high",
        # "renko_atr_low",
        # "renko_atr_close",
        # "renko_atr_volume",
        # "renko_atr_is_up",
        "zig_zag_zig_zag",
        "zig_zag_point_type",
        "zig_zag_retrace_high",
        "zig_zag_retrace_low",
    ]
    data = []
    for i, q in enumerate(quotes_list):
        data.append(
            [
                table,
                symbol,
                q.date.strftime("%Y-%m-%d"),
                tofl(fisher_transform[i].fisher),
                tofl(fisher_transform[i].trigger),
                tofl(heikin_ashi[i].open),
                tofl(heikin_ashi[i].high),
                tofl(heikin_ashi[i].low),
                tofl(heikin_ashi[i].close),
                tofl(heikin_ashi[i].volume),
                # tofl(renko[i].open),
                # tofl(renko[i].high),
                # tofl(renko[i].low),
                # tofl(renko[i].close),
                # tofl(renko[i].volume),
                # tofl(renko[i].is_up),
                # tofl(renko_atr[i].open),
                # tofl(renko_atr[i].high),
                # tofl(renko_atr[i].low),
                # tofl(renko_atr[i].close),
                # tofl(renko_atr[i].volume),
                # tofl(renko_atr[i].is_up),
                tofl(zig_zag[i].zig_zag),
                tofl(zig_zag[i].point_type, mapping),
                tofl(zig_zag[i].retrace_high),
                tofl(zig_zag[i].retrace_low),
            ]
        )
    df = pd.DataFrame(data, columns=columns)
    save_ta(ta_price_transforms, df)

def ma(quotes_list, symbol, table):
    c = count_ta("ta_ma", table, symbol)
    if c == len(quotes_list):
        return
    alma = indicators.get_alma(quotes_list)
    dema9 = indicators.get_dema(quotes_list, lookback_periods=9)
    dema20 = indicators.get_dema(quotes_list, lookback_periods=20)
    dema50 = indicators.get_dema(quotes_list, lookback_periods=50)
    epma5 = indicators.get_epma(quotes_list, lookback_periods=5)
    epma20 = indicators.get_epma(quotes_list, lookback_periods=20)
    epma100 = indicators.get_epma(quotes_list, lookback_periods=100)
    ema5 = indicators.get_ema(quotes_list, lookback_periods=5)
    ema20 = indicators.get_ema(quotes_list, lookback_periods=20)
    ema50 = indicators.get_ema(quotes_list, lookback_periods=50)
    ht_trendline = indicators.get_ht_trendline(quotes_list)
    hma9 = indicators.get_hma(quotes_list, lookback_periods=9)
    hma20 = indicators.get_hma(quotes_list, lookback_periods=20)
    hma50 = indicators.get_hma(quotes_list, lookback_periods=50)
    kama = indicators.get_kama(quotes_list)
    mama = indicators.get_mama(quotes_list)
    dynamic10 = indicators.get_dynamic(quotes_list, lookback_periods=10)
    dynamic30 = indicators.get_dynamic(quotes_list, lookback_periods=30)
    dynamic100 = indicators.get_dynamic(quotes_list, lookback_periods=100)
    smma5 = indicators.get_smma(quotes_list, lookback_periods=5)
    smma20 = indicators.get_smma(quotes_list, lookback_periods=20)
    smma100 = indicators.get_smma(quotes_list, lookback_periods=100)
    sma5 = indicators.get_sma(quotes_list, lookback_periods=5)
    sma20 = indicators.get_sma(quotes_list, lookback_periods=20)
    sma100 = indicators.get_sma(quotes_list, lookback_periods=100)
    t3 = indicators.get_t3(quotes_list)
    tema5 = indicators.get_tema(quotes_list, lookback_periods=5)
    tema20 = indicators.get_tema(quotes_list, lookback_periods=20)
    tema100 = indicators.get_tema(quotes_list, lookback_periods=100)
    vwap = indicators.get_vwap(quotes_list)
    vwma5 = indicators.get_vwma(quotes_list, lookback_periods=5)
    vwma20 = indicators.get_vwma(quotes_list, lookback_periods=20)
    vwma100 = indicators.get_vwma(quotes_list, lookback_periods=100)
    wma5 = indicators.get_wma(quotes_list, lookback_periods=5)
    wma20 = indicators.get_wma(quotes_list, lookback_periods=20)
    wma100 = indicators.get_wma(quotes_list, lookback_periods=100)
    columns = [
        "table",
        "symbol",
        "date",
        "alma_alma",
        "dema_dema9",
        "dema_dema20",
        "dema_dema50",
        "epma_epma5",
        "epma_epma20",
        "epma_epma100",
        "ema_ema5",
        "ema_ema20",
        "ema_ema50",
        "ht_trendline_trendline",
        "ht_trendline_dc_periods",
        "ht_trendline_smooth_price",
        "hma_hma9",
        "hma_hma20",
        "hma_hma50",
        "kama_efficiency_ratio",
        "kama_kama",
        "mama_mama",
        "mama_fama",
        "dynamic_dynamic10",
        "dynamic_dynamic30",
        "dynamic_dynamic100",
        "smma_smma5",
        "smma_smma20",
        "smma_smma100",
        "sma_sma5",
        "sma_sma20",
        "sma_sma100",
        "t3_t3",
        "tema_tema5",
        "tema_tema20",
        "tema_tema100",
        "vwap_vwap",
        "vwma_vwma5",
        "vwma_vwma20",
        "vwma_vwma100",
        "wma_wma5",
        "wma_wma20",
        "wma_wma100",
    ]
    data = []
    for i, q in enumerate(quotes_list):
        data.append(
            [
                table,
                symbol,
                q.date.strftime("%Y-%m-%d"),
                tofl(alma[i].alma),
                tofl(dema9[i].dema),
                tofl(dema20[i].dema),
                tofl(dema50[i].dema),
                tofl(epma5[i].epma),
                tofl(epma20[i].epma),
                tofl(epma100[i].epma),
                tofl(ema5[i].ema),
                tofl(ema20[i].ema),
                tofl(ema50[i].ema),
                tofl(ht_trendline[i].trendline),
                tofl(ht_trendline[i].dc_periods),
                tofl(ht_trendline[i].smooth_price),
                tofl(hma9[i].hma),
                tofl(hma20[i].hma),
                tofl(hma50[i].hma),
                tofl(kama[i].efficiency_ratio),
                tofl(kama[i].kama),
                tofl(mama[i].mama),
                tofl(mama[i].fama),
                tofl(dynamic10[i].dynamic),
                tofl(dynamic30[i].dynamic),
                tofl(dynamic100[i].dynamic),
                tofl(smma5[i].smma),
                tofl(smma20[i].smma),
                tofl(smma100[i].smma),
                tofl(sma5[i].sma),
                tofl(sma20[i].sma),
                tofl(sma100[i].sma),
                tofl(t3[i].t3),
                tofl(tema5[i].tema),
                tofl(tema20[i].tema),
                tofl(tema100[i].tema),
                tofl(vwap[i].vwap),
                tofl(vwma5[i].vwma),
                tofl(vwma20[i].vwma),
                tofl(vwma100[i].vwma),
                tofl(wma5[i].wma),
                tofl(wma20[i].wma),
                tofl(wma100[i].wma),
            ]
        )
    df = pd.DataFrame(data, columns=columns)
    save_ta(ta_ma, df)

def volume_based(quotes_list, symbol, table):
    c = count_ta("ta_volume_based", table, symbol)
    if c == len(quotes_list):
        return
    adl = indicators.get_adl(quotes_list, sma_periods=5)
    cmf = indicators.get_cmf(quotes_list)
    chaikin_osc = indicators.get_chaikin_osc(quotes_list)
    force_index = indicators.get_force_index(quotes_list, lookback_periods=5)
    kvo = indicators.get_kvo(quotes_list)
    mfi = indicators.get_mfi(quotes_list)
    obv = indicators.get_obv(quotes_list, sma_periods=5)
    pvo = indicators.get_pvo(quotes_list)
    columns = [
        "table",
        "symbol",
        "date",
        "adl_money_flow_multiplier",
        "adl_money_flow_volume",
        "adl_adl",
        "adl_adl_sma",
        "cmf_money_flow_multiplier",
        "cmf_money_flow_volume",
        "cmf_cmf",
        "chaikin_osc_money_flow_multiplier",
        "chaikin_osc_money_flow_volume",
        "chaikin_osc_adl",
        "chaikin_osc_oscillator",
        "force_index_force_index",
        "kvo_oscillator",
        "kvo_signal",
        "mfi_mfi",
        "obv_obv",
        "obv_obv_sma",
        "pvo_pvo",
        "pvo_signal",
        "pvo_histogram",
    ]
    data = []
    for i, q in enumerate(quotes_list):
        data.append(
            [
                table,
                symbol,
                q.date.strftime("%Y-%m-%d"),
                tofl(adl[i].money_flow_multiplier),
                tofl(adl[i].money_flow_volume),
                tofl(adl[i].adl),
                tofl(adl[i].adl_sma),
                tofl(cmf[i].money_flow_multiplier),
                tofl(cmf[i].money_flow_volume),
                tofl(cmf[i].cmf),
                tofl(chaikin_osc[i].money_flow_multiplier),
                tofl(chaikin_osc[i].money_flow_volume),
                tofl(chaikin_osc[i].adl),
                tofl(chaikin_osc[i].oscillator),
                tofl(force_index[i].force_index),
                tofl(kvo[i].oscillator),
                tofl(kvo[i].signal),
                tofl(mfi[i].mfi),
                tofl(obv[i].obv),
                tofl(obv[i].obv_sma),
                tofl(pvo[i].pvo),
                tofl(pvo[i].signal),
                tofl(pvo[i].histogram),
            ]
        )
    df = pd.DataFrame(data, columns=columns)
    save_ta(ta_volume_based, df)

def other_price_patterns(quotes_list, symbol, table):
    c = count_ta("ta_other_price_patterns", table, symbol)
    if c == len(quotes_list):
        return
    pivots = indicators.get_pivots(quotes_list)
    fractal = indicators.get_fractal(quotes_list)
    columns = [
        "table",
        "symbol",
        "date",
        "pivots_high_point",
        "pivots_low_point",
        "pivots_high_line",
        "pivots_low_line",
        "pivots_high_trend",
        "pivots_low_trend",
        "fractal_bear",
        "fractal_bull",
    ]
    data = []
    for i, q in enumerate(quotes_list):
        data.append(
            [
                table,
                symbol,
                q.date.strftime("%Y-%m-%d"),
                tofl(pivots[i].high_point),
                tofl(pivots[i].low_point),
                tofl(pivots[i].high_line),
                tofl(pivots[i].low_line),
                tofl(pivots[i].high_trend),
                tofl(pivots[i].low_trend),
                tofl(fractal[i].fractal_bear),
                tofl(fractal[i].fractal_bull),
            ]
        )
    df = pd.DataFrame(data, columns=columns)
    save_ta(ta_other_price_patterns, df)

def stop_reverse(quotes_list, symbol, table):
    c = count_ta("ta_stop_reverse", table, symbol)
    if c == len(quotes_list):
        return
    chandelier = indicators.get_chandelier(quotes_list)
    parabolic_sar = indicators.get_parabolic_sar(quotes_list)
    volatility_stop = indicators.get_volatility_stop(quotes_list)
    columns = [
        "table",
        "symbol",
        "date",
        "chandelier_chandelier_exit",
        "parabolic_sar_sar",
        "parabolic_sar_is_reversal",
        "volatility_stop_sar",
        "volatility_stop_is_stop",
        "volatility_stop_upper_band",
        "volatility_stop_lower_band",
    ]
    data = []
    for i, q in enumerate(quotes_list):
        data.append(
            [
                table,
                symbol,
                q.date.strftime("%Y-%m-%d"),
                tofl(chandelier[i].chandelier_exit),
                tofl(parabolic_sar[i].sar),
                tofl(parabolic_sar[i].is_reversal),
                tofl(volatility_stop[i].sar),
                tofl(volatility_stop[i].is_stop),
                tofl(volatility_stop[i].upper_band),
                tofl(volatility_stop[i].lower_band),
            ]
        )
    df = pd.DataFrame(data, columns=columns)
    save_ta(ta_stop_reverse, df)


def oscillators(quotes_list, symbol, table):
    c = count_ta("ta_oscillators", table, symbol)
    if c == len(quotes_list):
        return
    ao = indicators.get_awesome(quotes_list)
    cmo = indicators.get_cmo(quotes_list, lookback_periods=14)
    cci = indicators.get_cci(quotes_list)
    connors_rsi = indicators.get_connors_rsi(quotes_list)
    dpo = indicators.get_dpo(quotes_list, lookback_periods=20)
    stoch = indicators.get_stoch(quotes_list)
    rsi = indicators.get_rsi(quotes_list)
    stc = indicators.get_stc(quotes_list)
    smi = indicators.get_smi(quotes_list)
    stoch_rsi = indicators.get_stoch_rsi(
        quotes_list, rsi_periods=14, stoch_periods=14, signal_periods=5
    )
    trix = indicators.get_trix(quotes_list, lookback_periods=14)
    ultimate = indicators.get_ultimate(quotes_list)
    williams_r = indicators.get_williams_r(quotes_list)
    columns = [
        "table",
        "symbol",
        "date",
        "ao_oscillator",
        "ao_normalized",
        "cmo_cmo",
        "cci_cci",
        "connors_rsi_rsi_close",
        "connors_rsi_rsi_streak",
        "connors_rsi_percent_rank",
        "connors_rsi_connors_rsi",
        "dpo_sma",
        "dpo_dpo",
        "stoch_oscillator",
        "stoch_signal",
        "stoch_percent_j",
        "rsi_rsi",
        "stc_stc",
        "smi_smi",
        "smi_signal",
        "stoch_rsi_stoch_rsi",
        "stoch_rsi_signal",
        "trix_ema3",
        "trix_trix",
        "trix_signal",
        "ultimate_ultimate",
        "williams_r_williams_r",
    ]
    data = []
    for i, q in enumerate(quotes_list):
        data.append(
            [
                table,
                symbol,
                q.date.strftime("%Y-%m-%d"),
                tofl(ao[i].oscillator),
                tofl(ao[i].normalized),
                tofl(cmo[i].cmo),
                tofl(cci[i].cci),
                tofl(connors_rsi[i].rsi_close),
                tofl(connors_rsi[i].rsi_streak),
                tofl(connors_rsi[i].percent_rank),
                tofl(connors_rsi[i].connors_rsi),
                tofl(dpo[i].sma),
                tofl(dpo[i].dpo),
                tofl(stoch[i].oscillator),
                tofl(stoch[i].signal),
                tofl(stoch[i].percent_j),
                tofl(rsi[i].rsi),
                tofl(stc[i].stc),
                tofl(smi[i].smi),
                tofl(smi[i].signal),
                tofl(stoch_rsi[i].stoch_rsi),
                tofl(stoch_rsi[i].signal),
                tofl(trix[i].ema3),
                tofl(trix[i].trix),
                tofl(trix[i].signal),
                tofl(ultimate[i].ultimate),
                tofl(williams_r[i].williams_r),
            ]
        )
    df = pd.DataFrame(data, columns=columns)
    save_ta(ta_oscillators, df)

def price_channel(quotes_list, symbol, table):
    c = count_ta("ta_price_channel", table, symbol)
    if c == len(quotes_list):
        return
    bollinger = indicators.get_bollinger_bands(quotes_list)
    donchian = indicators.get_donchian(quotes_list)
    fcb = indicators.get_fcb(quotes_list)
    keltner = indicators.get_keltner(quotes_list)
    ma_envelopes = indicators.get_ma_envelopes(quotes_list, lookback_periods=10)
    pivot_points = indicators.get_pivot_points(quotes_list, PeriodSize.DAY)
    rolling_pivots = indicators.get_rolling_pivots(
        quotes_list, window_periods=5, offset_periods=5
    )
    starc_bands = indicators.get_starc_bands(quotes_list)
    stdev_channels = indicators.get_stdev_channels(quotes_list)
    columns = [
        "table",
        "symbol",
        "date",
        "bollinger_sma",
        "bollinger_upper_band",
        "bollinger_lower_band",
        "bollinger_percent_b",
        "bollinger_z_score",
        "bollinger_width",
        "donchian_upper_band",
        "donchian_center_line",
        "donchian_lower_band",
        "donchian_width",
        "fcb_upper_band",
        "fcb_lower_band",
        "keltner_upper_band",
        "keltner_center_line",
        "keltner_lower_band",
        "keltner_width",
        "ma_envelopes_center_line",
        "ma_envelopes_upper_envelope",
        "ma_envelopes_lower_envelope",
        "pivot_points_r3",
        "pivot_points_r2",
        "pivot_points_r1",
        "pivot_points_pp",
        "pivot_points_s1",
        "pivot_points_s2",
        "pivot_points_s3",
        "rolling_pivots_r3",
        "rolling_pivots_r2",
        "rolling_pivots_r1",
        "rolling_pivots_pp",
        "rolling_pivots_s1",
        "rolling_pivots_s2",
        "rolling_pivots_s3",
        "starc_bands_upper_band",
        "starc_bands_center_line",
        "starc_bands_lower_band",
        "stdev_channels_center_line",
        "stdev_channels_upper_channel",
        "stdev_channels_lower_channel",
        "stdev_channels_break_point",
    ]
    data = []
    for i, q in enumerate(quotes_list):
        data.append(
            [
                table,
                symbol,
                q.date.strftime("%Y-%m-%d"),
                tofl(bollinger[i].sma),
                tofl(bollinger[i].upper_band),
                tofl(bollinger[i].lower_band),
                tofl(bollinger[i].percent_b),
                tofl(bollinger[i].z_score),
                tofl(bollinger[i].width),
                tofl(donchian[i].upper_band),
                tofl(donchian[i].center_line),
                tofl(donchian[i].lower_band),
                tofl(donchian[i].width),
                tofl(fcb[i].upper_band),
                tofl(fcb[i].lower_band),
                tofl(keltner[i].upper_band),
                tofl(keltner[i].center_line),
                tofl(keltner[i].lower_band),
                tofl(keltner[i].width),
                tofl(ma_envelopes[i].center_line),
                tofl(ma_envelopes[i].upper_envelope),
                tofl(ma_envelopes[i].lower_envelope),
                tofl(pivot_points[i].r3),
                tofl(pivot_points[i].r2),
                tofl(pivot_points[i].r1),
                tofl(pivot_points[i].pp),
                tofl(pivot_points[i].s1),
                tofl(pivot_points[i].s2),
                tofl(pivot_points[i].s3),
                tofl(rolling_pivots[i].r3),
                tofl(rolling_pivots[i].r2),
                tofl(rolling_pivots[i].r1),
                tofl(rolling_pivots[i].pp),
                tofl(rolling_pivots[i].s1),
                tofl(rolling_pivots[i].s2),
                tofl(rolling_pivots[i].s3),
                tofl(starc_bands[i].upper_band),
                tofl(starc_bands[i].center_line),
                tofl(starc_bands[i].lower_band),
                tofl(stdev_channels[i].center_line),
                tofl(stdev_channels[i].upper_channel),
                tofl(stdev_channels[i].lower_channel),
                tofl(stdev_channels[i].break_point),
            ]
        )
    df = pd.DataFrame(data, columns=columns)
    save_ta(ta_price_channel, df)


def price_trends(quotes_list, symbol, table):
    c = count_ta("ta_price_trends", table, symbol)
    if c == len(quotes_list):
        return
    aroon = indicators.get_aroon(quotes_list)
    adx = indicators.get_adx(quotes_list)
    elder = indicators.get_elder_ray(quotes_list)
    gator = indicators.get_gator(quotes_list)
    hurst = indicators.get_hurst(quotes_list)
    ichimoku = indicators.get_ichimoku(quotes_list)
    macd = indicators.get_macd(quotes_list)
    super_trend = indicators.get_super_trend(quotes_list)
    vortex = indicators.get_vortex(quotes_list, lookback_periods=14)
    alligator = indicators.get_alligator(quotes_list)
    atr_stop = indicators.get_atr_stop(quotes_list)
    columns = [
        "table",
        "symbol",
        "date",
        "atr_stop",
        "aroon_up",
        "aroon_down",
        "aroon_oscillator",
        "adx_pdi",
        "adx_mdi",
        "adx_adx",
        "adx_adxr",
        "elder_ray_ema",
        "elder_ray_bull_power",
        "elder_ray_bear_power",
        "gator_upper",
        "gator_lower",
        "gator_is_upper_expanding",
        "gator_is_lower_expanding",
        "hurst_exponent",
        "ichimoku_tenkan_sen",
        "ichimoku_kijun_sen",
        "ichimoku_senkou_span_a",
        "ichimoku_senkou_span_b",
        "ichimoku_chikou_span",
        "macd_macd",
        "macd_signal",
        "macd_histogram",
        "macd_fast_ema",
        "macd_slow_ema",
        "super_trend_super_trend",
        "super_trend_upper_band",
        "super_trend_lower_band",
        "vortex_pvi",
        "vortex_nvi",
        "alligator_jaw",
        "alligator_teeth",
        "alligator_lips",
    ]
    data = []
    for i, q in enumerate(quotes_list):
        data.append(
            [
                table,
                symbol,
                q.date.strftime("%Y-%m-%d"),
                tofl(atr_stop[i].atr_stop),
                tofl(aroon[i].aroon_up),
                tofl(aroon[i].aroon_down),
                tofl(aroon[i].oscillator),
                tofl(adx[i].pdi),
                tofl(adx[i].mdi),
                tofl(adx[i].adx),
                tofl(adx[i].adxr),
                tofl(elder[i].ema),
                tofl(elder[i].bull_power),
                tofl(elder[i].bear_power),
                tofl(gator[i].upper),
                tofl(gator[i].lower),
                tofl(gator[i].is_upper_expanding),
                tofl(gator[i].is_lower_expanding),
                tofl(hurst[i].hurst_exponent),
                tofl(ichimoku[i].tenkan_sen),
                tofl(ichimoku[i].kijun_sen),
                tofl(ichimoku[i].senkou_span_a),
                tofl(ichimoku[i].senkou_span_b),
                tofl(ichimoku[i].chikou_span),
                tofl(macd[i].macd),
                tofl(macd[i].signal),
                tofl(macd[i].histogram),
                tofl(macd[i].fast_ema),
                tofl(macd[i].slow_ema),
                tofl(super_trend[i].super_trend),
                tofl(super_trend[i].upper_band),
                tofl(super_trend[i].lower_band),
                tofl(vortex[i].pvi),
                tofl(vortex[i].nvi),
                tofl(alligator[i].jaw),
                tofl(alligator[i].teeth),
                tofl(alligator[i].lips),
            ]
        )
    df = pd.DataFrame(data, columns=columns)
    save_ta(ta_price_trends, df)


def load_historical(symbol, alchemyEngine, anchor_table):
    query = f"""
        SELECT date DS, open, high, low, close, volume
        FROM {anchor_table}
        where symbol = %(symbol)s
        and date is not null
        and open <> 'nan' and open is not null
        and high <> 'nan' and high is not null
        and low <> 'nan' and low is not null
        and close <> 'nan' and close is not null
        and volume <> 'nan' and volume is not null
    """
    params = {"symbol": symbol}

    query += " order by DS"

    df = pd.read_sql(
        query,
        alchemyEngine,
        params=params,
        parse_dates=["ds"],
    )

    return df
