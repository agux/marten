import exchange_calendars as xcals
from sqlalchemy import text

def get_holiday_region(alchemyEngine, symbol):
    with alchemyEngine.connect() as conn:
        result = conn.execute(
            text("""SELECT "table" FROM symbol_dict WHERE symbol = :symbol"""),
            {"symbol": symbol},
        ).fetchone()
    if result:
        table = result[0] 
    else:
        raise ValueError(f"cannot infer holiday region for {symbol}")

    if "hk_" in table:
        return "HK"
    if "us_" in table:
        return "US"

    if "fund_etf_daily_em" in table:
        # this may contain QDII ETF for foreign markets
        match symbol:
            case "513800" | "513880" | "159866" | "513520" | "513000":
                return "JP"
            case '513080': # 法国CAC40ETF
                return "FR"
            case "513030":  # 华安德国(DAX)ETF(QDII)
                return "DE"
            # case _:
    
    return "CN"

def get_next_trade_dates(start_date, region, window):
    match region:
        case "HK":
            cal = xcals.get_calendar("XHKG")
        case "US":
            cal = xcals.get_calendar("XNYS")  # New York Stock Exchange
        case "JP":
            cal = xcals.get_calendar("XTKS")
        case "FR":
            cal = xcals.get_calendar("XPAR")
        case "DE":
            cal = xcals.get_calendar("XFRA")
        case _: 
            cal = xcals.get_calendar("XSHG")  # Default to China market

    return cal.sessions_window(
        start_date if isinstance(start_date, str) else start_date.strftime("%Y-%m-%d"),
        window+1,
    )[1:]
