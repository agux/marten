import requests
import pandas as pd


class SnowballAPI:
    @staticmethod
    def stock_bond_ratio_index():
        url = "https://danjuanfunds.com/djapi/fundx/base/index/stock/bond/line?time=60"
        headers = {
            "Host": "danjuanfunds.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.3",
        }
        r = requests.get(url, headers=headers)
        json_data = r.json()
        df = pd.DataFrame.from_dict(json_data["data"], orient="columns")
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.date
        df["quantile"] = pd.to_numeric(df["quantile"], errors="coerce")
        df["performance_benchmark"] = pd.to_numeric(
            df["performance_benchmark"], errors="coerce"
        )
        return df

if __name__ == "__main__":
    df = SnowballAPI.stock_bond_ratio_index()
    print(df)
