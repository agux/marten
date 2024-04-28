import requests
import pandas as pd

def stock_bond_ratio_index():
    url = "https://danjuanfunds.com/djapi/fundx/base/index/stock/bond/line?time=60"
    headers = {
        "Host": "danjuanfunds.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.3",
    }
    r = requests.get(url, headers=headers)
    json_data = r.json()
    df = pd.DataFrame.from_dict(json_data["data"], orient="columns")
    return df

if __name__ == "__main__":
    df = stock_bond_ratio_index()
    print(df)
