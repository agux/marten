import requests
import pandas as pd
import datetime
import py_mini_racer

from akshare.stock.cons import hk_js_decode

from marten.utils.url import make_request

zh_sina_bond_hs_hist_url = (
    "https://finance.sina.com.cn/realstock/company/{}/hisdata/klc_kl.js?d={}"
)

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.3"

class SinaAPI:

    @staticmethod
    def bond_zh_hs_daily(symbol: str = "sh010107") -> pd.DataFrame:
        """
        新浪财经-债券-沪深债券-历史行情数据, 大量抓取容易封 IP
        https://vip.stock.finance.sina.com.cn/mkt/#hs_z
        :param symbol: 沪深债券代码; e.g., sh010107
        :type symbol: str
        :return: 指定沪深债券代码的日 K 线数据
        :rtype: pandas.DataFrame
        """

        headers = {
            "User-Agent": user_agent,
        }
        # r = requests.get(
        #     zh_sina_bond_hs_hist_url.format(
        #         symbol, datetime.datetime.now().strftime("%Y_%m_%d")
        #     ),
        #     headers=headers,
        # )
        r = make_request(
            zh_sina_bond_hs_hist_url.format(
                symbol, datetime.datetime.now().strftime("%Y_%m_%d")
            ),
            headers=headers,
        )
        js_code = py_mini_racer.MiniRacer()
        js_code.eval(hk_js_decode)
        dict_list = js_code.call(
            "d", r.text.split("=")[1].split(";")[0].replace('"', "")
        )  # 执行 js 解密代码
        data_df = pd.DataFrame(dict_list)
        data_df["date"] = pd.to_datetime(data_df["date"], errors="coerce").dt.date
        data_df["open"] = pd.to_numeric(data_df["open"], errors="coerce")
        data_df["high"] = pd.to_numeric(data_df["high"], errors="coerce")
        data_df["low"] = pd.to_numeric(data_df["low"], errors="coerce")
        data_df["close"] = pd.to_numeric(data_df["close"], errors="coerce")
        return data_df
