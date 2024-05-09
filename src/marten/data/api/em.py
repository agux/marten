import requests
import pandas as pd
from tqdm import tqdm

from marten.utils.url import make_request
from akshare.utils import demjson
from bs4 import BeautifulSoup
from io import StringIO

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.3"


class EastMoneyAPI:

    @staticmethod
    def fund_portfolio_hold_em(
        symbol: str = "000001", date: str = "2023"
    ) -> pd.DataFrame:
        """
        天天基金网-基金档案-投资组合-基金持仓
        https://fundf10.eastmoney.com/ccmx_000001.html
        :param symbol: 基金代码
        :type symbol: str
        :param date: 查询年份
        :type date: str
        :return: 基金持仓
        :rtype: pandas.DataFrame
        """
        url = "http://fundf10.eastmoney.com/FundArchivesDatas.aspx"
        params = {
            "type": "jjcc",
            "code": symbol,
            "topline": "10000",
            "year": date,
            "month": "",
            "rt": "0.913877030254846",
        }
        headers = {
            "User-Agent": user_agent,
        }
        r = make_request(
            url,
            params=params,
            headers=headers,
            max_timeout=200,
        )
        # r = requests.get(url, params=params)
        data_text = r.text
        data_json = demjson.decode(data_text[data_text.find("{"): -1])
        soup = BeautifulSoup(data_json["content"], "lxml")
        item_label = [
            item.text.split("\xa0\xa0")[1]
            for item in soup.find_all("h4", attrs={"class": "t"})
        ]
        big_df = pd.DataFrame()
        for item in range(len(item_label)):
            temp_df = pd.read_html(StringIO(data_json["content"]), converters={"股票代码": str})[
                item
            ]
            del temp_df["相关资讯"]
            temp_df.rename(
                columns={"占净值 比例": "占净值比例"}, inplace=True
            )
            temp_df["占净值比例"] = (
                temp_df["占净值比例"].str.split("%", expand=True).iloc[:, 0]
            )
            temp_df.rename(
                columns={"持股数（万股）": "持股数", "持仓市值（万元）": "持仓市值"}, inplace=True
            )
            temp_df.rename(
                columns={"持股数 （万股）": "持股数", "持仓市值 （万元）": "持仓市值"}, inplace=True
            )
            temp_df.rename(
                columns={"持股数（万股）": "持股数", "持仓市值（万元人民币）": "持仓市值"}, inplace=True
            )
            temp_df.rename(
                columns={"持股数 （万股）": "持股数", "持仓市值 （万元人民币）": "持仓市值"}, inplace=True
            )

            temp_df["季度"] = item_label[item]
            temp_df = temp_df[
                [
                    "序号",
                    "股票代码",
                    "股票名称",
                    "占净值比例",
                    "持股数",
                    "持仓市值",
                    "季度",
                ]
            ]
            big_df = pd.concat([big_df, temp_df], ignore_index=True)
        big_df["占净值比例"] = pd.to_numeric(big_df["占净值比例"], errors="coerce")
        big_df["持股数"] = pd.to_numeric(big_df["持股数"], errors="coerce")
        big_df["持仓市值"] = pd.to_numeric(big_df["持仓市值"], errors="coerce")
        big_df["序号"] = range(1, len(big_df) + 1)
        return big_df

    @staticmethod
    def fund_portfolio_bond_hold_em(
        symbol: str = "000001", date: str = "2023"
    ) -> pd.DataFrame:
        """
        天天基金网-基金档案-投资组合-债券持仓
        https://fundf10.eastmoney.com/ccmx1_000001.html
        :param symbol: 基金代码
        :type symbol: str
        :param date: 查询年份
        :type date: str
        :return: 债券持仓
        :rtype: pandas.DataFrame
        """
        url = "http://fundf10.eastmoney.com/FundArchivesDatas.aspx"
        params = {
            "type": "zqcc",
            "code": symbol,
            "year": date,
            "rt": "0.913877030254846",
        }
        headers = {
            "User-Agent": user_agent,
        }
        r = make_request(
            url,
            params=params,
            headers=headers,
            max_timeout=200,
        )
        # r = requests.get(url, params=params)
        data_text = r.text
        data_json = demjson.decode(data_text[data_text.find("{") : -1])
        soup = BeautifulSoup(data_json["content"], "lxml")
        item_label = [
            item.text.split("\xa0\xa0")[1]
            for item in soup.find_all("h4", attrs={"class": "t"})
        ]
        big_df = pd.DataFrame()
        for item in range(len(item_label)):
            temp_df = pd.read_html(
                StringIO(data_json["content"]), converters={"债券代码": str}
            )[item]
            temp_df["占净值比例"] = (
                temp_df["占净值比例"].str.split("%", expand=True).iloc[:, 0]
            )
            temp_df.rename(columns={"持仓市值（万元）": "持仓市值"}, inplace=True)
            temp_df["季度"] = item_label[item]
            temp_df = temp_df[
                [
                    "序号",
                    "债券代码",
                    "债券名称",
                    "占净值比例",
                    "持仓市值",
                    "季度",
                ]
            ]
            big_df = pd.concat([big_df, temp_df], ignore_index=True)
        big_df["占净值比例"] = pd.to_numeric(big_df["占净值比例"], errors="coerce")
        big_df["持仓市值"] = pd.to_numeric(big_df["持仓市值"], errors="coerce")
        big_df["序号"] = range(1, len(big_df) + 1)
        return big_df

    @staticmethod
    def fund_fh_em(client=None) -> pd.DataFrame:
        """
        天天基金网-基金数据-分红送配-基金分红
        https://fund.eastmoney.com/data/fundfenhong.html#DJR,desc,1,,,
        :return: 基金分红
        :rtype: pandas.DataFrame
        """
        url = "http://fund.eastmoney.com/Data/funddataIndex_Interface.aspx"
        params = {
            "dt": "8",
            "page": "1",
            "rank": "BZDM",
            "sort": "asc",
            "gs": "",
            "ftype": "",
            "year": "",
        }
        headers = {
            "User-Agent": user_agent,
        }
        # r = requests.get(
        #     url,
        #     params=params,
        #     headers=headers,
        # )
        r = make_request(
            url,
            params=params,
            headers=headers,
        )
        data_text = r.text
        total_page = eval(data_text[data_text.find("=") + 1 : data_text.find(";")])[0]
        big_df = pd.DataFrame()

        def get_dividend_page(params):
            r = make_request(
                url,
                params=params,
                initial_timeout=60,
                max_timeout=300,
                max_attempts=30,
                headers=headers,
            )
            data_text = r.text
            temp_list = eval(
                data_text[data_text.find("[[") : data_text.find(";var jjfh_jjgs")]
            )
            return pd.DataFrame(temp_list)

        if client is None:
            for page in tqdm(range(1, total_page + 1), leave=False):
                params.update({"page": page})
                # r = requests.get(
                #     url,
                #     params=params,
                #     headers=headers,
                # )
                temp_df = get_dividend_page(params)
                big_df = pd.concat([big_df, temp_df], ignore_index=True)
        else:
            futures = []
            for page in range(1, total_page + 1):
                params.update({"page": page})
                # r = requests.get(
                #     url,
                #     params=params,
                #     headers=headers,
                # )
                futures.append(client.submit(
                    get_dividend_page,
                    params
                ))
            results = client.gather(futures)
            for temp_df in results:
                big_df = pd.concat([big_df, temp_df], ignore_index=True)
        big_df.reset_index(inplace=True)
        big_df["index"] = big_df.index + 1
        big_df.columns = [
            "序号",
            "基金代码",
            "基金简称",
            "权益登记日",
            "除息日期",
            "分红",
            "分红发放日",
            "-",
        ]
        big_df = big_df[
            [
                "序号",
                "基金代码",
                "基金简称",
                "权益登记日",
                "除息日期",
                "分红",
                "分红发放日",
            ]
        ]
        big_df["权益登记日"] = pd.to_datetime(big_df["权益登记日"]).dt.date
        big_df["除息日期"] = pd.to_datetime(big_df["除息日期"]).dt.date
        big_df["分红发放日"] = pd.to_datetime(big_df["分红发放日"]).dt.date
        big_df["分红"] = pd.to_numeric(big_df["分红"])
        return big_df
