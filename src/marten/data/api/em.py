import requests
import pandas as pd
from tqdm import tqdm

from marten.utils.url import make_request

class EastMoneyAPI:

    @staticmethod
    def fund_fh_em() -> pd.DataFrame:
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
        # r = requests.get(url, params=params)
        r = make_request(url, params=params)
        data_text = r.text
        total_page = eval(data_text[data_text.find("=") + 1: data_text.find(";")])[0]
        big_df = pd.DataFrame()
        for page in tqdm(range(1, total_page + 1), leave=False):
            params.update({"page": page})
            # r = requests.get(url, params=params)
            r = make_request(url, params=params)
            data_text = r.text
            temp_list = eval(
                data_text[data_text.find("[["): data_text.find(";var jjfh_jjgs")]
            )
            temp_df = pd.DataFrame(temp_list)
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
        big_df = big_df[["序号", "基金代码", "基金简称", "权益登记日", "除息日期", "分红", "分红发放日"]]
        big_df['权益登记日'] = pd.to_datetime(big_df['权益登记日']).dt.date
        big_df['除息日期'] = pd.to_datetime(big_df['除息日期']).dt.date
        big_df['分红发放日'] = pd.to_datetime(big_df['分红发放日']).dt.date
        big_df['分红'] = pd.to_numeric(big_df['分红'])
        return big_df
