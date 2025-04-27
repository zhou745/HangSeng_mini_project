import os
import tushare as ts
import json
import pandas as pd
from tqdm import tqdm
from datetime import date, timedelta
import time


def main():
    # save_root = "HK_5M"
    save_root = "CSI_D"

    file_path = "./data/csi300/code_list.json"
    code_list = json.load(open(file_path))

    token = 'da5ddd5b7fe5376d40d71e35bca0915cae494bd031b97ab488be3f75'

    ts.set_token(token)
    ts_api = ts.pro_api(token)
    os.makedirs(f"./data/{save_root}", exist_ok=True)
    for code in tqdm(code_list):
        symbol = code['Symbol']
        data = ts_api.daily(ts_code=symbol, start_date='20200101', end_date='20250422')
        data.to_pickle(f"./data/{save_root}/{code}.pkl")
        # quote_ctx = ft.OpenQuoteContext(host='127.0.0.1', port=11111)
        # os.makedirs(f"./data/{save_root}/{code}", exist_ok=True)
        # count = 73
        #
        # # start_date = date(2025, 1, 1)
        # # end_date = date(2025, 4, 22)
        # start_date = date(2025, 4, 23)
        # end_date = date(2025, 4, 24)
        # delta = timedelta(days=1)
        # while start_date <= end_date:
        #     start = start_date.strftime("%Y-%m-%d")
        #     end = start
        #
        #
        #
        #     ret, data, page_req_key = quote_ctx.request_history_kline(code,ktype=K_type, start=start, end=end,
        #                                                           max_count=record)
        #
        #     if isinstance(data,str):
        #         print(data)
        #         break
        #     if data.size>0:
        #         data.to_pickle(f"./data/{save_root}/{code}/{count}.pkl")
        #         count+=1
        #
        #     start_date += delta
        #     #intenstionally sleep 0.5 a second
        #     time.sleep(0.5)
        #
        # quote_ctx.close()
    print(f"finished code all")

if __name__ == '__main__':
    main()