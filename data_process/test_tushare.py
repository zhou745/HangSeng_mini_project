import tushare as ts
import json
import torch
from tqdm import tqdm
def main():
    token = 'da5ddd5b7fe5376d40d71e35bca0915cae494bd031b97ab488be3f75'

    ts.set_token(token)
    ts_api = ts.pro_api(token)
    #read all the stock code
    file_path = "./data/csi300/code_list.json"
    code_list = json.load(open(file_path))

    #download data for only one stock
    for code in tqdm(code_list):
        symbol = code["Symbol"]
        # symbol = "600015.SS"
        df = ts_api.daily(ts_code=symbol, start_date='20200101', end_date='20250422')
        try:
            a = torch.tensor(df['open'].to_numpy(),dtype=torch.float32)
            if a.shape[0]<60:
                print(f"find error at {code}")
        except:
            print(f"find error at {code}")
            print(df['open'])
        # break
    # print(df)


if __name__ == '__main__':
    main()