import pandas as pd
import os
from tqdm import tqdm

def main():
    root_path = "./data/HK_5M"
    stocks_list = os.listdir(root_path)
    stocks_list.sort()  # ensure order
    # stock_name = "./data/HK_5M/HK.00001/18.pkl"
    # df = pd.read_pickle(stock_name)
    # print(df['time_key'])
    for stock in tqdm(stocks_list):
        for i in range(72):
            stock_name = f"{root_path}/{stock}/{i}.pkl"
            df = pd.read_pickle(stock_name)

            feature = df['time_key']

            date_now = feature[0][0:10]
            if feature.shape[0]!=66:
                print(f"{date_now} at {stock_name} has record {feature.shape[0]}")
                break
            for i in range(feature.shape[0]):
                if date_now not in feature[i]:
                    print(f"find broken date: {date_now} at {stock_name}")
                    break


if __name__ == '__main__':
    main()
