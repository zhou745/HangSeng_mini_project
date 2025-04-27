import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import torch

def main():
    root_path = "./data/HK_5M"
    stocks_list = os.listdir(root_path)
    stocks_list.sort()  # ensure order

    step_size = 24
    thr = 10
    print(len(stocks_list))

    return_list_all = []
    weight = torch.tensor([(i/step_size+0.1)**2 for i in range(step_size)])
    for stock in tqdm(stocks_list):
        return_list = []
        volume_list = []
        change_ratio_list = []
        turnover_list = []
        num_list = []
        label_list = []
        for i in range(72):
            stock_name = f"{root_path}/{stock}/{i}.pkl"
            df = pd.read_pickle(stock_name)

            open = df['open']
            close = df['close']
            volume = df['volume']
            change_rate = df['change_rate']
            turnover = df['turnover']
            open_t = torch.tensor(open)
            close_t = torch.tensor(close)
            label = (close_t>open_t).to(torch.float32)



            if open.shape[0]==66:
                return_list.append((close-open)/open)
                return_list_all.append((close-open)/open)
                volume_list.append(volume)
                change_ratio_list.append(change_rate)
                turnover_list.append(turnover)
                all_t = torch.stack([open_t, close_t],dim=-1)
                # print(all_t)
                # break
                for j in range(66-step_size):
                    # num_list.append((label[j:j + step_size]*weight).sum().item())
                    num_list.append((open_t[j+step_size//2]<close_t[j + step_size-1]).to(torch.float32).item())
                    label_list.append(label[j + step_size].item())
        num_t = torch.tensor(num_list,dtype=torch.float32)
        label_t = torch.tensor(label_list,dtype=torch.float32)

        corr = ((num_t*label_t).mean()-num_t.mean()*label_t.mean())/(num_t.std()+label_t.std())
        print(corr)

        # print(num_t.mean(),num_t.std())
        # break
        # volume_arr = np.concatenate(volume_list, axis=0)
        # print(volume_arr.min())
        # change_ratio_arr = np.concatenate(change_ratio_list,axis=0)
        # print(change_ratio_arr.mean(),change_ratio_arr.std())
        # turnover_arr = np.concatenate(turnover_list, axis=0)
        # if turnover_arr.min()<0.:
        #     print(turnover_arr.max(),turnover_arr.min())
        # break
        # print(volume_arr.mean(), volume_arr.std())
    return_arr = np.concatenate(return_list_all, axis=0)
    print(return_arr.mean())
    print(return_arr.std())

    print(np.abs(return_arr).mean())
    print(np.abs(return_arr).std())

    num_ele = np.abs(return_arr)<0.008
    print(num_ele.sum()/len(num_ele))
    print(return_arr.max())
    print(return_arr.min())
    print("-------------------------------")



if __name__ == '__main__':
    main()
