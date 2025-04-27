import os
import math
import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset
from omegaconf import OmegaConf
import argparse, os, sys, datetime, glob, importlib
import json
from tqdm import tqdm
import pandas as pd

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))




class HK_5M_base(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)

        print(f"config: {self.config}")


        self.root_path = self.config['root_path']
        self.start_idx = self.config['start_idx']
        self.end_idx = self.config['end_idx']
        self.segment_length = self.config['segment_length']
        self.num_stock = self.config['num_stock']
        self.stock_start = self.config['stock_start']
        self.split = self.config['split']
        self.repeat = self.config['repeat']
        if isinstance(self.config['stock_list'],list):
            self.stock_list = [int(it) for it  in self.config['stock_list']]
        else:
            self.stock_list = [i for i in range(self.num_stock)]

        self.__prepare()

    def __prepare(self):
        # load sequences in the seq root file
        #list all the object in the dataset
        #list all stock index
        stocks_list = os.listdir(self.root_path)
        stocks_list.sort() #ensure order


        self.item_list = []
        self.label_list = []
        label_id = 0
        for stock in tqdm(stocks_list):
            if label_id<self.stock_start:
                label_id += 1
                continue

            if label_id not in self.stock_list:
                label_id += 1
                continue
            #intenstionally drop i=18 for half trading day

            self.item_list += [f"{self.root_path}/{stock}/{i}.pkl" for i in range(self.start_idx, self.end_idx) if i!=18]*self.repeat
            self.label_list += [ label_id for i in range(self.start_idx, self.end_idx) if i!=18]*self.repeat
            label_id +=1
            if label_id>=self.num_stock:
                break #only the firt num_stock is used



    def __len__(self):
        return(len(self.item_list))

    def __getitem__(self, i):
        # i: int, index of the data
        # sample a local block from the occupancy field

        stock_name = self.item_list[i]
        stock_label = self.label_list[i]
        #read the pose info mation
        df = pd.read_pickle(stock_name)
        #feature about price
        open_p = df['open'].to_numpy()
        close_p = df['close'].to_numpy()
        high_p = df['high'].to_numpy()
        low_p = df['low'].to_numpy()

        volume_p = df['volume'].to_numpy()
        turnover_p = df['turnover'].to_numpy()
        change_p = df['change_rate'].to_numpy()



        if self.segment_length>0:
            start_idx = np.random.randint(low=0, high=len(open_p)-self.segment_length)
            open_p = open_p[start_idx:start_idx+self.segment_length]
            close_p = close_p[start_idx:start_idx+self.segment_length]
            high_p = high_p[start_idx:start_idx+self.segment_length]
            low_p = low_p[start_idx:start_idx+self.segment_length]

            volume_p = volume_p[start_idx:start_idx+self.segment_length]
            turnover_p = turnover_p[start_idx:start_idx+self.segment_length]
            change_p = change_p[start_idx:start_idx+self.segment_length]

        #read the depth image
        example = dict()


        example['open'] = torch.tensor(open_p,dtype=torch.float32)
        example['close'] = torch.tensor(close_p,dtype=torch.float32)
        example['high'] = torch.tensor(high_p,dtype=torch.float32)
        example['low'] = torch.tensor(low_p,dtype=torch.float32)
        example['volume'] = torch.tensor(volume_p,dtype=torch.float32)
        example['turnover'] = torch.tensor(turnover_p,dtype=torch.float32)
        example['change_rate'] = torch.tensor(change_p,dtype=torch.float32)
        example['stock_id'] = torch.tensor(stock_label,dtype=torch.int64)
        return example
