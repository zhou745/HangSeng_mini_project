import os
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import yaml
import importlib
import matplotlib
import math
import torch.nn.functional as F


#first generate the predicted depth map
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

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    device = torch.device("cuda")
    print(os.getcwd())
    os.environ['PYTHONPATH'] = os.getcwd()
    print(os.environ['PYTHONPATH'])


    # config_path = f"./configs/HK_5M_base.yaml"
    # config_path = f"./configs/HK_5M_base_v1.yaml"
    # config_path = f"./configs/HK_5M_r_v0.yaml"
    # config_path = f"./configs/HK_5M_r_v1.yaml"
    # config_path = f"./configs/HK_5M_r_v4.yaml"
    # config_path = f"./configs/HK_5M_r_v5.yaml"
    # config_path = f"./configs/HK_5M_r_v6.yaml"
    # config_path = f"./configs/HK_5M_r_v7.yaml"
    config_path = f"./configs/HK_5M_r_linear_5_min.yaml"
    # model_name = "code_check"
    # model_name = "r_v4_amp_dir"
    # model_name = "r_v4_amp_dir_clamp"
    # model_name = "dir_amp_r_v5_v2"
    # model_name = "dir_amp_r_v5_focal"
    # model_name = "dir_amp_trend_label"
    # model_name = "dir_amp_shift_label"
    # model_name = "dir_amp_trend_10"
    # model_name = "op_cl_amp"
    model_name = "linear_base_s1"
    config = OmegaConf.load(config_path)

    config_data = config.data.params.train
    config_data_val = config.data.params.validation
    config_data.params.config.repeat = 1
    dataset_train = get_obj_from_str(config_data["target"])(**config_data.get("params", dict()))
    dataset_val = get_obj_from_str(config_data_val["target"])(**config_data_val.get("params", dict()))
    config_model = config.model
    model = get_obj_from_str(config_model["target"])(**config_model.get("params", dict()))

    #load ckpt
    state_dict = torch.load(f"./logs/exp_{model_name}/checkpoints/epoch=000004.ckpt",map_location="cpu")
    model.load_state_dict(state_dict['state_dict'])
    model.to(device)
    #create a batch and run inference once


    # loss = model.training_step(batch,0)
    # val_dict = model.validation_step(batch, 0)
    # print(loss)
    batch_val = dataset_val[1]
    print(len(dataset_val))
    for key in batch_val.keys():
        batch_val[key]=batch_val[key].unsqueeze(0).cuda()
    act_r,pred_r,act_dir,pred_dir,auroc_r = model.compute_statistics(batch_val)
    print(act_r)
    print(pred_r)
    print(pred_r.mean(),pred_r.std(),pred_r.shape)
    print(act_dir)
    print(pred_dir)

if __name__ == '__main__':
    main()