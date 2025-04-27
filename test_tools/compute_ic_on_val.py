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
    thr = 0.01
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    device = torch.device("cuda")
    print(os.getcwd())
    os.environ['PYTHONPATH'] = os.getcwd()
    print(os.environ['PYTHONPATH'])


    # config_path = f"./configs/HK_5M_base.yaml"
    # config_path = f"./configs/HK_5M_base_v1.yaml"
    # config_path = f"./configs/HK_5M_r_v0.yaml"
    # config_path = f"./configs/HK_5M_r_v1.yaml"
    # config_path = f"./configs/HK_5M_r_v2.yaml"
    # config_path = f"./configs/HK_5M_r_v2_fine.yaml"

    # config_path = f"./configs/HK_5M_r_v4.yaml"
    # config_path = f"./configs/HK_5M_r_v5.yaml"
    # config_path = f"./configs/HK_5M_r_v6.yaml"
    config_path = f"./configs/HK_5M_r_v7.yaml"
    # config_path = f"./configs/HK_5M_r_v7_10_min.yaml"
    # config_path = f"./configs/HK_5M_r_v7_15_min.yaml"
    # config_path = f"./configs/HK_5M_r_v7_30_min.yaml"
    # config_path = f"./configs/HK_5M_r_v7_1_h.yaml"
    # config_path = f"./configs/HK_5M_r_v7_2_h.yaml"
    # config_path = f"./configs/HK_5M_r_v7_2_h.yaml"
    # config_path = f"./configs/HK_5M_r_linear_5_min.yaml"
    # config_path = f"./configs/CSI_D_r_v7.yaml"
    # config_path = f"./configs/HK_5M_ABL_base.yaml"
    # config_path = f"./configs/HK_5M_ABL_norm.yaml"
    # config_path = f"./configs/HK_5M_ABL_norm_sig.yaml"
    # model_name = "code_check"
    # model_name = "loss_corr"
    # model_name = "loss_corr_ic"
    # model_name = "loss_corr_ic_fine"
    # model_name = "loss_corr_fine"
    # model_name = "loss_corr_cs_2"

    # model_name = "r_v4_amp_dir"
    # model_name = "r_v4_amp_dir_clamp"
    # model_name =   "dir_amp_r_v5"
    # model_name = "dir_amp_r_v5_v2"
    # model_name = "dir_amp_r_v5_focal"
    # model_name = "dir_amp_trend_label"
    # model_name = "dir_amp_shift_label"
    # model_name = "dir_amp_trend_10"
    # model_name = "op_cl_amp"
    # model_name = "op_cl_amp_10_min"
    # model_name = "op_cl_amp_15_min"
    # model_name = "op_cl_amp_30_min"
    # model_name = "op_cl_amp_1_h"
    # model_name = "op_cl_amp_2_h"
    # model_name = "linear_base"
    # model_name = "linear_base_s1"
    # model_name = "csi_d"
    # model_name = "abl_base"
    # model_name = "abl_norm"
    # model_name = "abl_sig"
    # model_name = "debug"
    model_name = "your model name"
    config = OmegaConf.load(config_path)

    config_data = config.data.params.train
    config_data_val = config.data.params.validation
    config_data.params.config.repeat = 1
    dataset_train = get_obj_from_str(config_data["target"])(**config_data.get("params", dict()))
    dataset_val = get_obj_from_str(config_data_val["target"])(**config_data_val.get("params", dict()))
    config_model = config.model
    model = get_obj_from_str(config_model["target"])(**config_model.get("params", dict()))

    #load ckpt
    state_dict = torch.load(f"./logs/exp_{model_name}/checkpoints/epoch=000002.ckpt",map_location="cpu")
    # state_dict = torch.load(f"./logs/exp_{model_name}/checkpoints/last.ckpt",map_location="cpu")
    model.load_state_dict(state_dict['state_dict'])
    model.to(device)
    #create a batch and run inference once
    list_act_r = []
    list_pred_r = []
    list_act_dir = []
    list_pred_dir = []
    list_auroc_r = []
    list_IC_day = []
    model.eval()
    stock_id=dataset_val[0]['stock_id'].item()

    pos_corr_list = []
    root_path = "./data/HK_5M"
    stocks_list = os.listdir(root_path)
    stocks_list.sort()
    average_list_corro = []
    average_list_corrd = []
    average_list_std = []
    average_list_rmse = []
    # for batch_idx in tqdm(range(len(dataset_val))):
    for batch_idx in range(len(dataset_val)):
        batch_val = dataset_val[batch_idx]

        if stock_id ==batch_val['stock_id'].item():

            for key in batch_val.keys():
                batch_val[key]=batch_val[key].unsqueeze(0).cuda()
            act_r,pred_r,act_dir,pred_dir,auroc_r = model.compute_statistics(batch_val)
            # act_r,pred_r,act_dir,pred_dir,auroc_r = model.validation_step(batch_val,0)
            IC_day = ((act_r * pred_r).mean() - act_r.mean() * pred_r.mean()) / (
                        (act_r.std() + 1e-6) * (pred_r.std() + 1e-6))
            list_act_r.append(act_r)
            list_pred_r.append(pred_r)
            list_act_dir.append(act_dir)
            list_pred_dir.append(pred_dir)
            list_auroc_r.append(auroc_r)
            list_IC_day.append(IC_day)
        else:
            act_r_all = torch.cat(list_act_r, dim=0)
            pred_r_all = torch.cat(list_pred_r, dim=0)

            act_dir_all = torch.cat(list_act_dir, dim=0)
            pred_dir_all = torch.cat(list_pred_dir, dim=0)
            auroc_r_all = torch.stack(list_auroc_r, dim=0)
            ic_day_all = torch.stack(list_IC_day, dim=0)

            IC = ((act_r_all * pred_r_all).mean() - act_r_all.mean() * pred_r_all.mean()) / (
                        (act_r_all.std() + 1e-6) * (pred_r_all.std() + 1e-6))
            if ic_day_all.mean()>thr and IC>thr:
                pos_corr_list.append(stock_id)
            print(f"{stocks_list[stock_id]} &  {IC.item():.5f} & {ic_day_all.mean().item():.5f} & {act_r_all.std().item():.5f} & {torch.sqrt((act_r_all - pred_r_all) ** 2).mean().item():.5f}")
            average_list_corro.append(IC.item())
            average_list_corrd.append(ic_day_all.mean())
            average_list_std.append(act_r_all.std())
            average_list_rmse.append(torch.sqrt((act_r_all - pred_r_all) ** 2).mean())

            # print(ic_day_all)
            stock_id = batch_val['stock_id'].item()
            list_act_r = []
            list_pred_r = []
            list_act_dir = []
            list_pred_dir = []
            list_auroc_r = []
            list_IC_day = []
            for key in batch_val.keys():
                batch_val[key] = batch_val[key].unsqueeze(0).cuda()
            act_r, pred_r, act_dir, pred_dir, auroc_r = model.compute_statistics(batch_val)
            # act_r,pred_r,act_dir,pred_dir,auroc_r = model.validation_step(batch_val,0)
            IC_day = ((act_r * pred_r).mean() - act_r.mean() * pred_r.mean()) / (
                    (act_r.std() + 1e-6) * (pred_r.std() + 1e-6))
            list_act_r.append(act_r)
            list_pred_r.append(pred_r)
            list_act_dir.append(act_dir)
            list_pred_dir.append(pred_dir)
            list_auroc_r.append(auroc_r)
            list_IC_day.append(IC_day)

            # break

    #compute IC
    act_r_all = torch.cat(list_act_r,dim=0)
    pred_r_all = torch.cat(list_pred_r,dim=0)

    act_dir_all = torch.cat(list_act_dir,dim=0)
    pred_dir_all = torch.cat(list_pred_dir,dim=0)
    auroc_r_all = torch.stack(list_auroc_r,dim=0)
    ic_day_all = torch.stack(list_IC_day,dim=0)


    IC = ((act_r_all*pred_r_all).mean()-act_r_all.mean()*pred_r_all.mean())/((act_r_all.std()+1e-6)*(pred_r_all.std()+1e-6))
    print(
        f"{stocks_list[stock_id]} &  {IC.item():.5f} & {ic_day_all.mean().item():.5f} & {act_r_all.std().item():.5f} & {torch.sqrt((act_r_all - pred_r_all) ** 2).mean().item():.5f}")

    average_list_corro.append(IC.item())
    average_list_corrd.append(ic_day_all.mean())
    average_list_std.append(act_r_all.std())
    average_list_rmse.append(torch.sqrt((act_r_all - pred_r_all) ** 2).mean())

    print(sum(average_list_corro)/len(average_list_corro))
    print(sum(average_list_corrd)/len(average_list_corrd))
    print(sum(average_list_std)/len(average_list_std))
    print(sum(average_list_rmse)/len(average_list_rmse))

    if ic_day_all.mean() > thr and IC>thr:
        pos_corr_list.append(stock_id)
    print(pos_corr_list)
    print(len(pos_corr_list))
    # print(auroc_r_all.mean())
if __name__ == '__main__':
    main()