
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import math
import numpy as np
import copy
import importlib
#this model does not require encoder or decoder, it a block model


base_offset = 1.

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

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000, warmup_steps=0
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0) or (step < warmup_steps):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

class GPT_stock_r(pl.LightningModule):
    def __init__(self,
                 moduleconfig,
                 feature_dim,
                 obeserve_length,
                 prediction_length,
                 test_step,
                 interval_num = 2, #2 for 10 min
                 ckpt_path = None
                 ):
        super().__init__()
        #this model is to check if the code is correct
        #instead of predicting the future, we fit the last five minites return
        #and see if the model can capture this property

        self.network = instantiate_from_config(moduleconfig)

        self.obeserve_length = obeserve_length
        self.prediction_length = prediction_length
        self.test_step = test_step
        self.pred_label = torch.tensor([[i for i in range(prediction_length)]],dtype=torch.long)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=[])
        self.trend_thr = 7
        self.interval_num = interval_num
        self.ic_list = []
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    #convert to rotation matrix

    def forward(self, x,stock_id,pred_label,feature_r_train):
        #encode image to latent code
        predictions = self.network(x,stock_id,pred_label,feature_r_train)
        return predictions

    def training_step(self, batch,batch_idx):

        #first group the datas

        feature_list = [batch['open'],batch['close'],batch['high'],batch['low'],
                        batch['volume'],batch['turnover'],batch['change_rate']]

        feature = torch.stack(feature_list, -1)

        pred_label = self.pred_label.to(feature.device)
        #we use a 24 tokens to 12 tokens prediction
        feature_train = feature[:, :-self.interval_num].clone()  #2 for 10 min

        B,N,_ = feature_train.shape
        feature_r_train = (feature_train[:,:,1]-feature_train[:,:,0])/feature_train[:,:,0]
        feature_r_train = feature_r_train.view(B,N,1).clamp(max=0.02,min=-0.02)
        feature_dir_train = (feature_r_train>0.).to(torch.float32)
        feature_r_10_min = (feature_train[:,self.interval_num-1:,1]-feature_train[:,:N-self.interval_num+1,0])/feature_train[:,:N-self.interval_num+1,0]
        #normalize the data
        mean_price = feature_train[:,:,:4].contiguous().view(feature_train.shape[0], -1).mean(1)
        std_price = feature_train[:,:,:4].contiguous().view(feature_train.shape[0], -1).std(1)+1e-3 #ensure no nan

        feature_train[:,:,:4] = (feature_train[:,:,:4]-mean_price[:,None,None])/std_price[:,None,None]
        feature_train[:, :, 4:6] = torch.log(feature_train[:, :, 4:6]+base_offset)/10   #10k as unit

        feature_train = torch.cat([feature_train,feature_dir_train],dim=-1)
        #now we regroup feature train
        feature_list = []
        feature_r_10_min_list = []

        for i in range(N-self.obeserve_length+1):
            feature_list.append(feature_train[:,i:i+self.obeserve_length])
            feature_r_10_min_list.append(feature_r_10_min[:,i:i+self.obeserve_length-self.interval_num+1])

        feature_r_train = torch.stack(feature_r_10_min_list, dim=1).contiguous()
        feature_r_train = feature_r_train.view(B*(N-self.obeserve_length+1),self.obeserve_length-self.interval_num+1,-1)
        feature_train = torch.stack(feature_list, dim=1).contiguous()
        feature_train = feature_train.view(B*(N-self.obeserve_length+1),self.obeserve_length,-1)

        pred_label = pred_label.repeat(B*(N-self.obeserve_length+1),1)

        feature_pred = feature[:, self.obeserve_length:] #fit the last minite return
        feature_pred = feature_pred[:,:,0:2] #only predict open and close value
        N_pre = feature_pred.shape[1]
        target = (feature_pred[:,self.interval_num-1:,1]-feature_pred[:,:N_pre-self.interval_num+1,0])/feature_pred[:,:N_pre-self.interval_num+1,0]


        target = target.contiguous()
        label = (target>0).to(torch.float32)
        stock_id = batch['stock_id'].view(B,1,1).repeat(1,(N-self.obeserve_length+1),1).contiguous().view(-1,1)

        # autoencode

        dir_pred, amp_pred = self(feature_train,stock_id,pred_label,feature_r_train)
        label_dir = label.view(-1,1)
        # label_dir = 0.5+(label.view(-1,1)*2.-1)*0.1
        loss_dir = F.binary_cross_entropy_with_logits(dir_pred,label_dir,reduction="mean")

        loss_amp = torch.abs(amp_pred-target.view(-1,1).clamp(max=0.008)).mean()

        # render with gsp
        loss = loss_amp+loss_dir

        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return(loss)
    def on_validation_epoch_start(self):
        self.ic_list = []

    def on_validation_epoch_end(self):
        ic_all = torch.stack(self.ic_list,dim=0)
        print(ic_all.mean())

    def validation_step(self, batch, batch_idx):

        # get gt render img
        #get gt render img
        # first group the datas
        feature_list = [batch['open'], batch['close'], batch['high'], batch['low'],
                        batch['volume'], batch['turnover'], batch['change_rate']]
        feature = torch.stack(feature_list, -1)
        stock_id = batch['stock_id']
        pred_label = self.pred_label.to(feature.device)

        # autoencode
        # we compute both corelation and auroc
        log_dict_val = dict()

        list_for_return_act = []
        list_for_return_pre = []
        list_for_return_amp = []
        list_for_dir_act = []
        list_for_dir_pre = []
        list_for_auroc = []
        # print(feature.shape)
        # print(self.obeserve_length)
        # print(self.test_step)
        for i in range(0,feature.shape[1]-self.obeserve_length,self.test_step):
            if (i + self.obeserve_length + self.interval_num > feature.shape[1]):
                break
            # we use a 24 tokens to 12 tokens prediction
            feature_train = feature[:, i:self.obeserve_length+i].clone()
            B, N, _ = feature_train.shape
            feature_r_train = (feature_train[:, :, 1] - feature_train[:, :, 0]) / feature_train[:, :, 0]
            feature_r_train = feature_r_train.view(B, N, 1)
            feature_r_10_min = (feature_train[:, self.interval_num - 1:, 1] - feature_train[:, :N-self.interval_num + 1,
                                                                              0]) / feature_train[:,
                                                                                    :N-self.interval_num + 1, 0]

            feature_dir_train = (feature_r_train > 0.).to(torch.float32)
            # normalize the data
            mean_price = feature_train[:, :, :4].contiguous().view(feature_train.shape[0], -1).mean(1)
            std_price = feature_train[:, :, :4].contiguous().view(feature_train.shape[0], -1).std(1)+1e-3

            feature_train[:, :, :4] = (feature_train[:, :, :4] - mean_price[:,None,None]) / std_price[:,None,None]
            feature_train[:, :, 4:6] = torch.log(feature_train[:, :, 4:6]+base_offset) / 10  # 10k as unit
            feature_train = torch.cat([feature_train, feature_dir_train], dim=-1)

            feature_pred = feature[:, i+self.obeserve_length:i+self.obeserve_length+self.interval_num]
            feature_pred = feature_pred[:, :, 0:2]  # only predict first 4 values

            # autoencode
            B = feature_train.shape[0]
            pred_label = pred_label.repeat(B, 1)
            #debug
            dir_pred,amp_pred = self(feature_train, stock_id,pred_label,feature_r_10_min)
            #compute return correlation  #for now use this approximate formula

            N_pre = feature_pred.shape[1]
            return_actual = (feature_pred[:,self.interval_num-1:,1]-feature_pred[:,:N_pre-self.interval_num+1,0])/feature_pred[:,:N_pre-self.interval_num+1,0]


            return_pred = dir_pred
            # return_pred = amp_pred
            list_for_return_act.append(return_actual)
            list_for_return_pre.append(return_pred)
            list_for_return_amp.append(amp_pred)
            list_for_dir_act.append((return_actual>0.).to(torch.float32))
            list_for_dir_pre.append((amp_pred>0.).to(torch.float32))


            # print(max_maxprice,min_minprice,min_maxprice,max_minprice)
            list_for_auroc.append(torch.tensor(-1.))

        #compute correlation and mean
        act_r = torch.stack(list_for_return_act, dim=-1)
        pred_r = torch.stack(list_for_return_pre, dim=-1)
        pred_amp = torch.stack(list_for_return_amp, dim=-1)
        act_dir = torch.stack(list_for_dir_act, dim=-1)
        pred_dir = torch.stack(list_for_dir_pre, dim=-1)
        auroc_r = torch.stack(list_for_auroc, dim=-1)

        #compute the mean ans std using the first 12 number
        # mean_pred = pred_r[:,:12].mean(dim=-1)
        # pred_r = pred_r-mean_pred[:,None]
        pred_r = ((pred_r>0.).to(torch.float32)*2-1)*pred_amp

        IC = ((act_r*pred_r).mean()-act_r.mean()*pred_r.mean())/((act_r.std()+1e-6)*(pred_r.std()+1e-6))
        self.ic_list.append(IC)
        # IC = ((act_dir*pred_dir).mean()-act_dir.mean()*pred_dir.mean())/((act_dir.std()+1e-3)*(pred_dir.std()+1e-3))
        AUROC = auroc_r.mean()
        log_dict_val['IC']=IC
        log_dict_val['AUROC'] = AUROC
        # render with gsp
        loss = torch.abs((dir_pred - (feature_pred[:,:,1]-feature_pred[:,:,0])/feature_pred[:,:,0])).mean()


        self.log("val/loss", loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/AUROC", AUROC,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/IC", IC,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_val)
        # print(log_dict_val)
        # return self.log_dict
        return(act_r,pred_r,act_dir,pred_dir,IC)

    def compute_statistics(self,batch):
        # get gt render img
        # get gt render img
        # first group the datas
        feature_list = [batch['open'], batch['close'], batch['high'], batch['low'],
                        batch['volume'], batch['turnover'], batch['change_rate']]
        feature = torch.stack(feature_list, -1)
        stock_id = batch['stock_id']
        pred_label = self.pred_label.to(feature.device)

        # autoencode
        # we compute both corelation and auroc
        log_dict_val = dict()

        list_for_return_act = []
        list_for_return_pre = []
        list_for_dir_act = []
        list_for_dir_pre = []
        list_for_return_amp = []
        list_for_auroc = []
        for i in range(0, feature.shape[1] - self.obeserve_length, self.test_step):
            if (i + self.obeserve_length + self.interval_num > feature.shape[1]):
                break
            # we use a 24 tokens to 12 tokens prediction
            feature_train = feature[:, i:self.obeserve_length + i].clone()
            B, N, _ = feature_train.shape
            feature_r_train = (feature_train[:, :, 1] - feature_train[:, :, 0]) / feature_train[:, :, 0]
            feature_r_train = feature_r_train.view(B, N, 1)
            feature_dir_train = (feature_r_train > 0.).to(torch.float32)

            feature_r_10_min = (feature_train[:, self.interval_num - 1:, 1] - feature_train[:, :N-self.interval_num + 1,
                                                                              0]) / feature_train[:,
                                                                                    :N-self.interval_num + 1, 0]

            # normalize the data
            mean_price = feature_train[:, :, :4].contiguous().view(feature_train.shape[0], -1).mean(1)
            std_price = feature_train[:, :, :4].contiguous().view(feature_train.shape[0], -1).std(1) + 1e-3

            feature_train[:, :, :4] = (feature_train[:, :, :4] - mean_price[:, None, None]) / std_price[:, None, None]
            feature_train[:, :, 4:6] = torch.log(feature_train[:, :, 4:6]+base_offset) / 10  # 10k as unit
            feature_train = torch.cat([feature_train, feature_dir_train], dim=-1)
            feature_pred = feature[:, i + self.obeserve_length:i + self.obeserve_length + self.interval_num]
            feature_pred = feature_pred[:, :, 0:2]  # only predict first 4 values

            # autoencode
            B = feature_train.shape[0]
            pred_label = pred_label.repeat(B, 1)
            dir_pred,amp_pred = self(feature_train, stock_id, pred_label,feature_r_10_min)

            N_pred = feature_pred.shape[1]

            return_actual = (feature_pred[:,self.interval_num-1:,1]-feature_pred[:,:N_pred-self.interval_num+1,0])/feature_pred[:,:N_pred-self.interval_num+1,0]
            return_pred = dir_pred
            list_for_return_act.append(return_actual)
            list_for_return_pre.append(return_pred)
            list_for_return_amp.append(amp_pred)
            list_for_dir_act.append((return_actual > 0.).to(torch.float32))
            list_for_dir_pre.append((dir_pred > 0.).to(torch.float32))
            # compute auroc

            # print(max_maxprice,min_minprice,min_maxprice,max_minprice)
            list_for_auroc.append(torch.tensor(-1.))

            # compute correlation and mean
        act_r = torch.stack(list_for_return_act, dim=-1).detach().cpu()
        pred_r = torch.stack(list_for_return_pre, dim=-1).detach().cpu()
        pred_amp = torch.stack(list_for_return_amp, dim=-1).detach().cpu()

        act_dir = torch.stack(list_for_dir_act, dim=-1).detach().cpu()
        pred_dir = torch.stack(list_for_dir_pre, dim=-1).detach().cpu()
        auroc_r = torch.stack(list_for_auroc, dim=-1).detach().cpu()

        # mean_pred = pred_r[:,:12].mean(dim=-1)
        # pred_r = pred_r-mean_pred[:,None]
        pred_r_new = torch.zeros_like(pred_r).to(pred_r.device)
        thr = 0.
        mask_p = pred_r>thr
        pred_r_new[mask_p] = 1.
        mask_n = pred_r<-thr
        pred_r_new[mask_n] = -1.
        # print(torch.abs(pred_r).mean(),torch.abs(pred_r).std())
        # a = input()
        pred_r = pred_r_new*pred_amp

        return(act_r,pred_r,act_dir,pred_dir,auroc_r)


    def configure_optimizers(self):
        lr = self.learning_rate #debug only scene

        # self.xyz_scheduler_args = get_expon_lr_func(lr/10, lr/1000,max_steps=50000)
        l = [p for p in self.network.parameters()]
        opt = torch.optim.Adam(l,lr=lr, betas=(0.5, 0.9),eps=1e-15)
        return opt


    def log_images(self, batch, **kwargs):
        log = dict()
        #need to convert rgb back to shs
        #implement later
        return log
