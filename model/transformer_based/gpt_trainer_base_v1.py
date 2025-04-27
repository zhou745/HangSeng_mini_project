
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import math
import numpy as np
import copy
import importlib
#this model does not require encoder or decoder, it a block model

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

class GPT_stock_base(pl.LightningModule):
    def __init__(self,
                 moduleconfig,
                 feature_dim,
                 obeserve_length,
                 prediction_length,
                 test_step,
                 ckpt_path = None
                 ):
        super().__init__()

        self.network = instantiate_from_config(moduleconfig)

        self.obeserve_length = obeserve_length
        self.prediction_length = prediction_length
        self.test_step = test_step
        self.pred_label = torch.tensor([[i for i in range(prediction_length)]],dtype=torch.long)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=[])

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

    def forward(self, x,stock_id,pred_label):
        #encode image to latent code
        predictions = self.network(x,stock_id,pred_label)
        return predictions

    def training_step(self, batch,batch_idx):

        #first group the datas

        feature_list = [batch['open'],batch['close'],batch['high'],batch['low'],
                        batch['volume'],batch['turnover'],batch['change_rate']]

        feature = torch.stack(feature_list, -1)

        pred_label = self.pred_label.to(feature.device)
        #we use a 24 tokens to 12 tokens prediction
        feature_train = feature[:, :self.obeserve_length].clone()

        #normalize the data
        mean_price = feature_train[:,:,:4].contiguous().view(feature_train.shape[0], -1).mean(1)
        std_price = feature_train[:,:,:4].contiguous().view(feature_train.shape[0], -1).std(1)+1e-3 #ensure no nan

        feature_train[:,:,:4] = (feature_train[:,:,:4]-mean_price[:,None,None])/std_price[:,None,None]
        feature_train[:, :, 5:7] = feature_train[:, :, 5:7]/10000   #10k as unit

        B = feature_train.shape[0]
        pred_label = pred_label.repeat(B,1)

        feature_pred = feature[:, self.obeserve_length:]
        feature_pred = feature_pred[:,:,2:4] #only predict first 2 values
        stock_id = batch['stock_id']
        # autoencode
        preditions = self(feature_train,stock_id,pred_label)
        preditions = preditions[:,self.obeserve_length:] #convert back to mean
        preditions = preditions[:,:,2:4]
        # render with gsp
        loss = torch.abs(preditions-(feature_pred-mean_price[:,None,None])/std_price[:,None,None]).mean()
        # print(preditions)
        # print((feature_pred-mean_price[:,None,None])/std_price[:,None,None])
        # print(feature_train[:,:self.obeserve_length,2:4])
        # print((feature[:,:,2:4]-mean_price[:,None,None])/std_price[:,None,None])
        # aeloss.backward()
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return(loss)

        # torch.cuda.synchronize()
        # del image_xyz
        # return aeloss
        # return(render_points,image_xyz,pred_depth_new,image_min,image_max,pred_img,pred_img_move)

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
        list_for_auroc = []
        for i in range(0,feature.shape[1]-self.obeserve_length,self.test_step):
            if (i + self.obeserve_length + self.prediction_length > feature.shape[1]):
                break
            # we use a 24 tokens to 12 tokens prediction
            feature_train = feature[:, i:self.obeserve_length+i].clone()

            # normalize the data
            mean_price = feature_train[:, :, :4].contiguous().view(feature_train.shape[0], -1).mean(1)
            std_price = feature_train[:, :, :4].contiguous().view(feature_train.shape[0], -1).std(1)+1e-3

            feature_train[:, :, :4] = (feature_train[:, :, :4] - mean_price[:,None,None]) / std_price[:,None,None]
            feature_train[:, :, 5:7] = feature_train[:, :, 5:7] / 10000  # 10k as unit

            feature_pred = feature[:, i+self.obeserve_length:i+self.obeserve_length+self.prediction_length]
            feature_pred = feature_pred[:, :, 2:4]  # only predict first 4 values

            # autoencode
            B = feature_train.shape[0]
            pred_label = pred_label.repeat(B, 1)
            preditions = self(feature_train, stock_id,pred_label)
            preditions = preditions[:,self.obeserve_length:]*std_price[:,None,None]+mean_price[:,None,None]
            preditions = preditions[:, :, 2:4]

            #compute return correlation  #for now use this approximate formula
            max_price = torch.max(feature_pred[:,:,0], dim=1)[0]
            min_price = torch.min(feature_pred[:,:,1], dim=1)[0]

            max_pred = torch.max(preditions[:,:,0], dim=1)[0]
            min_pred = torch.min(preditions[:,:,1], dim=1)[0]

            return_actual = (max_price-min_price)/min_price
            return_pred = (max_pred-min_pred)/min_pred
            list_for_return_act.append(return_actual)
            list_for_return_pre.append(return_pred)
            #compute auroc
            max_maxprice = torch.max(torch.stack((max_price,max_pred),dim=-1),dim=1)[0]
            min_minprice = torch.min(torch.stack((min_price, min_pred), dim=-1), dim=1)[0]

            max_minprice = min_pred+min_price-min_minprice
            min_maxprice = max_price+max_pred-max_maxprice
            # print(max_maxprice,min_minprice,min_maxprice,max_minprice)
            auroc = (min_maxprice-max_minprice).clamp(min=0.)/(max_maxprice-min_minprice)
            list_for_auroc.append(auroc)

        #compute correlation and mean
        act_r = torch.stack(list_for_return_act, dim=-1)
        pred_r = torch.stack(list_for_return_pre, dim=-1)
        auroc_r = torch.stack(list_for_auroc, dim=-1)

        IC = ((act_r*pred_r).mean()-act_r.mean()*pred_r.mean())/((act_r.std()+1e-3)*(pred_r.std()+1e-3))

        AUROC = auroc_r.mean()
        log_dict_val['IC']=IC
        log_dict_val['AUROC'] = AUROC
        # render with gsp
        loss = torch.abs((preditions - feature_pred)*std_price).mean()


        self.log("val/loss", loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/AUROC", AUROC,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/IC", IC,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_val)
        # print(log_dict_val)
        return self.log_dict

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
