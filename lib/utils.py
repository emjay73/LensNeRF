import os, math
import numpy as np
import scipy.signal
from typing import List, Optional

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from .masked_adam import MaskedAdam


''' Misc
'''
mse2psnr = lambda x : -10. * torch.log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# added -----------------------------------------------------------
def create_optimizer_or_freeze_model_and_cam(model, model_cam, cfg_train, global_step, i_outer):
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step/decay_steps)
    decay_factor = 1 # heck
    # decay_steps = 20000

    param_group = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model, k):
            continue
        
        # if k=='rgbnet':
        #     print('debug')

        param = getattr(model, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
            
        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            # param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
            param_group.append({'name':k, 'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
            # if ((k=='rgbnet') and (cfg_train.learning_dir_start <= global_step)):
            #     param_list = list(param)
            #     param_group.append({'params': param_list[0], 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
            # else:
            #     param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
    
    # added ----------------------------------
    
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model_cam, k):
            continue
        
        param = getattr(model_cam, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue
        if (k not in cfg_train.cam_do_not_learn) and  (cfg_train.N_outers != i_outer+1) and(
            #(k in cfg_train.train_from_the_beginning) or 
            (cfg_train.learning_cam_start <= global_step) and (global_step < cfg_train.learning_cam_end)):
            if k == 'col2pix_residual':
                lr = getattr(cfg_train, f'lrate_{k}') * (1e-2)**i_outer
            else:
                lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
            # lr = getattr(cfg_train, f'lrate_{k}') * 0.5 ** (global_step/decay_steps)
             
        else:
            lr = 0.0

        
        print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
        if isinstance(param, nn.Module):
            param = param.parameters()
        # param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
        param_group.append({'name':k,'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})

        if lr > 0:
            param.requires_grad = True
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
            
    # lr = cfg_train['lrate_camera'] * decay_factor
    # param_col2pix = getattr(model, 'col2pix')

    #lr *= decay_factor
    # if lr > 0:
    #     print(f"create_optimizer_or_freeze_model: param camera lr {lr}")
    #     param_group.append({'params': param_cam, 'lr': cfg_train['lrate_camera'], 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
    # else:
    #     print('create_optimizer_or_freeze_model: freeze camera parameters')
    #     for p in param_cam:
    #         if hasattr(p, 'requires_grad'):
    #             p.requires_grad = False    
    # print(f"create_optimizer_or_freeze_model: param camera lr {lr}")
    # param_group.append({'params': param_cam, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
    # if lr == 0:
    #     print('create_optimizer_or_freeze_model: freeze camera parameters')
    #     for p in param_cam:
    #         if hasattr(p, 'requires_grad'):
    #             p.requires_grad = False
    # else:
    #     for p in param_cam:
    #         if hasattr(p, 'requires_grad'):
    #             p.requires_grad = True
    # -----------------------------------------
    
    return MaskedAdam(param_group)
# ---------------------------------------------------------------------

def create_optimizer_or_freeze_model(model, cfg_train, global_step):
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model, k):
            continue

        param = getattr(model, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
    return MaskedAdam(param_group)


''' Checkpoint utils
'''
def load_checkpoint(model, optimizer, ckpt_path, no_reload_optimizer):
    ckpt = torch.load(ckpt_path)
    start = ckpt['global_step']
    model.load_state_dict(ckpt['model_state_dict'])
    if not no_reload_optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return model, optimizer, start

def load_checkpoints(model, model_cam, optimizer, ckpt_path, no_reload_optimizer):
    ckpt = torch.load(ckpt_path)
    start = ckpt['global_step']
    model.load_state_dict(ckpt['model_state_dict'])

    # hack
    if ('lensFp_residual' not in model_cam.state_dict().keys()) and ('lensFp_residual' in ckpt['model_cam_state_dict'].keys()):
        ckpt['model_cam_state_dict'].pop('lensFp_residual')

    # model_cam.load_state_dict(ckpt['model_cam_state_dict'], strict = False)

    # update model with same shape only
    model_cam_dict = model_cam.state_dict()
    pretrained_cam_dict = ckpt['model_cam_state_dict']
    pretrained_cam_dict = {k: v for k, v in pretrained_cam_dict.items() if (k in model_cam_dict) and (model_cam_dict[k].shape == pretrained_cam_dict[k].shape)} 
    model_cam_dict.update(pretrained_cam_dict) 
    model_cam.load_state_dict(model_cam_dict)

    if not no_reload_optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return model, model_cam, optimizer, start

def load_model(model_class, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model = model_class(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'])
    return model

# added ---------------------------------
def load_models(model_class, model_camera_class, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model = model_class(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'])
    
    model_cam= model_camera_class(**ckpt['model_cam_kwargs'])
    model_cam.load_state_dict(ckpt['model_cam_state_dict'])

    return model, model_cam

def load_models_except_buffer(model_class, model_cam, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model = model_class(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'])
    
    # load param only
    model_cam_dict = model_cam.state_dict() 
    model_cam_dict_trained = ckpt['model_cam_state_dict']
    model_cam_dict_trained = {k: v for k, v in model_cam_dict_trained.items() 
                                        if k in ['extrinsics_residual', 'intrinsics_residual', 'col2pix_residual']} 
      
    model_cam_dict.update(model_cam_dict_trained)     
    model_cam.load_state_dict(model_cam_dict) 

    return model, model_cam
    
''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()

