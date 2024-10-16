import os, sys, copy, glob, json, time, random, argparse
from shutil import copyfile
from typing_extensions import runtime
from xml.dom.pulldom import default_bufsize
from tqdm import tqdm, trange

import mmcv
#import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo, dcvgo, dmpigo
from lib.load_data import load_data

from torch_efficient_distloss import flatten_eff_distloss

# added --------------------------------------------
import wandb
from lib.render import do_rendering
from lib.compute_bbox import compute_bbox_by_cam_frustrm, compute_bbox_by_coarse_geo
from lib.aperture_camera import ApertureCamera
from lib.rays import gather_training_rays, gather_training_rays_from_index, make_indices_generator
from lib.aperture_rays import get_length_with_min_dist

from lib.dvgo import ndc_rays_aperture
from lib.render import DiskKernel

# ----------------------------------------------------

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_pose", action='store_true')
    parser.add_argument("--render_video_aperture", action='store_true')
    parser.add_argument("--render_video_pose_wo_apt", action='store_true')    

    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')
    parser.add_argument("--eval_dists", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')

    # aperture camera option
    parser.add_argument("--pinhole_camera", action='store_true')    
    parser.add_argument("--no_wandb", action='store_true')
    parser.add_argument("--exp_postfix", type=str, default='', help='specify wandb subversion')
    parser.add_argument("--render_mpi_defocus_image", action='store_true', help='make defocus image')
    parser.add_argument("--aperture_sample_rate", type=int, default=3,)    
    parser.add_argument("--aperture_sample_scale", type=int, default=11,)    
    parser.add_argument("--iter_train_single_ray_end", type=int, default=0,)

    parser.add_argument("--perturb_infocusZ", type=float, default=1.0,)    
    #parser.add_argument("--gamma", type=float, default=1/2.2)   
    parser.add_argument("--gamma", type=float, default=1)   
    
    return parser


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    
    # data_dict = load_data(cfg.data, args)
    data_dict = load_data(cfg, args)

    # remove useless field
    # modified ----------------------------------------------
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images', 
            'lens_info_list', 'intrinsics', 'col2X', 'infocusZ'}
    # original ----------------------------------------------
    # kept_keys = {
    #         'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
    #         'i_train', 'i_val', 'i_test', 'irregular_shape',
    #         'poses', 'render_poses', 'images'}
    #--------------------------------------------------------
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict

# added ---------------------------------------------------------
#def create_new_model_and_cam(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path, data_dict):
def create_new_model_and_optimizer(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path, model_cam, i_outer):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

    if cfg.data.ndc:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse multiplane images\033[0m')
        model = dmpigo.DirectMPIGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    elif cfg.data.unbounded_inward:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse contraced voxel grid (covering unbounded)\033[0m')
        model = dcvgo.DirectContractedVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    else:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        model = dvgo.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
    model = model.to(device)

    # added ---------------------------------
    # model_cam = ApertureCamera(data_dict)
    model_cam = model_cam.to(device)
    optimizer = utils.create_optimizer_or_freeze_model_and_cam(model, model_cam, cfg_train, global_step=0, i_outer=i_outer)
    return model, model_cam, optimizer
# -------------------------------------------------------------------------------------------
def create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

    if cfg.data.ndc:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse multiplane images\033[0m')
        model = dmpigo.DirectMPIGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    elif cfg.data.unbounded_inward:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse contraced voxel grid (covering unbounded)\033[0m')
        model = dcvgo.DirectContractedVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    else:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        model = dvgo.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
    model = model.to(device)
    optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    return model, optimizer

def load_existed_model(args, cfg, cfg_train, reload_ckpt_path):
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    
    model, model_cam = utils.load_models(model_class, ApertureCamera, reload_ckpt_path)
    model = model.to(device)
    model_cam = model_cam.to(device)
    if args.pinhole_camera:
    #     model = utils.load_model(model_class, reload_ckpt_path).to(device)
        optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    #     model, optimizer, start = utils.load_checkpoint(
    #         model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    #     return model, optimizer, start
    else:
        optimizer = utils.create_optimizer_or_freeze_model_and_cam(model, model_cam, cfg_train, global_step=0, i_outer=0)
    
    model, model_cam, optimizer, start = utils.load_checkpoints(
        model, model_cam, optimizer, reload_ckpt_path, args.no_reload_optimizer)    
    return model,model_cam,optimizer, start

def load_existed_model_except_buffer(args, cfg, cfg_train, model_cam, reload_ckpt_path):
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    
    model, model_cam = utils.load_models_except_buffer(model_class, model_cam, reload_ckpt_path)
    model = model.to(device)
    model_cam = model_cam.to(device)
    if args.pinhole_camera:
    #     model = utils.load_model(model_class, reload_ckpt_path).to(device)
        optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    #     model, optimizer, start = utils.load_checkpoint(
    #         model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    #     return model, optimizer, start
    else:
        optimizer = utils.create_optimizer_or_freeze_model_and_cam(model, model_cam, cfg_train, global_step=0, i_outer=0)
    
    model, model_cam, optimizer, start = utils.load_checkpoints(
        model, model_cam, optimizer, reload_ckpt_path, args.no_reload_optimizer)    
    return model,model_cam,optimizer, start

def make_last_reload_ckpt_path(args, cfg, stage='fine'):
    # find whether there is existing checkpoint path
    if cfg.expname_train is None:
        last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    else:
        last_ckpt_path = os.path.join(cfg.basedir, cfg.expname_train+args.exp_postfix, f'{stage}_last.tar')
    # if args.no_reload:
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None
    return last_ckpt_path, reload_ckpt_path

def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, ts_apt, data_dict, stage, model_cam, i_outer, last_ckpt_path = None, reload_ckpt_path=None, coarse_ckpt_path=None):
    time_before_loop = time.time()

    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    
    # modified ---------------------------------------------------------------------------------------
    if args.pinhole_camera:
        HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
            data_dict[k] for k in [
                'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
            ]
        ]
    else:
        # HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images, lens_info_list, intrinsics, col2X = [
        #     data_dict[k] for k in [
        #         'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images', 'lens_info_list', 'intrinsics', 'col2X'
        #     ]
        # ]
        
        #removed HW, Ks, poses, lens_info_list, intrinsics, col2X
        near, far, i_train, i_val, i_test, render_poses, images, infocusZ = [
            data_dict[k] for k in ['near', 'far', 'i_train', 'i_val', 'i_test', 'render_poses', 'images', 'infocusZ']]
    # original ---------------------------------------------------------------------------------------
    # HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
    #     data_dict[k] for k in [
    #         'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
    #     ]
    # ]

    # # find whether there is existing checkpoint path
    # if cfg.expname_train is None:
    #     last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_last.tar')
    # else:
    #     last_ckpt_path = os.path.join(cfg.basedir, cfg.expname_train+args.exp_postfix, f'{stage}_last.tar')
    # # if args.no_reload:
    # if args.no_reload:
    #     reload_ckpt_path = None
    # elif args.ft_path:
    #     reload_ckpt_path = args.ft_path
    # elif os.path.isfile(last_ckpt_path):
    #     reload_ckpt_path = last_ckpt_path
    # else:
    #     reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')

        if args.pinhole_camera:
            model, optimizer = create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path)
        else:            
            model, model_cam, optimizer = create_new_model_and_optimizer(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path, model_cam, i_outer)
        start = 0
        if cfg_model.maskout_near_cam_vox:
            model.maskout_near_cam_vox(poses[i_train,:3,3], near)
    else:
        print(f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        # if args.pinhole_camera:
        #     model, optimizer, start = load_existed_model(args, cfg, cfg_train, reload_ckpt_path)
        # else:
        #     model, model_cam, optimizer, start = load_existed_model(args, cfg, cfg_train, reload_ckpt_path)
        #model, model_cam, optimizer, start = load_existed_model(args, cfg, cfg_train, reload_ckpt_path)
        model, model_cam, optimizer, start = load_existed_model_except_buffer(args, cfg, cfg_train, model_cam, reload_ckpt_path)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }

    if args.pinhole_camera:
        #rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays(cfg, cfg_train, data_dict, images, poses, HW, Ks, i_train, device)

    else:
        total_n_pixels = model_cam.get_total_n_piexls(i_train)
        batch_index_sampler = make_indices_generator(total_n_pixels , cfg_train)
    # view-count-based learning rate
    if cfg_train.pervoxel_lr: # False
        def per_voxel_init():
            cnt = model.voxel_count_views(
                    rays_o_tr=rays_o_tr, rays_d_tr=rays_d_tr, imsz=imsz, near=near, far=far,
                    stepsize=cfg_model.stepsize, downrate=cfg_train.pervoxel_lr_downrate,
                    irregular_shape=data_dict['irregular_shape'])
            optimizer.set_pervoxel_lr(cnt)
            model.mask_cache.mask[cnt.squeeze() <= 2] = False
        per_voxel_init()

    if cfg_train.maskout_lt_nviews > 0:
        model.update_occupancy_cache_lt_nviews(
                rays_o_tr, rays_d_tr, imsz, render_kwargs, cfg_train.maskout_lt_nviews)

        
    # GOGO
    torch.cuda.empty_cache()
    psnr_lst = []    
    # psnr_lst_center = []
    

    time0 = time.time()
    global_step = -1
    
    if data_dict['irregular_shape']: # False
        rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
    else:
        rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
    
    time_before_loop = time.time() - time_before_loop 
    time_inside_loop_before_vrender=0
    time_inside_loop_vrender=0
    time_inside_loop_compute_loss_and_update_optimizer=0
    time_inside_loop_save_intermediate_results=0
    delta_inside_loop_before_vrender=0
    delta_inside_loop_vrender=0
    delta_inside_loop_compute_loss_and_update_optimizer=0
    delta_inside_loop_save_intermediate_results=0

    # pg_scale_with_reinit = cfg_train.pg_scale + list(np.array(cfg_train.pg_scale) + cfg_train.learning_cam_end)
    # pg_scale_with_reinit = cfg_train.pg_scale 
    # print(f"pg_scale_with_reinit: {pg_scale_with_reinit}")
    

    for global_step in trange(1+start, 1+cfg_train.N_iters):
    # for global_step in trange(start, 1+cfg_train.N_iters):
        
        if (cfg_train.N_outers != (i_outer+1)) and (global_step > cfg_train.learning_cam_end):
            break
        
        start_inside_loop_before_vrender=time.time()    
        
        # if global_step < args.iter_train_single_ray_end:
        # if (global_step < cfg_train.learning_cam_start) and (i_outer==0):
        if (global_step < args.iter_train_single_ray_end) and (i_outer==0):
            tf_single_ray = True
        else:
            tf_single_ray = False
        
        # renew occupancy grid
        if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
            model.update_occupancy_cache()

        # progress scaling checkpoint
        # if (global_step in cfg_train.pg_scale) or (global_step in (np.array(cfg_train.pg_scale) + cfg_train.learning_cam_end)): # False
        if global_step in cfg_train.pg_scale :
            # n_rest_scales = len(cfg_train.pg_scale)-cfg_train.pg_scale.index(global_step)-1
            # n_rest_scales = len(pg_scale_with_reinit) - pg_scale_with_reinit.index(global_step)-1
            n_rest_scales = len(cfg_train.pg_scale ) - cfg_train.pg_scale .index(global_step)-1
            # if n_rest_scales >= len(cfg_train.pg_scale):
            #     n_rest_scales = n_rest_scales-len(cfg_train.pg_scale)            
                
            cur_voxels = int(cfg_model.num_voxels / (2**n_rest_scales))
            if isinstance(model, (dvgo.DirectVoxGO, dcvgo.DirectContractedVoxGO)):
                model.scale_volume_grid(cur_voxels)
            elif isinstance(model, dmpigo.DirectMPIGO):
                model.scale_volume_grid(cur_voxels, model.mpi_depth)
            else:
                raise NotImplementedError
                            
            if args.pinhole_camera:
                optimizer = utils.create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
            else:   
                #optimizer = utils.create_optimizer_or_freeze_model_and_cam(model, model_cam, cfg_train, global_step=0)
                optimizer = utils.create_optimizer_or_freeze_model_and_cam(model, model_cam, cfg_train, global_step=global_step, i_outer=i_outer)
            model.act_shift -= cfg_train.decay_after_scale
            torch.cuda.empty_cache()
        
        if not args.pinhole_camera:
            # if (global_step not in cfg_train.pg_scale) and ((global_step == cfg_train.learning_cam_start) or (global_step == cfg_train.learning_cam_end)):
            if (global_step not in cfg_train.pg_scale ) and ((global_step == cfg_train.learning_cam_start) or (global_step == cfg_train.learning_cam_end) or (global_step==cfg_train.N_iters)):
                print(f"global_step: {global_step}, learning_cam_start: {cfg_train.learning_cam_start}, learning_cam_end: {cfg_train.learning_cam_end}")
                if (global_step == cfg_train.learning_cam_end) or (global_step == cfg_train.N_iters) :#and (not cfg.data.load_col2pix):
                    # testsavedir = os.path.join(cfg.basedir, cfg.expname+args.exp_postfix)
                    testsavedir = os.path.join(cfg.basedir, cfg.expname_train+args.exp_postfix) 
                    print(f'save col2pix: '+ os.path.join(testsavedir, "col2pix_last.npy"))     
                    # if cfg.expname_train is None:
                    #     ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
                    # else:
                    #     ckpt_path = os.path.join(cfg.basedir, cfg.expname_train+args.exp_postfix, 'fine_last.tar')
                    np.save( os.path.join(testsavedir, f"col2pix_{i_outer}.npy"), model_cam.get_col2pix().detach().cpu())
                    np.save( os.path.join(testsavedir, "col2pix_last.npy"), model_cam.get_col2pix().detach().cpu())
                    # if isinstance(cfg.data.datadir, list):
                    #     np.save( os.path.join(cfg.data.datadir[0], f"col2pix_{i_outer}.npy"), model_cam.get_col2pix().detach().cpu())
                    #     np.save( os.path.join(cfg.data.datadir[0], "col2pix_last.npy"), model_cam.get_col2pix().detach().cpu())
                    # else:
                    #     np.save( os.path.join(cfg.data.datadir, f"col2pix_{i_outer}.npy"), model_cam.get_col2pix().detach().cpu())
                    #     np.save( os.path.join(cfg.data.datadir, "col2pix_last.npy"), model_cam.get_col2pix().detach().cpu())
                    # if i_outer == 0:
                    
                    if cfg_train.N_outers != (i_outer+1):
                        break
                optimizer = utils.create_optimizer_or_freeze_model_and_cam(model, model_cam, cfg_train, global_step=global_step, i_outer=i_outer)
                torch.cuda.empty_cache()

        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()

            if args.pinhole_camera:    
                target = rgb_tr[sel_i]
                rays_o = rays_o_tr[sel_i]
                rays_d = rays_d_tr[sel_i]
                viewdirs = viewdirs_tr[sel_i]  
                rays_mask = None      
            else:

                rgb_tr, rays_o_tr, rays_d_tr, rays_mask_tr, viewdirs_tr, imsz = \
                        gather_training_rays_from_index(cfg, cfg_train, args, data_dict, rgb_tr_ori, i_train, device, model_cam, sel_i, tf_single_ray)
                target = rgb_tr
                rays_o = rays_o_tr
                rays_d = rays_d_tr
                rays_mask = rays_mask_tr
                viewdirs = viewdirs_tr        


        elif cfg_train.ray_sampler == 'random': # False
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            if rays_mask is not None:
                rays_mask = rays_mask.to(device)
            viewdirs = viewdirs.to(device)

        # volume rendering
        delta_inside_loop_before_vrender = time.time()-start_inside_loop_before_vrender
        time_inside_loop_before_vrender += delta_inside_loop_before_vrender
        start_inside_loop_vrender=time.time()
    
        sel_pos = model_cam.index2position(sel_i)
        sel_img_idx = sel_pos[:, 0]
        _, H, W, _ = rgb_tr_ori.shape # n tr imgs, h, w, 3
        sel_pix_idx = (sel_pos[:, 2:3]*W + sel_pos[:, 1:2]).flatten()
        
        # rays_o_ndc, rays_d_ndc = ndc_rays_aperture(H, W, model_cam.intrinsics[0,0].item(), 1., rays_o, rays_d)
        # render_result = model(
        #     rays_o_ndc.view(-1, 3), rays_d_ndc.view(-1, 3), viewdirs.view(-1, 3),
        #     global_step=global_step, is_train=True,
        #     **render_kwargs)
        render_result = model(
            rays_o.view(-1, 3), rays_d.view(-1, 3), viewdirs.view(-1, 3),
            global_step=global_step, is_train=True,
            **render_kwargs)
        delta_inside_loop_vrender = time.time()-start_inside_loop_vrender    
        time_inside_loop_vrender += delta_inside_loop_vrender
        start_inside_loop_compute_loss_and_update_optimizer = time.time()
    
        # gradient descent step
        optimizer.zero_grad(set_to_none=True)
        if not args.pinhole_camera:
            apt_sample_num = int( render_result["rgb_marched"].shape[0]/len(sel_i) )
            apt_sample_rate = int(np.sqrt(apt_sample_num))
            if tf_single_ray:
                apt_sample_scale = 1 #torch.ones_like(args.aperture_sample_scale)
                apt_sample_rate = 1
            else:
                apt_sample_scale = args.aperture_sample_scale
                apt_sample_rate = args.aperture_sample_rate
            # apt_sample_rate_wide = apt_sample_rate * apt_sample_scale
            apt_sample_rate_wide = apt_sample_scale
            apt_sample_num_wide = apt_sample_rate_wide**2
            if rays_mask is not None:
                #render_result["rgb_marched"]  = torch.pow(torch.sum(render_result["rgb_marched"].view(-1, apt_sample_num, 3)*rays_mask, dim=1)/ torch.sum(rays_mask, dim=1), 1/2.2)
                #rgb_marched_mean  = torch.pow(torch.sum(render_result["rgb_marched"].view(-1, apt_sample_num, 3)*rays_mask, dim=1)/ torch.sum(rays_mask, dim=1), 1/2.2)
                # rgb_marched_mean  = torch.mean(render_result["rgb_marched"].view(-1, apt_sample_num, 3)*rays_mask, dim=1)
                if apt_sample_scale != 1:
                    rgb_marched_mean=torch.nn.functional.interpolate(
                                    render_result["rgb_marched"].view(-1, apt_sample_rate, apt_sample_rate, 3).permute(0, 3, 1, 2), 
                                    # scale_factor=apt_sample_scale, 
                                    size = (apt_sample_scale,apt_sample_scale),
                                    mode='bilinear',
                                    align_corners=True,
                                    ).permute(0, 2,3,1)
                    rays_mask_wide=torch.nn.functional.interpolate(
                                        rays_mask.permute(0, 2, 1).reshape(-1, 1, apt_sample_rate,  apt_sample_rate), 
                                        # scale_factor=apt_sample_scale, 
                                        size = (apt_sample_scale,apt_sample_scale),
                                        mode='nearest',
                                        #align_corners=True
                                        ).permute(0, 2,3,1)
                else:
                    rgb_marched_mean = render_result["rgb_marched"].reshape(-1, apt_sample_rate, apt_sample_rate, 3)
                    rays_mask_wide = rays_mask.reshape(-1, apt_sample_rate,  apt_sample_rate, 1)

                if torch.any(torch.isnan(rgb_marched_mean)):
                    print('rgb_marched_mean includes nan')
                # disk_width=args.aperture_sample_rate*apt_sample_scale
                # disk_width=apt_sample_rate*apt_sample_scale
                disk_width=apt_sample_scale
                disk_kernel = DiskKernel(disk_width).view(1, disk_width, disk_width, 1).to(device=rays_mask_wide.device)
                rays_mask_wide = rays_mask_wide * disk_kernel 
                rgb_marched_mean  = (rgb_marched_mean*rays_mask_wide).reshape(-1, apt_sample_num_wide, 3).sum(dim=1) / torch.max(torch.sum(rays_mask_wide, dim=(1,2)), torch.tensor([1]))
                #rgb_marched_center  = torch.pow(render_result["rgb_marched"].view(-1, apt_sample_num, 3)[:, apt_sample_num//2, :], 1/2.2)
            else:
                # disk_width=args.aperture_sample_rate*apt_sample_scale
                # disk_width=apt_sample_rate*apt_sample_scale
                disk_width=apt_sample_scale
                rays_mask_wide = DiskKernel(disk_width).view(1, disk_width, disk_width, 1).to(device=rays_mask_wide.device)
                rgb_marched_mean=torch.nn.functional.interpolate(
                                render_result["rgb_marched"].view(-1, apt_sample_rate, apt_sample_rate, 3).permute(0, 3, 1, 2), 
                                # scale_factor=apt_sample_scale, 
                                size=(apt_sample_scale, apt_sample_scale),
                                mode='bilinear',
                                align_corners=True).permute(0, 2,3,1)                
                if torch.any(torch.isnan(rgb_marched_mean)):
                    # print('rays_rgb_wide includes nan')
                    raise RuntimeWarning(f"rays_rgb_wide includes nan")
                rgb_marched_mean  = (rgb_marched_mean*rays_mask_wide).reshape(-1, apt_sample_num_wide, 3).sum(dim=1) / torch.max(torch.sum(rays_mask_wide, dim=(1,2)), torch.tensor([1]))
            
            if args.gamma != 1:
                rgb_marched_mean  = torch.pow(rgb_marched_mean+torch.finfo(rgb_marched_mean.dtype).eps, args.gamma)


                # rgb_marched_mean = torch.mean(render_result["rgb_marched"].view(-1, apt_sample_num, 3), dim=1)
                #rgb_marched_mean = torch.pow(torch.mean(render_result["rgb_marched"].view(-1, apt_sample_num, 3), dim=1), 1/2.2)                
                #rgb_marched_center = torch.pow(render_result["rgb_marched"].view(-1, apt_sample_num, 3)[:, apt_sample_num//2, :], 1/2.2)
            
            loss = cfg_train.weight_main * F.mse_loss(rgb_marched_mean, target)
            psnr = utils.mse2psnr(loss.detach())

            # loss center
            # loss_center = cfg_train.weight_main * F.mse_loss(rgb_marched_center, target)
            # psnr_center = utils.mse2psnr(loss_center.detach())
            # loss += loss_center
            
        else:
            loss = cfg_train.weight_main * F.mse_loss(render_result['rgb_marched'], target)
            psnr = utils.mse2psnr(loss.detach())

        # # gradient descent step
        # optimizer.zero_grad(set_to_none=True)
        #


        # loss extrinsic residual
        if not args.pinhole_camera:
            if model_cam.col2pix_residual.requires_grad:
            #    loss_col = model_cam.get_col2mm_loss()
                loss_col = model_cam.get_col_loss()
                # loss_lensfp = model_cam.get_lensfp_loss()
                # loss += (loss_col*cfg_train.weight_colloss + loss_lensfp*cfg_train.weight_lensfp_loss)
                loss += (loss_col*cfg_train.weight_colloss)
            else:
               loss_col = torch.tensor([0])
               loss_lensfp = torch.tensor([0])
            # loss_col = torch.tensor([0])

            if model_cam.extrinsics_residual.requires_grad:
                loss_extrinsic = model_cam.get_extrinsics_loss()
                loss += loss_extrinsic
            else:
                loss_extrinsic = torch.tensor([0])

            if model_cam.intrinsics_residual.requires_grad:
                loss_intrinsic = model_cam.get_intrinsics_loss()
                loss += loss_intrinsic
            else:
                loss_intrinsic = torch.tensor([0])
        if cfg_train.weight_entropy_last > 0: # True
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_nearclip > 0: # False
            near_thres = data_dict['near_clip'] / model.scene_radius[0].item()
            near_mask = (render_result['t'] < near_thres)
            density = render_result['raw_density'][near_mask]
            if len(density):
                nearclip_loss = (density - density.detach()).sum()
                loss += cfg_train.weight_nearclip * nearclip_loss
        if cfg_train.weight_distortion > 0: # True
            n_max = render_result['n_max']
            s = render_result['s']
            w = render_result['weights']
            ray_id = render_result['ray_id']
            loss_distortion = flatten_eff_distloss(w, s, 1/n_max, ray_id)
            loss += cfg_train.weight_distortion * loss_distortion

        if args.pinhole_camera or (ts_apt is None):
            if cfg_train.weight_rgbper > 0:                 
                #render_result['ray_id'].view(-1, apt_sample_num, render_result['n_max'])[:, int(apt_sample_num/2), :].shape                
                #render_result['raw_rgb'].shape (N_rand*255, 3)
                #render_result['ray_id].shape (N_rand*255)
                rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)            
#                print('case1. rgbper.shape:'rgbper.shape)                
                rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
                loss += cfg_train.weight_rgbper * rgbper_loss
            else:
                rgbper_loss=torch.tensor([0])
        else:
            # if (cfg_train.weight_rgbper_aperture > 0) and (cfg_train.learning_dir_start > global_step) and (cfg_train.learning_cam_start<=global_step): 
            if (cfg_train.weight_rgbper_aperture > 0) : #and ((args.iter_train_single_ray_end <= global_step) or 0<i_outer):# and (global_step < cfg_train.learning_dir_start ) :             
                if ts_apt is not None:
                # if cfg_train.weight_rgbper_aperture > 0:
                
                
                    Ws = model_cam.get_Ws()
                    sel_pos = model_cam.index2position(sel_i)
                    sel_img_idx = sel_pos[:, 0]
                    sel_pix_idx = (sel_pos[:, 2:3]*Ws[sel_pos[:,0]] + sel_pos[:, 1:2]).flatten()
                    sel_ts = ts_apt[sel_img_idx, sel_pix_idx].unsqueeze(dim=-1)

                    mask_final = render_result['mask_final']
                    n_batch_pix, n_apt_sample, _ = rays_o.shape
                    n_pnt_sample = int(mask_final.shape[0]/(n_batch_pix*n_apt_sample))
                    mask_cnt = torch.sum(mask_final)
                    if True:
                        # apt_sample_rate = int(np.sqrt(apt_sample_num))

                        # center mask
                        mask_center = torch.zeros_like(mask_final, dtype=torch.bool)
                        for i in range(n_batch_pix):
                            mask_center[ i*( n_apt_sample*n_pnt_sample)+((n_apt_sample//2)*n_pnt_sample):
                                        i*( n_apt_sample*n_pnt_sample)+((n_apt_sample//2)*n_pnt_sample) + n_pnt_sample] = 1
                        mask_final_center = mask_center & mask_final

                        # all apt 
                        # mask_center = torch.ones_like(mask_final, dtype=torch.bool)
                        # mask_final_center = mask_final

                        idx_pixel_apt = torch.nonzero((mask_final_center.view(n_batch_pix*n_apt_sample, -1))).squeeze()[:, 0]
                        rays_d_masked = rays_d.view(-1, 3)[idx_pixel_apt]
                        
                        # pts_rel_len = torch.norm(rays_d_masked * render_result['s'][mask_center[mask_final]][:, None], dim=-1)                        
                        # sel_ts_masked = sel_ts.flatten()[idx_pixel_apt]
                        # weight_infocus = torch.exp(-(cfg_train.tau_rgbper_aperture*(sel_ts_masked - pts_rel_len).pow(2)))#.reshape(-1, 1)
                        # weight_infocus = weight_infocus.detach()
                        
                        sel_ts_masked = sel_ts.flatten()[idx_pixel_apt]
                        weight_infocus = torch.exp(-(cfg_train.tau_rgbper_aperture*(sel_ts_masked - render_result['s'][mask_center[mask_final]]).pow(2)))#.reshape(-1, 1)
                        weight_infocus = weight_infocus.detach()
                        
                        valid_pixel_apt_idx = torch.unique( idx_pixel_apt )
                        mask_allsample2pixelapt = (idx_pixel_apt[:, None].expand(-1, valid_pixel_apt_idx.shape[0]) == valid_pixel_apt_idx[None, :].expand(idx_pixel_apt.shape[0],-1))
                        weight_infocus_deno=torch.sum(weight_infocus[:, None] * mask_allsample2pixelapt, dim=0)+1e-7
                        weight_infocus = weight_infocus[:, None].repeat(1, valid_pixel_apt_idx.shape[0])*mask_allsample2pixelapt/weight_infocus_deno[None, :]
                        weight_infocus = weight_infocus[mask_allsample2pixelapt]
                        weight_density = render_result['weights'][mask_center[mask_final]].detach()

                        # trials -------------
                        #rgbper = weight_infocus*weight_density*(((render_result['raw_rgb'][mask_center[mask_final]] - target[(render_result['ray_id'][mask_center[mask_final]]//n_apt_sample)]).pow(2)).sum(-1))
                        #rgbper = weight_density*(((render_result['raw_rgb'][mask_center[mask_final]] - target[(render_result['ray_id'][mask_center[mask_final]]//n_apt_sample)]).pow(2)).sum(-1))
                        rgbper = weight_infocus*(((render_result['raw_rgb'][mask_center[mask_final]] - target[(render_result['ray_id'][mask_center[mask_final]]//n_apt_sample)]).pow(2)).sum(-1))
                        #rgbper = ((weight_infocus+weight_density)/2)*(((render_result['raw_rgb'][mask_center[mask_final]] - target[(render_result['ray_id'][mask_center[mask_final]]//n_apt_sample)]).pow(2)).sum(-1))
                        # -------------------

                    else: # all apt rays
                        idx_pixel_apt = torch.nonzero((mask_final.view(n_batch_pix*n_apt_sample, -1))).squeeze()[:, 0]
                        rays_d_masked = rays_d.view(-1, 3)[idx_pixel_apt]
                        #pts_rel_len = torch.norm(rays_d[:, :, None, :] * render_result['s'].view(n_batch_pix, n_apt_sample, -1, 1), dim=-1)
                        pts_rel_len = torch.norm(rays_d_masked * render_result['s'][mask_final][:, None], dim=-1)
                        
                        sel_ts_masked = sel_ts.flatten()[idx_pixel_apt]
                        weight_infocus = torch.exp(-(cfg_train.tau_rgbper_aperture*(sel_ts_masked - pts_rel_len).pow(2)))#.reshape(-1, 1)                        
                        weight_fullpnt = torch.zeros(n_batch_pix * n_apt_sample, n_pnt_sample)
                        weight_fullpnt[mask_final.view(n_batch_pix * n_apt_sample, n_pnt_sample)] = weight_infocus.detach()
                        weight_infocus = (weight_fullpnt/torch.sum(weight_fullpnt, dim=1, keepdim=True))[mask_final.view(n_batch_pix * n_apt_sample, n_pnt_sample)]
                        
                        # weight_density = render_result['weights'][mask_final].detach()
                        # trials -------------
                        #rgbper = weight_infocus*weight_density*(((render_result['raw_rgb'][mask_center[mask_final]] - target[(render_result['ray_id'][mask_center[mask_final]]//n_apt_sample)]).pow(2)).sum(-1))
                        #rgbper = weight_density*(((render_result['raw_rgb'][mask_center[mask_final]] - target[(render_result['ray_id'][mask_center[mask_final]]//n_apt_sample)]).pow(2)).sum(-1))
                        rgbper = weight_infocus*(((render_result['raw_rgb'][mask_final] - target[(render_result['ray_id'][mask_final]//n_apt_sample)]).pow(2)).sum(-1))
                        #rgbper = ((weight_infocus+weight_density)/2)*(((render_result['raw_rgb'][mask_center[mask_final]] - target[(render_result['ray_id'][mask_center[mask_final]]//n_apt_sample)]).pow(2)).sum(-1))
                        # -------------------
                        
                    # make it onehot --------------------------------
                    # weight_infocus_onehot = torch.zeros_like(weight_infocus)
                    # weight_infocus_onehot[torch.argmax(weight_infocus, dim=0), torch.arange(weight_infocus.shape[1])] =1
                    # weight_infocus = weight_infocus_onehot
                    # ---------------------------------
                    
                    # torch.zeros(idx_pixel_apt.shape[0], torch.unique( idx_pixel_apt ).shape[0])
                    # for ipa in torch.unique( idx_pixel_apt ):
                    #     mask_samples_with_samePA = (idx_pixel_apt == ipa)
                    #     weight_infocus[mask_samples_with_samePA]/=torch.sum(weight_infocus*mask_samples_with_samePA)

                    
                    #weight_infocus = weight_infocus.detach() / (weight_infocus.sum()+1e-7) #* render_result['weights'].detach()[:, None]

                    #rgbper = (weight*((render_result['raw_rgb'] - target[(render_result['ray_id']//n_apt_sample)]).pow(2))).sum(-1)
                    
                    #weight = weight.detach() * render_result['weights'][mask_center[mask_final]].detach()[:, None]
                    #weight = render_result['weights'][mask_center[mask_final]].detach()[:, None]
                    #rgbper = weight*(((render_result['raw_rgb'][mask_center[mask_final]] - target[(render_result['ray_id'][mask_center[mask_final]]//n_apt_sample)]).pow(2)).sum(-1, keepdim=True))
                    # weight_density = render_result['weights'][mask_center[mask_final]].detach()[:, None]                  
                    # rgbper = weight_infocus*weight_density*(((render_result['raw_rgb'][mask_center[mask_final]] - target[(render_result['ray_id'][mask_center[mask_final]]//n_apt_sample)]).pow(2)).sum(-1, keepdim=True))
                    # weight_density = render_result['weights'][mask_center[mask_final]].detach()
                    
                    # trials -------------
                    #rgbper = weight_infocus*weight_density*(((render_result['raw_rgb'][mask_center[mask_final]] - target[(render_result['ray_id'][mask_center[mask_final]]//n_apt_sample)]).pow(2)).sum(-1))
                    #rgbper = weight_density*(((render_result['raw_rgb'][mask_center[mask_final]] - target[(render_result['ray_id'][mask_center[mask_final]]//n_apt_sample)]).pow(2)).sum(-1))
                    # rgbper = weight_infocus*(((render_result['raw_rgb'][mask_center[mask_final]] - target[(render_result['ray_id'][mask_center[mask_final]]//n_apt_sample)]).pow(2)).sum(-1))
                    #rgbper = ((weight_infocus+weight_density)/2)*(((render_result['raw_rgb'][mask_center[mask_final]] - target[(render_result['ray_id'][mask_center[mask_final]]//n_apt_sample)]).pow(2)).sum(-1))
                    # -------------------
                    #rgbper = weight_infocus*(((render_result['raw_rgb'][mask_center[mask_final]] - target[(render_result['ray_id'][mask_center[mask_final]]//n_apt_sample)]).pow(2)).sum(-1, keepdim=True))
#                    print('case2. rgbper.shape:',rgbper.shape)
                    #rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
                    #rgbper_loss = rgbper.sum() / len(rays_o)
                    #rgbper_loss = rgbper.sum() / (n_batch_pix * n_apt_sample * n_pnt_sample*3)                    
                    #rgbper_loss = rgbper.sum() / (n_batch_pix * n_apt_sample)
                    #rgbper_loss = rgbper.sum() / n_batch_pix
                    #rgbper_loss = (torch.exp(-torch.tensor(global_step-1)*1.4/cfg_train.N_iters))*(rgbper.sum() / (n_apt_sample * n_batch_pix * n_pnt_sample * 3))
                    #rgbper_loss = (rgbper.sum() / (n_apt_sample * n_batch_pix * n_pnt_sample * 3))
                    rgbper_loss = (rgbper.sum() / (n_apt_sample * n_batch_pix))
                    # rgbper_loss *= (0.99)**global_step
                    #rgbper_loss = rgbper.sum() / torch.sum(mask_final_center)
                    loss += cfg_train.weight_rgbper_aperture * rgbper_loss
                else:
                    rgbper_loss = torch.tensor([0])
                    mask_final = torch.tensor([0])
                    mask_cnt = mask_final.sum()
            else:
                rgbper_loss = torch.tensor([0])
                mask_final = torch.tensor([0])
                mask_cnt = mask_final.sum()
        loss.backward()
        
        # if len(optimizer.param_groups) != 8:
        #     raise RuntimeError('unexpected number of opt params')

        if global_step<cfg_train.tv_before and global_step>cfg_train.tv_after and global_step%cfg_train.tv_every==0:
            if cfg_train.weight_tv_density>0:
                model.density_total_variation_add_grad(
                    cfg_train.weight_tv_density/len(rays_o), global_step<cfg_train.tv_dense_before)
            if cfg_train.weight_tv_k0>0:
                model.k0_total_variation_add_grad(
                    cfg_train.weight_tv_k0/len(rays_o), global_step<cfg_train.tv_dense_before)

        # do not update rgbnet dir weight by zeroing out grad
        #if cfg_train.learning_dir_start <= global_step:
        if cfg_train.learning_dir_start > global_step:
            for i_opt_g, param_group in enumerate(optimizer.param_groups):
                # if (not ('name' in param_group.keys())) or (param_group['name']!='rgbnet'):
                if (not ('name' in param_group.keys())) :
                    continue
                if (param_group['name']!='rgbnet'):
                    continue
                # initial weight gradient
                # param_group['params'][0].grad[:, model.k0_dim:]=0 
                param_group['params'][0].grad[:, -3:]=0 
        optimizer.step()
        
        psnr_lst.append(psnr.item())

        # if not args.pinhole_camera:
        #     psnr_lst_center.append(psnr_center.item())
        
        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            # if param_group['name'] == 'col2pix_residual':
            #     param_group['lr'] = param_group['lr'] * (0.5 ** (1/decay_steps))
            #     continue

            param_group['lr'] = param_group['lr'] * decay_factor
        
        delta_inside_loop_compute_loss_and_update_optimizer = time.time()-start_inside_loop_compute_loss_and_update_optimizer
        time_inside_loop_compute_loss_and_update_optimizer += delta_inside_loop_compute_loss_and_update_optimizer
        start_inside_loop_save_intermediate_results = time.time()
        # check log & save
        if global_step%args.i_print==0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
            
            if not args.no_wandb:
                if args.pinhole_camera:
                    wandb.log({'Loss':loss.item(), 'PSNR': np.mean(psnr_lst)}, step=global_step)
                else:
                    wandb.log({'Loss':loss.item(), 
                        'loss_intrinsic': loss_intrinsic.item(),
                        'loss_extrinsic': loss_extrinsic.item(),
                        'loss_col': loss_col.item(),
                        # 'loss_lensfp':loss_lensfp.item(),
                        'loss_rgbper': rgbper_loss.item(),
                        'PSNR': np.mean(psnr_lst), 

                        # about memory and speed                        
                        f'mask_cnt(max=255*nray*nbatch={mask_final.shape[0]})':mask_cnt.item(),
                        'memory_allocated(Byte)':torch.cuda.memory_allocated(),
                        'max_memory_allocated(Byte)':torch.cuda.max_memory_allocated(),
                        'time_inside_loop_vrender(Sec)':delta_inside_loop_vrender,
                        'time_loop(Sec)':delta_inside_loop_before_vrender+delta_inside_loop_vrender+delta_inside_loop_compute_loss_and_update_optimizer,#+time_inside_loop_save_intermediate_results,

                        # 'PSNR_center': np.mean(psnr_lst_center), 
                        #'imgFX':model_cam.intrinsics[0,0].item(),
                        'imgFX':model_cam.intrinsics[0,0].item() + model_cam.intrinsics[0,0].item() * model_cam.intrinsics_residual[0,0].item(),
                        #'imgFY':model_cam.intrinsics[0,1].item(),
                        'imgFY':model_cam.intrinsics[0,1].item() + model_cam.intrinsics[0,1].item() * model_cam.intrinsics_residual[0,1].item(),
                        #'imgCX':model_cam.intrinsics[0,2].item(),
                        'imgCX':model_cam.intrinsics[0,2].item() + model_cam.intrinsics[0,2].item() * model_cam.intrinsics_residual[0,2].item(),
                        #'imgCY':model_cam.intrinsics[0,3].item(),
                        'imgCY':model_cam.intrinsics[0,3].item() + model_cam.intrinsics[0,3].item() * model_cam.intrinsics_residual[0,3].item(),
                        'col2pixX': model_cam.get_col2pix()[0].item(),
                        'col2pixY': model_cam.get_col2pix()[1].item(),
                        'T_norm_mean':  model_cam.get_RTs()[:, :3, 3].norm(dim=1).mean().item() },
                        
                        step=global_step + (i_outer * cfg_train.N_iters))
            
            if args.pinhole_camera:
                tqdm.write( f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                            f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '                       
                            f'Eps: {eps_time_str}')
            else:
                tqdm.write( f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                            f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / '# PSNR_center: {np.mean(psnr_lst_center):5.2f} / '                       
                            f'Eps: {eps_time_str}')            

        if global_step%args.i_weights==0:
            if cfg.expname_train is None:
                path = os.path.join(cfg.basedir, cfg.expname, f'{stage}_{global_step:06d}.tar')
            else:
                path = os.path.join(cfg.basedir, cfg.expname_train+args.exp_postfix, f'{stage}_{global_step:06d}.tar')
            # torch.save({
            #     'global_step': global_step,
            #     'model_kwargs': model.get_kwargs(),
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            # }, path)
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'model_cam_kwargs': model_cam.get_kwargs(),
                'model_cam_state_dict': model_cam.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', path)
        delta_inside_loop_save_intermediate_results=time.time()-start_inside_loop_save_intermediate_results
        time_inside_loop_save_intermediate_results+=delta_inside_loop_save_intermediate_results

    time_outside_loop_save_checkpoint = time.time()
    if global_step != -1:
        # if args.pinhole_camera:
        #     torch.save({
        #         'global_step': global_step,
        #         'model_kwargs': model.get_kwargs(),
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #     }, last_ckpt_path)
        # else:
        torch.save({
            'global_step': global_step,
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'model_cam_kwargs': model_cam.get_kwargs(),
            'model_cam_state_dict': model_cam.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        print(f'scene_rep_reconstruction ({stage}): saved checkpoints at', last_ckpt_path)
    
    time_outside_loop_save_checkpoint = time.time()-time_outside_loop_save_checkpoint
    sec2str=lambda eps_time:f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print(f'time_before_loop: {sec2str(time_before_loop)}')
    print(f'time_inside_loop_before_vrender: {sec2str(time_inside_loop_before_vrender)}')
    print(f'time_inside_loop_vrender: {sec2str(time_inside_loop_vrender)}')
    print(f'time_inside_loop_compute_loss_and_update_optimizer: {sec2str(time_inside_loop_compute_loss_and_update_optimizer)}')
    print(f'time_inside_loop_save_intermediate_results: {sec2str(time_inside_loop_save_intermediate_results)}')
    print(f'time_outside_loop_save_checkpoint: {sec2str(time_outside_loop_save_checkpoint)}')
    torch.cuda.empty_cache()

def train(args, cfg, data_dict, i_outer, last_ckpt_path, reload_ckpt_path):

    # init
    print('train: start')
    eps_time = time.time()

    if cfg.expname_train is None:
        os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
        with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))
    else:
        train_path = os.path.join(cfg.basedir, cfg.expname_train + args.exp_postfix)
        os.makedirs(train_path , exist_ok=True)
        with open(os.path.join(train_path , 'args.txt'), 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        cfg.dump(os.path.join(train_path, 'config.py'))


    # added -------------------------
    # create camera model
    model_cam = ApertureCamera(data_dict, aperture_sample_rate = args.aperture_sample_rate, aperture_sample_scale = args.aperture_sample_scale)
    # --------------------------------
    
    # coarse geometry searching (only works for inward bounded scenes)
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse, ts_apt = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, model_cam=model_cam, **data_dict)
    if cfg.coarse_train.N_iters > 0: # False
        scene_rep_reconstruction(
                args=args, cfg=cfg,
                cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse, ts_apt=ts_apt,
                data_dict=data_dict, stage='coarse',
                # added ------------------------
                model_cam=model_cam
                # -----------------------
                )
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse geometry searching in', eps_time_str)
        if cfg.expname_train is None:
            coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_last.tar')
        else:
            coarse_ckpt_path = os.path.join(cfg.basedir, cfg.expname_train+args.exp_postfix, f'coarse_last.tar')
    else:
        print('train: skip coarse geometry searching')
        coarse_ckpt_path = None

    # fine detail reconstruction
    eps_fine = time.time()
    if cfg.coarse_train.N_iters == 0: # True (fortress)
        xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
    else:
        xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                model_class=dvgo.DirectVoxGO, model_path=coarse_ckpt_path,
                thres=cfg.fine_model_and_render.bbox_thres)
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
            xyz_min=xyz_min_fine, xyz_max=xyz_max_fine, ts_apt=ts_apt,
            data_dict=data_dict, stage='fine',
            # added --------------------------
            model_cam = model_cam,
            i_outer = i_outer,
            # -------------------------------------
            last_ckpt_path=last_ckpt_path,
            reload_ckpt_path=reload_ckpt_path,
            coarse_ckpt_path=coarse_ckpt_path,

            )
    eps_fine = time.time() - eps_fine
    eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
    print('train: fine detail reconstruction in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')

def pre_train(args, cfg):
        # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    # export scene bbox and camera poses in 3d for debugging and visualization
    if args.export_bbox_and_cams_only: # False
        print('Export bbox and cameras...')
        xyz_min, xyz_max, ts_final = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)
        poses, HW, Ks, i_train = data_dict['poses'], data_dict['HW'], data_dict['Ks'], data_dict['i_train']
        near, far = data_dict['near'], data_dict['far']
        if data_dict['near_clip'] is not None:
            near = data_dict['near_clip']
        cam_lst = []
        for c2w, (H, W), K in zip(poses[i_train], HW[i_train], Ks[i_train]):
            rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                    H, W, K, c2w, cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,)
            cam_o = rays_o[0,0].cpu().numpy()
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]].cpu().numpy()
            cam_lst.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))
        np.savez_compressed(args.export_bbox_and_cams_only,
            xyz_min=xyz_min.cpu().numpy(), xyz_max=xyz_max.cpu().numpy(),
            cam_lst=np.array(cam_lst))
        print('done')
        sys.exit()

    if args.export_coarse_only: # False
        print('Export coarse visualization...')
        with torch.no_grad():
            if cfg.expname_train is None:
                ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_last.tar')
            else:
                ckpt_path = os.path.join(cfg.basedir, cfg.expname_train+args.exp_postfix, 'coarse_last.tar')
            model = utils.load_model(dvgo.DirectVoxGO, ckpt_path).to(device)
            alpha = model.activate_density(model.density.get_dense_grid()).squeeze().cpu().numpy()
            rgb = torch.sigmoid(model.k0.get_dense_grid()).squeeze().permute(1,2,3,0).cpu().numpy()
        np.savez_compressed(args.export_coarse_only, alpha=alpha, rgb=rgb)
        print('done')
        sys.exit()

    return data_dict

if __name__=='__main__':
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    # load setup
    parser = config_parser()
    args = parser.parse_args()
    
    if args.aperture_sample_rate % 2 == 0:
        args.aperture_sample_rate += 1
    if args.aperture_sample_scale % 2 == 0:
        args.aperture_sample_scale += 1
    
    if not os.path.isfile(args.config):
        print(f'I got the config file:{args.config}')
        dataname_full= os.path.basename(args.config.split('_train')[0])         
        dataname = dataname_full.split('_fake')[0]
        config_file_path = args.config.split(dataname)[0] + args.config.split(dataname)[1]
        default_config=True
    else:
        print(f'using the config file named.. :{args.config}')
        dataname_full= os.path.basename(args.config.split('_train')[0])         
        dataname = dataname_full.split('_fake')[0]
        config_file_path = args.config
        default_config=False
    cfg = mmcv.Config.fromfile(config_file_path)
    
    if default_config:
        cfg.expname = dataname+cfg.expname
        cfg.expname_train = dataname+cfg.expname_train
        if isinstance(cfg.data['datadir'], list):
            cfg.data['datadir'][0] = cfg.data['datadir'][0].split('/_')[0] + '/'+ dataname + '_'+ cfg.data['datadir'][0].split('/_')[1]
            cfg.data['datadir'][1] = cfg.data['datadir'][1].split('/_')[0] + '/'+ dataname + '_'+ cfg.data['datadir'][1].split('/_')[1]
        else:
            cfg.data['datadir'] = cfg.data['datadir'].split('/_')[0] + '/'+ dataname + '_'+ cfg.data['datadir'].split('/_')[1]

    if not args.no_wandb:
        print('wandb tags:',list(filter(None, args.exp_postfix.split('_'))))
        wandb_dir=os.path.join(cfg.basedir,'../wandb')
        print('wandb_dir:', wandb_dir)
        os.makedirs(wandb_dir, exist_ok=True)	
        wandb.init(
            # settings=wandb.Settings(start_method="fork"),
            #name=cfg.expname + '_trial7',
            name=cfg.expname + args.exp_postfix,
            # project="dvgo",  
            project="wacv",  
            # added ---------
            entity="anerf",
            #entity=cfg.wandb_entity,#'emjay73',
#            dir=cfg.wandb_init_dir,
            dir=wandb_dir,
            tags=list(filter(None, args.exp_postfix.split('_')))+[dataname],
            #settings=wandb.Settings(start_method='fork')
            # --------------
        )

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()


    # train
    torch.autograd.set_detect_anomaly(True)
    # remove col2pix
    if isinstance(cfg.data.datadir, list):
        basedir = cfg.data.datadir[0]
        basedir_test = cfg.data.datadir[1]
    else:
        basedir = cfg.data.datadir
        basedir_test = None
    
    # testsavedir = os.path.join(cfg.basedir, cfg.expname+args.exp_postfix)
    testsavedir = os.path.join(cfg.basedir, cfg.expname_train+args.exp_postfix) 
    os.makedirs(testsavedir, exist_ok=True)
    # for f in glob.glob(os.path.join(testsavedir, "col2pix*.npy")):
    # # for f in glob.glob(os.path.join(basedir, "col2pix*.npy")):
    #     print(f"remove {f}")
    #     os.remove(f)
    if not args.render_only:
        
        i_outer_start = 0
        # i_outer_start_path = os.path.join(basedir, 'i_outer_last.npy')        
        i_outer_start_path = os.path.join(testsavedir, 'i_outer_last.npy')               
        
        if os.path.exists(i_outer_start_path):
            print('load i_outer_start from '+os.path.join(testsavedir, 'i_outer_last.npy'))
            i_outer_start = np.load(os.path.join(testsavedir, 'i_outer_last.npy'))


        last_ckpt_path, reload_ckpt_path = make_last_reload_ckpt_path(args, cfg)
        reload_ckpt_path_inner = reload_ckpt_path
        for i in range(i_outer_start, cfg.fine_train.N_outers):
            print(f'i_outer: {i}')
            np.save(i_outer_start_path, i)
            data_dict = pre_train(args, cfg)
            # last_ckpt_path, reload_ckpt_path = make_last_reload_ckpt_path(args, cfg)            
            # train(args, cfg, data_dict, i, last_ckpt_path, reload_ckpt_path)
            train(args, cfg, data_dict, i, last_ckpt_path, reload_ckpt_path = reload_ckpt_path_inner)
            # np.save(i_outer_start_path, i+1)
            reload_ckpt_path_inner = None
            torch.cuda.empty_cache()
            
            # rendering ==================================================================================================
            print('BeforeRender')
            do_rendering(args, cfg, data_dict, device, i)
            print('Done')

            # if (i==0) and (reload_ckpt_path is not None):
            #     break
        
        if i_outer_start>=cfg.fine_train.N_outers:
            data_dict = pre_train(args, cfg)
            print('BeforeRender')
            do_rendering(args, cfg, data_dict, device, cfg.fine_train.N_outers)
            print('Done')
    else:
        data_dict = pre_train(args, cfg)
        print('BeforeRender')
        do_rendering(args, cfg, data_dict, device, cfg.fine_train.N_outers)
        print('Done')
