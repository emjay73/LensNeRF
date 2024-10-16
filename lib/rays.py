
from typing_extensions import runtime
from lib import utils, dvgo, dcvgo, dmpigo

# init batch rays sampler
def gather_training_rays(cfg, cfg_train, data_dict, images, poses, HW, Ks, i_train, device):
    if data_dict['irregular_shape']: # False
        rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
    else:
        rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

    if cfg_train.ray_sampler == 'in_maskcache': # False(fortress)
        raise RuntimeError('ray_sampler=in_maskcache option is not handled in lensnerf yet')
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train],
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                model=model, render_kwargs=render_kwargs)

    elif cfg_train.ray_sampler == 'flatten': # True (fortress)
        
        # if args.pinhole_camera:
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
            rgb_tr_ori=rgb_tr_ori,
            train_poses=poses[i_train],
            HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)

    else:
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
            rgb_tr=rgb_tr_ori,
            train_poses=poses[i_train],
            HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
    index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
    batch_index_sampler = lambda: next(index_generator)
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

def make_indices_generator(N, cfg_train): # N: number of pixels in the whole images
    index_generator = dvgo.batch_indices_generator(N, cfg_train.N_rand)
    batch_index_sampler = lambda: next(index_generator)
    return batch_index_sampler

def gather_training_rays_from_index(cfg, cfg_train, args, data_dict, rgb_tr_ori, i_train, device, model_cam, sel_i, tf_single_ray):
    # poses = model_cam.get_RTs()
    
    # if data_dict['irregular_shape']: # False
    #     rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
    # else:
    #     rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

    if cfg_train.ray_sampler == 'in_maskcache': # False(fortress)
        raise RuntimeError('ray_sampler=in_maskcache option is not handled in lensnerf yet')
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train],
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                model=model, render_kwargs=render_kwargs)

    elif cfg_train.ray_sampler == 'flatten': # True (fortress)
        
        if not args.pinhole_camera:            
            #removed HW, Ks, poses, lens_info_list, intrinsics, col2X
            #rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_batch_flatten_aperture(
            rgb_tr, rays_o_tr, rays_d_tr, rays_mask_tr, viewdirs_tr, imsz = \
                dvgo.get_training_rays_batch_flatten_aperture(
            rgb_tr_ori=rgb_tr_ori,
            # modified -------------------------------------------
            model_cam=model_cam,
            i_train=i_train,
            sel_i = sel_i,      
            # apt_sample_rates = model_cam.apt_sample_rates,
            tf_single_ray = tf_single_ray,
            # ----------------------------------------------------
            ndc=cfg.data.ndc,             
            inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        # --------------------------------------------------------------------------------------------
        else:
            raise RuntimeError('unhandled case')
    else:
        raise RuntimeError('unhandled case')
        # rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
        #     rgb_tr=rgb_tr_ori,
        #     train_poses=poses[i_train],
        #     HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
        #     flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
    # index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
    # batch_index_sampler = lambda: next(index_generator)
    return rgb_tr, rays_o_tr, rays_d_tr, rays_mask_tr, viewdirs_tr, imsz#, batch_index_sampler
