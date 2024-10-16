
import os
import torch
import numpy as np
import imageio
import wandb

from lib import utils, dvgo, dcvgo, dmpigo
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
# added ---------------------
from lib.aperture_camera import ApertureCamera
import math, cv2
import time
from lib.dists_pt import dists_obj
from datetime import datetime
from skimage.draw import disk
def DiskKernel(dim):
    kernelwidth = dim
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
    circleCenterCoord = int(dim / 2)
    circleRadius = circleCenterCoord +1
    # circleRadius = circleCenterCoord
    
    rr, cc = disk((circleCenterCoord, circleCenterCoord), circleRadius)
    kernel[rr,cc]=1
    kernel = torch.from_numpy(kernel)
    return kernel
# --------------------------


@torch.no_grad()
def render_viewpoints(  model, render_poses, HW, Ks, 
                        ndc, render_kwargs,
                        gt_imgs=None, savedir=None, dump_images=False,
                        render_factor=0, render_video_flipy=False, render_video_rot90=0,
                        eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False, eval_dists=False, use_mask=True):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)    

    if render_factor!=0: # False
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    dists = []
    lpips_alex = []
    lpips_vgg = []

    # added -------------------
    weights = []
    raw_rgbs = []
    ss = []
    # -------------------------

    if not use_mask:
        mask_cache_temp = model.mask_cache
        fast_color_thres_temp = model.fast_color_thres
        model.mask_cache = None
        model.fast_color_thres = 0
    print('debug', render_poses.shape)
    #for i, c2w in enumerate(tqdm(render_poses)):    
    # 
    render_time_total = 0
    time0=time.time()    
    for i in range(render_poses.shape[0]):
        c2w = render_poses[i]
        # print('debug')
        H, W = HW[i]        
        K = Ks[i]        
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, 
                c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=render_kwargs['flip_x'], flip_y=render_kwargs['flip_y'])                
               
        
        # added --------------------------------------------------------        
        keys_raw = ['weights', 'raw_rgb', 's']        
        # original ---------------------------------------------------------
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        # --------------------------------------------------------------------
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)

        # modified -------------------------------------------------------------------------------------        
        if not use_mask:            
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, use_mask=use_mask, **render_kwargs).items() if (k in keys or k in keys_raw)}
                for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
            ]

        else:
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, use_mask=use_mask, **render_kwargs).items() if (k in keys)}
                for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
            ]
        # render_result = {
        #     k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
        #     for k in keys
        # }
        # original -------------------------------------------------------------------------------------
        # render_result_chunks = [
        #     {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
        #     for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        # ]        
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }        
        # ------------------------------------------------------------------------------------------------
        
        rgb = render_result['rgb_marched'].cpu().numpy()

        render_time = time.time() - time0        
        render_time_total += render_time
        render_time_str = f'{render_time//3600:02.0f}:{render_time//60%60:02.0f}:{render_time%60:02.0f}'
        print(f'render_time_str:{render_time_str}')
        time0 = time.time()
        
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()



        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)

        # added -------------------------------------------
        if not use_mask:
            raw_rgbs.append(render_result['raw_rgb'].cpu().numpy())
            weights.append(render_result['weights'].cpu().numpy())
            ss.append(render_result['s'].cpu().numpy())        
        # -------------------------------------------------

        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            print('Testing psnr', p, f'({i})')            

            if eval_ssim:
                ssim_ = utils.rgb_ssim(rgb, gt_imgs[i], max_val=1)
                print('Testing ssim', ssim_, f'({i})')
                ssims.append(ssim_)
            if eval_lpips_alex:
                lpips_alex_ = utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device)
                print('Testing lpips_alex', lpips_alex_, f'({i})')
                lpips_alex.append(lpips_alex_)
            if eval_lpips_vgg:
                lpips_vgg_ = utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device)
                print('Testing lpips_vgg', lpips_vgg_, f'({i})')
                lpips_vgg.append(lpips_vgg_)
            if eval_dists:
                # calculate DISTS between X, Y (a batch of RGB images, data range: 0~1)
                # X: (N,C,H,W) 
                # Y: (N,C,H,W) 
                dists_value = dists_obj(
                    torch.from_numpy(rgb).unsqueeze(0).permute(0,3,1,2), 
                    torch.from_numpy(gt_imgs[i]).unsqueeze(0).permute(0,3,1,2))
                dists_value = dists_value.item()
                print('Testing dists', dists_value, f'({i})')
                dists.append(dists_value)
    
    print(f'render time avr:{render_time_total/len(render_poses)}s')
    
    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')        
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')
        if eval_dists: print('Testing dists', np.mean(dists), '(avg)')

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    print(f'savedir:{savedir}, dump_image:{dump_images}')
    if savedir is not None and dump_images:
        os.makedirs(os.path.join(savedir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(savedir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(savedir, 'depths'), exist_ok=True)
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, 'images', '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            if not use_mask:
                filename = os.path.join(savedir, 'raw', f'raw_rgb_depth_weight_{i:03}.npz')
                if not os.path.exists(filename):
                    np.savez(filename, raw_rgb=raw_rgbs[i], raw_depth=ss[i], raw_weights=weights[i])
                filename = os.path.join(savedir, 'raw', 'gt_{:03d}.png'.format(i))
                if not os.path.exists(filename):
                    rgb8 = utils.to8b(gt_imgs[i])
                    imageio.imwrite(filename, rgb8)

        for i in trange(len(depths)):
            depth8 = utils.to8b(1 - depths / np.max(depths))[i].squeeze()
            filename = os.path.join(savedir, 'depths', f'{i:03d}.png')#.format(i))
            imageio.imwrite(filename, depth8)
            
        # dump csv
        if eval_ssim and eval_dists and eval_lpips_alex:
            fpath = os.path.join(savedir, f'metric_avr.csv')
            with open(fpath, 'w') as f:
                # f.write(f"{fpath},{np.mean(psnrs)},{np.mean(ssims)},{np.mean(lpips_alex)},{np.mean(dists)}\n")
                f.write(f"{fpath}\n")
                f.write(f"{render_time_total/len(render_poses)},{np.mean(psnrs)},{np.mean(ssims)},{np.mean(lpips_alex)},{np.mean(dists)}\n")

                for i, (psnr, ssim, lpipsa, dist) in enumerate(zip(psnrs, ssims, lpips_alex, dists)):                    
                    f.write(f"{i},{psnr:.8f},{ssim:.8f},{lpipsa:.8f},{dist:.8f}\n")
    del raw_rgbs
    del ss
    del weights
    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    if not use_mask:
        model.mask_cache = mask_cache_temp
        model.fast_color_thres = fast_color_thres_temp 
        
    return rgbs, depths, bgmaps

# added -------------------------------------------------------------------
def inverse_RT(c2w):
    w2c = torch.zeros_like(c2w)
    R = c2w[:3, :3]
    T = c2w[:3, -1]

    # newR = R^t    
    for r in range(0,3):
        for c in range(0,3):
            w2c[c,r] = R[r,c]
    
    # newT = -(R^t)@T    
    w2c[:3, -1] = torch.sum(-T[..., np.newaxis, :] * w2c[:3, :3], -1) 

    return w2c

@torch.no_grad()
def render_aperture_rays(  model, render_poses, Hs, Ws, Ks, 
                        # added ------------------------------------
                        apt_sample_rate, lensFps, lensDps, col2pix, col2mm,
                        # ------------------------------------------
                        ndc, render_kwargs,
                        gt_imgs=None, savedir=None, dump_images=False,
                        render_factor=0, render_video_flipy=False, render_video_rot90=0,
                        eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False, eval_dists=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    #assert len(render_poses) == len(HW) and len(HW) == len(Ks)
    #assert len(render_poses) == len(HW) and len(HW) == len(Ks) and len(Ks) == len(lens_info_list)
    assert len(render_poses) == len(Hs)
    assert len(render_poses) == len(Ws) 
    assert len(render_poses) == len(Ks) 

    if render_factor!=0: # False
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, (c2w, H, W, K, lensFp, lensDp) in enumerate(tqdm(zip(render_poses, Hs, Ws, Ks, lensFps, lensDps))):

        #H, W = HW[i]
        #K = Ks[i]

        # added -------------------------------
        #lens_info = lens_info_list[i]
        #col2pix = col2pix_list[i]
        # -------------------------------------

        c2w = torch.Tensor(c2w)
        rays_o, rays_d, apt_sample_num = dvgo.get_rays_aperture(
                    H, W, K, 
                    # added --------------------
                    apt_sample_rate, lensFp, lensDp, col2pix,
                    # --------------------------
                    c2w, inverse_y=render_kwargs['inverse_y'], 
                    flip_x=render_kwargs['flip_x'], flip_y=render_kwargs['flip_y'])
        #col2mmX, col2mmY = col2X[1]
        #col2pixX, col2pixY = col2X[0]
        col2mmX, col2mmY = col2mm[0], col2mm[1]
        col2pixX, col2pixY = col2pix[0], col2pix[1]
        # convert back to cam coordinate (mm)
        w2c = inverse_RT(c2w)
        #col2mmX, col2mmY = col2pix[0] /lens_info['mm2pixX'], col2pix[1] /lens_info['mm2pixY']
        
        rays_o_cam = torch.sum(rays_o[..., np.newaxis, :] * w2c[:3, :3], -1) + w2c[:3, -1].view(1,1,1,3)    # (n_apt_sample x n_pixel) x 3         
        rays_d_cam = torch.sum(rays_d[..., np.newaxis, :] * w2c[:3, :3], -1)
        rays_d_cam = rays_d_cam/rays_d_cam.norm(dim=-1, keepdim=True)
        rays_o_cam_sel = rays_o_cam[[int(H/2),int(H/2),int(H/2)], [0, int(W/2), W-1], :, :] # col scale
        rays_d_cam_sel = rays_d_cam[[int(H/2),int(H/2),int(H/2)], [0, int(W/2), W-1], :, :]
        rays_o_cam_sel *= col2mmX # mm # display only       

        # world origin to cam coordinate
        world_origin_cam = torch.sum(torch.zeros((1,3)) * w2c[:3, :3], -1) + w2c[:3, -1].view(1,1,1,3)
        world_origin_cam *= col2mmX
        world_origin_cam = world_origin_cam.squeeze()

        # draw origins
        os.makedirs(os.path.join(savedir, 'ray_display'), exist_ok=True)
        fig = plt.figure() # x,z
        
        file_name = f"F{lens_info['FNumber']:.02f}_IF{K[0][0]/lens_info['mm2pixX']:.02f}mm_LF{lens_info['lensFXp']/lens_info['mm2pixX']:.02f}mm.png"
        plt.title(file_name)
        plt.scatter(world_origin_cam[0].item(), -world_origin_cam[2].item(), c='black')
        for ray_o_cam in rays_o_cam_sel:
            for apt in ray_o_cam:
                plt.scatter(apt[0].item(), -apt[2].item(), c='black')        

        # draw directions
        colors = ['lightcoral', 'coral', 'khaki', 'forestgreen', 'royalblue', 'mediumblue', 'mediumpurple'] # roughly rainbow colors
        #rayd_scaleup = abs(world_origin_cam[2]).item()
        rayd_scaleup = (lens_info['lensFXp']/lens_info['mm2pixX'])*(K[0][0]/lens_info['mm2pixX'])/(K[0][0]/lens_info['mm2pixX']-lens_info['lensFXp']/lens_info['mm2pixX'])
        rayd_scaleup*=1.1
        plt.ylim([0, rayd_scaleup])
        for ip, (op, dp) in enumerate(zip(rays_o_cam_sel, rays_d_cam_sel)): # for each pixel
            for opa, dpa in zip(op, dp): # for each aperture position
                plt.arrow(  opa[0].item(), -opa[2].item(), dpa[0].item()*rayd_scaleup , -dpa[2].item()*rayd_scaleup , color = colors[ip])
        plt.savefig(os.path.join(savedir, 'ray_display', file_name))
        plt.close()
        viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
        
        if ndc:
            rays_o, rays_d = dvgo.ndc_rays_aperture(H, W, K[0][0], 1., rays_o, rays_d)
            #rays_o, rays_d = dvgo.ndc_rays_aperture(H, W, W/2, 1., rays_o, rays_d)
        
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        #apt_sample_num = int( render_result["rgb_marched"].shape[0]/len(sel_i) )
        if gamma !=1 : 
	        rgb = torch.pow(torch.mean(render_result["rgb_marched"].view(-1, apt_sample_num, 3), dim=1), gamma)
        else:
	        rgb = torch.mean(render_result["rgb_marched"].view(-1, apt_sample_num, 3), dim=1)
        # rgb = render_result['rgb_marched'].cpu().numpy()
        # depth = render_result['depth'].cpu().numpy()
        # bgmap = render_result['alphainv_last'].cpu().numpy()

        #rgb = torch.pow(torch.mean(render_result['rgb_marched'].view(H,W,-1,3), dim=2), 1/2.2).cpu().numpy()
        # rgb = torch.mean(render_result['rgb_marched'].view(H,W,-1,3), dim=2).cpu().numpy()
        depth = render_result['depth'][...,int(apt_sample_num/2)].cpu().numpy()
        bgmap = render_result['alphainv_last'][...,int(apt_sample_num/2)].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if render_video_flipy: # False
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0: # False
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images: # False
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps

@torch.no_grad()
def render_viewpoints_aperture(  model, render_poses, 
                        # added ------------------------------------
                        Hs, Ws, Ks, apt_sample_rates, apt_sample_scales, gamma, lensFps, lensDps, col2pix, col2mms, distortions, i_outer,
                        # ------------------------------------------
                        ndc, render_kwargs,
                        gt_imgs=None, savedir=None, dump_images=False,
                        render_factor=0, render_video_flipy=False, render_video_rot90=0,
                        eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False, eval_dists=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(Hs)
    assert len(render_poses) == len(Ws) 
    assert len(render_poses) == len(Ks)
    assert len(render_poses) == len(apt_sample_rates)
    #assert len(render_poses) == len(HW) and len(HW) == len(Ks) and len(Ks) == len(lens_info_list)

    if render_factor!=0: # False
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    rgbs_txt = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    dists = []
    render_time_total = 0
    
    for i, (c2w, H, W, K, lensFp, lensDp, apt_sample_rate, apt_sample_scale, col2mm, distortion) in enumerate(tqdm(zip(render_poses, Hs, Ws, Ks, lensFps, lensDps, apt_sample_rates, apt_sample_scales, col2mms, distortions))):
        time0=time.time()
        # H, W = HW[i]
        # K = Ks[i]

        # added -------------------------------
        #lens_info = lens_info_list[i]
        #col2pix = col2pix_list[i]
        # -------------------------------------

        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs, apt_sample_num = dvgo.get_rays_of_a_view_aperture(
                H.item(), W.item(), K, distortion,
                # added --------------------------------                
                apt_sample_rate, lensFp.view(-1,2), lensDp.view(-1,2), col2pix, 
                # --------------------------------------
                c2w.view(-1, 3, 4), ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=render_kwargs['flip_x'], flip_y=render_kwargs['flip_y'])
                #flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        # rays_o, rays_d = dvgo.ndc_rays_aperture(H.item(), W.item(), K[0,0].item(), 1., rays_o, rays_d)
        keys = ['rgb_marched', 'depth', 'alphainv_last']        
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)
        
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        #apt_sample_num = int( render_result["rgb_marched"].shape[0]/len(sel_i) )
        #render_result["rgb_marched"] = torch.pow(torch.mean(render_result["rgb_marched"].view(-1, apt_sample_num, 3), dim=1), 2.2)
        # rgb = render_result['rgb_marched'].cpu().numpy()
        # depth = render_result['depth'].cpu().numpy()
        # bgmap = render_result['alphainv_last'].cpu().numpy()
        #rgb = torch.pow(torch.mean(render_result['rgb_marched'].view(H,W,-1,3), dim=2), 1/2.2).cpu().numpy()

        # apt_sample_rate_wide = apt_sample_rate * apt_sample_scale
        apt_sample_rate_wide = apt_sample_scale
        apt_sample_num_wide = apt_sample_rate_wide**2
        
        rgb_marched_mean=torch.nn.functional.interpolate(
                        render_result["rgb_marched"].view(-1, apt_sample_rate, apt_sample_rate, 3).permute(0, 3, 1, 2), 
                        # scale_factor=apt_sample_scale.item(), 
                        size = (apt_sample_scale, apt_sample_scale),
                        mode='bilinear',
                        align_corners=True
                        ).permute(0, 2,3,1)
        # disk_width = apt_sample_rate * apt_sample_scale
        disk_width = apt_sample_scale
        rays_mask_wide = DiskKernel(disk_width).view(1, disk_width, disk_width, 1).to(device=rgb_marched_mean.device)            
            
        rgb_marched_mean  = (rgb_marched_mean*rays_mask_wide).reshape(-1, apt_sample_num_wide, 3).sum(dim=1) / torch.max(torch.sum(rays_mask_wide, dim=(1,2)), torch.tensor([1]))
        if gamma != 1:
            rgb_marched_mean = torch.pow(rgb_marched_mean+torch.finfo(rgb_marched_mean.dtype).eps, gamma)
            
        rgb = rgb_marched_mean.reshape(H,W,3).cpu().numpy()
        # rgb = torch.mean(rgb_marched_mean.view(H,W,-1,3), dim=2).cpu().numpy()
        # rgb = torch.mean(render_result['rgb_marched'].view(H,W,-1,3), dim=2).cpu().numpy()

        render_time = time.time() - time0        
        render_time_total += render_time
        render_time_str = f'{render_time//3600:02.0f}:{render_time//60%60:02.0f}:{render_time%60:02.0f}'
        print(f'render_time_str:{render_time_str}')
        # time0 = time.time()

        depth = render_result['depth'][...,int(apt_sample_num/2)].cpu().numpy()
        bgmap = render_result['alphainv_last'][...,int(apt_sample_num/2)].cpu().numpy()

        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            print('Testing psnr', p, f'({i})')
            
            # if eval_ssim:
            #     ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            # if eval_lpips_alex:
            #     lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            # if eval_lpips_vgg:
            #     lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

            if eval_ssim:
                ssim_ = utils.rgb_ssim(rgb, gt_imgs[i], max_val=1)
                print('Testing ssim', ssim_, f'({i})')
                ssims.append(ssim_)
            if eval_lpips_alex:
                lpips_alex_ = utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device)
                print('Testing lpips_alex', lpips_alex_, f'({i})')
                lpips_alex.append(lpips_alex_)
            if eval_lpips_vgg:
                lpips_vgg_ = utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device)
                print('Testing lpips_vgg', lpips_vgg_, f'({i})')
                lpips_vgg.append(lpips_vgg_)
            if eval_dists:
                # calculate DISTS between X, Y (a batch of RGB images, data range: 0~1)
                # X: (N,C,H,W) 
                # Y: (N,C,H,W) 
                dists_value = dists_obj(
                    torch.from_numpy(rgb).unsqueeze(0).permute(0,3,1,2), 
                    torch.from_numpy(gt_imgs[i]).unsqueeze(0).permute(0,3,1,2))
                dists_value = dists_value.item()
                print('Testing dists', dists_value, f'({i})')
                dists.append(dists_value)
    
    # print(f'render time avr:{render_time_total/len(render_poses)}s')
        
        # rgb_text =cv2.putText(img=np.copy(rgb), 
        #     text=f"imgF{(K[0,0]/col2pix[0]*col2mm[0]).item():.02f}mm_lensF{(lensFp[0]/col2pix[0]*col2mm[0]).item():.02f}mm_lensD{(lensDp[0]/col2pix[0]*col2mm[0]).item():.02f}mm_sampleN{apt_sample_num.item()}", 
        #     org=(50,50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(150,150,150), thickness=3)
        rgb_text =cv2.putText(img=np.concatenate((np.zeros((80, rgb.shape[1], 3)), np.copy(rgb)), axis=0),
            # text=f"F-num {int(lensFp[0].item()/lensDp[0].item()):00d} ImgF {K[0,0].item():.02f}pix LensF {(lensFp[0]).item():.02f}pix LensR {(lensDp[0]).item()/2:.02f}pix sampleN {apt_sample_num.item()**2}", 
            text=f"F-num {(lensFp[0].item()/lensDp[0].item()):04.1f} ImgF {K[0,0].item():.01f}pix LensF {(lensFp[0]).item():.01f}pix LensR {(lensDp[0]).item()/2:05.1f}pix sampleN {apt_sample_num.item()}", 
            org=(30,50), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(150,150,150), thickness=1)
        # rgb8 = utils.to8b(rgb_text)        
        # imageio.imwrite('rgb_text.png', rgb8)

        rgbs_txt.append(rgb_text)
        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)

    print(f'render time avr:{render_time_total/len(render_poses)}s')

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')        
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')
        if eval_dists: print('Testing dists', np.mean(dists), '(avg)')
    
    if render_video_flipy: # False
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0: # False
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0,1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0,1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0,1))

    if savedir is not None and dump_images: # False
        os.makedirs(os.path.join(savedir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(savedir, 'depths'), exist_ok=True)
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, 'images', f'outer_{i_outer}_{i:03d}.png')#.format(i))
            imageio.imwrite(filename, rgb8)
        for i in trange(len(rgbs_txt)):
            rgb8 = utils.to8b(rgbs_txt[i])
            filename = os.path.join(savedir, 'images', f'outer_{i_outer}_{i:03d}_txt.png') #.format(i))
            imageio.imwrite(filename, rgb8)
        for i in trange(len(depths)):
            depth8 = utils.to8b(1 - depths / np.max(depths))[i].squeeze()
            filename = os.path.join(savedir, 'depths', f'outer_{i_outer}_{i:03d}.png')#.format(i))
            imageio.imwrite(filename, depth8)
            
        # dump csv
        if eval_ssim and eval_dists and eval_lpips_alex:
            fpath = os.path.join(savedir, f'outer_{i_outer}_metric_avr.csv')
            with open(fpath, 'w') as f:
                # f.write(f"{fpath},{np.mean(psnrs)},{np.mean(ssims)},{np.mean(lpips_alex)},{np.mean(dists)}\n")
                f.write(f"{fpath}\n")
                f.write(f"{render_time_total/len(render_poses)},{np.mean(psnrs)},{np.mean(ssims)},{np.mean(lpips_alex)},{np.mean(dists)}\n")

                for i, (psnr, ssim, lpipsa, dist) in enumerate(zip(psnrs, ssims, lpips_alex, dists)):                    
                    f.write(f"{i},{psnr:.8f},{ssim:.8f},{lpipsa:.8f},{dist:.8f}\n")

    rgbs = np.array(rgbs_txt)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps
# -------------------------------------------------------------------------------
@torch.no_grad()
def do_rendering(args, cfg, data_dict, device, i_outer):
    print('args.render_test: ', args.render_test)
    print('args.render_train: ', args.render_train)
    print('args.render_video: ', args.render_video)
    print('args.render_video_aperture: ', args.render_video_aperture)
    print('args.render_video_pose: ', args.render_video_pose)
    print('args.render_video_pose_wo_apt: ', args.render_video_pose_wo_apt)

    # load model for rendring
    if not (args.render_test or args.render_train or args.render_video or args.render_video_aperture or args.render_video_pose or args.render_video_pose_wo_apt):
        print('nothing to render')
        return
    print('Start Rendering')

    if args.ft_path:
        ckpt_path = args.ft_path
    else:
        if cfg.expname_train is None:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname_train+args.exp_postfix, 'fine_last.tar')

    ckpt_name = ckpt_path.split('/')[-1][:-4]
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO

    if args.pinhole_camera:
        model = utils.load_model(model_class, ckpt_path).to(device)
    else:
        model_cam = ApertureCamera(data_dict, aperture_sample_rate = args.aperture_sample_rate, aperture_sample_scale= args.aperture_sample_scale)
        model_cam = model_cam.eval()
        #model, model_cam= utils.load_models(model_class, ApertureCamera, ckpt_path)
        model, model_cam= utils.load_models_except_buffer(model_class, model_cam, ckpt_path)
        model = model.to(device)
        model_cam = model_cam.to(device)

    stepsize = cfg.fine_model_and_render.stepsize
    render_viewpoints_kwargs = {
        'model': model,
        'ndc': cfg.data.ndc,
        'render_kwargs': {
            'near': data_dict['near'],
            'far': data_dict['far'],
            'bg': 1 if cfg.data.white_bkgd else 0,
            'stepsize': stepsize,
            'inverse_y': cfg.data.inverse_y,
            'flip_x': cfg.data.flip_x,
            'flip_y': cfg.data.flip_y,
            'render_depth': True,
        },
    }
    # os.environ["IMAGEIO_FFMPEG_EXE"] = "/scratch/x2438a03/.conda/envs/lensnerf_go/bin/ffmpeg"
    # render trainset and eval
    if args.render_train:
        print('Render Train..')
        if args.pinhole_camera:
            trainsavedir = os.path.join(cfg.basedir, cfg.expname+args.exp_postfix, f'render_train_{ckpt_name}')
            os.makedirs(trainsavedir, exist_ok=True)
            print('All results are dumped into', trainsavedir)
            print(data_dict['poses'][data_dict['i_train']].shape)
            rgbs, depths, bgmaps = render_viewpoints(
                    render_poses=data_dict['poses'][data_dict['i_train']],
                    HW=data_dict['HW'][data_dict['i_train']],
                    Ks=data_dict['Ks'][data_dict['i_train']],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                    savedir=trainsavedir, dump_images=args.dump_images,
                    eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg, eval_dists=args.eval_dists,
                    use_mask=False, # added
                    **render_viewpoints_kwargs)
            imageio.mimwrite(os.path.join(trainsavedir, f'outer_{i_outer}_video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(trainsavedir, f'outer_{i_outer}_video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)
        
        else:
            i_train = data_dict['i_train']        
            render_poses = data_dict['poses'][i_train]
            Hs = model_cam.get_Hs(i_train)
            Ws = model_cam.get_Ws(i_train)
            Ks = model_cam.get_Ks(i_train)
            lensFps = model_cam.get_lensFps(i_train)
            lensDps = model_cam.get_lensDps(i_train)
            col2pix = model_cam.get_col2pix()
            col2mms = model_cam.get_col2mms()
            distortions = model_cam.get_distortions(i_train)
            apt_sample_rates = model_cam.get_apt_sample_rates(i_train)
            apt_sample_scales = model_cam.get_apt_sample_scales(i_train)
            rgbs, depths, bgmaps = render_viewpoints_aperture(                                        
                    render_poses = render_poses,                
                    Hs=Hs, Ws=Ws, Ks=Ks,
                    # added ---------------------------------------                        
                    apt_sample_rates = apt_sample_rates,
                    apt_sample_scales = apt_sample_scales,
                    gamma=args.gamma,
                    lensFps = lensFps, lensDps= lensDps, col2pix = col2pix, col2mms=col2mms, distortions=distortions,
                    i_outer = i_outer,
                    # ---------------------------------------------
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],                        
                    savedir=testsavedir, dump_images=args.dump_images,
                    eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg, eval_dists=args.eval_dists,
                    **render_viewpoints_kwargs)
                
    # render testset and eval
    if args.render_test:      
        testsavedir = os.path.join(cfg.basedir, cfg.expname+args.exp_postfix, f'render_test_{ckpt_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)

        if args.pinhole_camera:
            rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg, eval_dists=args.eval_dists,
                **render_viewpoints_kwargs)
        else:
            i_test = data_dict['i_test']
            #render_poses = model_cam.get_RTs(i_test)
            render_poses = data_dict['poses'][i_test]
            Hs = model_cam.get_Hs(i_test)
            Ws = model_cam.get_Ws(i_test)
            Ks = model_cam.get_Ks(i_test)
            lensFps = model_cam.get_lensFps(i_test)
            lensDps = model_cam.get_lensDps(i_test)
            col2pix = model_cam.get_col2pix()
            col2mms = model_cam.get_col2mms()
            apt_sample_rates = model_cam.get_apt_sample_rates(i_test)
            apt_sample_scales = model_cam.get_apt_sample_scales(i_test)
            distortions = model_cam.get_distortions(i_test)
            rgbs, depths, bgmaps = render_viewpoints_aperture(                        
                    #render_poses=data_dict['poses'][data_dict['i_test']],
                    render_poses = render_poses,
                    #H=Hs[0], W=Ws[0], K=Ks[0],
                    Hs=Hs, Ws=Ws, Ks=Ks,
                    # added ---------------------------------------                        
                    apt_sample_rates = apt_sample_rates,
                    apt_sample_scales = apt_sample_scales,
                    gamma = args.gamma,
                    lensFps = lensFps, lensDps= lensDps, col2pix = col2pix, col2mms=col2mms, distortions = distortions,
                    i_outer = i_outer,
                    #lens_info_list = [e for i, e in enumerate(data_dict['lens_info_list']) if i in data_dict['i_test']], 
                    #intrinsics = data_dict['intrinsics'],
                    #center_z = data_dict['center_z'],
                    #col2pix_list = [e for i, e in enumerate(data_dict['col2pix_list']) if i in data_dict['i_test']], 
                    #col2X = data_dict['col2X'],
                    # ---------------------------------------------
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],                        
                    savedir=testsavedir, dump_images=args.dump_images,
                    eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg, eval_dists=args.eval_dists,
                    **render_viewpoints_kwargs)
        if not args.no_wandb:
            wandb.log({'test/rgb[0]':wandb.Image(utils.to8b(rgbs)[0])})
            wandb.log({'test/gt_rgb[0]':wandb.Image(utils.to8b(data_dict['images'][data_dict['i_test'][0]].cpu().numpy()))})                
            wandb.log({'test/depth[0]':wandb.Image(utils.to8b(1 - depths / np.max(depths))[0])})
            
        imageio.mimwrite(os.path.join(testsavedir, f'outer_{i_outer}_video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, f'outer_{i_outer}_video.depth.mp4'), utils.to8b(1 - depths / np.max(depths)), fps=30, quality=8)

    # render video
    if args.render_video or args.render_video_aperture or args.render_video_pose or args.render_video_pose_wo_apt:
        i_test = data_dict['i_test']
        testsavedir = os.path.join(cfg.basedir, cfg.expname+args.exp_postfix, f'render_video_{ckpt_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(testsavedir, exist_ok=True)
        print('All results are dumped into', testsavedir)
        
        with open(os.path.join(testsavedir, 'video_render_poses.npy'), 'wb') as f:    
            rp = data_dict['render_poses'].detach().cpu().numpy()  # torch.Size([240, 3, 4]) 
            np.save(f, rp)

        if args.pinhole_camera:
            rgbs, depths, bgmaps = render_viewpoints(
                render_poses=data_dict['render_poses'],
                HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),

                render_factor=args.render_video_factor,
                render_video_flipy=args.render_video_flipy,
                render_video_rot90=args.render_video_rot90,
                savedir=testsavedir, dump_images=args.dump_images,
                **render_viewpoints_kwargs)

            imageio.mimwrite(os.path.join(testsavedir, f'outer_{i_outer}_video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
            # imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=15, quality=8)
            import matplotlib.pyplot as plt
            depths_vis = depths * (1-bgmaps) + bgmaps
            dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
            depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
            imageio.mimwrite(os.path.join(testsavedir, f'outer_{i_outer}_video.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)
            # imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=15, quality=8)
        else:             
            # render aperture variation -----------------------------------------------------------------
            n_poses = len(data_dict['render_poses'])

            # get references 
            # test index 0 image
            render_poses = model_cam.get_RTs(i_test)[0:1]
            render_poses = render_poses.expand(n_poses, -1, -1)

            # test index render pose 1/4 point
            idx=int(data_dict['render_poses'].shape[0]*0.25)
            render_poses2=data_dict['render_poses'][idx:idx+1]
            render_poses2 = render_poses2.expand(n_poses, -1, -1)

            lensFps = model_cam.get_lensFps(i_test)[0:1] 
            lensFps = lensFps.expand(n_poses, -1)              
            Hs = model_cam.get_Hs(i_test)[0:1] 
            Hs = Hs.expand(n_poses, -1)              
            Ws = model_cam.get_Ws(i_test)[0:1] 
            Ws = Ws.expand(n_poses, -1)              
            Ks = model_cam.get_Ks(i_test)[0:1] 
            distortions = model_cam.get_distortions(i_test)[0:1]
            distortions = distortions.expand(n_poses, -1) 
            # Ks = Ks.expand(n_poses, -1, -1) # oops...     
            col2pix = model_cam.get_col2pix()                
            col2mms = model_cam.get_col2mms()[0:1] 
            col2mms = col2mms.expand(n_poses, -1)              
            mm2pixs = model_cam.get_mm2pix()[0:1] 
            mm2pixs = mm2pixs.expand(n_poses, -1)              
            pix2mms = 1/mm2pixs
            n_scene = 4
            lensDps = torch.zeros((n_poses, 2))                
            Ks_gen = torch.zeros(n_poses, 3, 3)
            # #f_nums = np.linspace(3, 22, n_poses//n_scene)
            # # f_nums = np.linspace(1, 22, n_poses//n_scene)
            # f_nums = np.logspace(np.log10(4), np.log10(22), (n_poses//n_scene)//2) # array([ 4.        ,  4.24218632,  4.4990362 ,  4.77143745,  5.06033167,...       17.39016446, 18.44307945, 19.55974484, 20.7440205 , 22.        ])
            # # f_nums = np.linspace(2, 22, n_poses//n_scene) # array([ 2.        ,  4.22222222,  6.44444444,  8.66666667, 10.88888889, 13.11111111, 15.33333333, 17.55555556, 19.77777778, 22.        ])
            # # f_nums = np.linspace(4, 22, n_poses//n_scene) 
            # for ipos, fnum in enumerate(f_nums):       
            #     pos0 = ipos
            #     pos1 = 2*((n_poses//n_scene)//2)+ipos         
                
            #     lensDps[pos0, :] = lensFps[pos0]/fnum.item()
            #     lensDps[pos1, :] = lensFps[pos1]/fnum.item()
                
            #     Ks_gen[pos0] = Ks[0].clone()
            #     Ks_gen[pos1] = Ks[0].clone()

            # f_nums = np.linspace(22, 1, n_poses//n_scene)
            # f_nums = np.logspace(np.log(22), np.log(2), n_poses//n_scene)
            # f_nums = np.logspace(np.log10(22), np.log10(2), n_poses//n_scene) #array([22., 16.85436452, 12.9122547 ,  9.89217489,  7.57846916, 5.80592189,  4.44796018,  3.40761556,  2.61059976,  2.])
            # f_nums = np.logspace(np.log10(3), np.log10(22), (n_poses//n_scene)) # array([22.        , 19.77777778, 17.55555556, 15.33333333, 13.11111111, 10.88888889,  8.66666667,  6.44444444,  4.22222222,  2.        ])            
            # for ipos, fnum in enumerate(f_nums):                
            #     pos1=ipos
            #     # pos2 = len(f_nums)*2-1-ipos #((n_poses//n_scene))+ipos         
            #     pos2 = len(f_nums)+ipos #((n_poses//n_scene))+ipos         
                
            #     lensDps[pos1, :] = lensFps[pos1]/fnum.item()
            #     lensDps[pos2, :] = lensFps[pos2]/f_nums[-ipos-1].item()

            #     Ks_gen[pos1] = Ks[0].clone()
            #     Ks_gen[pos2] = Ks[0].clone()

            f_nums = np.array([ 2.        ,  2.17240147,  2.35966408,  2.56306886,  2.78400728,
                                3.02399075,  3.28466098,  3.56780117,  3.87534826,  4.20940613,
                                4.57226003,  4.96639221,  5.39449887,  5.85950865,  6.3646026 ,
                                6.91323603,  7.50916206,  8.15645735,  8.85954998,  9.6232497 ,
                                10.45278091, 11.35381831, 12.3325258 , 13.3955986 , 14.55030905,
                                15.80455639, 17.16692078, 18.64672198, 20.25408313, 22.,
                                1233.22537816, 1019.42765155,  842.69490001,  696.6013659 ,
                                575.835291  ,  476.00578838,  393.48319582,  325.26710636,
                                268.87727761,  222.26345364,  183.73082048,  151.87838505,
                                125.54803699,  103.78244137,   85.79023133,   70.91723508,
                                58.62269111,   48.45958686,   40.05840595,   33.1136931 ,
                                27.3729482 ,   22.62744572,   18.70464576,   15.46192078,
                                12.78136979,   10.56553167,    8.73384162,    7.21970194,
                                5.9680606 ,    4.93340967 ])
            f_nums = np.clip(f_nums.repeat(2),2, 22)
            for ipos, fnum in enumerate(f_nums):                
                # pos1=ipos                
                # pos2 = len(f_nums)+ipos #((n_poses//n_scene))+ipos        
                lensDps[ipos, :] = lensFps[ipos]/fnum.item()                
                Ks_gen[ipos] = Ks[0].clone()
                
            
            # change focal length ----------------------------------------------------------------
            fnum = 4
            # img_f_mm = np.logspace( np.log10(lensFps[0,0].item()*pix2mms[0,0].item()*1.09), 
            #                         np.log10(lensFps[0,0].item()*pix2mms[0,0].item()*1.3), n_poses//n_scene) 
            # img_fx = img_f_mm*mm2pixs[0,0].item()
            # img_fy = img_f_mm*mm2pixs[0,1].item()
            # img_fx = np.linspace( Ks[0,0,0].item(), Ks[0,0,0].item()*0.95, (n_poses//n_scene)//2)        #0.93 # 0.85   
            # img_fy = np.linspace( Ks[0,1,1].item(), Ks[0,1,1].item()*0.95, (n_poses//n_scene)//2)
#            img_fx = np.linspace( Ks[0,0,0].item(), Ks[0,0,0].item()*0.97, (n_poses//n_scene)//2)        #0.93 # 0.85   
#            img_fy = np.linspace( Ks[0,1,1].item(), Ks[0,1,1].item()*0.97, (n_poses//n_scene)//2)
            #img_fx = np.linspace( Ks[0,0,0].item(), Ks[0,0,0].item()*0.95, (n_poses//n_scene)//2)        #0.93 # 0.85   
            #img_fy = np.linspace( Ks[0,1,1].item(), Ks[0,1,1].item()*0.95, (n_poses//n_scene)//2)
            img_fx = np.linspace( Ks[0,0,0].item(), Ks[0,0,0].item()*0.96, (n_poses//n_scene)//2)        #0.93 # 0.85   
            img_fy = np.linspace( Ks[0,1,1].item(), Ks[0,1,1].item()*0.96, (n_poses//n_scene)//2)
            f_nums3 = np.ones_like(img_fx) * fnum
            f_nums3 = np.ones_like(img_fx) * fnum
            f_nums3 = np.ones_like(img_fx) * fnum
            
            for ipos, (fx, fy) in enumerate(zip(img_fx, img_fy)):
                pos=(n_poses//n_scene)*2+ipos                
                Ks_gen[pos, 0,0], Ks_gen[pos, 1,1] = fx, fx
                Ks_gen[pos, 0,2], Ks_gen[pos, 1,2] = Ks[0, 0, 2], Ks[0, 1, 2]                
                Ks_gen[pos, 2,2] = 1
                lensDps[pos, :] = lensFps[pos]/fnum      

                pos=(n_poses//n_scene)*2+((n_poses//n_scene)//2)+ipos                
                Ks_gen[pos, 0,0], Ks_gen[pos, 1,1] = img_fx[-ipos-1], img_fx[-ipos-1]
                Ks_gen[pos, 0,2], Ks_gen[pos, 1,2] = Ks[0, 0, 2], Ks[0, 1, 2]                
                Ks_gen[pos, 2,2] = 1
                lensDps[pos, :] = lensFps[pos]/fnum      

            # img_f_mm = np.logspace( np.log10(lensFps[0,0].item()*pix2mms[0,0].item()*1.3), 
            #                         # np.log10(lensFps[0,0].item()*pix2mms[0,0].item()*1.05), n_poses//n_scene) 
            #                         # np.log10(lensFps[0,0].item()*pix2mms[0,0].item()*1.07), n_poses//n_scene) 
            #                         np.log10(lensFps[0,0].item()*pix2mms[0,0].item()*1.09), n_poses//n_scene) 
            # img_fx = img_f_mm*mm2pixs[0,0].item()
            # img_fy = img_f_mm*mm2pixs[0,1].item()
            # img_fx = np.linspace( Ks[0,0,0].item(), Ks[0,0,0].item()*1.05, (n_poses//n_scene)//2)      #1.07#1.14      
            # img_fy = np.linspace( Ks[0,1,1].item(), Ks[0,1,1].item()*1.05, (n_poses//n_scene)//2)
            #img_fx = np.linspace( Ks[0,0,0].item(), Ks[0,0,0].item()*1.03, (n_poses//n_scene)//2)      #1.07#1.14      
            #img_fy = np.linspace( Ks[0,1,1].item(), Ks[0,1,1].item()*1.03, (n_poses//n_scene)//2)
            #img_fx = np.linspace( Ks[0,0,0].item(), Ks[0,0,0].item()*1.05, (n_poses//n_scene)//2)      #1.07#1.14      
            #img_fy = np.linspace( Ks[0,1,1].item(), Ks[0,1,1].item()*1.05, (n_poses//n_scene)//2)
            img_fx = np.linspace( Ks[0,0,0].item(), Ks[0,0,0].item()*1.04, (n_poses//n_scene)//2) # I never tried it, but 1.03 was the safe choice but not fancy and 1.05 was slightly grid-y 
            img_fy = np.linspace( Ks[0,1,1].item(), Ks[0,1,1].item()*1.04, (n_poses//n_scene)//2)
            f_nums4 = np.ones_like(img_fx) * fnum
            f_nums4 = np.ones_like(img_fx) * fnum
            f_nums4 = np.ones_like(img_fx) * fnum
            # fnum = 2          
            # for ipos, (fx, fy) in enumerate(zip(reversed(img_fx), reversed(img_fy))):
            for ipos, (fx, fy) in enumerate(zip(img_fx, img_fy)):
                pos = (n_poses//n_scene)*3+ipos                
                Ks_gen[pos, 0, 0], Ks_gen[pos, 1, 1] = fx, fx
                Ks_gen[pos, 0,2], Ks_gen[pos, 1,2] = Ks[0, 0, 2], Ks[0, 1, 2]                
                Ks_gen[pos, 2,2] = 1
                lensDps[pos, :] = lensFps[pos]/fnum        

                pos = (n_poses//n_scene)*3+((n_poses//n_scene)//2)+ipos                
                Ks_gen[pos, 0, 0], Ks_gen[pos, 1, 1] = img_fx[-ipos-1], img_fx[-ipos-1]
                Ks_gen[pos, 0,2], Ks_gen[pos, 1,2] = Ks[0, 0, 2], Ks[0, 1, 2]                
                Ks_gen[pos, 2,2] = 1
                lensDps[pos, :] = lensFps[pos]/fnum        

            if 0:
                # save ray directions for debugging purpose
                rgbs_rays = render_aperture_rays(
                    render_poses=data_dict['render_poses'][[0]].repeat(len(data_dict['render_poses']), 1, 1),
                    HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), axis=0),
                    Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), axis=0),

                    # added -------------------------------------------------------
                    apt_sample_rate = args.apt_sample_rate,                        
                    lens_info_list = lens_info_list,
                    intrinsics = data_dict['intrinsics'],
                    #col2pix_list = [e for i, e in enumerate(data_dict['col2pix_list']) if i in data_dict['i_test']], 
                    col2X=data_dict['col2X'],
                    # -------------------------------------------------------------

                    render_factor=args.render_video_factor,
                    render_video_flipy=args.render_video_flipy,
                    render_video_rot90=args.render_video_rot90,
                    savedir=testsavedir, dump_images=args.dump_images,
                    **render_viewpoints_kwargs)
            
            if args.render_video or args.render_video_aperture or args.render_video_pose:
                # apt_sample_rates = torch.ceil(22/(lensFps/lensDps))
                # apt_sample_rates += (apt_sample_rates%2 == 0).int()
                # apt_sample_rates = torch.clamp_max(apt_sample_rates[:, 0].int(), 9)           

                # print('debug')
                
                # apt_sample_rates = torch.ceil(38/(lensFps/lensDps))                
                apt_sample_rates = args.aperture_sample_rate*torch.ones_like(lensFps)
                apt_sample_rates += (apt_sample_rates%2 == 0).int()                
                apt_sample_rates = torch.clamp_max(apt_sample_rates[:, 0].int(), 17)
                # apt_sample_scales = args.aperture_sample_scale*torch.ones_like(apt_sample_rates)

                # f_nums_static = np.concatenate((f_nums, f_nums2, f_nums3, f_nums3, f_nums4, f_nums4), axis=0)
                f_nums_static = np.concatenate((f_nums, f_nums3, f_nums3, f_nums4, f_nums4), axis=0)
                apt_sample_scales = model_cam.compute_apt_sample_scales(torch.from_numpy(f_nums_static).unsqueeze(1))
                
                # apt_sample_rates = torch.clamp_max(apt_sample_rates[:, 0].int(), 1) # debugging
                # apt_sample_rates = args.aperture_sample_rate
                print(f'Fnumber Min: {(lensFps/lensDps).min()}')
                print(f'Fnumber Max: {(lensFps/lensDps).max()}')
                print(f'Apt Sample Rate Min: {apt_sample_rates.min()}')
                print(f'Apt Sample Rate Max: {apt_sample_rates.max()}')
                print(f'Apt Sample Scale Min: {apt_sample_scales.min()}')
                print(f'Apt Sample Scale Max: {apt_sample_scales.max()}')

            if args.render_video or args.render_video_aperture:
                print('render video - aperture and focal length variation')                     
                # second pose                
                rgbs, depths, bgmaps = render_viewpoints_aperture(
                    render_poses=render_poses2,
                    Hs=Hs, Ws=Ws, Ks=Ks_gen,                    
                    apt_sample_rates = apt_sample_rates,
                    apt_sample_scales = apt_sample_scales,
                    gamma=args.gamma,
                    lensFps = lensFps, lensDps=lensDps, col2pix=col2pix, col2mms=col2mms, distortions=distortions, 
                    i_outer = i_outer,                
                    # -------------------------------------------------------------
                    render_factor=args.render_video_factor,
                    render_video_flipy=args.render_video_flipy,
                    render_video_rot90=args.render_video_rot90,
                    savedir=testsavedir, dump_images=args.dump_images,
                    **render_viewpoints_kwargs)
                
                imageio.mimwrite(os.path.join(testsavedir, f'outer_{i_outer}_video_ours.apt.rgb.1over4.fx.min{apt_sample_rates.min()}.max{apt_sample_rates.max()}.mp4'), utils.to8b(rgbs[:n_poses//2]), fps=30, quality=8)
                imageio.mimwrite(os.path.join(testsavedir, f'outer_{i_outer}_video_ours.f.rgb.1over4.fx.min{apt_sample_rates.min()}.max{apt_sample_rates.max()}.mp4'), utils.to8b(rgbs[n_poses//2:]), fps=30, quality=8)

                # first pose
                rgbs, depths, bgmaps = render_viewpoints_aperture(
                    render_poses=render_poses,
                    Hs=Hs, Ws=Ws, Ks=Ks_gen,                    
                    apt_sample_rates = apt_sample_rates,
                    apt_sample_scales = apt_sample_scales,
                    gamma=args.gamma,
                    lensFps = lensFps, lensDps=lensDps, col2pix=col2pix, col2mms=col2mms, distortions=distortions, 
                    i_outer = i_outer,                
                    # -------------------------------------------------------------
                    render_factor=args.render_video_factor,
                    render_video_flipy=args.render_video_flipy,
                    render_video_rot90=args.render_video_rot90,
                    savedir=testsavedir, dump_images=args.dump_images,
                    **render_viewpoints_kwargs)
                
                #imageio.mimwrite(os.path.join(testsavedir, 'video_apt.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
                # imageio.mimwrite(os.path.join(testsavedir, 'video_ours.apt.rgb.0.fx.mp4'), utils.to8b(rgbs[:rgbs.shape[0]//2]), fps=15, quality=8)
                # imageio.mimwrite(os.path.join(testsavedir, 'video_ours.f.rgb.0.fx.mp4'), utils.to8b(rgbs[rgbs.shape[0]//2:]), fps=15, quality=8)
                # imageio.mimwrite(os.path.join(testsavedir, f'outer_{i_outer}_video_ours.apt.rgb.0.fx.min{apt_sample_rates.min()}.max{apt_sample_rates.max()}.mp4'), utils.to8b(rgbs[:rgbs.shape[0]//2]), fps=30, quality=8)
                # imageio.mimwrite(os.path.join(testsavedir, f'outer_{i_outer}_video_ours.f.rgb.0.fx.min{apt_sample_rates.min()}.max{apt_sample_rates.max()}.mp4'), utils.to8b(rgbs[rgbs.shape[0]//2:]), fps=30, quality=8)
                imageio.mimwrite(os.path.join(testsavedir, f'outer_{i_outer}_video_ours.apt.rgb.0.fx.min{apt_sample_rates.min()}.max{apt_sample_rates.max()}.mp4'), utils.to8b(rgbs[:n_poses//2]), fps=30, quality=8)
                imageio.mimwrite(os.path.join(testsavedir, f'outer_{i_outer}_video_ours.f.rgb.0.fx.min{apt_sample_rates.min()}.max{apt_sample_rates.max()}.mp4'), utils.to8b(rgbs[n_poses//2:]), fps=30, quality=8)
                # imageio.mimwrite(os.path.join(testsavedir, 'video_ours.apt.rgb.0.wh.mp4'), utils.to8b(rgbs[:rgbs.shape[0]//2]), fps=15, quality=8)
                # imageio.mimwrite(os.path.join(testsavedir, 'video_ours.f.rgb.0.wh.mp4'), utils.to8b(rgbs[rgbs.shape[0]//2:]), fps=15, quality=8)
                import matplotlib.pyplot as plt
                depths_vis = depths * (1-bgmaps) + bgmaps
                dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
                depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
                #imageio.mimwrite(os.path.join(testsavedir, 'video_apt.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)
                # imageio.mimwrite(os.path.join(testsavedir, 'video_ours.depth.mp4'), utils.to8b(depth_vis), fps=15, quality=8)
                imageio.mimwrite(os.path.join(testsavedir, f'outer_{i_outer}_video_ours.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)

                

            # render pose variation ------------------------------------------------------------------
            if args.render_video or args.render_video_pose:
                print('render video - pose variation')
                # for ipos, fnum in enumerate(f_nums):
                #     pos=(n_poses//n_scene)*2+ipos
                #     lensDps[pos, :] = lensDps[ipos, :] 
                #     pos=(n_poses//n_scene)*3+ipos
                #     lensDps[pos, :] = lensDps[(n_poses//n_scene)+ipos, :] 
                    
                rgbs, depths, bgmaps = render_viewpoints_aperture(
                    render_poses=data_dict['render_poses'],
                    # Hs=Hs, Ws=Ws, Ks=Ks.expand(n_poses, -1, -1),    
                    Hs=Hs, Ws=Ws, Ks=Ks_gen,                        
                    apt_sample_rates = apt_sample_rates,
                    apt_sample_scales = apt_sample_scales,
                    gamma=args.gamma,
                    lensFps = lensFps, lensDps=lensDps, col2pix=col2pix, col2mms=col2mms, distortions=distortions,
                    i_outer = i_outer,
                    # -------------------------------------------------------------
                    render_factor=args.render_video_factor,
                    render_video_flipy=args.render_video_flipy,
                    render_video_rot90=args.render_video_rot90,
                    savedir=testsavedir, dump_images=args.dump_images,
                    **render_viewpoints_kwargs)

                #imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
                # imageio.mimwrite(os.path.join(testsavedir, 'video_ours.rgb.mp4'), utils.to8b(rgbs), fps=15, quality=8)
                imageio.mimwrite(os.path.join(testsavedir, f'outer_{i_outer}_video_ours.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
                import matplotlib.pyplot as plt
                depths_vis = depths * (1-bgmaps) + bgmaps
                dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
                depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
                #imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)
                # imageio.mimwrite(os.path.join(testsavedir, 'video_ours.depth.mp4'), utils.to8b(depth_vis), fps=15, quality=8)
                imageio.mimwrite(os.path.join(testsavedir, f'outer_{i_outer}_video_ours.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)
            
            if args.render_video or args.render_video_pose_wo_apt:
                print('render video - pose variation wo apt')
                fnum=22
                # for ipos in enumerate(Ks.shape[0]):
                #     Ks_gen[ipos] = Ks[0]
                lensDps=lensFps/fnum
                # apt_sample_rates = torch.ceil(38/(lensFps/lensDps))                
                # apt_sample_rates += (apt_sample_rates%2 == 0).int()
                # apt_sample_rates = torch.clamp_max(apt_sample_rates[:, 0].int(), 19)
                apt_sample_rates=torch.ones_like(lensFps)
                apt_sample_rates = torch.clamp_max(apt_sample_rates[:, 0].int(), 1)
                apt_sample_scales = torch.ones_like(apt_sample_rates)
                rgbs, depths, bgmaps = render_viewpoints_aperture(
                    render_poses=data_dict['render_poses'],
                    Hs=Hs, Ws=Ws, Ks=Ks.expand(n_poses, -1, -1),
                    apt_sample_rates = apt_sample_rates,
                    apt_sample_scales = apt_sample_scales,
                    gamma=args.gamma,
                    lensFps = lensFps, lensDps=lensDps, col2pix=col2pix, col2mms=col2mms, distortions=distortions,
                    i_outer = i_outer,
                    # -------------------------------------------------------------
                    render_factor=args.render_video_factor,
                    render_video_flipy=args.render_video_flipy,
                    render_video_rot90=args.render_video_rot90,
                    savedir=testsavedir, dump_images=args.dump_images,
                    **render_viewpoints_kwargs)

                #imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
                # imageio.mimwrite(os.path.join(testsavedir, 'video_ours.fixapt.fixK.rgb.mp4'), utils.to8b(rgbs), fps=15, quality=8)
                imageio.mimwrite(os.path.join(testsavedir, f'outer_{i_outer}_video_ours.fixapt.fixK.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
                import matplotlib.pyplot as plt
                depths_vis = depths * (1-bgmaps) + bgmaps
                dmin, dmax = np.percentile(depths_vis[bgmaps < 0.1], q=[5, 95])
                depth_vis = plt.get_cmap('rainbow')(1 - np.clip((depths_vis - dmin) / (dmax - dmin), 0, 1)).squeeze()[..., :3]
                #imageio.mimwrite(os.path.join(testsavedir, 'video.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)
                # imageio.mimwrite(os.path.join(testsavedir, 'video_ours.fixapt.fixK.depth.mp4'), utils.to8b(depth_vis), fps=15, quality=8)
                imageio.mimwrite(os.path.join(testsavedir, f'outer_{i_outer}_video_ours.fixapt.fixK.depth.mp4'), utils.to8b(depth_vis), fps=30, quality=8)


        
