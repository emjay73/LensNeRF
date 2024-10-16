import torch
import numpy as np
import time

from lib import utils, dvgo, dcvgo, dmpigo
from lib.aperture_rays import get_length_with_min_dist

def _compute_bbox_by_cam_frustrm_bounded(
        #cfg, HW, Ks, 
        cfg, Hs, Ws, Ks, 
        poses, i_train, near, far
        ):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    #for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):        
    for H, W, K, c2w in zip(Hs[i_train], Ws[i_train], Ks[i_train], poses[i_train]):        
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H.item(), W=W.item(), K=K, 
                c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max

def _compute_bbox_by_cam_frustrm_unbounded(
        #cfg, HW, Ks, 
        cfg, Hs, Ws, Ks, 
        poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max

# added --------------------------------------------------------------
@torch.no_grad()
def _compute_bbox_by_cam_frustrm_bounded_aperture(
        cfg, Hs, Ws, Ks, distortions, 
        # added ---------------------------        
        apt_sample_rates, lensFps, lensDps, col2pix, 
        # ---------------------------------
        poses, i_train, near, far
        ):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    # ts_final = None
    #for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
    #lens_info_train_list = [e for i, e in enumerate(lens_info_list) if i in i_train]    
    #col2pix_train_list = [e for i, e in enumerate(col2pix_list) if i in i_train]    
    #for (H, W), K, c2w, lens_info in zip(HW[i_train], Ks[i_train], poses[i_train], lens_info_train_list):
    #for H, W, K, c2w, lens_info in zip(Hs[i_train], Ws[i_train], Ks[i_train], poses[i_train], lens_info_train_list):
    ts_list = []
    for H, W, K, c2w, lensFp, lensDp, apt_sample_rate, distortion in zip(Hs[i_train], Ws[i_train], Ks[i_train], poses[i_train], lensFps[i_train], lensDps[i_train], apt_sample_rates[i_train], distortions[i_train]):
        rays_o, rays_d, viewdirs, apt_sample_num = dvgo.get_rays_of_a_view_aperture(
                H=H.item(), W=W.item(), K=K, distortion=distortion,
                # added ---------------------------                
                apt_sample_rate=apt_sample_rate, lensFps=lensFp.view(1,-1), lensDps=lensDp.view(1,-1), col2pix=col2pix, 
                # ---------------------------------
                c2ws=c2w.view(-1,3,4),
                #ndc=cfg.data.ndc, 
                ndc=False, 
                inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc: # True
            rays_o_ndc, rays_d_ndc = dvgo.ndc_rays_aperture(H.item(), W.item(), K[0][0], 1., rays_o, rays_d) 
            pts_nf = torch.stack([rays_o_ndc+rays_d_ndc*near, rays_o_ndc+rays_d_ndc*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
        #xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2,3)))
        #xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2,3)))        
        
        # ts= ((K[0,0]*lensFp[0])/(K[0,0]-lensFp[0]))/col2pix[0]
        n_pix, n_sample, _ = rays_o.shape
        if n_sample > 1:
            ts = torch.zeros((n_pix, n_sample))
            for iray in range(n_sample):
                if iray==int(n_sample//2):
                    continue
                t0, t1 = get_length_with_min_dist(
                    rays_o[:, int(n_sample//2), :], 
                    rays_d[:, int(n_sample//2), :],
                    rays_o[:, iray, :],
                    rays_d[:, iray, :],
                )
                ts[:, int(n_sample//2)] += t0
                ts[:, iray] = t1
            ts[:, int(n_sample//2)] = ts[:, int(n_sample//2)]/(n_sample-1)
            # if ts_final is None:
            #     ts_final = ts.clone()
            # else:
            #     ts_final = torch.max(ts, ts_final)

                # convert to ndc
            
        else:
            # ts_final=None
            ts = None

        if ts is not None and cfg.data.ndc:
            
            rays_d_normalized = rays_d/torch.linalg.norm(rays_d, dim=-1, keepdim=True)
            
            # ts in ndc
            ts = (ts[:, :, None]*rays_d_normalized[:, :, -1:])/(rays_o[:, :, -1:]+ts[:, :, None]*rays_d_normalized[:, :, -1:])
            ts = ts[:, :, 0] # npix x napt x 1 -> npix x napt

        ts_list.append(ts)
    if any([t==None for t in ts_list]):
        ts = None
    else:
        ts = torch.stack(ts_list, dim=0)

    return xyz_min, xyz_max, ts #ts_final

@torch.no_grad()
def _compute_bbox_by_cam_frustrm_unbounded_aperture(
        cfg, Hs, Ws, Ks, distortions,
        # added ---------------------------        
        apt_sample_rates, lensFps, lensDps, col2pix, 
        # ---------------------------------
        poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    
    # lens_info_train_list = [e for i, e in enumerate(lens_info_list) if i in i_train]    
    #col2pix_train_list = [e for i, e in enumerate(col2pix_list) if i in i_train]    
    #for (H, W), K, c2w, lens_info in zip(HW[i_train], Ks[i_train], poses[i_train], lens_info_train_list):
    for H, W, K, c2w, lensFp, lensDp, apt_sample_rate, distortion in zip(Hs[i_train], Ws[i_train], Ks[i_train], poses[i_train], lensFps[i_train], lensDps[i_train], apt_sample_rates[i_train], distortions[i_train]):
        rays_o, rays_d, viewdirs, apt_sample_num = dvgo.get_rays_of_a_view_aperture(
                H=H.item(), W=W.item(), K=K, distortion = distortion,
                # added ---------------------------                
                apt_sample_rate=apt_sample_rate, lensFp=lensFp, lensDp=lensDp, col2pix=col2pix, 
                #-----------------------------------            
                c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        # xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        # xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1,2)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max
# --------------------------------------------------------------------
#def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
@torch.no_grad()
def compute_bbox_by_cam_frustrm(args, cfg, model_cam, i_train, near, far,  **kwargs):
    print('compute_bbox_by_cam_frustrm: start')

    Hs = model_cam.get_Hs()
    Ws = model_cam.get_Ws()
    Ks = model_cam.get_Ks()
    lensFps = model_cam.get_lensFps()
    lensDps = model_cam.get_lensDps()   
    col2pix = model_cam.get_col2pix() 
    poses = model_cam.get_RTs()
    distortions = model_cam.get_distortions()
    apt_sample_rates= model_cam.apt_sample_rates 
    ts_final = None

    if args.pinhole_camera:
        if cfg.data.unbounded_inward: # False
            xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
                    #cfg, HW, Ks,                     
                    cfg, Hs, Ws, Ks,                     
                    poses, i_train, kwargs.get('near_clip', None))

        else:
            xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
                    #cfg, HW, Ks,                     
                    cfg, Hs, Ws, Ks,                     
                    poses, i_train, near, far)
    else:
        if cfg.data.unbounded_inward: # False
            xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded_aperture(
                #cfg, HW, Ks, 
                cfg, Hs, Ws, Ks, 
                # added ---------------------------
                #kwargs.get('lens_info_list'), kwargs.get('intrinsics'), kwargs.get('col2X'),
                apt_sample_rates, lensFps, lensDps, col2pix, 
                # ---------------------------------
                poses, i_train, kwargs.get('near_clip', None))

        else: # True
            xyz_min, xyz_max, ts_final = _compute_bbox_by_cam_frustrm_bounded_aperture(
                    #cfg, HW, Ks, 
                    cfg, Hs, Ws, Ks, distortions,
                    # added ---------------------------
                    # args.apt_sample_rate, kwargs.get('lens_info_list'), kwargs.get('intrinsics'), kwargs.get('col2X'),
                    #args.apt_sample_rate, lensFps, lensDps, col2pix,
                    apt_sample_rates, lensFps, lensDps, col2pix,
                    # ---------------------------------
                    poses, i_train, near, far)
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max, ts_final

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.world_size[0]),
        torch.linspace(0, 1, model.world_size[1]),
        torch.linspace(0, 1, model.world_size[2]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max

