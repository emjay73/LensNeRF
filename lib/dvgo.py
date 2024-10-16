import os
import time
import functools
from tkinter import E
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# added --------------------------
import math, copy
# --------------------------------

from torch_scatter import segment_coo

from . import grid
from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)


'''Model'''
class DirectVoxGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 mask_cache_path=None, mask_cache_thres=1e-3, mask_cache_world_size=None,
                 fast_color_thres=0,
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=4,
                 **kwargs):
        super(DirectVoxGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1/(1-alpha_init) - 1)]))
        print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(
                density_type, channels=1, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.density_config)

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        self.rgbnet_full_implicit = rgbnet_full_implicit
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                    k0_type, channels=self.k0_dim, world_size=self.world_size,
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    config=self.k0_config)
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*viewbase_pe*2)
            if self.rgbnet_full_implicit:
                pass
            elif rgbnet_direct:
                dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim-3
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('dvgo: feature voxel grid', self.k0)
            print('dvgo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = grid.MaskGrid(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2]),
            ), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(
                path=None, mask=mask,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'density_type': self.density_type,
            'k0_type': self.k0_type,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near_clip):
        # maskout grid points that between cameras and their near planes
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density.grid[nearest_dist[None,None] <= near_clip] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
            ), -1)
            self_alpha = F.max_pool3d(self.activate_density(self.density.get_dense_grid()), kernel_size=3, padding=1, stride=1)[0,0]
            self.mask_cache = grid.MaskGrid(
                    path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        print('dvgo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.mask_cache.mask.shape[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.mask_cache.mask.shape[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.mask_cache.mask.shape[2]),
        ), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None,None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('dvgo: voxel_count_views start')
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.world_size.cpu())+1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.get_dense_grid())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
                rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
                ones(rays_pts).sum().backward()
            with torch.no_grad():
                count += (ones.grid.grad > 1)
        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self.world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)

    def hit_coarse_geo(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Check whether the rays hit the solved coarse geometry or not'''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
        return hit.reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # skip known free space
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for alpha w/ post-activation
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for color
        if self.rgbnet_full_implicit:
            pass
        else:
            k0 = self.k0(ray_pts)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            if self.rgbnet_direct:
                k0_view = k0
            else:
                k0_view = k0[:, 3:]
                k0_diffuse = k0[:, :3]
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]
            rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            if self.rgbnet_direct:
                rgb = torch.sigmoid(rgb_logit)
            else:
                rgb = torch.sigmoid(rgb_logit + k0_diffuse)

        # Ray marching
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')
        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                        src=(weights * step_id),
                        index=ray_id,
                        out=torch.zeros([N]),
                        reduce='sum')
            ret_dict.update({'depth': depth})

        return ret_dict


''' Misc
'''
class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None

class Raw2Alpha_nonuni(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        exp, alpha = render_utils_cuda.raw2alpha_nonuni(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_nonuni_backward(exp, grad_back.contiguous(), interval), None, None

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


''' Ray and batch
'''
def get_rays(   H, W, K,                 
                c2w, inverse_y, flip_x, flip_y, mode='center'):
    
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center': # True
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x: # False
        i = i.flip((1,))
    if flip_y: # False
        j = j.flip((0,))
    if inverse_y: # False
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d

# added ----------------------------------------------------------------------------------------
def get_rays_aperture(   H, W, K, 
                # added ------------------------
                #apt_sample_rate, lens_info, intrinsics, col2X,
                apt_sample_rate, lensFp, lensDp, col2pix,
                # ------------------------------
                c2w, inverse_y, flip_x, flip_y, mode='center'):
    # h, w, imgFXp, imgFYp, cx, cy, k1, k2, p1, p2 = intrinsics # just in case when I might take distortion into account
    imgFXp, imgFYp = K[0][0], K[1][1]

    #lensFXp = lens_info['lensFXp']
    #lensFYp = lens_info['lensFYp']
    lensFXp, lensFYp = lensFp[0], lensFp[1]
    if lensFXp > imgFXp or lensFYp > imgFYp:
        raise RuntimeWarning(f"lensFXp({lensFXp}) > imgFXp({imgFXp}) or lensFYp({lensFYp}) > imgFYp({imgFYp})")
        imgFXp = lensFXp +1e-7
        imgFYp = lensFYp +1e-7
    
    # lensDXp = lens_info['lensDXp']
    # lensDYp = lens_info['lensDYp']
    lensDXp, lensDYp = lensDp[0], lensDp[1]

    #1/(center_z*col2pix) + 1/imgFXp = 1/lensFXp
    #1/(center_z*col2pix) = 1/lensFXp - 1/imgFXp = (imgFXp-lensFXp)/lensFXp*imgFXp
    #center_z*col2pix = (lensFXp*imgFXp)/(imgFXp-lensFXp)
    #col2pixX = (lensFXp*imgFXp)/((imgFXp-lensFXp)*center_z)
    #col2pixY = (lensFYp*imgFYp)/((imgFYp-lensFYp)*center_z)
    #col2pixX, col2pixY = col2pix[0], col2pix[1]
    
    # col2pixX, col2pixY = col2X[0]
    # col2mmX, col2mmY = col2X[1]
    col2pixX, col2pixY = col2pix[0], col2pix[1]
    
    lensDXc = lensDXp/col2pixX
    lensDYc = lensDYp/col2pixY

    
    #apt_sample_rate = 3
    apt_sample_num  = apt_sample_rate*apt_sample_rate
    apt_list = torch.zeros(apt_sample_num, 3)
    # deltaX = math.sqrt(2)*lensDXc/(2*(apt_sample_rate-1)) if apt_sample_rate>1 else 0
    # deltaY = math.sqrt(2)*lensDYc/(2*(apt_sample_rate-1)) if apt_sample_rate>1 else 0
    deltaX = lensDXc/(2*(apt_sample_rate-1)) if apt_sample_rate>1 else 0
    deltaY = lensDYc/(2*(apt_sample_rate-1)) if apt_sample_rate>1 else 0

    y, x = torch.meshgrid(
                torch.linspace(0, apt_sample_rate-1, apt_sample_rate, device=c2w.device),
                torch.linspace(0, apt_sample_rate-1, apt_sample_rate, device=c2w.device)) 
    y=(y-int(apt_sample_rate/2))*deltaY
    x=(x-int(apt_sample_rate/2))*deltaX
    apt_list[:, 0] =  x.flatten()
    apt_list[:, 1] = -y.flatten() # y axis upward

    #orisP = orisC = apt_list.reshape(-1, 1, 3).expand(-1, H*W, -1).reshape(-1, 3)
    #orisP = orisC = apt_list.reshape(1, -1, 3).expand(H*W, -1, -1).reshape(-1, 3)
    #orisC = apt_list.reshape(1, -1, 3).repeat(H*W, 1, 1).reshape(H, W, -1, 3)
    #orisC = apt_list.view(1, -1, 3).expand(H*W, -1, -1).view(H, W, -1, 3)
    orisC = apt_list.view(1, -1, 3).expand(H*W, -1, -1)
    col2pix = torch.ones((1,3))
    col2pix[0][0] = col2pixX
    col2pix[0][1] = col2pixY
    
    # orisP = copy.deepcopy(orisC)
    # orisP[..., 0] *= col2pixX
    # orisP[..., 1] *= col2pixY
        
    # ------------------------------------------------------            
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center': # True
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x: # False
        i = i.flip((1,))
    if flip_y: # False
        j = j.flip((0,))
    if inverse_y: # False
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1) # h, w, 3

    # compute refracted ray direction ------------------------------------------
    #K1_x, K1_y = torch.tensor( (1/lensFXp) - (1/imgFXp) , device=c2w.device),  torch.tensor((1/lensFYp) - (1/imgFYp), device=c2w.device)
    K1_x, K1_y = (1/lensFXp) - (1/imgFXp),  (1/lensFYp) - (1/imgFYp)
    K1_x = K1_x.to(c2w.device)
    K1_y = K1_y.to(c2w.device)
    refraction_tensor = torch.ones((3), device=K1_x.device)
    refraction_tensor[0] = K1_x
    refraction_tensor[1] = K1_y
    #dirs = -refraction_tensor.reshape(1, 3) * orisP.clone().reshape(-1, 3) + dirs[None, ...].repeat(apt_sample_num, 1, 1, 1).reshape(-1, 3) # n_apt_sample x n_pixel x 3
    #dirs = -refraction_tensor.reshape(1, 3) * orisP.clone().reshape(-1, 3) + dirs[None, ...].expand(apt_sample_num, -1, -1, -1).reshape(-1, 3) # n_apt_sample x n_pixel x 3
    #dirs = dirs[..., None, :].expand(-1, -1, apt_sample_num, -1).reshape(-1, 3) -refraction_tensor.view(1, 3) * orisP.view(-1, 3) # n_apt_sample x n_pixel x 3    
    dirs = dirs[..., None, :].expand(-1, -1, apt_sample_num, -1).reshape(-1, 3) -refraction_tensor.view(1, 3) * orisC.reshape(-1, 3) * col2pix # n_apt_sample x n_pixel x 3    
    #dirs = dirs[..., None, :].expand(-1, -1, apt_sample_num, -1).view(-1, 3) -refraction_tensor.view(1, 3) * orisC.view(-1, 3) * col2pix # n_apt_sample x n_pixel x 3    
    dirs = dirs.reshape(H, W, apt_sample_num, 3)
    # dirs /= (-dirs[:, :, :, -1])

    # Rotate ray directions from camera frame to the world frame

    # rays_d.shape : torch.Size([h, w, apt_sample_num, 3])
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    
    # rays_d.shape : torch.Size([h, w, apt_sample_num, 3])
    rays_o = torch.sum(orisC[..., np.newaxis, :] * c2w[:3, :3], -1) + c2w[:3, -1].view(1,1,1,3)    # (n_apt_sample x n_pixel) x 3         
    return rays_o, rays_d, apt_sample_num

def undistort(p_d1, k1, k2, p1, p2):

    def distort_normal(x, y, k1,k2, p1,p2):
        r2 = x*x + y*y
        #radial_d = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        radial_d = 1 + k1*r2 + k2*r2*r2
        x_d = radial_d*x + 2*p1*x*y + p2*(r2 + 2*x*x)
        y_d = radial_d*y + p1*(r2 + 2*y*y) + 2*p2*x*y

        return torch.stack([x_d, y_d, -torch.ones_like(x_d)], dim=1)
        # return (x_d, y_d)

    # p_d1 = normalize(p_d)
    p_u1 = p_d1.clone()
    err_threshold=0.00001
    while 1:
        #err = distort_normal(p_u1) - p_d1
        #err = distort_normal(p_u1) - p_u1
        err = distort_normal(p_u1[:, 0], p_u1[:, 1], k1, k2, p1, p2) - p_d1
        p_u1 = p_u1 - err

        if torch.all(torch.abs(err)<err_threshold):
            break

    return p_u1
    # p_u = denormalize(p_u1)
    # return p_u

def get_rays_batch_aperture(   
                #H, W, 
                K, ij, distortion,
                # added ------------------------
                #apt_sample_rate, lens_info, intrinsics, col2X,
                apt_sample_rate, lensFps, lensDps, col2pix,
                # ------------------------------
                c2ws, inverse_y, flip_x, flip_y, mode='center'):

    n_pixels = ij.shape[0]
    # h, w, imgFXp, imgFYp, cx, cy, k1, k2, p1, p2 = intrinsics # just in case when I might take distortion into account
    imgFXp, imgFYp = K[0][0], K[1][1]

    #lensFXp = lens_info['lensFXp']
    #lensFYp = lens_info['lensFYp']
    lensFXp, lensFYp = lensFps[:,0], lensFps[:,1]
    if torch.all(lensFXp > imgFXp) and torch.all(lensFYp > imgFYp):
        raise RuntimeWarning(f"lensFXp({lensFXp}) > imgFXp({imgFXp}) or lensFYp({lensFYp}) > imgFYp({imgFYp})")
        imgFXp = lensFXp +1e-7
        imgFYp = lensFYp +1e-7
    
    # lensDXp = lens_info['lensDXp']
    # lensDYp = lens_info['lensDYp']
    lensDXp, lensDYp = lensDps[:,0], lensDps[:,1]

    #1/(center_z*col2pix) + 1/imgFXp = 1/lensFXp
    #1/(center_z*col2pix) = 1/lensFXp - 1/imgFXp = (imgFXp-lensFXp)/lensFXp*imgFXp
    #center_z*col2pix = (lensFXp*imgFXp)/(imgFXp-lensFXp)
    #col2pixX = (lensFXp*imgFXp)/((imgFXp-lensFXp)*center_z)
    #col2pixY = (lensFYp*imgFYp)/((imgFYp-lensFYp)*center_z)
    #col2pixX, col2pixY = col2pix[0], col2pix[1]
    
    # col2pixX, col2pixY = col2X[0]
    # col2mmX, col2mmY = col2X[1]
    col2pixX, col2pixY = col2pix[0], col2pix[1]
    
    lensDXc = lensDXp/col2pixX
    lensDYc = lensDYp/col2pixY
    
    #apt_sample_rate = 3
    apt_sample_num  = apt_sample_rate*apt_sample_rate
    apt_list = torch.zeros(n_pixels, apt_sample_num, 3)
    # deltaX = math.sqrt(2)*lensDXc/(2*(apt_sample_rate-1)) if apt_sample_rate>1 else torch.zeros((1,1))
    # deltaY = math.sqrt(2)*lensDYc/(2*(apt_sample_rate-1)) if apt_sample_rate>1 else torch.zeros((1,1))
    deltaX = lensDXc/(2*(apt_sample_rate-1)) if apt_sample_rate>1 else torch.zeros((1,1))
    deltaY = lensDYc/(2*(apt_sample_rate-1)) if apt_sample_rate>1 else torch.zeros((1,1))

    y, x = torch.meshgrid(
                torch.linspace(0, apt_sample_rate-1, apt_sample_rate, device=c2ws.device),
                torch.linspace(0, apt_sample_rate-1, apt_sample_rate, device=c2ws.device)) 
    #y=(y-int(apt_sample_rate/2))*deltaY
    y=(y-int(apt_sample_rate/2)).flatten().view(1, -1)*deltaY.view(-1, 1)
    #x=(x-int(apt_sample_rate/2))*deltaX
    x=(x-int(apt_sample_rate/2)).flatten().view(1, -1)*deltaX.view(-1, 1)
    # apt_list[:, 0] =  x.flatten()
    # apt_list[:, 1] = -y.flatten() # y axis upward
    apt_list[:, :, 0] =  x 
    apt_list[:, :, 1] = -y # y axis upward

    #orisP = orisC = apt_list.reshape(-1, 1, 3).expand(-1, H*W, -1).reshape(-1, 3)
    #orisP = orisC = apt_list.reshape(1, -1, 3).expand(H*W, -1, -1).reshape(-1, 3)
    #orisC = apt_list.reshape(1, -1, 3).repeat(H*W, 1, 1).reshape(H, W, -1, 3)
    #orisC = apt_list.view(1, -1, 3).expand(H*W, -1, -1).view(H, W, -1, 3)
    #orisC = apt_list.view(1, -1, 3).expand(n_pixels, -1, -1)
    orisC = apt_list
    col2pix3d = torch.ones((1,3))
    col2pix3d[0][0] = col2pixX
    col2pix3d[0][1] = col2pixY
    
    # orisP = copy.deepcopy(orisC)
    # orisP[..., 0] *= col2pixX
    # orisP[..., 1] *= col2pixY
        
    # ------------------------------------------------------            
    # i, j = torch.meshgrid(
    #     torch.linspace(0, W-1, W, device=c2w.device),
    #     torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = ij[:, 0]
    j = ij[:, 1]
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center': # True
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x: # False
        i = i.flip((1,))
    if flip_y: # False
        j = j.flip((0,))
    if inverse_y: # False
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1) # h, w, 3

    # normalize
    k1, k2, p1, p2 = distortion
    dirs = undistort(dirs, k1, k2, p1, p2)
    # compute refracted ray direction ------------------------------------------
    #K1_x, K1_y = torch.tensor( (1/lensFXp) - (1/imgFXp) , device=c2w.device),  torch.tensor((1/lensFYp) - (1/imgFYp), device=c2w.device)
    K1_x, K1_y = (1/lensFXp) - (1/imgFXp),  (1/lensFYp) - (1/imgFYp)
    K1_x = K1_x.to(c2ws.device)
    K1_y = K1_y.to(c2ws.device)
    #refraction_tensor = torch.ones((3), device=K1_x.device)
    refraction_tensor = torch.ones((n_pixels,3), device=K1_x.device)
    refraction_tensor[:, 0] = K1_x
    refraction_tensor[:, 1] = K1_y
    #dirs = -refraction_tensor.reshape(1, 3) * orisP.clone().reshape(-1, 3) + dirs[None, ...].repeat(apt_sample_num, 1, 1, 1).reshape(-1, 3) # n_apt_sample x n_pixel x 3
    #dirs = -refraction_tensor.reshape(1, 3) * orisP.clone().reshape(-1, 3) + dirs[None, ...].expand(apt_sample_num, -1, -1, -1).reshape(-1, 3) # n_apt_sample x n_pixel x 3
    #dirs = dirs[..., None, :].expand(-1, -1, apt_sample_num, -1).reshape(-1, 3) -refraction_tensor.view(1, 3) * orisP.view(-1, 3) # n_apt_sample x n_pixel x 3    
    #dirs = dirs[..., None, :].expand(-1, -1, apt_sample_num, -1).reshape(-1, 3) -refraction_tensor.view(1, 3) * orisC.reshape(-1, 3) * col2pix # n_apt_sample x n_pixel x 3    
    #dirs = dirs[..., None, :].expand(-1, apt_sample_num, -1).reshape(-1, 3) -refraction_tensor.view(1, 3) * orisC.reshape(-1, 3) * col2pix # n_apt_sample x n_pixel x 3    
    dirs = dirs[..., None, :].expand(-1, apt_sample_num, -1) -refraction_tensor.view(-1,1,3) * orisC * col2pix3d.view(1,1,3) # n_apt_sample x n_pixel x 3    
    #dirs = dirs[..., None, :].expand(-1, -1, apt_sample_num, -1).view(-1, 3) -refraction_tensor.view(1, 3) * orisC.view(-1, 3) * col2pix # n_apt_sample x n_pixel x 3    
    #dirs = dirs.reshape(H, W, apt_sample_num, 3)
    #dirs = dirs.reshape(n_pixels, apt_sample_num, 3)
    # dirs /= (-dirs[:, :, :, -1])

    # Rotate ray directions from camera frame to the world frame

    # rays_d.shape : torch.Size([h, w, apt_sample_num, 3])
    #rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2ws[:, np.newaxis, :3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    
    # rays_d.shape : torch.Size([h, w, apt_sample_num, 3])
    #rays_o = torch.sum(orisC[..., np.newaxis, :] * c2w[:3, :3], -1) + c2w[:3, -1].view(1,1,1,3)             
    rays_o = torch.sum(orisC[..., np.newaxis, :] * c2ws[:, np.newaxis, :3, :3], -1) + c2ws[:, :3, -1].view(-1,1,3)    
    return rays_o, rays_d, apt_sample_num # n_pixel x n_apt_sample x 3         
# ----------------------------------------------------------------------------------------------------

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2] # if we multiply this to the next equ.,
    rays_o = rays_o + t[...,None] * rays_d # rays_o's z component will be -near, which is -1 # step towards rays_d by -near amount

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2] # instead of divide it by abs(rays_o[...,2]), put -1
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2] # 1+2*near/(-near) = -1

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2]) # divided by W/2 so the d2 is 2
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d

# added ---------------------------------------------------------------------------
def ndc_rays_aperture(H, W, focal, near, rays_o, rays_d):
    apt_center_idx = int(rays_o.shape[2]/2) 
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2] # if we multiply this to the next equ.,

    # rays o conversion--------------------------
    rays_o = rays_o + t[...,None] * rays_d # rays_o's z component will be -near, which is -1 # step towards rays_d by -near amount

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2] # instead of divide it by abs(rays_o[...,2]), put -1
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2] # 1+2*near/(-near) = -1

    # rays d conversion -----------------------------------
    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d
# --------------------------------------------------------------------------------

def get_rays_of_a_view( H, W, K,                         
                        c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(
                        H, W, K, 
                        c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs

# added -----------------------------------------------------------
def get_rays_of_a_view_aperture( H, W, K, distortion,
                        # added ---------------------------
                        #apt_sample_rate, lens_info, intrinsics, col2X,
                        apt_sample_rate, lensFps, lensDps, col2pix, 
                        # ---------------------------------
                        c2ws, ndc, inverse_y, flip_x, flip_y, mode='center'):

    # i, j = torch.meshgrid(
    #     torch.linspace(0, W-1, W, device=c2ws.device),
    #     torch.linspace(0, H-1, H, device=c2ws.device))  # pytorch's meshgrid has indexing='ij'
    y, x = torch.meshgrid(
        torch.linspace(0, H-1, H, device=c2ws.device),
        torch.linspace(0, W-1, W, device=c2ws.device))  # pytorch's meshgrid has indexing='ij'

    rays_o, rays_d, apt_sample_num = get_rays_batch_aperture(
                K, torch.stack([x.flatten(), y.flatten()], dim=1), distortion,
                apt_sample_rate, lensFps, lensDps, col2pix,                
                c2ws, inverse_y, flip_x, flip_y, mode)

    # rays_o, rays_d, apt_sample_num = get_rays_aperture(
    #                     H, W, K, 
    #                     # added --------------------
    #                     #apt_sample_rate, lens_info, intrinsics, col2X,
    #                     apt_sample_rate, lensFp, lensDp, col2pix,
    #                     # --------------------------
    #                     c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays_aperture(H, W, K[0][0], 1., rays_o, rays_d)        

    # rays_o = rays_o.reshape(H*W, -1, 3)
    # rays_d = rays_d.reshape(H*W, -1, 3)
    # viewdirs = viewdirs.reshape(H*W, -1, 3)
    
    return rays_o, rays_d, viewdirs, apt_sample_num

def get_rays_of_a_batch_aperture( 
                        Hs, Ws, Ks, distortions, xys,
                        # added ---------------------------
                        #apt_sample_rate, lens_info, intrinsics, col2X,
                        apt_sample_rates, lensFps, lensDps, col2pix, shared_intrinsic,
                        # ---------------------------------
                        c2ws, ndc, inverse_y, flip_x, flip_y, mode='center'):
    
    assert len(Hs) == len(Ws)
    assert len(Hs) == len(Ks) 
    assert len(Hs) == len(xys)
    assert len(Hs) == len(lensFps) 
    assert len(Hs) == len(lensDps)
    assert len(Hs) == len(c2ws)
    assert len(Hs) == len(apt_sample_rates)
    n_pix = xys.shape[0]
    apt_sample_rate_max = apt_sample_rates.max()



    # rays_o = []
    # rays_d = []
    # viewdirs = []
    # rays_mask = []

    if shared_intrinsic and all(apt_sample_rates == apt_sample_rates[0]):
        H = Hs[0]
        W = Ws[0]
        K = Ks[0]
        distortion = distortions[0]
        #rays_mask = None
        rays_mask = torch.ones((n_pix, apt_sample_rate_max * apt_sample_rate_max, 1))        
        rays_o, rays_d, apt_sample_num = get_rays_batch_aperture(
                                K, xys.view(-1,2), distortion,
                                # added --------------------
                                #apt_sample_rate, lens_info, intrinsics, col2X,
                                apt_sample_rates[0], lensFps, lensDps, col2pix,
                                # --------------------------
                                c2ws.view(-1,3,4), inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
        viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
        if ndc:
            rays_o, rays_d = ndc_rays_aperture(H, W, K[0][0], 1., rays_o, rays_d)
            #rays_o, rays_d = ndc_rays_aperture(H, W, W/2, 1., rays_o, rays_d)
    else:
        rays_o = torch.zeros((n_pix, apt_sample_rate_max * apt_sample_rate_max, 3))
        rays_d = torch.zeros_like(rays_o)
        viewdirs = torch.zeros_like(rays_o)
        rays_mask = torch.zeros((n_pix, apt_sample_rate_max * apt_sample_rate_max, 1))
        for i, (H, W, K, xy, lensFp, lensDp, c2w, apt_sample_rate) in enumerate(zip(Hs, Ws, Ks, xys, lensFps, lensDps, c2ws, apt_sample_rates)):
            ray_o, ray_d, apt_sample_num = get_rays_batch_aperture(
                                K, xy.view(-1,2),distortions,
                                # added --------------------
                                #apt_sample_rate, lens_info, intrinsics, col2X,
                                apt_sample_rate, lensFp.view(-1,2), lensDp.view(-1,2), col2pix,
                                # --------------------------
                                c2w.view(-1,3,4), inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
            viewdir = ray_d / ray_d.norm(dim=-1, keepdim=True)
            if ndc:
                ray_o, ray_d = ndc_rays_aperture(H, W, K[0][0], 1., ray_o, ray_d)
                #ray_o, ray_d = ndc_rays_aperture(H, W, W/2, 1., ray_o, ray_d)
            
            n_pix, n_apt, _ = ray_o.shape
            apt_position = range((apt_sample_rate_max*apt_sample_rate_max-n_apt)//2,(apt_sample_rate_max*apt_sample_rate_max-n_apt)//2+n_apt)

            rays_o[i, apt_position, :] = ray_o
            rays_d[i, apt_position, :] = ray_d
            viewdirs[i, apt_position, :] = viewdir
            rays_mask[i, apt_position, :] = 1
            
            # ray_o_temp = torch.zeros(n_pix, apt_sample_rate_max*apt_sample_rate_max, 3)
            # ray_o_temp[:, apt_position, :] = ray_o
            # ray_d_temp = torch.zeros_like(ray_o_temp)
            # ray_d_temp[:, apt_position, :] = ray_d
            # ray_mask_temp = torch.zeros_like(ray_o_temp)
            # ray_mask_temp[:, apt_position, :] = 1

            #rays_o.append(ray_o)
            #rays_d.append(ray_d)
            #viewdirs.append(viewdir)
            # rays_o.append(ray_o_temp)
            # rays_d.append(ray_d_temp)
            # rays_mask.append(ray_mask_temp)
            # viewdirs.append(viewdir)

            # rays_o = rays_o.reshape(H*W, -1, 3)
            # rays_d = rays_d.reshape(H*W, -1, 3)
            # viewdirs = viewdirs.reshape(H*W, -1, 3)
    # rays_o = torch.cat(rays_o, dim=0)
    # rays_d = torch.cat(rays_d, dim=0)
    # rays_mask = torch.cat(rays_mask, dim=0)
    # viewdirs = torch.cat(viewdirs, dim=0)

    return rays_o, rays_d, rays_mask, viewdirs, apt_sample_num
# -----------------------------------------------------------------------

@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz

# added --------------------------------------------------------------------


def get_training_rays_batch_flatten_aperture(rgb_tr_ori,         
        # train_poses, 
        # HW, Ks, 
        # # added --------------------        
        # lens_info_list, intrinsics, col2X,
        model_cam, i_train, sel_i, tf_single_ray, #apt_sample_rates, 
        # --------------------------
        ndc, inverse_y, flip_x, flip_y):
    #print('get_training_rays_flatten: start')
    
    Hs = model_cam.get_Hs(i_train)
    Ws = model_cam.get_Ws(i_train)
    Ks = model_cam.get_Ks(i_train)
    train_poses = model_cam.get_RTs(i_train)
    lensFps = model_cam.get_lensFps(i_train)
    lensDps = model_cam.get_lensDps(i_train)
    col2pix = model_cam.get_col2pix()
    positions = model_cam.index2position(sel_i, i_train)
    apt_sample_rates = model_cam.get_apt_sample_rates(i_train)
    if tf_single_ray:
        apt_sample_rates = torch.ones_like(apt_sample_rates)
    apt_sample_rate_max = apt_sample_rates.max()
    distortions = model_cam.get_distortions(i_train)
    # modified -----------------------------------------------------------------------------------------------------
    assert len(rgb_tr_ori) == len(train_poses) 
    assert len(rgb_tr_ori) == len(Ks) 
    assert len(rgb_tr_ori) == len(Hs) and len(rgb_tr_ori) == len(Ws)
    assert len(rgb_tr_ori) == len(lensFps) 
    assert len(rgb_tr_ori) == len(lensDps) 
    #assert len(lens_info_list) == len(col2pix_list)
    # ----------------------------------------------------------------------------------------------------------------
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    #N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    N = len(sel_i)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    # rays_o_tr = torch.zeros_like(rgb_tr)
    # rays_d_tr = torch.zeros_like(rgb_tr)
    # viewdirs_tr = torch.zeros_like(rgb_tr)

    # rays_o_tr   = torch.zeros([N, apt_sample_rate_max*apt_sample_rate_max, 3], device=DEVICE)
    # rays_d_tr   = torch.zeros_like(rays_o_tr)
    # viewdirs_tr = torch.zeros_like(rays_o_tr)
    imsz = []
    top = 0
    #for c2w, img, (H, W), K, lens_info in zip(train_poses, rgb_tr_ori, HW, Ks, lens_info_list):
    #for c2w, img, H, W, K, lensFp, lensDp in zip(train_poses, rgb_tr_ori, Hs, Ws, Ks, lensFps, lensDps):
    idx_imgs = positions[:,0]
    #assert img.shape[:2] == (H, W)            
    # img = rgb_tr_ori[idx_img]
    
    xs, ys =positions[:,1], positions[:,2]
    
    #rays_o, rays_d, viewdirs, apt_sample_num = get_rays_of_a_batch_aperture(
    rays_o, rays_d, rays_mask, viewdirs, apt_sample_num = get_rays_of_a_batch_aperture(
            Hs=Hs[idx_imgs], Ws=Ws[idx_imgs], Ks=Ks[idx_imgs], distortions = distortions[idx_imgs],
            xys=positions[:,1:],
            # added --------------------------
            apt_sample_rates = apt_sample_rates[positions[:, 0]],
            # lens_info=lens_info, 
            # intrinsics=intrinsics,
            # col2X = col2X,
            lensFps=lensFps[idx_imgs], 
            lensDps=lensDps[idx_imgs],
            col2pix = col2pix,
            shared_intrinsic=model_cam.shared_intrinsic,                
            # ---------------------------------
            c2ws=train_poses[idx_imgs], ndc=ndc,
            inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
    #n_pixel = H.item() * W.item()
    n_pixel = N
    img = rgb_tr_ori[idx_imgs,ys,xs]
    
    #rgb_tr[top:top+n_pixel].copy_(img.flatten(0,1))
    #rgb_tr.copy_(img)
    rgb_tr=img
    #rays_o_tr[top:top+n_pixel].copy_(rays_o.flatten(0,1).to(DEVICE))
    # rays_o_tr.copy_(rays_o.to(DEVICE))
    rays_o_tr = rays_o.to(DEVICE)
    #rays_d_tr[top:top+n_pixel].copy_(rays_d.flatten(0,1).to(DEVICE))
    #rays_d_tr.copy_(rays_d.to(DEVICE))
    rays_d_tr=rays_d.to(DEVICE)

    if rays_mask is not None:
        rays_mask_tr = rays_mask.to(DEVICE)
    else:
        rays_mask_tr = None
    #viewdirs_tr[top:top+n_pixel].copy_(viewdirs.flatten(0,1).to(DEVICE))
    #viewdirs_tr.copy_(viewdirs.to(DEVICE))
    viewdirs_tr = viewdirs.to(DEVICE)
    
    # rgb_tr.copy_(img.flatten(0,1))
    # rays_o_tr.copy_(rays_o.flatten(0,1).to(DEVICE))
    # rays_d_tr.copy_(rays_d.flatten(0,1).to(DEVICE))
    # viewdirs_tr.copy_(viewdirs.flatten(0,1).to(DEVICE))

    #imsz.append(n)
    imsz.append(n_pixel)

    #top += n
    # top += n_pixel

    # assert top == N
    eps_time = time.time() - eps_time
    #print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, rays_mask_tr, viewdirs_tr, imsz
# ---------------------------------------------------------------------------------

@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i+CHUNK] = model.hit_coarse_geo(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS

