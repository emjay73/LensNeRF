import numpy as np

from .load_llff import load_llff_data
from .load_blender import load_blender_data
from .load_nsvf import load_nsvf_data
from .load_blendedmvs import load_blendedmvs_data
from .load_tankstemple import load_tankstemple_data
from .load_deepvoxels import load_dv_data
from .load_co3d import load_co3d_data
from .load_nerfpp import load_nerfpp_data

# added -----------------------------------
from .load_lensnerf import compute_infocusZ, load_lensnerf_data, load_lensnerf_images, load_lens_info_from_exif, compute_colmap_to_X, load_colmap_to_X #load_center_z #load_labcal_after_colmap, load_labcal
import os
# ------------------------------------------

def load_data(cfg, real_args):
    args = cfg.data    
    testsavedir = os.path.join(cfg.basedir, cfg.expname_train+real_args.exp_postfix) 
    K, depths = None, None
    near_clip = None

    # load lensnerf -------------------------------------------------------
    if isinstance(args.datadir, list):
        basedir = args.datadir[0]
        basedir_test = args.datadir[1]
    else:
        basedir = args.datadir
        basedir_test = None

    exif_list = None
    labcal = None
    if args.dataset_type == 'lensnerf':
        images, depths, intrinsics, poses, bds, render_poses, i_test, sc = load_lensnerf_data(
            #args.datadir, args.factor, args.width, args.height,
            basedir, args.factor, args.width, args.height,
            recenter=True, bd_factor=args.bd_factor,
            spherify=args.spherify,
            load_depths=args.load_depths,
            movie_render_kwargs=args.movie_render_kwargs)
        #exif_list = load_exifs(args.datadir, args.factor, args.width, args.height)
        #lens_info_list = load_lens_info_from_exif(args.datadir, args.factor, args.width, args.height)
        lens_info_list = load_lens_info_from_exif(basedir, args.factor, args.width, args.height)
        #labcal = load_labcal(args.datadir)
        #labcal = load_labcal_after_colmap(args.datadir, args.factor)


        #col2X = compute_colmap_to_X(args.datadir, intrinsics, lens_info_list, sc)        
        if not real_args.pinhole_camera:            
            col2X = compute_colmap_to_X(basedir, intrinsics, lens_info_list, sc, real_args.perturb_infocusZ)        
            print("compute col2X: ", col2X)
        else:
            col2X = ([1.0,1.0],[1.0,1.0])
        # if args.load_col2pix:
        if os.path.exists(os.path.join(testsavedir, 'col2pix_last.npy')):
            col2X = load_colmap_to_X(testsavedir, lens_info_list)        
            print("load col2X from: "+os.path.join(testsavedir, 'col2pix_last.npy'))
            print("load_col2X: ", col2X)

        
        if not real_args.pinhole_camera:            
            infocusZ = compute_infocusZ(basedir, sc, real_args.perturb_infocusZ)
        else:
            infocusZ = 0.0

        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        #print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        #print('Loaded llff', images.shape, render_poses.shape, intrinsics, args.datadir)
        print('Loaded llff', images.shape, render_poses.shape, intrinsics, basedir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])

        # if datadir_test exist, replace images.
        if basedir_test is not None:
            imgs_test = load_lensnerf_images(basedir_test, factor=args.factor, width=args.width, height=args.height, load_depths=args.load_depths)
            lens_info_list_test = load_lens_info_from_exif(basedir_test, args.factor, args.width, args.height)
            for it in i_test:
                images[it] = imgs_test[it].astype(np.float32)
                lens_info_list[it] = lens_info_list_test[it]
            
        print('DEFINING BOUNDS')
        if args.ndc:
            near = 0.
            far = 1.
        else:
            near_clip = max(np.ndarray.min(bds) * .9, 0)
            _far = max(np.ndarray.max(bds) * 1., 0)
            near = 0
            far = inward_nearfar_heuristic(poses[i_train, :3, 3])[1]
            print('near_clip', near_clip)
            print('original far', _far)
        print('NEAR FAR', near, far)
    # ------------------------------------------------------------------
    elif args.dataset_type == 'llff':
        images, depths, poses, bds, render_poses, i_test = load_llff_data(
                args.datadir, args.factor, args.width, args.height,
                recenter=True, bd_factor=args.bd_factor,
                spherify=args.spherify,
                load_depths=args.load_depths,
                movie_render_kwargs=args.movie_render_kwargs)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.ndc:
            near = 0.
            far = 1.
        else:
            near_clip = max(np.ndarray.min(bds) * .9, 0)
            _far = max(np.ndarray.max(bds) * 1., 0)
            near = 0
            far = inward_nearfar_heuristic(poses[i_train, :3, 3])[1]
            print('near_clip', near_clip)
            print('original far', _far)
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = 2., 6.

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'blendedmvs':
        images, poses, render_poses, hwf, K, i_split = load_blendedmvs_data(args.datadir)
        print('Loaded blendedmvs', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

        assert images.shape[-1] == 3

    elif args.dataset_type == 'tankstemple':
        images, poses, render_poses, hwf, K, i_split = load_tankstemple_data(
                args.datadir, movie_render_kwargs=args.movie_render_kwargs)
        print('Loaded tankstemple', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0)

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'nsvf':
        images, poses, render_poses, hwf, i_split = load_nsvf_data(args.datadir)
        print('Loaded nsvf', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3])

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]

    elif args.dataset_type == 'deepvoxels':
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.scene, basedir=args.datadir, testskip=args.testskip)
        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R - 1
        far = hemi_R + 1
        assert args.white_bkgd
        assert images.shape[-1] == 3

    elif args.dataset_type == 'co3d':
        # each image can be in different shapes and intrinsics
        images, masks, poses, render_poses, hwf, K, i_split = load_co3d_data(args)
        print('Loaded co3d', args.datadir, args.annot_path, args.sequence_name)
        i_train, i_val, i_test = i_split

        near, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0)

        for i in range(len(images)):
            if args.white_bkgd:
                images[i] = images[i] * masks[i][...,None] + (1.-masks[i][...,None])
            else:
                images[i] = images[i] * masks[i][...,None]

    elif args.dataset_type == 'nerfpp':
        images, poses, render_poses, hwf, K, i_split = load_nerfpp_data(args.datadir)
        print('Loaded nerf_pp', images.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near_clip, far = inward_nearfar_heuristic(poses[i_train, :3, 3], ratio=0.02)
        near = 0

    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    h, w, fx, fy, cx, cy, k1, k2, p1, p2 = intrinsics

    if K is None:
        # K = np.array([
        #     [focal, 0, 0.5*W],
        #     [0, focal, 0.5*H],
        #     [0, 0, 1]
        # ])
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks,
        near=near, far=far, near_clip=near_clip,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths,
        irregular_shape=irregular_shape,
        #imgid2exif=imgid2exif
        lens_info_list=lens_info_list, 
        intrinsics = intrinsics, 
        col2X = col2X,
        infocusZ = infocusZ,
    )
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
                      # it is only used to determined scene bbox
                      # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far

