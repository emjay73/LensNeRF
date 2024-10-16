# emjay: newly created file 

from ast import USub
import os, json
from collections import defaultdict
import numpy as np
import os, imageio
import torch
import scipy
from PIL import Image
from .load_llff import recenter_poses, spherify_poses, rerotate_poses, normalize, poses_avg, render_path_spiral, _minify, imread, depthread

# added to remove errors when loading data from huggingface
# https://blog.csdn.net/gg864461719/article/details/126016571
imageio.plugins.freeimage.download()

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True, load_depths=False):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    if poses_arr.shape[1] == 17:
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    elif poses_arr.shape[1] == 14:
        poses = poses_arr[:, :-2].reshape([-1, 3, 4]).transpose([1,2,0])
    else:
        raise NotImplementedError
    bds = poses_arr[:, -2:].transpose([1,0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if height is not None and width is not None:
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif factor is not None and factor != 1:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    print(f'Loading images from {imgdir}')
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print()
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        names = set(name[:-4] for name in np.load(os.path.join(basedir, 'poses_names.npy')))
        assert len(names) == poses.shape[-1]
        print('Below failed files are skip due to SfM failure:')
        new_imgfiles = []
        for i in imgfiles:
            fname = os.path.split(i)[1][:-4]
            if fname in names:
                new_imgfiles.append(i)
            else:
                print('==>', i)
        imgfiles = new_imgfiles

    if len(imgfiles) < 3:
        print('Too few images...')
        import sys; sys.exit()

    sh = imageio.imread(imgfiles[0]).shape
    if poses.shape[1] == 4:
        poses = np.concatenate([poses, np.zeros_like(poses[:,[0]])], 1)
        #poses[2, 4, :] = np.load(os.path.join(basedir, 'hwf_cxcy.npy'))[2]
        intrinsics = np.load(os.path.join(basedir, 'intrinsics.npy')) # h, w, fx, fy, cx, cy, k1, k2, p1, p2
        poses[2, 4, :] = intrinsics[2]
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    intrinsics[0:2] = np.array(sh[:2]) # hw
    intrinsics[2:6] = intrinsics[2:6] * 1./factor # fxfycxcy

    if not load_imgs:
        return poses, bds

    # modified ----------------------------------------
    imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    # original ------------------------------------------
    #imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    # --------------------------------------------------
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:,-1,0])

    if not load_depths:
        return intrinsics, poses, bds, imgs

    depthdir = os.path.join(basedir, 'stereo', 'depth_maps')
    assert os.path.exists(depthdir), f'Dir not found: {depthdir}'

    depthfiles = [os.path.join(depthdir, f) for f in sorted(os.listdir(depthdir)) if f.endswith('.geometric.bin')]
    assert poses.shape[-1] == len(depthfiles), 'Mismatch between imgs {} and poses {} !!!!'.format(len(depthfiles), poses.shape[-1])

    depths = [depthread(f) for f in depthfiles]
    depths = np.stack(depths, -1)
    print('Loaded depth data', depths.shape)
    return intrinsics, poses, bds, imgs, depths

def load_lensnerf_images(basedir, factor=8, width=None, height=None, load_depths=False):
    _, _, _, imgs, *depths = _load_data(
        basedir, factor=factor, width=width, height=height, load_depths=load_depths)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)    
    imgs = imgs.astype(np.float32)
    return imgs

def load_lensnerf_data(basedir, factor=8, width=None, height=None,
                   recenter=True, rerotate=True,
                   bd_factor=.75, spherify=False, path_zflat=False, load_depths=False,
                   movie_render_kwargs={}):
    
    intrinsics, poses, bds, imgs, *depths = _load_data(
        basedir, factor=factor, width=width, height=height,
        load_depths=load_depths) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    if load_depths:
        depths = depths[0]
    else:
        depths = 0

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    if bds.min() < 0 and bd_factor is not None:
        print('Found negative z values from SfM sparse points!?')
        print('Please try bd_factor=None')
        import sys; sys.exit()
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    depths *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify: # False
        poses, radius, bds, depths = spherify_poses(poses, bds, depths)
        if rerotate:
            poses = rerotate_poses(poses)

        ### generate spiral poses for rendering fly-through movie
        centroid = poses[:,:3,3].mean(0)
        radcircle = movie_render_kwargs.get('scale_r', 1) * np.linalg.norm(poses[:,:3,3] - centroid, axis=-1).mean()
        centroid[0] += movie_render_kwargs.get('shift_x', 0)
        centroid[1] += movie_render_kwargs.get('shift_y', 0)
        centroid[2] += movie_render_kwargs.get('shift_z', 0)
        new_up_rad = movie_render_kwargs.get('pitch_deg', 0) * np.pi / 180
        target_y = radcircle * np.tan(new_up_rad)

        render_poses = []

        for th in np.linspace(0., 2.*np.pi, 200):
            camorigin = np.array([radcircle * np.cos(th), 0, radcircle * np.sin(th)])
            if movie_render_kwargs.get('flip_up', False):
                up = np.array([0,1.,0])
            else:
                up = np.array([0,-1.,0])
            vec2 = normalize(camorigin)
            vec0 = normalize(np.cross(vec2, up))
            vec1 = normalize(np.cross(vec2, vec0))
            pos = camorigin + centroid
            # rotate to align with new pitch rotation
            lookat = -vec2
            lookat[1] = target_y
            lookat = normalize(lookat)
            vec2 = -lookat
            vec1 = normalize(np.cross(vec2, vec0))

            p = np.stack([vec0, vec1, vec2, pos], 1)

            render_poses.append(p)

        render_poses = np.stack(render_poses, 0)
        render_poses = np.concatenate([render_poses, np.broadcast_to(poses[0,:3,-1:], render_poses[:,:3,-1:].shape)], -1)

    else:

        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz * movie_render_kwargs.get('scale_f', 1)

        # Get radii for spiral path
        zdelta = movie_render_kwargs.get('zdelta', 0.5)
        zrate = movie_render_kwargs.get('zrate', 1.0)
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0) * movie_render_kwargs.get('scale_r', 1)
        c2w_path = c2w
        # N_views = 120
        N_views = 240
        N_rots = movie_render_kwargs.get('N_rots', 1)
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=zrate, rots=N_rots, N=N_views)

    render_poses = torch.Tensor(render_poses)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
       
    return images, depths, intrinsics, poses, bds, render_poses, i_test, sc


#def load_exifs(basedir, factor, width, height):
def load_lens_info_from_exif(basedir, factor, width, height):

    exifs_path = os.path.join(basedir, 'exifs.json')
    with open(exifs_path, 'r') as fp:
        json_data = json.load(fp)
        img_data  = json_data['images']
        exif_data = json_data['exifs']

    useful_key_list = [
        "ResolutionUnit", 
        "Model", 
        "ExifVersion", 
        "ShutterSpeedValue", 
        "ApertureValue", 
        "FocalLength", 
        "ExifImageWidth",
        "ExifImageHeight",
        "FocalPlaneXResolution",
        "FocalPlaneYResolution",
        "ExposureTime",
        "FNumber",
        "ISOSpeedRatings",
        ]
    #imgid2exif = defaultdict(dict)
    exif_list = [{} for _ in exif_data]
    runit2string = {1:'None', 2:'inch', 3:'cm'}
    #for edata in exif_data:
    for idata, edata in zip(img_data, exif_data):
        #imgid2exif[edata['image_id']]['FileName'] = edata['exif']['File_name']
        #exif_list[edata['image_id']]['FileName']    = edata['exif']['File_name']
        exif_list[edata['image_id']]['FileName']    = idata['file_name']
        exif_list[edata['image_id']]['ImageWidth']  = idata['width']
        exif_list[edata['image_id']]['ImageHeight'] = idata['height']

        for k, v in edata['exif'].items():
            if k not in useful_key_list:
                continue
            if k == "ResolutionUnit":
                if v in runit2string.keys():
                    v = runit2string[v]
                else:
                    raise RuntimeError("unknown resolution unit")

            if (k=='Model') and (v=='Canon EOS 550D'):
                # imgid2exif[edata['image_id']]['SensorWmm'] = 22.3
                # imgid2exif[edata['image_id']]['SensorHmm'] = 14.9
                exif_list[edata['image_id']]['SensorWmm'] = 22.3
                exif_list[edata['image_id']]['SensorHmm'] = 14.9

            #imgid2exif[edata['image_id']][k] = v
            exif_list[edata['image_id']][k] = v

    # get factor ---------------------------------------------------
    for f in sorted(os.listdir(os.path.join(basedir, 'images'))):
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png'):
            img0 = os.path.join(basedir, 'images', f) 
            break    
    sh = imageio.imread(img0).shape
    
    factorX = None
    factorY = None
    if factor is not None and factor != 1:        
        factorX = factor
        factorY = factor
    elif (height is not None) and (width is not None):
        factorY = sh[0] / float(height)        
        factorX = sh[1] / float(width)        
    else:
        raise RuntimeError('unable to compute factorXY')
        factor = 1

    # update exif based on factor --------------------------------
    lens_info_list = [{} for _ in exif_data]
    for iexif, EXIF in enumerate(exif_list):
        EXIF['FocalLengthX']=EXIF['FocalLength']
        EXIF['FocalLengthY']=EXIF['FocalLength']
        EXIF.pop('FocalLength')
        
        if factorX !=1 or factorY != 1:
            EXIF['ImageWidth']  /= factorX
            EXIF['ImageHeight'] /= factorY
            EXIF['FocalPlaneXResolution']/=factorX
            EXIF['FocalPlaneYResolution']/=factorY
            

        # added ------------------------------------------------
        mm2pixX = EXIF['FocalPlaneXResolution']
        mm2pixY = EXIF['FocalPlaneYResolution']
        if EXIF['ResolutionUnit'] == 'inch':
            mm2pixX /= 25.4
            mm2pixY /= 25.4
        elif EXIF['ResolutionUnit'] == 'cm':
            mm2pixX /= 10
            mm2pixY /= 10
        else:
            raise RuntimeError('unknown focalplaneresolution unit')

        lens_info_list[iexif]['lensFXp'] = EXIF['FocalLengthX']*mm2pixX
        lens_info_list[iexif]['lensFYp'] = EXIF['FocalLengthY']*mm2pixY                
        
        lens_info_list[iexif]['lensDXp'] = EXIF['FocalLengthX']*mm2pixX/EXIF['FNumber'] # lens diameter pixel unit
        lens_info_list[iexif]['lensDYp'] = EXIF['FocalLengthY']*mm2pixY/EXIF['FNumber'] # lens diameter pixel unit

        lens_info_list[iexif]['FNumber'] = EXIF['FNumber']
        lens_info_list[iexif]['mm2pixX'] = mm2pixX
        lens_info_list[iexif]['mm2pixY'] = mm2pixY
        lens_info_list[iexif]['expT']    = EXIF['ExposureTime']
    
    #return imgid2exif
    return lens_info_list
            
# def load_labcal(basedir):
    
#     exifs_path = os.path.join(basedir, 'labcal.json')
#     with open(exifs_path, 'r') as fp:
#         labcal_data = json.load(fp)
#         return labcal_data

# def load_labcal_after_colmap(basedir):
#     exifs_path = os.path.join(basedir, 'labcal_after_colmap.json')
#     with open(exifs_path, 'r') as fp:
#         labcal_data = json.load(fp)
#         return labcal_data

def load_center_z(basedir):
    #return float(np.load(os.path.join(basedir, 'zcenter_mean.npy')))
    return float(np.load(os.path.join(basedir, 'z_median.npy')))

def compute_infocusZ(datadir, sc, perturb=1.0):
    center_z = load_center_z(datadir)
    center_z *= sc
    return center_z*perturb

def compute_colmap_to_X(datadir, intrinsics, lens_info_list, sc, perturb_infocusZ=1.0):
        
    center_z = compute_infocusZ(datadir, sc, perturb_infocusZ)
    # col2pix_list = []    
    # for iinfo, info in enumerate(lens_info_list):
    #     imgFXp, imgFYp = intrinsics[2], intrinsics[3]
    #     col2pix_list.append(
    #         [(info['lensFXp']*imgFXp)/((imgFXp-info['lensFXp'])*center_z),
    #          (info['lensFYp']*imgFYp)/((imgFYp-info['lensFYp'])*center_z)]
    #          )
    col2pix = [0,0]    
    info = lens_info_list[0]
    imgFXp, imgFYp = intrinsics[2], intrinsics[3]
    col2pix[0] = (info['lensFXp']*imgFXp)/((imgFXp-info['lensFXp'])*center_z)
    col2pix[1] = (info['lensFYp']*imgFYp)/((imgFYp-info['lensFYp'])*center_z)

    col2mm = [0, 0]
    col2mm[0] = col2pix[0]/info['mm2pixX']
    col2mm[1] = col2pix[1]/info['mm2pixY']

    #return col2pix_list
    return col2pix, col2mm

def load_colmap_to_X(datadir, lens_info_list):
    
    col2pix=  np.load(os.path.join(datadir, 'col2pix_last.npy'))
    col2pix = [col2pix[0].item(), col2pix[1].item()]
    info = lens_info_list[0]
    col2mm = [0, 0]
    col2mm[0] = col2pix[0]/info['mm2pixX']
    col2mm[1] = col2pix[1]/info['mm2pixY']

    #return col2pix_list
    return col2pix, col2mm
