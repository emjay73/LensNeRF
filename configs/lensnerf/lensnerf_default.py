_base_ = '../default.py'


basedir = './logs/lensnerf'
wandb_init_dir = './wandb'

import os
#wandb_entity=os.getlogin() # username
wandb_entity='lensnerf'

expname_train=None
data = dict(
    dataset_type='lensnerf',
    ndc=True,
    width=1296,
    height=864,
    
    # added -----------------
    load2gpu_on_the_fly=False, #speed
    #load2gpu_on_the_fly=True, #memory
    # llffhold=2,                   # testsplit
    # load_col2pix=False,
    # ------------------------
)

coarse_train = dict(
    N_iters=0,
    
    # added --------------
    weight_rgbper=0,
    #lrate_camera=1e-3,    
    learning_cam_start = 8000,    
    #lrate_cam = [0, 0.001],
    lrate_col2pix_residual = 1e-3,
    lrate_intrinsics_residual = 1e-3,
    lrate_extrinsics_residual = 1e-1,
    lrate_expT = 1e-1,
    lrate_expS = 1e-1,
    # --------------------
)

fine_train = dict(
    N_iters=120000, #80000,
    N_rand=4096,
    weight_distortion=0.01,
    pg_scale=[2000,4000,6000,8000],
    ray_sampler='flatten',
    tv_before=1e9,
    tv_dense_before=10000,
    weight_tv_density=1e-5,
    weight_tv_k0=1e-6,
        
    # added --------------
    N_outers = 3, 

    #weight_rgbper=0,
    weight_rgbper=0.02,
    # weight_rgbper_aperture=0.04,#0,#0.02,#1,#0.02, #0.02, #1, #0.02,#10,
    
    

    # col on/off ----------------------------------
    #off
    #weight_colloss = 0.0,
    #train_from_the_beginning = [],  
    #cam_do_not_learn = ['col2pix_residual', 'expT', 'expS', 'intrinsics_residual', 'extrinsics_residual'],

    # on w/o lensfp
    weight_lensfp_loss = 0.0,
    weight_colloss = 0.0,
    # train_from_the_beginning = ['col2pix_residual'],  
    cam_do_not_learn = ['expT', 'expS', 'intrinsics_residual', 'extrinsics_residual', 'lensFp_residual'],
    learning_cam_start = 0, #20000,#20000, #0, #100000,
    learning_cam_end = 40000, #70000, #40000, #30000, #0, #100000,

    #on with lensfp   
    # weight_lensfp_loss = 0.0,
    # weight_colloss = 0.0,
    # train_from_the_beginning = ['col2pix_residual', 'lensFp_residual'],  
    # cam_do_not_learn = ['expT', 'expS', 'intrinsics_residual', 'extrinsics_residual'],

    # infocus on/off -------------------------------
    # off
    # learning_dir_start = 0, #0,
    # weight_rgbper_aperture=0.0,#0,#0.2,
    # tau_rgbper_aperture = 5, # #weight_infocus.min() tensor(0.2711)

    #on
    learning_dir_start = 40000, #70000, #50000,# 30000, #0,
    weight_rgbper_aperture=0.4, #0.2,#0,#0.2,
    tau_rgbper_aperture = 5, # #weight_infocus.min() tensor(0.2711)    
    # learning_cam_start = 0,#20000, #0, #100000,
    
    lrate_col2pix_residual = 1e-4, #1e-2, #1e-4, #1e-1, # 1e-3
    # lrate_lensFp_residual = 1e-2,
    lrate_expT = 1e-1,
    lrate_expS = 1e-1,

    lrate_intrinsics_residual = 1e-2, #1e-4,
    lrate_extrinsics_residual = 1e-2, #1e-4, #1e-1,    
    # --------------------
)

fine_model_and_render = dict(
    num_voxels=256**3,
    mpi_depth=128,
    rgbnet_dim=9,
    rgbnet_width=64,
    world_bound_scale=1,
    fast_color_thres=1e-3,

    
)

# aperture camera related arguments -------------
pinhole_camera = False
