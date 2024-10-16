_base_ = './lensnerf_default.py'

# -------------------------------------------------------
expname = '_train_debug_test_debug'
expname_train = expname.split('_test')[0]

data = dict(
    datadir= 
        './data/lensnerf_dataset/real_defocus_dataset_with_zmean/_F4' ,
    
)

fine_train = dict( 
    # debugging   
    # N_iters=1000,
    N_iters=500,
    N_rand=4, 

    # N_iters=60000,         
    # N_rand=2048, 

    learning_cam_start = 200,
    # learning_cam_start = 0,
    learning_cam_end = 300,

    #on
    learning_dir_start = 400,

    pg_scale=[20,40,60,80],
)
