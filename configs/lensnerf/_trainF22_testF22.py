_base_ = './lensnerf_default.py'

expname = '_train_F22_test_F22'
expname_train = expname.split('_test')[0]

data = dict(
    datadir= './data/lensnerf_dataset/real_defocus_dataset_with_zmean/_F22',
    # load2gpu_on_the_fly=True
)

fine_train = dict(    
    # N_iters=100000, #80000, 
    #N_iters=120000,
    N_rand=2048, 
)
