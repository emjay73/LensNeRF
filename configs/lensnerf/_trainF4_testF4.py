_base_ = './lensnerf_default.py'

expname = '_train_F4_test_F4'
expname_train = expname.split('_test')[0]

data = dict(
    datadir= 
    './data/lensnerf_dataset/real_defocus_dataset_with_zmean/_F4'    
    
)
fine_train = dict( 
    # debugging   
    # N_iters=1000,
    # N_rand=32, 

    # N_iters=100000, #80000, 
    #N_iters=120000,
    N_rand=2048, 
)
