_base_ = './lensnerf_default.py'


# expname = '_train_F4_test_F22'
expname = '_train_F4_test_F5dot6'
expname_train = expname.split('_test')[0]

data = dict(
    #datadir='./data/lensnerf_dataset/real_defocus_dataset/Gink_F4',
    datadir=[
        './data/lensnerf_dataset/real_defocus_dataset_with_zmean/_F4', # train
        './data/lensnerf_dataset/real_defocus_dataset_with_zmean/_F5dot6' # test
        ]
)

fine_train = dict(    
    # N_iters=100000, #80000, 
    #N_iters=120000,
    N_rand=2048, 
)
