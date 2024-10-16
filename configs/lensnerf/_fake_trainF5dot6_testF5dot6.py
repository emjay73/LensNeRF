_base_ = './lensnerf_default.py'

# -------------------------------------------------------
expname = '_fake_train_F5dot6_test_F5dot6'
expname_train = expname.split('_test')[0]

data = dict(
    datadir= 
    './data/lensnerf_dataset/fake_defocus_dataset/_F5dot6',
    load2gpu_on_the_fly=True
)
fine_train = dict(    
    #N_iters=60000, 
    N_rand=2048, # trial14, trial16
    N_outers = 1, 
)
# --------------------------------------------------------
