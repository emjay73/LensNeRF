_base_ = './lensnerf_default.py'

# -------------------------------------------------------
expname = '_fake_train_F4_test_F4'
expname_train = expname.split('_test')[0]

data = dict(
    datadir= 
    './data/lensnerf_dataset/fake_defocus_dataset/_F4'    
)
fine_train = dict(    
    #N_iters=60000, 
    #N_rand=1024, # trial14
    N_rand=2048, # trial16
    N_outers = 1, 
)
