_base_ = './lensnerf_default.py'

expname = '_fakedeblur_train_F4_test_F22'
expname_train = expname.split('_test')[0]

data = dict(
    datadir=[
        './data/lensnerf_dataset/fake_deblur_dataset/_F4', # train
        './data/lensnerf_dataset/fake_deblur_dataset/_F22' # test
        ]
)

fine_train = dict(    
    #N_iters=60000,     
    #N_rand=1024, # trial14
    N_rand=2048, # trial16
    N_outers = 1, 
)

