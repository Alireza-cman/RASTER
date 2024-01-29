import RASTER_multivariate as RASTERM
import RASTER_univariate as RASTERU
import numpy as np

def RASTER(x_train,y_train, x_test, y_test,n_features = 10_000,verbose = False):
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    assert len(x_train.shape) == len(x_test.shape), 'train and test shape are not the same'
  
    if len(x_train.shape)  == 2:
        parameter = RASTERU.fit(x_train,num_features =n_features)
        x_train_trans_org = RASTERU.transform(x_train, parameter,'ter')
        x_test_trans_org = RASTERU.transform(x_test, parameter,'ter')
        
        
    elif len(x_train.shape) == 3:
        if verbose: 
            print('Check if you pass correct form (samples, channels , length)')
        parameter = RASTERM.fit(x_train,num_features =n_features)
        x_train_trans_org = RASTERM.transform(x_train, parameter)
        x_test_trans_org = RASTERM.transform(x_test, parameter)
    
    return x_train_trans_org, x_test_trans_org