import os
import sys
import pickle as pk
import numpy as np
import pandas as pd
import random as rn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

######################################################
#Script-specific functions
def feature_scale(data):
    data_mean = np.nanmean(data, axis=0)
    data_std = np.nanstd(data, axis=0)
    data_x = (data - data_mean)/data_std
    data_x[data_x!=data_x] = 0.0
        
    return data_x

######################################################




plt.ion()
seed = 2018
np.random.seed(seed)
rn.seed(seed)

train_val_split = True
do_save = True

data_dir = '../data/EHT_pointing/'
data = np.load(os.path.join(data_dir,'training_data_2013-2016.npy'))
targets = np.load(os.path.join(data_dir,'training_targets_2013-2016.npy'))

#data2 = np.load(os.path.join(data_dir,'mat5a_training_data.npy'))
#targets2 = np.load(os.path.join(data_dir,'mat5a_training_targets.npy'))


data_x = feature_scale(data)
#data_x2 = feature_scale(data2)

val_size = 0.2
batch_size = 1
epochs = 3

if train_val_split:
    train_x, val_x, train_y, val_y = train_test_split(data_x,
                                                      targets,
                                                      test_size=val_size,
                                                      random_state=seed,
                                                      shuffle=True)

else:
    train_size = np.ceil(len(data)*(1-val_size))
    train_x = data[0:train_size,:]
    train_y = targets[0:train_size,:]
    val_x = data[train_size:,:]
    val_y = targets[train_size:,:]
    
    

regr = RandomForestRegressor(max_depth=None, min_samples_split=2,
                             n_estimators=100, max_features=data_x.shape[1])

regr.fit(train_x, train_y[:,0])
residuals_X = (regr.predict(val_x) - val_y[:,0])*3600.
mean_residuals_X = np.mean(residuals_X, axis=0)
std_residuals_X = np.std(residuals_X, axis=0)

regr.fit(train_x, train_y[:,1])
residuals_Y = (regr.predict(val_x) - val_y[:,1])*3600.
mean_residuals_Y = np.mean(residuals_Y, axis=0)
std_residuals_Y = np.std(residuals_Y, axis=0)

out = {'residuals_X':residuals_X,
       'mean_residuals_X':mean_residuals_X,
       'std_residuals_X':std_residuals_X,
       'expected_val_X':val_y[:,0],
       'expected_train_X':train_y[:,0],
       'residuals_Y':residuals_Y,
       'mean_residuals_Y':mean_residuals_Y,
       'std_residuals_Y':std_residuals_Y,
       'expected_val_Y':val_y[:,1],
       'expected_train_Y':train_y[:,1]}

print 'STD X before: ', np.std(val_y[:,0], axis=0)*3600.
print 'STD X Residual Offsets (arsec): ', std_residuals_X
print 'STD Y before: ', np.std(val_y[:,1], axis=0)*3600.
print 'STD Y Residual Offsets (arsec): ', std_residuals_Y

if do_save:
    
    rf_all = RandomForestRegressor(n_estimators=10, n_jobs=5)
    rf_all.fit(data_x, targets[:,0])
    pk.dump(rf_all, open('thermolin_model_X.pkl','w'))
    test_residuals_X = (rf_all.predict(data_x) - targets[:,0])*3600.

    rf_all.fit(data_x, targets[:,1])
    pk.dump(rf_all, open('thermolin_model_Y.pkl','w'))
    test_residuals_Y = (rf_all.predict(data_x) - targets[:,1])*3600.
    

    print 'Training residuals X: ', np.std(test_residuals_X, axis=0)
    print 'Training residuals Y: ', np.std(test_residuals_Y, axis=0)
