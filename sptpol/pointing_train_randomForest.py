import os
import sys
import pickle as pk
import numpy as np
import pandas as pd
import random as rn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot

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
seed = 2019
np.random.seed(seed)
rn.seed(seed)

train_val_split = True
do_save = False
plot_tree = False

data_dir = '../data/EHT_pointing/'
data = np.load(os.path.join(data_dir,'training_data_2013-2016.npy'))
targets = np.load(os.path.join(data_dir,'training_targets_2013-2016.npy'))

#Try without Mean Az and El.
#data = np.load(os.path.join(data_dir,'training_data_2013-2016_noMeanAzMeanEl.npy'))
#targets = np.load(os.path.join(data_dir,'training_targets_2013-2016_noMeanAzMeanEl.npy'))
#Just 2013.
#data = np.load(os.path.join(data_dir,'training_data_2013-2013.npy'))
#targets = np.load(os.path.join(data_dir,'training_targets_2013-2013.npy'))

#Just 2014.
#data = np.load(os.path.join(data_dir,'training_data_2014-2014.npy'))
#targets = np.load(os.path.join(data_dir,'training_targets_2014-2014.npy'))

#Just 2015
#data = np.load(os.path.join(data_dir,'training_data_2015-2015.npy'))
#targets = np.load(os.path.join(data_dir,'training_targets_2015-2015.npy'))

#Just 2016
#data = np.load(os.path.join(data_dir,'training_data_2016-2016.npy'))
#targets = np.load(os.path.join(data_dir,'training_targets_2016-2016.npy'))

#Just 2015-2016
#data = np.load(os.path.join(data_dir,'training_data_2015-2016.npy'))
#targets = np.load(os.path.join(data_dir,'training_targets_2015-2016.npy'))


#Convert targets to arcsec
targets = targets * 3600.

#What are the feature names?
feature_names = ['focus1', 'focus2','focus3',
                 'ext_temp', 'ext_pressure', 'wind_dir', 'wind_speed',
                 'mean_az', 'mean_el',
                 'r1', 'r2', 'l1', 'l2']
for i in range(1,61):
    feature_names.append('scu_temp'+str(i))


#data_x = feature_scale(data)
data_x = data

val_size = 0.3
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
    
    

regr = RandomForestRegressor(max_depth=None, min_samples_split=4,
                             n_estimators=100, max_features=data_x.shape[1])

#Fit the model
regr.fit(train_x, train_y)

#Extract example tree if desired
if plot_tree:
    # Extract the small tree
    tree_small0 = regr.estimators_[0]
    tree_small1 = regr.estimators_[1]
    tree_small2 = regr.estimators_[2]
    tree_small3 = regr.estimators_[3]

    # Save the tree as a png image
    export_graphviz(tree_small0, out_file = 'small_tree0.dot',
                    feature_names = feature_names, rounded = True,
                    precision = 1)
    (graph, ) = pydot.graph_from_dot_file('small_tree0.dot')
    graph.write_png('small_tree0.png');

    export_graphviz(tree_small1, out_file = 'small_tree1.dot',
                    feature_names = feature_names, rounded = True,
                    precision = 1)
    (graph, ) = pydot.graph_from_dot_file('small_tree1.dot')
    graph.write_png('small_tree1.png');

    export_graphviz(tree_small2, out_file = 'small_tree2.dot',
                    feature_names = feature_names, rounded = True,
                    precision = 1)
    (graph, ) = pydot.graph_from_dot_file('small_tree2.dot')
    graph.write_png('small_tree2.png');

    export_graphviz(tree_small3, out_file = 'small_tree3.dot',
                    feature_names = feature_names, rounded = True,
                    precision = 1)
    (graph, ) = pydot.graph_from_dot_file('small_tree3.dot')
    graph.write_png('small_tree3.png');

#What is the importance of the variables?
#importances = regr.feature_importances_
##idx = np.argsort(importances)
##important_features = (feature_names[idx[::-1]], importances[idx[::-1]])

#important_features = sorted(zip(map(lambda x: round(x, 4),
#                                    importances), feature_names), reverse=True)
#print('Important Features: ', important_features)


# Get numerical feature importances
importances = list(regr.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_names, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)[0:30]
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];



residuals_test = (regr.predict(val_x) - val_y)
mean_residuals_test = np.mean(residuals_test, axis=0)
std_residuals_test = np.std(residuals_test, axis=0)

residuals_train = (regr.predict(train_x) - train_y)
mean_residuals_train = np.mean(residuals_train, axis=0)
std_residuals_train = np.std(residuals_train, axis=0)


out = {'residuals_test':residuals_test,
       'mean_residuals_test':mean_residuals_test,
       'std_residuals_test':std_residuals_test,
       'residuals_train':residuals_train,
       'mean_residuals_train':mean_residuals_train,
       'std_residuals_train':std_residuals_train,
       'expected_val':val_y,
       'expected_train':train_y}

print('STD (X,Y) before: ', np.round(np.std(val_y, axis=0),1))
print('Mean Starting Offsets (arsec): ', np.round(np.mean(val_y, axis=0),1))
print('STD Residual Offsets TEST (arsec): ', np.round(std_residuals_test,1))
print('Mean Residual Offsets TEST (arsec): ', np.round(mean_residuals_test,1))
print('STD Residual Offsets TRAIN (arsec): ', np.round(std_residuals_train,1))
print('Mean Residual Offsets TRAIN (arsec): ', np.round(mean_residuals_train,1))

plt.figure(1)
plt.clf()
plt.axes().set_aspect('equal')
plt.plot(val_y[:,0], val_y[:,1], 'ro', alpha=0.075, label='Offline Residuals')
plt.plot(residuals_test[:,0], residuals_test[:,1], 'bo', alpha=0.075, label='RFR Residuals')
plt.legend(loc='best', fontsize=15)
plt.xlabel('Offset Az*cos(El) [arcsec]', fontsize=20)
plt.ylabel('Offset El [arcsec]', fontsize=20)
#plt.title('Test Set Residuals')
plt.savefig('test_residuals.pdf')


plt.figure(2)
plt.clf()
plt.axes().set_aspect('equal')
plt.plot(train_y[:,0], train_y[:,1], 'ro', alpha=0.075, label='Offsets')
plt.plot(residuals_train[:,0], residuals_train[:,1], 'bo', alpha=0.075, label='Residuals')
plt.legend(loc='best')
plt.xlabel('Offset Az*cos(el) [arcsec]')
plt.ylabel('Offset El [arcsec]')
plt.title('Training Set Residuals')
plt.savefig('train_residuals.pdf')

if do_save:
    
    rf_all = RandomForestRegressor(n_estimators=100, n_jobs=5)
    rf_all.fit(data_x, targets)
    pk.dump(rf_all, open('thermolin_model.pkl','w'))

    test_residuals = (rf_all.predict(data_x) - targets)*3600.

    print('Training residuals: ', np.std(test_residuals, axis=0))
