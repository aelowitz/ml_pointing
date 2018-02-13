import os
import sys
import pickle as pk
import numpy as np
import pandas as pd
import random as rn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import optimizers, losses, regularizers
from keras import callbacks
from keras.utils import to_categorical
from keras.datasets import mnist, fashion_mnist, boston_housing

import matplotlib.pyplot as plt

plt.ion()
seed = 2018
np.random.seed(seed)
rn.seed(seed)

units = [512] #[128, 256, 512]
activation = ['tanh']

def feature_scale(data):
    data_mean = np.nanmean(data, axis=0)
    data_std = np.nanstd(data, axis=0)
    data_x = (data - data_mean)/data_std
    data_x[data_x!=data_x] = 0.0
        
    return data_x
                    

#data = np.load('training_data.npy')
#targets = np.load('training_targets.npy')

data = np.load('rcw38_training_data_None.npy')
targets = np.load('rcw38_training_targets_None.npy')

data_x = feature_scale(data)

val_size = 0.2
batch_size = 100
epochs = 10
train_x, val_x, train_y, val_y = train_test_split(data_x,
                                                  targets,
                                                  test_size=val_size,
                                                  random_state=seed,
                                                  shuffle=True)


callbacks_list = [callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                              factor = 0.33,
                                              patience = 20,
                                              verbose = 1),
    
                  callbacks.CSVLogger(filename='training.log',
                                      separator=',',
                                      append=False)
                 ]


all_hist = []
hyper_params = []
for i in range(len(units)):
    for j in range(len(activation)):
        this_units = units[i]
        this_activation = activation[j]
        
        model = Sequential()
        model.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        #model.add(Dropout(0.25))
        model.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        #model.add(Dropout(0.125))
        model.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model.add(Dense(units=2, activation=this_activation))
        model.compile(loss='mean_squared_error', optimizer='sgd')#, metrics=['loss'])

        hist = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                               validation_data = (val_x, val_y),
                               callbacks = callbacks_list,
                               shuffle = True)     
        all_hist.append(hist)
        hyper_params.append([units[i], activation[j]])
        
        model_name = 'trained_pointing_units'+str(units[i])+'_'+activation[j]+'.h5'
        model.save(model_name)

        plt.figure(1)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(hist.history['loss'], label='Training')
        plt.plot(hist.history['val_loss'], label='Validation')
        plt.title('Loss')
        #plt.legend(['Training', 'Validation'])

plt.figure(1)
plt.legend(loc='best')

all_hist = np.array(all_hist)
#What hyperparameters gave the smallest difference between test and validation loss.

#Compare prediction to validation set.
prediction = model.predict(val_x)*3600.
prediction_train = model.predict(train_x)*3600.
val_y *= 3600.
train_y *= 3600.
plt.figure(3)
plt.plot(prediction[:,0], prediction[:,1], 'o', alpha=0.5, label='Predicted')
plt.plot(val_y[:,0], val_y[:,1], 'o', alpha=0.5, label='Expected')
plt.title('Az+El')
plt.legend(loc='best')

residuals = (prediction - val_y)
residuals_train = (prediction_train - train_y)
mean_residuals = np.mean(residuals, axis=0)
std_residuals = np.std(residuals, axis=0)
mean_residuals_train = np.mean(residuals_train, axis=0)
std_residuals_train = np.std(residuals_train, axis=0)

out = {'residuals':residuals,
       'residuals_train':residuals_train,
       'mean_residuals':mean_residuals,
       'std_residuals':std_residuals,
       'mean_residuals_train':mean_residuals_train,
       'std_residuals_train':std_residuals_train,
       'prediction':prediction,
       'prediction_train':prediction_train,
       'expected_val':val_y,
       'expected_train':train_y}

pk.dump(out,open('residuals_epochs'+str(epochs)+'_batchSize'+str(batch_size)+'_units'+str(units[0])+'.pkl','wb'))

print('Mean Residual Offsets (arsec): ', mean_residuals)
print('STD Residual Offsets (arsec): ', std_residuals)

print('Trained Mean Residual Offsets (arsec): ', mean_residuals_train)
print('Trained STD Residual Offsets (arsec): ', std_residuals_train)

plt.figure(4)
plt.plot(residuals[:,0], residuals[:,1], 'o', alpha=0.1, label='Validation')
plt.plot(residuals_train[:,0], residuals_train[:,1], 'o', alpha=0.1, label='Trained')
plt.legend(loc='best')



