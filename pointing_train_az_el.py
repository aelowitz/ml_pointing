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
epochs = 3

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


all_hist_az = []
all_hist_el = []
hyper_params = []
for i in range(len(units)):
    for j in range(len(activation)):
        this_units = units[i]
        this_activation = activation[j]
        
        model_az = Sequential()
        model_az.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model_az.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model_az.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model_az.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model_az.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model_az.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model_az.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model_az.add(Dense(units=1, activation=this_activation))
        model_az.compile(loss='mean_squared_error', optimizer='sgd')#, metrics=['loss'])

        hist_az = model_az.fit(train_x, train_y[:,0], epochs=epochs, batch_size=batch_size,
                               validation_data = (val_x, val_y[:,0]),
                               callbacks = callbacks_list,
                               shuffle = True)     
        all_hist_az.append(hist_az)
        hyper_params.append([units[i], activation[j]])
        
        model_az_name = 'az_trained_pointing_units'+str(units[i])+'_'+activation[j]+'.h5'
        model_az.save(model_az_name)

        plt.figure(1)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(hist_az.history['loss'], label='Training')
        plt.plot(hist_az.history['val_loss'], label='Validation')
        plt.title('Az Loss')
        #plt.legend(['Training', 'Validation'])


        model_el = Sequential()
        model_el.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model_el.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model_el.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model_el.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model_el.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model_el.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model_el.add(Dense(units=this_units, activation=this_activation, input_dim=73))
        model_el.add(Dense(units=1, activation=this_activation))
        model_el.compile(loss='mean_squared_error', optimizer='sgd')#, metrics=['loss'])

        hist_el = model_el.fit(train_x, train_y[:,1], epochs=epochs, batch_size=batch_size,
                            validation_data = (val_x, val_y[:,1]),
                            callbacks = callbacks_list,
                            shuffle = True)     
        all_hist_el.append(hist_el)
        hyper_params.append([units[i], activation[j]])
        
        model_el_name = 'el_trained_pointing_units'+str(units[i])+'_'+activation[j]+'.h5'
        model_el.save(model_el_name)

        plt.figure(2)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(hist_el.history['loss'], label='Training')
        plt.plot(hist_el.history['val_loss'], label='Validation')
        plt.title('El Loss')
        #plt.legend(['Training', 'Validation'])

plt.figure(1)
plt.legend(loc='best')
plt.figure(2)
plt.legend(loc='best')


all_hist_az = np.array(all_hist_az)
all_hist_el = np.array(all_hist_el)
#What hyperparameters gave the smallest difference between test and validation loss.

#Compare prediction to validation set.
prediction_az = model_az.predict(val_x)*3600.
prediction_az_train = model_az.predict(train_x)*3600.
val_y *= 3600.
train_y *= 3600.

prediction_el = model_el.predict(val_x)*3600.
prediction_el_train = model_el.predict(train_x)*3600.
val_y *= 3600.
train_y *= 3600.


residuals_az = (prediction_az - val_y[:,0])
residuals_az_train = (prediction_az_train - train_y[:,0])
mean_az_residuals = np.mean(residuals_az, axis=0)
std_az_residuals = np.std(residuals_az, axis=0)
mean_az_residuals_train = np.mean(residuals_az_train, axis=0)
std_az_residuals_train = np.std(residuals_az_train, axis=0)

residuals_el = (prediction_el - val_y[:,1])
residuals_el_train = (prediction_el_train - train_y[:,1])
mean_el_residuals = np.mean(residuals_el, axis=0)
std_el_residuals = np.std(residuals_el, axis=0)
mean_el_residuals_train = np.mean(residuals_el_train, axis=0)
std_el_residuals_train = np.std(residuals_el_train, axis=0)



out = {'residuals_az':residuals_az,
       'residuals_az_train':residuals_az_train,
       'mean_az_residuals':mean_az_residuals,
       'std_az_residuals':std_az_residuals,
       'mean_az_residuals_train':mean_az_residuals_train,
       'std_az_residuals_train':std_az_residuals_train,
       'prediction_az':prediction_az,
       'prediction_az_train':prediction_az_train,
       'residuals_el':residuals_el,
       'residuals_el_train':residuals_el_train,
       'mean_el_residuals':mean_el_residuals,
       'std_el_residuals':std_el_residuals,
       'mean_el_residuals_train':mean_el_residuals_train,
       'std_el_residuals_train':std_el_residuals_train,
       'prediction_el':prediction_el,
       'prediction_el_train':prediction_el_train,
       'expected_val':val_y,
       'expected_train':train_y}

pk.dump(out,open('residuals_azEl_epochs'+str(epochs)+'_batchSize'+str(batch_size)+'_units'+str(units[0])+'.pkl','wb'))

print('Mean Az Residual Offsets (arsec): ', mean_az_residuals)
print('STD Az Residual Offsets (arsec): ', std_az_residuals)
print('')
print('Mean El Residual Offsets (arsec): ', mean_el_residuals)
print('STD El Residual Offsets (arsec): ', std_el_residuals)

plt.figure(3)
plt.plot(residuals_az, residuals_el, 'o', alpha=0.1, label='Validation')
plt.plot(residuals_az_train, residuals_el_train, 'o', alpha=0.1, label='Trained')
plt.legend(loc='best')



