import numpy as np
import pickle as pk
import pylab as py
import glob
import os
import sys

'''looks at pkls of offsets and pulls out features we care about, saves them in a convenient format for training'''

    
def extract_features(data):
    '''Take in data and extract the features we think are interesting.
    '''
    
    #Now let's grab interesting info about the observation.
    focus = np.array(data['scu.benchoff'][0:3]) #0-2
    ext_temp = np.array([data['observation.temp_avg']]) #3
    ext_pressure = np.array([data['observation.pressure_avg']]) #4
    wind_dir = np.array([data['observation.wind_dir_avg']]) #5
    wind_speed = np.array([data['observation.wind_speed_avg']]) #6
    mean_az = np.array([data['observation.mean_az']]) #7
    mean_el = np.array([data['observation.mean_el']]) #8
    med_r1 = np.array([data['tracker.linear_sensor_avg'][0]]) #9
    med_r2 = np.array([data['tracker.linear_sensor_avg'][1]]) #10
    med_l1 = np.array([data['tracker.linear_sensor_avg'][2]]) #11
    med_l2 = np.array([data['tracker.linear_sensor_avg'][3]]) #12
    scu_temps = np.array(data['scu.temp']) #13-

    #These are the offsets we're trying to zero.
    this_x = data['az_offset_proj0'] #az*cos(el)
    #this_x = data['az_offset_divCosEl'] #az
    this_y = data['el_offset_proj0'] #el
    
    targets = np.array([[this_x, this_y]], dtype=np.float)
    
    these_features = np.concatenate((focus,ext_temp,ext_pressure,wind_dir,wind_speed,
                                     #mean_az,mean_el,
                                     med_r1, med_r2, med_l1, med_l2,
                                     scu_temps))
    
    return these_features.reshape((1,len(these_features))), targets


#################################
#Load in data


out_dir = '/Users/jhenning/codes/data/EHT_pointing'
data_dir = '/Users/jhenning/codes/raw_data/EHT_pointing'
#years = ['2013','2014','2015','2016']
#years = ['2013']
#years = ['2014']
#years = ['2015']
#years = ['2016']
years = ['2015','2016']

sources = ['rcw38','mat5a']

data = []
targets = []
#Grab features we think are interesting.
for k in range(len(sources)):
    for l in range(len(years)):
        this_data = []
        this_targets = []
        filenames = np.sort(glob.glob(os.path.join(data_dir,sources[k],years[l],'*.pkl')))
        for m in range(len(filenames)):
            d = pk.load(open(os.path.join(data_dir,filenames[m]),'r'))
    
            this, this_target = extract_features(d)

            if this[0][0]==this[0][0]:
                if this_data == []:
                    this_data = [this]
                    this_targets = [this_target]
                else:
                    this_data.append(this)
                    this_targets.append(this_target)

        this_data = np.array(this_data)
        this_targets = np.array(this_targets)
        this_data = this_data.reshape(this_data.shape[0],this_data.shape[-1])
        this_targets = this_targets.reshape(this_targets.shape[0],this_targets.shape[-1])
        
        
        this_targets = this_targets - np.nanmedian(this_targets, axis=0)

        if data == []:
            data = np.array(this_data)
            targets = np.array(this_targets)
        else:
            data = np.concatenate((data, this_data))
            targets = np.concatenate((targets, this_targets))

data = np.array(data)
targets = np.array(targets)
data = data.reshape(data.shape[0],data.shape[-1])
targets = targets.reshape(targets.shape[0],targets.shape[-1])


#targets_median = np.nanmedian(targets, axis=0)
#targets = targets - targets_median


#Make an outlier cut
good_idx = np.ones(len(targets), dtype=int)

good_idx[np.sqrt(targets[:,0]**2 + targets[:,1]**2) > 0.025] = 0
#bad1 = np.where((np.abs(targets[:,0]) > 0.05))[0]
#bad2 = np.where((np.abs(targets[:,1]) > 0.025))[0]
#bad = np.concatenate((bad1, bad2))
#good_idx[bad] = 0
good_idx = np.array(good_idx, dtype=bool)


if len(sources) != 1:
    np.save(os.path.join(out_dir,'training_data_'+str(years[0])+'-'+str(years[-1])+'.npy'), data[good_idx], allow_pickle=False)
    np.save(os.path.join(out_dir,'training_targets_'+str(years[0])+'-'+str(years[-1])+'.npy'), targets[good_idx], allow_pickle=False)

else:
    np.save(os.path.join(out_dir,sources[0]+'_training_data_'+str(years[0])+'.npy'), data[good_idx], allow_pickle=False)
    np.save(os.path.join(out_dir,sources[0]+'_training_targets_'+str(years[0])+'.npy'), targets[good_idx], allow_pickle=False)
