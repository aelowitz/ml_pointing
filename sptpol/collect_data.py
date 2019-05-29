import numpy as np
import pickle as pk
import pylab as py
import glob
import os
import sys

def correct_az_tilt(data, pointing_params):
    a2 = pointing_params[2]
    a3 = pointing_params[3]
    az_rad = data['mean_az']*np.pi/180.
    el_rad = data['mean_el']*np.pi/180.
    
    d_az = -((a2*np.cos(az_rad) + a3*np.sin(az_rad))*np.tan(el_rad))
    d_el = (a2*np.sin(az_rad) - a3*np.cos(az_rad))

    return d_az, d_el

def correct_boom_flexure(data,pointing_params):
    az_rad = data['mean_az']*np.pi/180.
    el_rad = data['mean_el']*np.pi/180.
    a0 = pointing_params[0]
    a1 = pointing_params[1]
    
    d_az = 0.0
    d_el = a0*np.sin(el_rad) + a1*np.cos(el_rad)

    return d_az, d_el

def correct_el_tilt(data,pointing_params):
    az_rad = data['mean_az']*np.pi/180.
    el_rad = data['mean_el']*np.pi/180.
    a4 = pointing_params[4]
    
    d_az = a4*np.tan(el_rad)
    d_el = 0.0

    return d_az, d_el

def correct_az_collimation(data,pointing_params):
    az_rad = data['mean_az']*np.pi/180.
    el_rad = data['mean_el']*np.pi/180.
    a5 = pointing_params[5]
    
    d_az = - a5/np.cos(el_rad)
    d_el = 0.0

    return d_az, d_el

    
def correct_el_collimation(data, pointing_params):
    a6 = pointing_params[6]

    d_az = 0.0
    d_el = -a6

    return d_az, d_el


def correct_az_zero(data, pointing_params):
    az0 = pointing_params[7]
    
    d_az = - az0
    d_el = 0.0

    return d_az, d_el


def correct_refraction(data):
    el_rad = data['mean_el']*np.pi/180.
    P = data['pressure_avg']
    T = data['temp_avg']
    n0_minus_1 = (77.6e-6)*(P/T)

    d_az = 0.0
    d_el = n0_minus_1/np.tan(el_rad)*180./np.pi

    return d_az, d_el

    
def extract_features(data, bs_offsets,
                     pointing_params,
                     source='rcw38'):
    '''Take in data and extract the features we think are interesting.
    '''

    response_cut_low_long=1.25e-13
    response_cut_high_long=1.7e-13
    response_cut_low_short=1.25e-13
    response_cut_high_short=1.7e-13

    #Bump the response cuts up if you're looking at mat5a.
    if source =='mat5a':
        response_cut_low_long += 0.25e-13
        response_cut_high_long += 0.25e-13

    if source == 'cena':
        response_cut_low_long = 0.3e-13
        response_cut_high_long = 0.6e-13

    if data['nscans'] > 50:
        wh_good_response = np.arange(1599, dtype=np.float)
        n_goodResponse = 1599          
        response_cut_low = response_cut_low_long
        response_cut_high = response_cut_high_long
    else:
        response_cut_low = response_cut_low_short
        response_cut_high = response_cut_high_short

    #Work on amplitude cut.
    this_amp = data['amp']

    if np.std(this_amp[this_amp==this_amp]) != 0.:
        wh_good_response = np.array(np.nonzero((np.abs(this_amp) > response_cut_low) & \
                                               (np.abs(this_amp) < response_cut_high))[0])
    else:
        wh_good_response = np.zeros(1599)
    

    #Work on position cut.
    #Let's only include bolos with 0.5 deg of nominal boresight.
    #This is a static cut, and doesn't care about the response to sources on 
    # any given observation.
    common_xoff = np.array([bs_offsets[key]['xdeg'] for key in bs_offsets])
    common_yoff = np.array([bs_offsets[key]['ydeg'] for key in bs_offsets])
    
    dr = np.sqrt((common_xoff)**2. + (common_yoff)**2.)
    wh_dr = np.nonzero(dr < 0.5)[0]

    #If NSCANS < 50, consider this a "very_fast" "stripe scan", and
    #only pay attention to those detectors near the stripes.
    if data['nscans'] < 50.:
        stripe1 = 0.28 #Relative to boresight, NOT source
        stripe2 = -0.28 #Relative to boresight, NOT source
        d_stripe = 0.02#25
        wh_stripe = np.nonzero((np.abs(common_yoff-stripe1) < d_stripe) | \
                               (np.abs(common_yoff-stripe2) < d_stripe))[0]
    else: wh_stripe = np.arange(1599, dtype=np.float)

    wh_spatial = np.intersect1d(wh_dr, wh_stripe)

    good_bolos = np.array(np.intersect1d(wh_spatial, wh_good_response), dtype=int)

    good_keys = data['id'][good_bolos]

    
    if data['source'] != source:
        return np.zeros((1,73))*np.nan, np.zeros((1,2))*np.nan
    #Grab BS-corrected bolo offsets for each bolo.
    bolos = good_keys #bs_offsets.keys() #data['id']
    offsets = {}
    for i in range(len(bolos)):
        offsets[bolos[i]] = {'xdeg':data['xdeg'][good_bolos[i]], 'ydeg':data['ydeg'][good_bolos[i]]}

        
    cleanx = []
    cleany = []
    try:
        for key in offsets.keys():
            d_az = 0.0
            d_el = 0.0
            
            #Remove boresight offsets
            offsets[key]['xdeg'] -= bs_offsets[key]['xdeg']
            offsets[key]['ydeg'] -= bs_offsets[key]['ydeg']

            #Remove az tilt
            dd_az, dd_el = correct_az_tilt(data, pointing_params)
            d_az += dd_az
            d_el += dd_el

            #Remove el tilt
            #dd_az, dd_el = correct_el_tilt(data, pointing_params)
            #d_az += dd_az
            #d_el += dd_el

            #Remove boom flexure
            dd_az, dd_el = correct_boom_flexure(data, pointing_params)
            d_az += dd_az
            d_el += dd_el

            #Remove az collimation
            dd_az, dd_el = correct_az_collimation(data, pointing_params)
            d_az += dd_az
            d_el += dd_el

            #Remove el collimation
            #dd_az, dd_el = correct_el_collimation(data, pointing_params)
            #d_az += dd_az
            #d_el += dd_el

            #Remove az0 offset
            dd_az, dd_el = correct_az_zero(data, pointing_params)
            d_az += dd_az
            d_el += dd_el

            #Remove refraction
            #dd_az, dd_el = correct_refraction(data)
            #d_az += dd_az
            #d_el += dd_el
            
            
            offsets[key]['xdeg'] -= d_az
            offsets[key]['ydeg'] -= d_el
            
            cleanx.append(offsets[key]['xdeg'])
            cleany.append(offsets[key]['ydeg'])
        cleanx = np.array(cleanx)
        cleany = np.array(cleany)
    except KeyError:
        return np.zeros((1,73))*np.nan, np.zeros((1,2))*np.nan
    
    #These are the offsets we're trying to zero.
    this_x = np.nanmedian(cleanx)
    this_y = np.nanmedian(cleany)
    targets = np.array([[this_x, this_y]])
    
    #Now let's grab interesting info about the observation.
    focus = np.array(data['focus_position'][0:3]) #0-2
    ext_temp = np.array([data['temp_avg']]) #3
    ext_pressure = np.array([data['pressure_avg']]) #4
    wind_dir = np.array([data['wind_dir_avg']]) #5
    wind_speed = np.array([data['wind_speed_avg']]) #6
    mean_az = np.array([data['mean_az']]) #7
    mean_el = np.array([data['mean_el']]) #8
    med_r1 = np.array([data['med_r1']]) #9
    med_r2 = np.array([data['med_r2']]) #10
    med_l1 = np.array([data['med_l1']]) #11
    med_l2 = np.array([data['med_l2']]) #12
    scu_temps = np.array(data['mean_scu_temp']) #13-
    
    these_features = np.concatenate((focus,ext_temp,ext_pressure,wind_dir,wind_speed,
                                     mean_az,mean_el,med_r1, med_r2, med_l1, med_l2,
                                     scu_temps))
    
    return these_features.reshape((1,len(these_features))), targets


#################################
#Load in data
filenames = [#'source_scan_structure_20150126_201200.pkl',
             'source_scan_structure_20150126_201300.pkl',
             'source_scan_structure_20150126_201400.pkl',
             #'source_scan_structure_20161209_201500.pkl',
             #'source_scan_structure_20161209_201600.pkl']
]

out_dir = '/Users/jhenning/codes/data/pointing'
data_dir = '/Users/jhenning/codes/raw_data/'

sources = ['rcw38']#, 'mat5a']
focus = None #-6.4 

bs_offsets = pk.load(open(os.path.join(data_dir,'boresight_bolo_offsets.pkl'),'r'))

data = []
#Grab features we think are interesting.
for k in range(len(filenames)):
    d = pk.load(open(os.path.join(data_dir,filenames[k]),'r'))
    pointing_params = np.load(os.path.join(data_dir,'pointing_params_'+filenames[k].split('.pkl')[0]+'.npy'))
    for i in range(len(d)):
        for j in range(len(sources)):
            this, this_target = extract_features(d[i], bs_offsets,
                                                 pointing_params[i],
                                                 source=sources[j])

            if (focus == None) or (np.abs(focus - this[0][0]) < 0.01):
                if this[0][0]==this[0][0]:
                    if data == []:
                        data = [this]
                        targets = [this_target]
                    else:
                        data.append(this)
                        targets.append(this_target)


data = np.array(data)
targets = np.array(targets)

data = data.reshape(data.shape[0],data.shape[-1])
targets = targets.reshape(targets.shape[0],targets.shape[-1])

#Cut NaNs
good_idx1 = np.zeros(len(targets))
good_idx1[np.array(list(set(np.where(targets == targets)[0])))] += 1

#Cut large outliers.
good_idx2 = np.ones(len(targets))
targets_median = np.nanmedian(targets, axis=0)
y = targets - targets_median
y[y!=y] = 1000.
#good_idx2[np.where(y[:,0] > 0.03)[0]] -= 1
#good_idx2[np.where(y[:,0] < -0.03)[0]] -= 1
#good_idx2[np.where(y[:,1] > 0.0125)[0]] -= 1
#good_idx2[np.where(y[:,1] < -0.02)[0]] -= 1
#good_idx2[np.where((y[:,0] < 0.005) & (y[:,1] < -0.0075))[0]] -= 1

good_idx2[np.where(targets[:,0] > 0.02)[0]] -= 1
good_idx2[np.where(targets[:,0] < -0.105)[0]] -= 1
good_idx2[np.where(targets[:,1] > 0.23)[0]] -= 1
good_idx2[np.where(targets[:,1] < 0.19)[0]] -= 1
good_idx2[good_idx2 < 1] = 0

good_idx = np.array(good_idx1*good_idx2, dtype=bool)

if len(sources) != 1:
    np.save(os.path.join(out_dir,'training_data_'+str(focus)+'.npy'), data[good_idx], allow_pickle=False)
    np.save(os.path.join(out_dir,'training_targets_'+str(focus)+'.npy'), targets[good_idx], allow_pickle=False)

else:
    np.save(os.path.join(out_dir,sources[0]+'_training_data_'+str(focus)+'.npy'), data[good_idx], allow_pickle=False)
    np.save(os.path.join(out_dir,sources[0]+'_training_targets_'+str(focus)+'.npy'), targets[good_idx], allow_pickle=False)
