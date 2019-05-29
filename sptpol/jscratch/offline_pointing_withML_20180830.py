import numpy as np
import pylab as py
import pickle as pk
import os
import copy
import sptpol_software.util.time as time
from sptpol_software.util.tools import struct
import sptpol_software.analysis.pointing.pointing_tools as pt
import sptpol_software.analysis.pointing.eht as eht

#####################################################################################################
#AUX FILE READER - GRABS PARAMETERS
#####################################################################################################
'''
   This module contains functions for obtaining a list of parameters 
   to pass to the offline pointing model module.  When correcting the az/el, 
   these parameters are pulled by the called model to calculate the pointing correction.

   For the moment, this module also contains the offline pointing models 
   and data handler, though we may want to place them in different modules in the future.
'''

def get_az_tilt_params(date, config_file, config_filename_out=None, return_measured_tilts=False,
                       extrap_forward=False):
    '''
    Look at an config file with az bearing tilt fits for the season.  
    Interpolate between fits to get the necessary parameters for the offline 
    pointing model (a2 and a3).  If the requested date is before or after
    the first (last) tilt fit data, then fit a second-order polynomial to the 
    parameter fits and extrapolate to the requested date.

    INPUTS:
        date: array of dates, from start of the observation to the end of the observation.
        
        config_file: name of config file containing az tilt fit data.
        
        config_filename_out [None]: If this is a list, the name of the config file read
            will be appended to it.
        
        return_measured_tilts [False]: If set to true, return the az tilt fit 
            results along with the interpolated values of a2 and a3.
         
        extrap_forward [False] : Should always be False for the production pointing 
            map runs.  If set to True (used for daily map checks at Pole), extrapolate
            the fitted tilt parameters forward in time to get better pointing corrections
            for maps immediately after data acquisition.

    OUTPUTS:
        Returns a2 and a3, where a2 and a3 are the interpolated model parameters over 
        the mjd times in question.

    WRITTEN:
        2012-08-05 by Jason W. Henning.
    '''
    # Convert the input time to mjd. Assume that it starts as an array either
    # of SPTDatetime objects or MJDs.
    try:
        date = np.array(map(lambda x: x.mjd, date))
    except AttributeError:
        pass
    
    #Read in the list of measured az tilt parameter fits.
    print 'Reading from tilt config file: ', config_file

    try:
        date_utc, meas_a2, meas_a3, tilt_ang = np.array(files.readConfig(config_file, start_date=np.min(date),
                                       stop_date=np.max(date), has_header=False,
                                       filename_out=config_filename_out))
        have_tilt_angle = True
    except ValueError:
        have_tilt_angle = False
        date_utc, meas_a2, meas_a3 = np.array(files.readConfig(config_file, start_date=np.min(date),
                                       stop_date=np.max(date), has_header=False,
                                       filename_out=config_filename_out))
    #Check if the arrays are increasing in time.  If not, re-order them.
    date_length = len(date_utc)
    sorted_indices = sorted(range(date_length), key=lambda k: date_utc[k])

    date_utc = date_utc[sorted_indices]
    meas_a2 = meas_a2[sorted_indices]
    meas_a3 = meas_a3[sorted_indices]
    if have_tilt_angle:
        tilt_ang = tilt_ang[sorted_indices]

    #Check to see the last input date is smaller than the last date read in from the config file.
    #If it isn't, throw an exception letting the user know the data they are trying to pointing
    #correct doesn't have config file info for, unless extrapolation into the future is 
    #explicitly turned on.
    if extrap_forward == False:
        print 'Tilt parameter extrapolation is currently set to OFF.  Normally, this is right.'
        if date[-1] > date_utc[-1]:
            raise Exception("This observation is after the latest processed date in the az tilt offline pointing " + \
                            "config files.  If you need this observation to be pointing corrected, please wait " + \
                            "a few days for the config files to be updated.  Otherwise, you can look at the data " + \
                            "without pointing corrections by using the flag 'correct_global_pointing=False' " + \
                            "when calling readData().")

    else:
        print 'Tilt parameter extrapolation is currently set to ON.  This is only right for daily map quality checks at Pole.'
        #Least-squares fit the tilt data with a 6th-order poly (way overkill) to extrapolate forward for 
        #correcting pointing tilt before the *official* pointing corrections are calculated.

        print 'Last a2/a3 in config file: ', meas_a2[-1], '/', meas_a3[-1]
        these_dates = np.where(date_utc > 56350)[0]
        p2 = np.polyfit(date_utc[these_dates], meas_a2[these_dates], 6)
        p3 = np.polyfit(date_utc[these_dates], meas_a3[these_dates], 6)

        poly_date = np.arange(10) + date[0]
        meas_a2 = p2[0]*poly_date**6. + p2[1]*poly_date**5. + p2[2]*poly_date**4. + \
                  p2[3]*poly_date**3. + p2[4]*poly_date**2. + p2[5]*poly_date + p2[6]

        meas_a3 = p3[0]*poly_date**6. + p3[1]*poly_date**5. + p3[2]*poly_date**4. + \
                  p3[3]*poly_date**3. + p3[4]*poly_date**2. + p3[5]*poly_date + p3[6]

        print 'a2 range: ', np.min(meas_a2), '/', np.max(meas_a2)
        print 'a3 range: ', np.min(meas_a3), '/', np.max(meas_a3)

        date_utc = poly_date
        

    #There's a scale factor, measured by Ryan Keisler.  After the scaling, the RMS difference
    #between optical tilts and bearing tilts is 3 arcseconds.  This also accounts for a mapping
    #between tilt_ha, tilt_lat and a2,a3 I mistakenly introduced in the tilt config files.
    sf = 0.88
    meas_a3_temp = sf*meas_a2
    meas_a2_temp = -sf*meas_a3

    #Interpolate to get the fits.  
    a2 = np.interp(date, date_utc, meas_a2_temp)
    a3 = np.interp(date, date_utc, meas_a3_temp)    
    
    if return_measured_tilts==False:
        return a2, a3
    else:
        return a2, a3, date, date_utc, meas_a2_temp, meas_a3_temp
#----------------------------------------------------------------------------------------------------
def get_hii_params(date, config_file, config_filename_out=None, return_measured_params=False,
                   use_median=False):
    """
    Look at a config file with HII region pointing parameter.  Interpolate between fits to get the
    necessary parameters for the offline pointing model (a4, a5, a6).

    INPUTS:
        date: array of dates, from start of the observation to the end of the observation.
        
        config_file: name of config file containing HII source fit parameter data.
        
        config_filename_out [None]: If this is a list, the name of the config file read
            will be appended to it.

    OUTPUTS:
        Returns pointing model parameters a4, a5, and a6 interpolated over the mjd times of the
        observation.

    WRITTEN:
        2012-11-05 by Jason W. Henning.
    """

    #Convert the input time to mjd. Assume that it starts as an array either
    # of SPTDatetime objects or MJDs.
    try:
        date = np.array(map(lambda x: x.mjd, date))
    except AttributeError:
        pass

    #Read in the list of measured az tilt parameter fits.
    print 'Reading from hii config file: ', config_file

    #Read in the list of measured HII region pointing parameters.
    date_utc, meas_a0, meas_a1, meas_a4, meas_a5, meas_a6, meas_az0 = \
              np.array(files.readConfig(config_file, start_date=np.min(date),
                                        stop_date=np.max(date), has_header=False,
                                        filename_out=config_filename_out))

    #Check if the arrays are increasing in time.  If not, re-order them.
    date_length = len(date_utc)
    sorted_indices = sorted(range(date_length), key=lambda k: date_utc[k])

    date_utc = date_utc[sorted_indices]
    meas_a0 = meas_a0[sorted_indices]
    meas_a1 = meas_a1[sorted_indices]
    meas_a4 = meas_a4[sorted_indices]
    meas_a5 = meas_a5[sorted_indices]
    meas_a6 = meas_a6[sorted_indices]
    meas_az0 = meas_az0[sorted_indices]

    #Check to see the last input date is smaller than the last date read in from the config file.
    #If it isn't, throw an exception letting the user know the data they are trying to pointing
    #correct doesn't have config file info for.
    if date[-1] > date_utc[-1]:
        raise Exception("This observation is after the latest processed date in the HII offline pointing " + \
                        "config files.  If you need this observation to be pointing corrected, please wait " + \
                        "a few days for the config files to be updated.  Otherwise, you can look at the data " + \
                        "without pointing corrections by using the flag 'correct_global_pointing=False' " + \
                        "when calling readData().")

    #For now assume the dates are within the dates we have HII region parameters for.
    #Interpolate to get the right values for these dates.
    a0 = np.interp(date, date_utc, meas_a0)
    a1 = np.interp(date, date_utc, meas_a1)
    a4 = np.interp(date, date_utc, meas_a4)
    a5 = np.interp(date, date_utc, meas_a5)
    a6 = np.interp(date, date_utc, meas_a6)
    az0 = np.interp(date, date_utc, meas_az0)
    
    if use_median:
        a4 = np.nanmedian(meas_a4)
        a5 = np.nanmedian(meas_a5)
        a6 = np.nanmedian(meas_a6)

    if return_measured_params == False:
        return a0, a1, a4, a5, a6, az0
    else:
        return a0, a1, a4, a5, a6, az0, date_utc, meas_a0, meas_a1, meas_a4, meas_a5, meas_a6, meas_az0
#----------------------------------------------------------------------------------------------------

def get_lin_sens(date, nointerp=True, HII_fit=False, in_data=None):
    """
    Translated from the IDL get_lin_sens.pro written by RK.

    The purpose of this function is to return the linear sensor and thermometry sensor data
    from a given time window.  Linearly interpolate over dropouts in the 
    sensor data.
    
    INPUTS
        date: Array of dates.

    OUTPUTS
        S: a dictionary with the following substructures:
            'utc': The 100 Hz UTC.
            'l1': The 100 Hz L1 length, in mm.
            'l2': The 100 Hz L2 length, in mm.
            'r1': The 100 Hz R1 length, in mm.
            'r2': The 100 Hz R2 length, in mm.
            'del': The 100 Hz elevation correction, in arcseconds.
            'daz': The 100 Hz azimuth correction, in arcseconds.
            'det': The 100 Hz elevation tilt correction, in arcseconds.
            'temp': The thermometry data, which is an array with [nthermos, nsamples].

     Translated: October 2012, JWH.
     Originally Written: April 2008, RK.
     Modifications: Take linear sensor data in place of the 'date' input. 7 Dec 2012, SH
    """

    # Check if the "date" input is an array of dates, or already the linear sensor data from those dates.
    # An array of dates will be 1D, while linear sensor data are 2D. 
    if date.ndim==1:
        #Convert times to MJD if not already.
        start_time = date[0]
        stop_time = date[-1]
    
        #Make sure the dates make sense.
        if start_time >= stop_time:
            print 'Your start date is after your stop date.'
            print 'Quitting!'
            return {'utc':0., 
                    'l1':0., 'l2':0., 'r1':0., 'r2':0.,
                    'del':0., 'daz':0., 'det':0., 'temp':0.}
    
        #Load in pointing-model pertinent registers (defined in the pointing registers config file).
        if in_data==None:
            d = pt.readPointingRegisters(start_time, stop_time, HII_fit=HII_fit)
        else:
            d = struct({'linsens_avg':in_data['tracker.linear_sensor_avg'],
                        'track_utc':in_data['track_utc'],
                        'scu_temp':in_data['scu.temp']
                })
            d.linsens_avg = d.linsens_avg.reshape(4,1)
        
        lin = d.linsens_avg
        utc = d.track_utc
    
        #Let's interpolate over dropouts in the linear sensor data.
        npts = len(lin[0])
        linsens_zero = 9.5
        if nointerp==True:
            for i in range(0,4):
                whnodrop1 = np.nonzero((lin[i] != linsens_zero) & (lin[i] != 0.))[0]
                nnodrop = len(whnodrop1)
                if nnodrop < npts/2.:
                    print 'Warning: no good linear sensor data.'
                    continue
                thisdata = lin[i]
                thisdata = pt.interp_over_dropouts(thisdata, whnodrop=whnodrop1)
                lin[i] = thisdata
    
        #Assign the 4 different measured lengths, and interpolate to match the input date array.
        # First convert the input time to mjd. Assume that it starts as an array either
        # of SPTDatetime objects or MJDs.
        try:
            date = np.array(map(lambda x: x.mjd, date))
        except AttributeError:
            pass
        if in_data==None:
            l1 = np.interp(date, utc, lin[0])
            l2 = np.interp(date, utc, lin[1])
            r1 = np.interp(date, utc, lin[2])
            r2 = np.interp(date, utc, lin[3])
        else:
            l1 = lin[0]
            l2 = lin[1]
            r1 = lin[2]
            r2 = lin[3]
    
        #Interpolate the temperature sensor array to match the input date array.  This is a
        #separate set fron above because d.scu_temp is an array with shape (60,N), where N is
        #the number of frames.  We have to loop over the 60 theremometers and interpolate each
        #component.
        if in_data==None:
            temp = []
            for i in range(len(d.scu_temp)):
                temp.append(np.interp(date, d.utc,d.scu_temp[i]))
            temp = np.array(temp)
        else:
            temp = d.scu_temp
    else:
        l1 = date[0]
        l2 = date[1]
        r1 = date[2]
        r2 = date[3]
        temp = [] # Assume that if whoever called me has linear sensor data, they also have scu temperature data.

    #Yoke dimensions in mm.
    Rs = 1652.
    Rh = 3556.
    Ry = 6782.

    #Calculate corrections in arcsec.
    DEL = (1./(2.*Rs))*(l2 - l1 + r2 - r1)*(3600.*180./np.pi)
    DAZ = (Rh/(Ry*Rs))*(l1 - l2 - r1 + r2)*(3600.*180./np.pi)
    DET = (1./(2.*Ry))*(r1 + r2 - l1 - l2)*(3600.*180./np.pi)

    # We'd like these corrections to have a mean = 0, so let's subtract off
    # the mean (in fact we are subtracting off the median).  Note that
    # this isn't a huge effect when applying these corrections with a 
    # pointing model.  If we were NOT removing the mean, there would just 
    # be an additional DC offset in these terms in the pointing model.
    # These are to to mean = 0 because then there isn't a spike in the
    # pointing model parameters when we go from not having linear sensor
    # data to having linear sensor data.
    #DAZ -= 16.2
    #DEL -= 14.6
    #DET -= 25.8

    #Medians calculated from RCW38 observations.
    DAZ -= 38.7
    DEL -= 27.6
    DET -= 18.6

    #Fill the output dictionary.
    s = {#'utc':utc, 
         'l1':l1, 'l2':l2, 'r1':r1, 'r2':r2,
         'del':DEL, 'daz':DAZ, 'det':DET, 'temp':temp}

    return s
#----------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------

def computeLinSensParams(lin):
    """
    Take linear sensor data and spit out pointing model parameters DET, DEL, and DAZ.

    INPUTS:
        lin: The lineare sensor data for the observation.  Assumed to be the same length of track_utc,
             which it will be if you only run this function with linear sensor data read in at the 
             same time as the bolodata when processing a map.  Otherwise, use get_lin_sens
    """

    #If this is for HII region fitting, interpolate to 100 Hz.
    l1 = lin[0]
    l2 = lin[1]
    r1 = lin[2]
    r2 = lin[3]

    #Yoke dimensions in mm.
    Rs = 1652.
    Rh = 3556.
    Ry = 6782.

    #Calculate corrections in arcsec.
    DEL = (1./(2.*Rs))*(l2 - l1 + r2 - r1)*(3600.*180./np.pi)
    DAZ = (Rh/(Ry*Rs))*(l1 - l2 - r1 + r2)*(3600.*180./np.pi)
    DET = (1./(2.*Ry))*(r1 + r2 - l1 - l2)*(3600.*180./np.pi)

    # We'd like these corrections to have a mean = 0, so let's subtract off
    # the mean (in fact we are subtracting off the median).  Note that
    # this isn't a huge effect when applying these corrections with a 
    # pointing model.  If we were NOT removing the mean, there would just 
    # be an additional DC offset in these terms in the pointing model.
    # These are to to mean = 0 because then there isn't a spike in the
    # pointing model parameters when we go from not having linear sensor
    # data to having linear sensor data.
    #DAZ -= 16.2
    #DEL -= 14.6
    #DET -= 25.8 

    #Medians calculated from 2014 RCW38 observations.
    DAZ -= 38.7
    DEL -= 27.6
    DET -= 18.6


    #Fill the output dictionary.
    s = {'l1':l1, 'l2':l2, 'r1':r1, 'r2':r2,
         'del':DEL, 'daz':DAZ, 'det':DET}

    return s
#----------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------
def thermo2pointing(scu_temp_in, mjd, thermometry_config_file='thermometer_pointing_coefficients',
                    config_filename_out=None, nointerp=True):
    """
    The purpose of this function is to provide pointing corrections DET and DEL, given an
    input array of structure thermometry and/or linear sensor data.  The model is just linear 
    in the thermometry + linear sensors.  The coefficients for the model are stored in an 
    external common txt file.

    INPUTS:
        scu_temp_in: the array of thermometry + linear sensor data.  It has
                     dimensions of (63, nsamples) where there are 60 thermometers
                     and 3 linear sensors.  The thermometry should be raw (degrees C).
        mjd: the MJD (Modified Julian Date).  If this vector has nsamples elements, then
             obviously each element corresponds to each "sample."  Otherwise, (like if MJD is
             a single number or a vector of the wrong length), the first MJD is applied to all
             samples.
        
        thermometry_config_file ['thermometer_pointing_coefficients'] : (string) The name of
            a config file where we can find coefficients for each of the thermometers.
        
        config_filename_out [None]: If this is a list, the name of the config file read
            will be appended to it.

    OUTPUTS:
        s - a dictionary with the following fields:
            DET: the DET (elevation axis tilt) correction, in arcseconds.
            DEL: the DEL (plain old elevation) correction, in arcseconds.


    EXCEPTIONS
        ValueError if the config file doesn't match the size of the scu_temp register data. 

    Translated to python from the original IDL written by RK, Jan 2009.
    Translated by JWH October 2012.
    """

    scu_temp = scu_temp_in

    # Read in a config file that contains the coefficients for going from
    # structure temperatures to pointing offsets DET and DEL.
    index, det_coeff, del_coeff, neighbors1, neighbors2 = files.readConfig(thermometry_config_file,
                                                                           start_date=time.SptDatetime(mjd[0]),
                                                                           has_header=False,
                                                                           filename_out=config_filename_out)
    
    #If scu_temp is only for 1 time sample, and is 1D, let's make it 2D.
    try:
        ndim = scu_temp.shape[1]
    except IndexError:
        warnings.warn('Thermometry array only 1D.  Converting to 2D...', RuntimeWarning)
        ndim=1
    if ndim==1: scu_temp = scu_temp.reshape((scu_temp.shape[0], -1))

    nsamples = scu_temp.shape[1]
    nthermo = scu_temp.shape[0]
    if nthermo != len(det_coeff)-1 or nthermo != len(del_coeff)-1:
        raise ValueError('# of thermometers in scu_temp does not match # of coefficients')


    #Interpolate over dropouts
    npts = nsamples
    thermo_zero = -200.0
    if nointerp==True and nsamples > 1:
        for i in range(nthermo-3):
            whnodrop1 = np.nonzero((scu_temp[i] != thermo_zero) &
                                   (scu_temp[i] != 0.0) &
                                   (scu_temp[i] > -150.) &
                                   (scu_temp[i] < 40.))[0]
            nnodrop = len(whnodrop1)
            if nnodrop < npts/2.:
                continue
            thisdata = scu_temp[i]
            thisdata = pt.interp_over_dropouts(thisdata, whnodrop=whnodrop1)
            scu_temp[i] = thisdata

    #The thermometer indexed in IDL by i=40 begain to have problems in 2011.
    #The cause of these problems are unknown, but basically the temperatures
    #that are recorded are crazy.  This can screw up the pointing, since the 
    #offline pointing model depends on the telescope temperatures.  So if it's
    #2011 or later, and the i=40 thermometer looks crazy, let's replace its data
    #with that of a nearby thermometer, i=42.
    if (np.abs(np.median(scu_temp[40]) - np.median(scu_temp[42])) > 10.):
        #if (type(mjd) != np.array): # I don't think this if statement is necessary. And it can cause crashes. SH
        #    mjd = np.array([mjd]).squeeze()
        if (time.SptDatetime(mjd[0]).mjd > 5562.000):
                #warnings.warn('The data from thermometer 40 look bad! Replacing its data with that from thermometer 42!', RuntimeWarning)
                scu_temp[40] = scu_temp[42]

    # Look through each thermometer, checking to see if there are any
    # which are still equal to the "thermometer zero" value, typically
    # -200.0 C, which were not interpolated over because there's no good
    # data to use for the interpolation.  For these thermometers we want
    # to replace their output with that of their "neighbors", where 
    # neighbor is defined as a thermometer that historically had a 
    # similar temperature.
    scu_tempo = scu_temp.copy()
    for i in range(nthermo-3):
        this_temp = np.array(scu_temp[i]).reshape(len(scu_temp[i]))
        wh_zero = np.nonzero(this_temp == thermo_zero)[0]
        n_zero = len(wh_zero)
        if n_zero > 0:
            #This thermometer is returning the "zero" value.  Replace its
            #data from that from its closest possible neighbor with similar data.
            this_n1 = neighbors1[i]
            this_temp_n = scu_temp[int(this_n1)]#.reshape(len(scu_temp[i]))
            wh_zero_n = np.nonzero(this_temp_n == thermo_zero)[0]
            n_zero_n = len(wh_zero_n)  
            if n_zero_n == 0:
                #warnings.warn('Replacing bad thermometer output with 1st neighbor output! Replacing %d with %d.' % (i, this_n1), RuntimeWarning)
                scu_temp[i] = this_temp_n
            else:
                this_n2 = int(neighbors2[i])
                this_temp_n = np.array(scu_temp[this_n2])#.reshape(len(scu_temp[i]))
                wh_zero_n = np.nonzero(this_temp_n == thermo_zero)[0]
                n_zero_n = len(wh_zero_n)
                if n_zero_n == 0:
                    #warnings.warn('Replacing bad thermometer output with 2nd neighbor output! Replacing %d with %d.' % (i, this_n2), RuntimeWarning)
                    scu_temp[i] = this_temp_n
                else:
                    if (i<=25) or (i >= 40):
                        warnings.warn('Was unable to replace bad thermometer output for thermometer %d with a neighbor. Setting it to the median of all thermometers!' % i, RuntimeWarning)
                    wh_not_zero = np.nonzero(scu_tempo != thermo_zero)[0]
                    n_not_zero = len(wh_not_zero)
                    wh_not_zero = 0.
                    if n_not_zero > 0:
                        scu_temp[i] = ( np.zeros(len(scu_temp[i])) + 
                                        np.median(scu_tempo[scu_tempo != thermo_zero]) )

    #Convert thermometry to degrees Kelvin.
    nthermo = len(scu_temp)
    scu_temp[:-3] += 273.

    #Restructure the coefficients
    det_dc = det_coeff[-1]
    del_dc = del_coeff[-1]
    det_coeff = np.matrix(det_coeff[:-1])
    del_coeff = np.matrix(del_coeff[:-1])
    scu_temp = np.matrix(scu_temp)

    #Subtract off the median (calculated from 2014 RCW38 obs) so we have mean zero corrections.
    DET = np.array(det_coeff*scu_temp + det_dc) - -20.2
    DEL = np.array(del_coeff*scu_temp + del_dc) - -7.0

    return {'det':DET, 'del':DEL}
#----------------------------------------------------------------------------------------------------

#####################################################################################################
#POINTING MODELS
#####################################################################################################
def model1(date, az, el, T, P, config_file_tilt, config_file_hii, config_file_thermometry,
           config_filename_out=None, lin=None, scu_temp=None, 
           flags=['tilts', 'boom_flex'], realtime_params=True,
           extrap_forward=False):
    '''
   model1 is the model used for SPTsz.  The full model is as follows:
       \delta_az = (a2*cos[az] + a3*sin[az])*tan[el] + (a4 - dET)*tan[el] + a5/cos[el]
       \delta_el = a0*sin[el] + a1*cos[el] - a6 - (a2*sin[az] - a3*cos[az]) - dEL - theta_refr

       a0,a1: Boom flexure terms, known from SPTsz (but we can fit for new values).  Turn on by 
              invoking the flag 'boom_flex'.
       a2,a3: Az bearing tilt parameters.  Obtained from fitting az bearing tilt measurements.  
              Invoked by the flag 'tilts'.
       a4,a5,a6: Obtained from fits to processed HII soure measurements.  Invoked with the flag 
                 'HII_params'.
       dET, dEL: Parameters calculated from linear sensor data.  Invoked with the flag 'lin_sen'

   INPUTS:
          date: array of times over which we're correcting the az/el of the observation.
          az: raw az array of the observation.
          el: raw el array of the observation.
          lin: linear sensor data during the observation.
          scu_temp: Telescope temperature readings. If either lin or scu_temp are not given, 
              they will be read in from archive files.
          config_file_tilt, config_file_hii, config_file_thermometry: Names of files containing model 
              parameters.  Should be in data structure.
          config_filename_out [None]: If this is a list, the name of the config file read
              will be appended to it.
          flags: List of strings naming components of the model to turn on.  By default flag is 
                 'all', which turns on every component of the model.
          realtime_params[False]: If set to false, set realtime parameters 
                                  (a4,a5,a6, dET,dEL, and theta) to zero regardless of flags called.
          extrap_forward[False]: If set to True, allow extrapolation forward in time for az tilt
                                 parameters.  Generally, we do not want this to be True.

   OUTPUTS:
           az_c: az array corrected by the model.
           el_c: el array corrected by the model.
           d_az: The difference between the input and corrected az arrays.
           d_el: The difference between the input and corrected el arrays.

    EXCEPTIONS:
        ValueError if 'flags' contains an unrecognized flag.

   AUTHOR: Jason W. Henning, August 5, 2012.
    '''
    valid_flags = ['tilts','HII_params','lin_sen','boom_flex','all',
                   'just_a5','just_refraction','ml_thermolin', 'median_HII']
    # Error-check the "flags" input.
    invalid_flags = set()
    for flag in flags:
        if flag not in valid_flags:
            invalid_flags.add(flag)
    if invalid_flags:
        raise ValueError("You gave me the flag names %s, which I do not recognize." % str(list(invalid_flags)))
    
    
    #First, we need to grab the model parameters for the dates in question.
    #1) Obtain the tilt parameters a2 and a3 (obtained roughly daily and interpolated).
    if 'tilts' in flags or 'all' in flags:
        a2, a3 = get_az_tilt_params(date, config_file_tilt, config_filename_out=config_filename_out,
                                    extrap_forward=extrap_forward)

    #2) Get the parameters from the realtime data and HII region measurements.
    if 'HII_params' in flags or 'all' in flags:
        if 'median_HII' in flags: 
            use_median = True
        else:
            use_median = False
        a0, a1, a4, a5, a6, az0 = get_hii_params(date, config_file_hii, config_filename_out=config_filename_out,
                                                 use_median=use_median)


        #Until new HII param config files are generated, we'll hardcode the az encoder offset.
        #az0 = -0.304527

    if 'lin_sen' in flags or 'all' in flags:
        #4) Read in the linear sensor data.
        if lin is not None and scu_temp is not None:
            lin_data = get_lin_sens(lin)
        else:
            lin_data = get_lin_sens(date)
            scu_temp = lin_data['temp']
        this_lin = np.array([lin_data['daz'], lin_data['del'], lin_data['det']])
        th = thermo2pointing(np.concatenate((scu_temp, this_lin)), date, 
                             thermometry_config_file=config_file_thermometry, 
                             config_filename_out=config_filename_out)

        DET = np.array(lin_data['det'])/3600. #Calculated in arcsec, so convert to degrees.
        DEL = np.array(lin_data['del'])/3600. #Calculated in arcsec, so convert to degrees.
        th_DET = np.array(th['det'])[0]/3600.
        th_DEL = np.array(th['del'])[0]/3600.
        lin_data = 0.

        total_DET = (DET + th_DET)
        total_DEL = (DEL + th_DEL)

        #Grab the refraction correction
        refract = pt.quick_refr(el,T+273.15,P)

    if ('ml_thermolin' in flags):
        az_ml, el_ml = eht.get_thermolin_correction2(date)

    #3) Put az and el into radians
    az_rad = az*np.pi/180.
    el_rad = el*np.pi/180.

    #4) Now that we have the model parameters, calculate the change in az and el.
    if ('boom_flex' in flags) or ('all' in flags):
        daz1 = np.zeros(len(date))
        del1 = a0*np.sin(el_rad) + a1*np.cos(el_rad)
    else:
        daz1 = np.zeros(len(date))
        del1 = np.zeros(len(date))

    if ('tilts' in flags) or ('all' in flags):
        daz2 = -((a2*np.cos(az_rad) + a3*np.sin(az_rad))*np.tan(el_rad))
        del2 = (a2*np.sin(az_rad) - a3*np.cos(az_rad))
    else:
        daz2 = np.zeros(len(date))
        del2 = np.zeros(len(date))

    if ('HII_params' in flags) or ('all' in flags):
        if ('just_a5' in flags):
            a4 *= 0.
            a6 *= 0.
        daz3 = a4*np.tan(el_rad) - a5/np.cos(el_rad) - az0
        del3 = -a6
    else:
        daz3 = np.zeros(len(date))
        del3 = np.zeros(len(date))

    if ('lin_sen' in flags) or ('all' in flags):
        if ('just_refraction' in flags):
            total_DET *= 0.
            total_DEL *= 0.

        if ('ml_thermolin' in flags):
            total_DET *= 0.
            total_DEL *= 0.
        else:
            az_ml = 0.0
            el_ml = 0.0
        #The default sign for the ml corrections is negative.
        daz4 = -total_DET*np.tan(el_rad) + az_ml/np.cos(el_rad)
        del4 = -total_DEL - refract + el_ml
        
    else:
        daz4 = np.zeros(len(date))
        del4 = np.zeros(len(date))

    d_az = (daz1 + daz2 + daz3 + daz4)
    d_el = (del1 + del2 + del3 + del4)

    az_c = az + d_az
    el_c = el + d_el

    print 'Mean AZ/EL corrections: ', np.mean(d_az), np.mean(d_el)
        
    return az_c, el_c, d_az, d_el

#----------------------------------------------------------------------------------------------------

#####################################################################################################
#POINTING DATA HANDLER
#####################################################################################################
import numpy as np
from scipy.optimize import curve_fit
from numpy import log, abs, sqrt, mean
import warnings
from time import clock
#from .. import constants
#from .. import float_type, util
from sptpol_software.util import tools, files, math
#from ..util.tools import struct
#from ..analysis import filters
#from ..util.physics import trjToTcmb
from sptpol_software.analysis import cuts as ct
#from ..util.time import SptDatetime
import scipy.stats.stats as st   #st.nanmedian()

#import pointing_models 
import sptpol_software.observation.telescope as telescope

def applyOfflinePointing(data, model='SPT', flags=['all'], overwrite_global=False, \
                         hii_config=None, \
                         overwrite=False, onlyEphemeris=False,
                         extrap_forward=False):

    """
    Applies Offline Pointing model to data structure.  Changes pointing within
    given structure.  

    INPUTS:
        data: (SPTDataReader) The data from observations

        model ['SPT']: Offline pointing model to apply.  "None" or "SPT" applies 
                      the SPT model as a default.  Others may or may not be available.

        flags ['all']: flags to be passed to given model.  name individually, or say 'all'

        overwrite_global [False]: If global pointing correction (based off point source obs)
                                   has already been applied, overwrite it and delete its flag
                 
        hii_config ['None']: If you want to test different hii region fit parameters besides what, 
                             is in the sptpol_software/config_files directory, pass
                             the name and full path to the config file you want to use.
                  
        overwrite [False] : If True, overwrite a previous application of the offline 
            pointing model (if any).
                                   
        onlyEphemeris [False]: When set to False, applies the offline pointing model coordinate 
                               corrections along with ephemeris corrections. If set to True, 
                               only make ephemeris corrections.

        extrap_forward[False]: If set to True, allow extrapolation forward in time for az tilt
                                 parameters.  Generally, we do not want this to be True.

    OUTPUT:
        None.  Applies corrections to Az/EL values in given data structure.  Also generates
        RA/DEC using PyEphem in J2000 epoch and saves data in structure.  Also adds model
        and flags used to data structure.  We need to check accuracy of PyEphem.
        
    EXCEPTIONS
        RuntimeError if the offline pointing has already been applied or the currectGlobalPointing function has been run.
        ValueError if an invalid model is requested.

    AUTHOR: Jason Austermann, August 6, 2012
    
    CHANGES:
        25 Apr 2014: Get temp_avg and pressure_avg from data.telescope.location rather than data.observation.
                     This function gets called earlier now -- before data.observation is populated.
                     Calling earlier allows things like mean_ra to use offline pointing.
    """

    # warn if global point source correction already applied.. overwrite?
    if 'correctGlobalPointing' in data.header['processing_applied']:
        if overwrite_global == True:
            del data.header['processing_applied']['correctGlobalPointing']
        else:
            raise RuntimeError("Global pointing correction (pt sources) already applied. Use overwrite_global keyword to proceed")

    # get model corrections
    if onlyEphemeris==False:
        if model in ['SPT', 'spt', 'Spt', 'SPTsz', 'SPT-sz', None]:
            
            # If the ACU times are SptDatetimes, convert to MJD. If they're
            # not SptDatetimes, then assume that they're already MJD.
            if isinstance(data.antenna.track_utc[0], time.SptDatetime):
                track_utc_mjd = np.array(map(lambda x: x.mjd, data.antenna.track_utc))
            else:
                track_utc_mjd = data.antenna.track_utc
            
            scu_temp=None
            if 'lin_sen' in flags or 'all' in flags:
                # We'll need to upsample the temperature data. Do that now.
                try:
                    scu_temp = []
                    for i_temp in xrange(len(data.antenna['scu_temp'])):
                        scu_temp.append(np.interp(track_utc_mjd, data.data['antenna']['scu_time'],data.data['antenna']['scu_temp'][i_temp]))
                    scu_temp = np.array(scu_temp)
                except KeyError:
                    # If 'scu_time' or 'scu_temp' aren't in the antenna substructure, then 
                    # we'll have to read them in from the archive file.
                    scu_temp=None

            #For testing purposes, we can read in a chosen HII params file, or we can let the software
            #choose the right config file by validity dates (hII_config == None).
            if hii_config == None:
                config_file_hii =  data.config.get('offline_hii_config','sptpol_offline_hii_params')
            else:
                config_file_hii = hii_config
            
            #Which pressure field should we use?
            data_pressure = copy.deepcopy(data.telescope.location.pressure)

            if data_pressure == 0.0:
                print 'Warning: data.telescope.location.pressure = 0.0.  Due to a bug, this may affect IDFs made between Apr 2014 and Dec 2014 that applied offline pointing.  (This affects the 2012+2013 deepfield BB IDFs).  Using data.observation.pressure_avg instead.'
                data_pressure = copy.deepcopy(data.observation.pressure_avg)

            config_files_read = []
            az_c,el_c,d_az,d_el = model1(data.antenna.track_utc, 
                                         data.antenna.track_actual[0], 
                                         data.antenna.track_actual[1],
                                         data.telescope.location.temp, #data.observation.temp_avg,
                                         data_pressure,
                                         lin=data.antenna.track_linsens_avg,
                                         scu_temp=scu_temp,
                                         config_file_hii = config_file_hii,
                                         config_file_tilt = data.config.get('offline_tilts_config',
                                                                            'sptpol_offline_tilts'),
                                         config_file_thermometry=data.config.get('offline_thermometry_config',
                                                                                 'thermometer_pointing_coefficients'),
                                         config_filename_out=config_files_read,
                                         flags=flags,
                                         extrap_forward=extrap_forward)
            data.config['all_configs']['files_used'].extend(config_files_read)
            data.data['header']['dependencies'].update(config_files_read)
            
    #elif (add other model options here, if you make them)
        else:
            raise ValueError("Model given is not currently valid.  Feel free to add it to archive!")
    else:
        az_c, el_c = data.antenna.track_actual[0], data.antenna.track_actual[1]
        d_az = np.zeros(len(az_c))
        d_el = np.zeros(len(el_c))
    
    # get corrected RA/DEC
    # For now, use PyEphem, but need to check how accurate it is
    # takes temperature in C, pressure in mBar
    # Set pressure to zero to disable pyephem's atmospheric refraction correction.
    ra_new, dec_new = data.telescope.toRaDec(data.antenna.track_utc, az_c, el_c, 
                                             input_temp=np.array([data.telescope.location.temp]*len(az_c)),
                                             input_pressure=np.zeros(len(az_c)))

    #Pyephem replaces data.telescope.pressure with the input_presure in the call above.  Fix that by replacing
    #data.telescope.pressure with the copied raw version.
    data.telescope.location.pressure = data_pressure
    
    # unwrap in case crosses 360->0 barrier. Do in place to save memory
    tools.unwrapAz(az_c,   modify_in_place=True)
    tools.unwrapAz(ra_new, modify_in_place=True)

    # make sure model hasn't already been applied to data, otherwise store
    if 'applyOfflinePointing' in data.header['processing_applied'] and not overwrite:
        raise RuntimeError("The offline pointing model has already been applied to this data")
    else:
        data.header['processing_applied']['applyOfflinePointing'] = \
                   {'model':model, 'flags':flags, 'overwrite_global':overwrite_global, \
                    'onlyEphemeris':onlyEphemeris, 'extrap_forward':extrap_forward}
        data.antenna.offline_az_correction = d_az
        data.antenna.offline_el_correction = d_el
        data.antenna['track_actual_corrected'] = np.array([az_c,el_c])
        data.antenna.ra = ra_new
        data.antenna.dec = dec_new
        
    return 
#----------------------------------------------------------------------------------------------------
    
