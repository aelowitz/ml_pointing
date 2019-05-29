"""
This is a script for the SPT autoprocessor. It is intended to be used by 
the autoprocess_spt_data module. 

This script is a demonstration of scripting a map run. It's designed with 
observations of a single bright source in mind (planets, RCW38, etc.), but
could easily be modified for CMB fields.

Before using this script, set the variables "source_name", "subdirectory_name", and 
"nscans_min" to appropriate values.

This script creates finely binned (0.1 arcmin/pixel) maps of a source.
This script first makes a map to find the location of the source in RA/dec,
writes a pointsource config file with that location, and re-makes the
map, using the pointsource mask rather than a dynamic mask.


"""

__metaclass__ = type
__author__    = ["Stephen Hoover"]
__email__     = ["hoover@kicp.uchicago.edu"]
__version__   = "1.0"
__date__      = "2013-04-08"


import os, tempfile
import numpy as np
import pickle as pk
from script_defaults import * # This module defines defaults for various variables.
from sptpol_software.autotools import logs, files
from sptpol_software.analysis import quicklook, processing
from sptpol_software.util.tools import struct
from sptpol_software.data.readout import SPTDataReader
from sptpol_software.util.time import SptDatetime
from sptpol_software.util.files import read
from sptpol_software.data import mapidf
import sptpol_software.analysis.offline_pointing as op


# INPUTS #

directories['output_dir'] = '/data32/jhenning/ptsrc/deepfield/'
directories['idf_dir'] = '/data/sptdat/idf'

#my_name = 'rcw38'
#my_name = 'mat5a'
my_name = '0537-441'
#my_name = '0521-365'
#my_name = '2255-282'

month_flag = '2014'
#test_flag = 'EHT_test'
#test_flag = 'boom_flexure'
#test_flag = 'full_pointing'
test_flag = 'all'

correct_pointing = True
read_from_idf = False

if my_name == 'rcw38':
    map_center = [134.77, -47.51]
elif my_name == 'mat5a':
    map_center = [167.88625, -61.362222] 
elif my_name == '0537-441':
    map_center = [84.7075, -44.0858333] 
elif my_name == '0521-365':
    map_center = [80.74160417, -36.45856972] 
elif my_name == '2255-282':
    map_center = [344.52484537, -27.97257132] 
    #map_center = [344.52484537 + 5./60., -27.97257132 + 5./60.] 

if test_flag == 'EHT_test':
    #flags = ['all','just_a5', 'just_refraction']
    flags = ['all','just_refraction']
    if month_flag == '2013':
        subdirectory_name = 'thermolin/2013' # Store results in $OUTPUT_DIR/map/(subdirectory_name)
        autoprocess_start_date = SptDatetime('30-Apr-2013:06:00:00') # 
        autoprocess_stop_date = SptDatetime('29-Oct-2013:04:45:00') # 
    elif month_flag == '2014':
        subdirectory_name = 'test/2014' # Store results in $OUTPUT_DIR/map/(subdirectory_name)
        autoprocess_start_date = SptDatetime('25-Mar-2014') # 
        autoprocess_stop_date = SptDatetime('13-Dec-2014') # 
    elif month_flag == '201404':
        subdirectory_name = 'thermolin/201404' # Store results in $OUTPUT_DIR/map/(subdirectory_name)
        autoprocess_start_date = SptDatetime('01-Apr-2014') # 
        autoprocess_stop_date = SptDatetime('06-Apr-2014') # 

    elif month_flag == '201405':
        subdirectory_name = 'thermolin/201405' # Store results in $OUTPUT_DIR/map/(subdirectory_name)
        autoprocess_start_date = SptDatetime('01-May-2014') # 
        autoprocess_stop_date = SptDatetime('06-May-2014') # 

    elif month_flag == '201406':
        subdirectory_name = 'thermolin/201406' # Store results in $OUTPUT_DIR/map/(subdirectory_name)
        autoprocess_start_date = SptDatetime('01-Jun-2014') # 
        autoprocess_stop_date = SptDatetime('06-Jun-2014') # 
    
    elif month_flag == '201407':
        subdirectory_name = 'thermolin/201407' # Store results in $OUTPUT_DIR/map/(subdirectory_name)
        autoprocess_start_date = SptDatetime('01-Jul-2014') # 
        autoprocess_stop_date = SptDatetime('06-Jul-2014') # 

    elif month_flag == '201408':
        subdirectory_name = 'thermolin/201408' # Store results in $OUTPUT_DIR/map/(subdirectory_name)
        autoprocess_start_date = SptDatetime('01-Aug-2014') # 
        autoprocess_stop_date = SptDatetime('06-Aug-2014') # 

    elif month_flag == '201409':
        subdirectory_name = 'thermolin/201409' # Store results in $OUTPUT_DIR/map/(subdirectory_name)
        autoprocess_start_date = SptDatetime('01-Sep-2014') # 
        autoprocess_stop_date = SptDatetime('06-Sep-2014') # 

    elif month_flag == '201410':
        subdirectory_name = 'thermolin/201410' # Store results in $OUTPUT_DIR/map/(subdirectory_name)
        autoprocess_start_date = SptDatetime('01-Oct-2014') # 
        autoprocess_stop_date = SptDatetime('06-Oct-2014') # 

    elif month_flag == '201411':
        subdirectory_name = 'thermolin/201411' # Store results in $OUTPUT_DIR/map/(subdirectory_name)
        autoprocess_start_date = SptDatetime('01-Nov-2014') # 
        autoprocess_stop_date = SptDatetime('06-Nov-2014') # 

if test_flag == 'boom_flexure':
    flags = ['tilts','lin_sen','just_refraction']
    if month_flag == '2014':
        subdirectory_name = 'EHT_test/2014_new' # Store results in $OUTPUT_DIR/map/(subdirectory_name)
        autoprocess_start_date = SptDatetime('01-Aug-2014') # 
        autoprocess_stop_date = SptDatetime('15-Dec-2014') # 

if test_flag == 'full_pointing':
    flags = ['all','just_refraction']
    if month_flag == '2014':
        subdirectory_name = 'full_pointing/2014' # Store results in $OUTPUT_DIR/map/(subdirectory_name)
        autoprocess_start_date = SptDatetime('15-Nov-2014') # 
        autoprocess_stop_date = SptDatetime('15-Dec-2014') # 
    elif month_flag == '201408':
        subdirectory_name = 'full_pointing/2014' # Store results in $OUTPUT_DIR/map/(subdirectory_name)
        autoprocess_start_date = SptDatetime('01-Aug-2014') # 
        autoprocess_stop_date = SptDatetime('01-Sep-2014') # 

if test_flag == 'all':
    flags = ['all']
    if month_flag == '2014':
        subdirectory_name = 'all_flags/2014' # Store results in $OUTPUT_DIR/map/(subdirectory_name)
        autoprocess_start_date = SptDatetime('25-Mar-2014') # 
        autoprocess_stop_date = SptDatetime('15-Dec-2014') # 


poly_order = 2
nscans_min = 10 # Exclude aborted observations
nscans_max = 0 # 0 max means no limit on the max number of scans.
use_fits=False
maximum_time_to_read = timedelta(seconds=12000) # Refuse to read more data than this.

do_left_right = False # If True, make three sets of maps: only leftgoing scans, only rightgoing scans, and all scans.

#Sources in 'SPTsz" coordinates.
ra_cut = [0.,360.0] # only process maps with mean_ra within this range, e.g. to only do lead or trail

#In degrees, mask this region around the source for the polynomial fits.
# Set to zero to skip making a pointsource mask. 
mask_radius = 0./60 
force_source_location = map_center  #use this known source location rather than finding it (good choice if source is weak in a single map, but known)

# Do polynomial filtering inside quickmap (use the 'timestream_filtering' argument), instead of
# directly through the cFiltering function. Doing it through quickmap lets the quickmapper
# tell the C code exactly what the map parameters are, making it more likely that the
# point source mask will be correct.
#   The 'pointsource_file' argument is blank now, but we'll fill it in inside the analyzeData function. 
analysis_function = quicklook.quickmap
analysis_function_kwargs = {'good_bolos':['flagged','timestream','calibrator',
                                          'elnod','has_pointing','has_polcal',
                                          'full_pixel','source_response','no_c4',
                                          #'has_calibration',
                                          'good_timestream_weight'],
                            'reso_arcmin':0.25,
                            'proj':0,
                            'map_shape':[2.0,2.0],
                            'map_center':map_center,
                            't_only':True,
                            'inverse_noise_weighted_map':True,
                            'timestream_filtering':{'poly_order':poly_order,
                                                    'dynamic_source_mask':False,
                                                    'pointsource_file':''}}

preprocessing_function = []
preprocessing_function_kwargs = []

# We always read out with SPTDataReader.readData, so no need to specify a function for that.
readout_kwargs = {'timestream_units':'watts', 
                  'correct_global_pointing':True}
                  
                  
                  
                  
######################## functions below:
#########   getTimeInterval()
#########   getFilenameOut()
#########   analyzeData()
#########   writeData()                  
                  

def getTimeInterval():
    """
    Replace generic getTimeInterval with one that calls logs.readSourceScans instead.
    """
    return logs.readSourceScanTimes(autoprocess_start_date, autoprocess_stop_date, my_name,
                                    nscans_min=nscans_min, nscans_max=nscans_max,
                                    log_summary_dir=directories['log_summary_dir'])
    

def getFilenameOut(time_interval):
    """
    Replace generic getFilenameOut with a version that outputs to fits_dir/map/my_name/$SOURCENAME_map_date_150ghz.hdf5.
    We'll end up writing two maps, a 150ghz and a 090ghz version, but this function only needs to return
    one of them.
    """
    try:
        start_time = time_interval[0]
    except TypeError:
        start_time = time_interval
    filename = "%s_%s_%03ighz.%s" % (my_name, start_time.file, 150, ('fits' if use_fits else 'hdf5'))
    return os.path.join(directories['output_dir'], my_name, subdirectory_name, filename)


def analyzeData(time_interval, idf_band=None):
    """
    Nearly the generic function, but we return the result of the analysis function instead of 
    the data object that we read out.
    """
    # This is a bit of a kludge to let us analyze both 150 GHz and 90 GHz IDFs with the same script.
    # When the autoprocessor calls analyzeData, we'll get the default argument idf_band=None.
    # This bit of code will separately run the 150 GHz and 90 GHz mapmaking, then combine the
    # results and return them.
    if read_from_idf and idf_band is None:
        if do_left_right:
            # Do mapmaking for each of the bands in turn.
            analysis_result150, analysis_result150_left, analysis_result150_right = analyzeData(time_interval, idf_band=150)
            analysis_result90, analysis_result90_left, analysis_result90_right = analyzeData(time_interval, idf_band=90)
        
            # Combine the dictionaries of maps.
            analysis_result150.update(analysis_result90)
            analysis_result150_left.update(analysis_result90_left)
            analysis_result150_right.update(analysis_result90_right)
        
            # Return the combined results.
            return analysis_result150, analysis_result150_left, analysis_result150_right
                   
        else:
            # Do mapmaking for each of the bands in turn.
            analysis_result150 = analyzeData(time_interval, idf_band=150)
            analysis_result90 = analyzeData(time_interval, idf_band=90)
        
            # Combine the dictionaries of maps.
            analysis_result150.update(analysis_result90)
        
            # Return the combined results.
            return analysis_result150 


    # Check if we're reading from an IDF or going directly from archive files.
    if read_from_idf:
        if my_name == 'ra0hdec-57.5':
            print '2013 field...'
            idf_filename = os.path.join(directories['idf_dir'],'data','%s_idf_%s_%03ighz.h5' % (my_name, time_interval[0].file, idf_band))
        else:
            idf_filename = os.path.join(directories['idf_dir'],my_name,'data','%s_idf_%s_%03ighz.h5' % (my_name, time_interval[0].file, idf_band))
        data = read(idf_filename, timeit=True)
        if not data:
            raise ValueError("I couldn't find an IDF for the %s observation taken at %s. (No file at %s .)"
                             % (my_name, time_interval[0].archive, idf_filename))
        # test if data is in ra range you want
        if ra_cut:
            if (data.observation.mean_ra < ra_cut[0]) or (data.observation.mean_ra > ra_cut[1]):   
                raise ValueError("Skipping this IDF b/c wrong RA range: %s observation taken at %s. (file at %s .)"
                             % (my_name, time_interval[0].archive, idf_filename))

        data._readConfigs() # We need the auxdata_keys information.
        
        # The IDF generator may have flagged some scans in some timestreams as bad. 
        # With such a huge source, the RMS flagger's output is suspect (in fact, very 
        # bad over the source), so remove all timestream flags.
        for scan in data.scan:
            scan.is_bad_channel[:]=False

        #Correct pointing after the fact.
        if correct_pointing:
            op.applyOfflinePointing(data, model='SPT', overwrite_global=True,
                                flags=flags, overwrite=True)

        #Grab the thermometry and metrology data.
        thermo_data = {'tracker.encoder_off':np.median(data.antenna.track_enc_off, axis=1), \
                       'tracker.horiz_mount':np.median(data.antenna.track_hor_mnt, axis=1), \
                       'tracker.horiz_off':np.median(data.antenna.track_hor_off, axis=1), \
                       #'tracker.tilt_xy_avg':np.median(data.antenna.track_tilt_xy_avg, axis=1), \
                       'tracker.linear_sensor_avg':np.median(data.antenna.track_linsens_avg, axis=1), \
                       'scu.temp':np.median(data.antenna.scu_temp, axis=1),
                       'scu.benchoff':data.antenna.scu_benchoff[:,0],
                       'observation.temp_avg':data.observation.temp_avg,
                       'observation.pressure_avg':data.observation.pressure_avg,
                       'observation.mean_az':data.observation.mean_az,
                       'observation.mean_el':data.observation.mean_el,
                       'observation.wind_dir_avg':data.observation.wind_dir_avg,
                       'observation.wind_speed_avg':data.observation.wind_speed_avg
                       }

        this_filename = "%s_%s_thermolinear.pkl" % (my_name, time_interval[0].file)
        filename_out_dir = os.path.join(directories['output_dir'], my_name, subdirectory_name)
        filename = os.path.join(filename_out_dir, this_filename)

        pk.dump(thermo_data, open(filename, 'w'))

    else:
        data = SPTDataReader(time_interval[0], time_interval[1], 
                             experiment=experiment,
                             master_configfile=master_config)
        data.readData(obstype=my_name, **readout_kwargs)

        #Correct pointing after the fact.
        if correct_pointing:
            op.applyOfflinePointing(data, model='SPT', overwrite_global=True,
                                flags=flags, overwrite=True)

        #Grab the thermometry and metrology data.
        thermo_data = {'tracker.encoder_off':np.median(data.antenna.track_enc_off, axis=1), \
                       'tracker.horiz_mount':np.median(data.antenna.track_hor_mnt, axis=1), \
                       'tracker.horiz_off':np.median(data.antenna.track_hor_off, axis=1), \
                       #'tracker.tilt_xy_avg':np.median(data.antenna.track_tilt_xy_avg, axis=1), \
                       'tracker.linear_sensor_avg':np.median(data.antenna.track_linsens_avg, axis=1), \
                       'scu.temp':np.median(data.antenna.scu_temp, axis=1),
                       'scu.benchoff':data.antenna.scu_benchoff[:,0],
                       'observation.temp_avg':data.observation.temp_avg,
                       'observation.pressure_avg':data.observation.pressure_avg,
                       'observation.mean_az':data.observation.mean_az,
                       'observation.mean_el':data.observation.mean_el,
                       'observation.wind_dir_avg':data.observation.wind_dir_avg,
                       'observation.wind_speed_avg':data.observation.wind_speed_avg
                       }

        this_filename = "%s_%s_thermolinear.pkl" % (my_name, time_interval[0].file)
        filename_out_dir = os.path.join(directories['output_dir'], my_name, subdirectory_name)
        filename = os.path.join(filename_out_dir, this_filename)

        pk.dump(thermo_data, open(filename, 'w'))

    
    if mask_radius:
        if force_source_location:
            ra, dec = force_source_location[0], force_source_location[1]
        else:
            # Find the location of the source. Do a linear fit subtraction now. Make an IDF
            # for the sourcefinding, as a quick and easy way to be able to do light filtering
            # on the sourcefinding map without filtering the real data.
            temp_idf = (data.copy() if read_from_idf else mapidf.MapIDF(data))
            finder_map = quicklook.quickmap(temp_idf, good_bolos=['flagged','pointing'],
                                            reso_arcmin=0.25,
                                            proj=5,
                                            map_shape=[0.8,1.],
                                            map_center=map_center,
                                            t_only=True,
                                            inverse_noise_weighted_map=True,
                                            timestream_filtering={'poly_order':1, 
                                                                  'dynamic_source_mask':True,
                                                                  'outlier_mask_nsigma':3.0})
            temp_idf = None # Discard the temporary IDF now that we're done with it.
            # Discard the map we're not creating. We only need one of them.
            finder_map = finder_map[str(idf_band)] if idf_band is not None else finder_map['150'] 
        
            # Find the RA and dec of the maximum point in the map.
            ra, dec = finder_map.pix2Ang(np.unravel_index(np.argmax(np.abs(finder_map.map)), finder_map.shape))
    

        # Make a temporary pointsource config file, and write it in the arguments
        # to be passed to quickmap.
        ptsrc_configfile = tempfile.NamedTemporaryFile(mode='w', suffix='.txt')
        ptsrc_configfile.write('1 %f %f %f' % (ra, dec, mask_radius))
        ptsrc_configfile.flush()
        print "  Writing temporary pointsource config file, %s . Contents:" % ptsrc_configfile.name
        print '1 %f %f %f\n' % (ra, dec, mask_radius)
        analysis_function_kwargs['timestream_filtering']['pointsource_file'] = ptsrc_configfile.name 
 
        
    for func, kwargs in zip(preprocessing_function, preprocessing_function_kwargs):
        func(data, **kwargs)
        
    print "  Processing sum map.\n"   
    print "  analysis_function_kwargs : %s" % str(analysis_function_kwargs)
    analysis_result = analysis_function(data, **analysis_function_kwargs)
    
    if do_left_right:
        # Now we've done filtering once, no need to do it again.
        lr_analysis_function_kwargs = analysis_function_kwargs.copy() # Don't alter the original kwargs!
        lr_analysis_function_kwargs['timestream_filtering'] = {}
        
        print "  Processing leftgoing map.\n"   
        analysis_result_left = analysis_function(data, use_leftgoing=True, **lr_analysis_function_kwargs)
        
        print "  Processing rightgoing map.\n"   
        analysis_result_right = analysis_function(data, use_leftgoing=False, **lr_analysis_function_kwargs)
    
    if mask_radius:
        ptsrc_configfile.close()
    
    if read_from_idf:
        # If this is an IDF, it's only got one band. Delete maps from the other(s) so we
        # don't save empty maps.
        if str(data.band) not in analysis_result:
            raise RuntimeError("I was expecting to see %s in the output maps, but it's not there!" % str(data.band))
        for band in analysis_result.keys():
            if band!=str(data.band):
                del analysis_result[band]
                if do_left_right:
                    del analysis_result_left[band]
                    del analysis_result_right[band]
    
    if do_left_right:
        return analysis_result, analysis_result_left, analysis_result_right
    else:
        return analysis_result

def writeData(data, time_interval, overwrite=False):
    # Create filenames.
    try:
        start_time = time_interval[0]
    except TypeError:
        start_time = time_interval
        
    filename_out_dir = os.path.join(directories['output_dir'], my_name, subdirectory_name)
    if not os.path.exists(filename_out_dir): os.makedirs(filename_out_dir)
    
    # Split up the input into the three maps.
    if do_left_right:
        map_sum, map_left, map_right = data
    else:
        map_sum = data
    
    # Write files to disk.
    filenames, filenames_left, filenames_right = struct(), struct(), struct()
    for band in map_sum:
        this_filename = "%s_%s_%03ighz.%s" % (my_name, start_time.file, int(band), ('fits' if use_fits else 'hdf5'))
        filenames[band] = os.path.join(filename_out_dir, this_filename)
        
        if do_left_right:
            this_filename = "%s_leftgoing_%s_%03ighz.%s" % (my_name, start_time.file, int(band), ('fits' if use_fits else 'hdf5'))
            filenames_left[band] = os.path.join(filename_out_dir, this_filename)
            
            this_filename = "%s_rightgoing_%s_%03ighz.%s" % (my_name, start_time.file, int(band), ('fits' if use_fits else 'hdf5'))
            filenames_right[band] = os.path.join(filename_out_dir, this_filename)
    
    for band, _map in map_sum.iteritems():
        if use_fits:
            _map.writeToFITS(filenames[band], overwrite=overwrite)
            if do_left_right:
                map_left[band].writeToFITS(filenames_left[band], overwrite=overwrite)
                map_right[band].writeToFITS(filenames_right[band], overwrite=overwrite)
        else:
            _map.writeToHDF5(filenames[band], overwrite=overwrite, use_compression=False)
            if do_left_right:
                map_left[band].writeToHDF5(filenames_left[band], overwrite=overwrite, use_compression=False)
                map_right[band].writeToHDF5(filenames_right[band], overwrite=overwrite, use_compression=False)

    return filenames.get('150', filenames[0])
