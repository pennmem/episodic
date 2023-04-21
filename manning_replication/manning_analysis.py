"""
manning_analysis.py

Author:
    Daniel Schonhaut
    Computational Memory Lab
    University of Pennsylvania
    daniel.schonhaut@gmail.com

Description: 
    Functions that perform the core data processing and analyses for the Manning
    et al., J Neurosci 2009 paper.

Last Edited: 
    6/21/19
"""
import sys
import os
import glob
import pickle
from time import time
from time import strftime
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
import statsmodels.api as sm

import mne
from ptsa.data.TimeSeriesX import TimeSeries
from ptsa.data.filters import ButterworthFilter
from ptsa.data.filters import MorletWaveletFilter
import manning_utils

def epoch_fr_power(subj_sess,
                   lfp_proc, 
                   fr_df,
                   chan_to_clus,
                   chans=None,
                   log_power=False,
                   z_power='',
                   freq_params={'low': 2, 'high': 200, 'num': 50}, 
                   epoch_size=1000, 
                   epoch_cut=3,
                   input_dir='/data3/scratch/dscho/frLfp/data/lfp/morlet',
                   output_dir='/data3/scratch/dscho/frLfp/data/epoch',
                   power_file_suffix='_width5_2-200Hz-50log10steps',
                   load_existing=True,
                   save_files=False,
                   log_dir='/data3/scratch/dscho/frLfp/logs'):
    """Divide each channel into epochs, and calculate the mean firing rate and
    power (at each frequency) within each epoch.
    
    Also calculate narrowband and broadband power for each epoch, using a
    robust regression model for the latter.
    
    Parameters
    ----------
    subj_sess : str
        e.g. 'U367_env2'
    lfp_proc : ptsa.data.timeseries.TimeSeries
        An n_channels x n_timepoints TimeSeries of the processed LFP data.
    fr_df : pandas.core.frame.DataFrame
        A DataFrame version of session_spikes, of sorts. n_clusters long and
        stores spikes, fr, and interp_mask for each cluster across all channels 
        in the session, along with some metadata.
    chan_to_clus : collections.OrderedDict
        Mapping between each unique channel in the recording session and a list
        of corresponding (across channel) cluster numbers (only has keys for
        channels with spikes!).
    chans : list or numpy.ndarray
        A list of channel names to process. Default is to use 
        all channels in lfp_proc.
    log_power : bool
        If True, power values are log10 transformed after epoching.
    z_power : str
        'withinfreq' -- Power values are Z-scored across epochs, 
            separately for each frequency.
        'acrossfreq' -- Power values are Z-scored across epochs 
            and frequencies. 
        Transform is applied after epoching. If log_power is True, 
        log transform is done before Z-scoring.
    freq_params : dict
        Keys must be 'low', 'high', and 'num'. Gives the frequency
        range and number of frequencies used in calculating power
        from LFP data through Morlet decomposition.
    epoch_size : int
        The number of timepoints that will comprise each epoch.
    epoch_cut : int
        The number of epochs to cut from the beginning and end
        of the recording session (to remove edge effects).
    input_dir : str
        Directory where input data are loaded.
    output_dir : str
        Directory where output files are loaded/saved.
    power_file_suffix : str
        The suffix to use in selecting power files to load from
        input_dir.
    load_existing : bool
        If True and output files exist, load and return them
        rather than reprocessing epochs.
    save_files : bool
        If True, output TimeSeries objects are saved as hdf5 files.
        If files already exist, load_existing is False, and save_files
        is True, then existing files are overwritten!
    log_dir : str
        Directory where the log file is saved.
        
    Returns
    -------
    epoch_fr : ptsa.data.timeseries.TimeSeries
        Mean firing rate for each epoch, for each cluster in the
        session. Dims are cluster x epoch.
    epoch_power : ptsa.data.timeseries.TimeSeries
        Mean power for each frequency, for each epoch, for each
        channel in the session. Dims are channel x epoch x freq.
    epoch_band_power : ptsa.data.timeseries.TimeSeries
        Mean power for frequencies within delta, theta, alpha, beta,
        and gamma bands; along with the slope and intercept of a robust
        robust regression model predicting power at each frequency for
        a given channel/epoch. Broadband power is also reported as the
        mean predicted value from the regression fit. Dims are 
        channel x epoch x freq.
    """
    # Setup logging.
    logger = logging.getLogger(sys._getframe().f_code.co_name)
    logger.handlers = []
    log_f = os.path.join(log_dir, '{}_{}_{}.log'.format(subj_sess, 
                                                        sys._getframe().f_code.co_name, 
                                                        strftime('%m-%d-%Y-%H-%M-%S')))
    handler = logging.FileHandler(log_f)
    handler.setLevel(logging.DEBUG)
    formatting = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                   datefmt='%m-%d-%Y %H:%M:%S')
    handler.setFormatter(formatting)
    logger.addHandler(handler)
    
    # Load epoched data if it exists.
    # -----------------------------------
    
    # Generate tag to explain processing steps (done in L->R order).
    fstr = ''
    if log_power:
        fstr += '_log10'
    if z_power:
        fstr += '_Z-{}'.format(z_power)
    if lfp_proc.samplerate >= epoch_size:
        epoch_rate_str = '{:.0f}'.format(lfp_proc.samplerate.data / epoch_size)
    else:
        epoch_rate_str = '{:.02f}'.format(lfp_proc.samplerate.data / epoch_size)
    process_tag = ('power' + power_file_suffix 
                   + '_epoch-{}Hz-cut{}'.format(epoch_rate_str, epoch_cut)
                   + fstr)
    
    epoch_fr_file = os.path.join(output_dir, '{}_fr_epoch-{}Hz-cut{}.hdf'
                                 .format(subj_sess, epoch_rate_str, epoch_cut))
    epoch_power_file = os.path.join(output_dir, '{}_{}.hdf'.format(subj_sess, process_tag))
    epoch_band_power_file = os.path.join(output_dir, '{}_{}_freqbands.hdf'.format(subj_sess, process_tag))
    files_exist = (os.path.exists(epoch_fr_file) 
                   and os.path.exists(epoch_power_file) 
                   and os.path.exists(epoch_band_power_file))
    if files_exist and load_existing:
        logger.info('Loading epoched spike and power data.')
        epoch_fr = TimeSeries.from_hdf(epoch_fr_file)
        epoch_power = TimeSeries.from_hdf(epoch_power_file)
        epoch_band_power = TimeSeries.from_hdf(epoch_band_power_file)
        return epoch_fr, epoch_power, epoch_band_power
    
    # Process data into epochs.
    # -------------------------
    start_time = time()
    logger.info('Processing spike and power data into epochs.')
    
    # Get channels and corresponding clusters to process.
    if chans is None:
        chans = lfp_proc.channel.data
    clusters = []
    for chan in chans:
        if chan in chan_to_clus.keys():
            clusters += chan_to_clus[chan]
    
    # Get time bins to divide the data into 500ms epochs, cutting
    # out the first and last (+remainder) 3000 timepoints (1500ms).
    n_timepoints = lfp_proc.shape[1]    
    time_bins = manning_utils.get_epochs(np.arange(n_timepoints, dtype=np.int32), 
                                         epoch_size=epoch_size, 
                                         cut=epoch_cut)

    # Get wavelet frequencies and assign them to bands of interest.
    freqs, freq_bands = manning_utils.get_freqs(low=freq_params['low'], 
                                                high=freq_params['high'], 
                                                num=freq_params['num'])
    # X is the independent variable for the robust regression.
    X = sm.add_constant(np.log10(freqs))
    
    # Get the mean firing rate for each epoch, for each cluster.
    # ----------------------------------------------------------
    epoch_fr = []
    cluster_channels = []
    cluster_locs = []
    for clus in clusters:
        # Get the mean firing rate (in Hz) for each epoch.
        cluster_channels.append(fr_df.loc[fr_df.clus==clus, 'chan'].iat[0])
        cluster_locs.append(fr_df.loc[fr_df.clus==clus, 'location'].iat[0])
        clus_fr = fr_df.loc[fr_df.clus==clus, 'fr'].iat[0]
        epoch_fr_row = []
        for epoch_start, epoch_stop in time_bins:
            epoch_fr_row.append(2 * np.sum(clus_fr[epoch_start:epoch_stop]))
        epoch_fr.append(epoch_fr_row)

    epoch_fr = TimeSeries(np.array(epoch_fr), name=lfp_proc.name,
                          dims=['cluster', 'epoch'],
                          coords={'cluster': clusters,
                                  'epoch': np.arange(len(time_bins)),
                                  'samplerate': lfp_proc.samplerate.data / epoch_size},
                          attrs={'epoch_bins': time_bins,
                                 'channel': cluster_channels,
                                 'location': cluster_locs})
    
    # Get narrowband and broadband power for each epoch, for each channel.
    # --------------------------------------------------------------------
    epoch_power = []
    epoch_band_power = []
    for i, chan in enumerate(chans):
        if i % 8 == 0:
            logger.info('Loading power data for {} channel {}.'.format(subj_sess, chan))
        f = os.path.join(input_dir, '{}_ch{}_power{}.hdf'.format(subj_sess, chan, power_file_suffix))
        power = TimeSeries.from_hdf(f).data.squeeze() # frequency x time
        epoch_power_row = []
        epoch_band_power_row = []
        # For timepoints in each epoch, get the mean power over time, 
        # at each frequency.
        for epoch_start, epoch_stop in time_bins:
            epoch_power_row.append(np.mean(power[:, epoch_start:epoch_stop], axis=1))
        
        epoch_power_row = np.array(epoch_power_row) # n_epochs x n_freqs
        
        # Transform epoched power values.
        if log_power:
            epoch_power_row = np.log10(epoch_power_row)
            if i == 0:
                logger.info('Log transforming epoched power values.')
        if z_power == 'withinfreq':
            epoch_power_row = (epoch_power_row - np.mean(epoch_power_row, axis=0)) / np.std(epoch_power_row, axis=0)
            if i == 0:
                logger.info('Z-scoring epoched power values within each frequency.')
        elif z_power == 'acrossfreq':
            epoch_power_row = (epoch_power_row - np.mean(epoch_power_row)) / np.std(epoch_power_row)
            if i == 0:
                logger.info('Z-scoring epoched power values across frequencies.')
            
        for epoch in range(epoch_power_row.shape[0]):
            # Get mean delta, theta, alpha, beta, and gamma power.   
            delta_power = np.mean(epoch_power_row[epoch, (freqs<4)])
            theta_power = np.mean(epoch_power_row[epoch, (freqs>=4) & (freqs<8)])
            alpha_power = np.mean(epoch_power_row[epoch, (freqs>=8) & (freqs<12)])
            beta_power = np.mean(epoch_power_row[epoch, (freqs>=12) & (freqs<30)])
            gamma_power = np.mean(epoch_power_row[epoch, (freqs>=30)])

            # Fit a robust linear regression to estimate the intercept 
            # and slope (broadband tilt) of the power spectrum. Broadband
            # power is the mean of the predicted values.
            y = epoch_power_row[epoch, :]
            huber_t = sm.RLM(y, X, M=sm.robust.norms.HuberT())
            hub_results = huber_t.fit()
            intercept, slope = hub_results.params
            epoch_band_power_row.append(
                    [delta_power, theta_power, alpha_power, beta_power, gamma_power, 
                     intercept, slope, np.mean(hub_results.predict())]
                )

        epoch_power.append(epoch_power_row) # n_channels x n_epochs x n_freqs
        epoch_band_power.append(epoch_band_power_row) # n_channels x n_epochs x (n_freq_bands + 3)

    epoch_power = TimeSeries(np.array(epoch_power), name=lfp_proc.name, 
                             dims=['channel', 'epoch', 'freq'],
                             coords={'channel': chans,
                                     'epoch': np.arange(len(time_bins)),        
                                     'freq': freqs,
                                     'samplerate': lfp_proc.samplerate.data / epoch_size},
                             attrs={'epoch_bins': time_bins,
                                    'process_tag': process_tag})
    
    epoch_band_power = TimeSeries(np.array(epoch_band_power), name=lfp_proc.name, 
                                  dims=['channel', 'epoch', 'freq'],
                                  coords={'channel': chans,
                                          'epoch': np.arange(len(time_bins)),        
                                          'freq': (list(freq_bands.keys()) 
                                                   + ['intercept', 'bband_tilt', 'bband_power']),
                                          'samplerate': lfp_proc.samplerate.data / epoch_size},
                                  attrs={'epoch_bins': time_bins,
                                         'process_tag': process_tag})
    
    if save_files:
        epoch_fr.to_hdf(epoch_fr_file)
        epoch_power.to_hdf(epoch_power_file)
        epoch_band_power.to_hdf(epoch_band_power_file)
        logger.info('Saved epoched data to files:\n\t{}\n\t{}\n\t{}.'
                     .format(epoch_fr_file, epoch_power_file, epoch_band_power_file))

    duration = time() - start_time
    logger.info('Done in {} secs.'.format(int(duration)))
        
    return epoch_fr, epoch_power, epoch_band_power
   
def get_fr_df(subj_sess, session_spikes):
    """Format session_spikes as a DataFrame.
    
    Return fr_df, clus_to_chan, chan_to_clus.
    """
    fr_dat = []
    clus = 0
    for chan in session_spikes.keys():
        for chan_clus in range(len(session_spikes[chan]['spikes'])):
            fr_dat.append([subj_sess, 
                           chan, 
                           session_spikes[chan]['location'], 
                           session_spikes[chan]['pct_interp'], 
                           clus, 
                           chan_clus,
                           session_spikes[chan]['fr'][chan_clus, :], 
                           session_spikes[chan]['spikes'][chan_clus, :],
                           session_spikes[chan]['interp_mask']])
            clus += 1
            
    col_names = ['subj_sess', 'chan', 'location', 'pct_interp', 
                 'clus', 'chan_clus', 'fr', 'spikes', 'interp_mask']
    fr_df = pd.DataFrame(fr_dat, columns=col_names)
    fr_df.insert(6, 'mean_fr', fr_df.spikes.apply(np.mean) * 2000) 
    clus_to_chan = OrderedDict(fr_df[['clus', 'chan']].values)
    chan_to_clus = OrderedDict(fr_df.groupby('chan', sort=False).clus.apply(list))  
    
    return fr_df, clus_to_chan, chan_to_clus
     
def preprocess_session(subj_sess, subj_df, subj_df_file, overwrite=False, 
                       log_dir='/data3/scratch/dscho/frLfp/logs'):
    """Process the raw (2000 Hz) LFP data for each channel from the session.
    
    A 4th-order Butterworth filter is applied from 58-62 Hz. 
    Spikes are then removed from the LFP by linear interpolation. 
    Raw LFP and processed LFP data are saved as n_channels by n_timepoints 
    TimeSeries objects in hdf5 format.
    
    Spike time data are loaded and processed (see Returns:'session_spikes').
    
    Parameters
    ----------
    subj_sess : str
        e.g. 'U367_env2'
    subj_df : pandas.core.frame.DataFrame
        The DataFrame of subject metadata for each channel.
    subj_df_file : str
        File path to the subj_df; note, NOT where the subj_df output is saved.
        A separate excel file with rows just for the subj_sess given is saved
        in os.path.join(os.path.dirname(subj_df_file), 'metadata').
    overwrite : bool
        If True and output files already exist, the data are reprocessed 
        and 3 output files are saved over (lfp_proc, sesssion_spikes, 
        and subj_df). If False and output files already exist, these files
        are loaded in memory and returned.
    log_dir : str
        Directory where the log file is saved.
        
    Returns
    -------
    subj_df : pandas.core.frame.DataFrame
        Subjects DataFrame for the Manning replication project. Each row has 
        info for one channel, and the total length is the sum of all channels
        from all sessions from all subjects in the study.
    lfp_raw : ptsa.data.timeseries.TimeSeries
        An n_channels x n_timepoints TimeSeries of the raw LFP data.
    lfp_proc : ptsa.data.timeseries.TimeSeries
        An n_channels x n_timepoints TimeSeries of the processed LFP data.
    session_spikes : collections.OrderedDict[collections.OrderedDict]
        Dictionary keys are channel names for the session, and each value is a
        dictionary with the following keys:
            'location' : str
                Location of channel in the brain.
            'pct_interp' : float
                Percent of timepoints in the processed LFP data with 
                interpolated values.
            'spike_times' : dict[numpy.ndarray]
                n_clusters x n_spikes timepoint of each spike. Dict keys are
                the integer indices of each cluster in the arbitrary order
                that they appeared in the original wave_clus output file.
            'spikes' : numpy.ndarray
                n_clusters x n_timepoints Boolean array of spikes (i.e. spike 
                was present or absent for a given cluster at a given timepoint).
            'fr' : numpy.ndarray
                n_clusters x n_timepoints array of firing rates, calculated by
                convolving the 'spikes' array with a Gaussian kernel.
            'spike_lfp' : list[numpy.ndarray] 
                The list has n_clusters elements, in the same order as for 
                'spike_times', 'spikes', and 'fr'. The list elements are 
                n_spikes x 21 arrays that contain the raw LFP trace from -2
                to 8 ms after each spike event for a given cluster.
            'interp_mask' : numpy.ndarray
                n_timepoints long Boolean array of values to interpolate around
                each spike, combining across clusters for a given channel.
    fr_df : pandas.core.frame.DataFrame
        A DataFrame version of session_spikes, of sorts. n_clusters long and
        stores spikes, fr, and interp_mask for each cluster across all channels 
        in the session, along with some metadata.
    clus_to_chan : collections.OrderedDict
        Mapping between each unique cluster in the recording session and its
        corresponding channel.
    chan_to_clus : collections.OrderedDict
        Mapping between each unique channel in the recording session and a list
        of corresponding (across channel) cluster numbers (only has keys for
        channels with spikes!).
    """                                   
    # Get dirs.
    dirs = manning_utils.get_dirs()
    data_dir = dirs['data']
    
    # Get channel indices for the recording session.
    df_inds = subj_df.query("(subj_sess=='{}')".format(subj_sess)).index.values
    
    # Figure out if processed files already exist.
    proc_lfp_file = subj_df.at[df_inds[0], 'proc_lfp_file']
    raw_lfp_file = os.path.join(os.path.dirname(proc_lfp_file), 
                                '{}_raw.hdf'.format(subj_sess))
    session_spikes_file = subj_df.at[df_inds[0], 'session_spikes_file']
    subj_df_file = os.path.join(os.path.dirname(subj_df_file), 'metadata', 
                                'subj_df_{}.xlsx'.format(subj_sess))
    proc_files_exist = np.all((os.path.exists(raw_lfp_file),
                               os.path.exists(proc_lfp_file), 
                               os.path.exists(session_spikes_file),
                               os.path.exists(subj_df_file)))
    if proc_files_exist and not overwrite:
        print('Loading subj_df, raw LFP, processed LFP, and spike data for {}.\n'.format(subj_sess))
        
        # Load the subject DataFrame.
        subj_df = pd.read_excel(subj_df_file, converters={'chan': str})
        
        # Load raw LFP.
        lfp_raw = TimeSeries.from_hdf(raw_lfp_file)
        
        # Load processed LFP (notch-filtered and spike interpolated).
        lfp_proc = TimeSeries.from_hdf(proc_lfp_file)
        
        # Load session_spikes.
        with open(session_spikes_file, 'rb') as f:
            session_spikes = pickle.load(f)   
            
        # Create a DataFrame with n_clusters rows to hold smoothed firing rate data
        # (array of firing rates at each timepoint) for each cluster
        fr_df, clus_to_chan, chan_to_clus = get_fr_df(subj_sess, session_spikes)
    else:
        print('Processing LFP and spike data for {}.\n'.format(subj_sess))
        config = manning_utils.get_config()
        sampling_rate = config['samplingRate']
        ms_before = 2.0
        ms_after = 4.0
        steps_before = int(2 * ms_before) # 2 ms
        steps_after = int((2 * ms_after) + 1) # 4 ms
        
        # Process the spike data.
        subj_df, session_spikes, fr_df, clus_to_chan, chan_to_clus = process_spikes(
                subj_sess, subj_df, steps_before, steps_after, sampling_rate
            )
        
        # Process the raw LFP data.
        lfp_raw, lfp_proc = process_lfp(subj_sess=subj_sess, 
                                        subj_df=subj_df, 
                                        sampling_rate=sampling_rate,
                                        notch_freqs=[60, 120, 180],
                                        interpolate=True,
                                        session_spikes=session_spikes, 
                                        ms_before=ms_before, 
                                        ms_after=ms_after)
        
        # Save raw LFP data as an hdf5 file.
        lfp_raw.to_hdf(raw_lfp_file)
        print('Saved raw LFP data for {} to file: {}'.format(subj_sess, raw_lfp_file))
        
        # Save processed LFP data as an hdf5 file.
        lfp_proc.to_hdf(proc_lfp_file)
        print('Saved processed LFP data for {} to file: {}'.format(subj_sess, proc_lfp_file))

        # Save session_spikes as a pickle file.
        with open(session_spikes_file, 'wb') as f:
            pickle.dump(session_spikes, f, pickle.HIGHEST_PROTOCOL)
        print('Saved spikes data for {} to file: {}'.format(subj_sess, session_spikes_file))

        # Save the subj DataFrame (keeping only rows for the subj)
        subj_df = subj_df.query("(subj_sess=='{}')".format(subj_sess))
        writer = pd.ExcelWriter(subj_df_file)
        subj_df.to_excel(writer, index=True)
        writer.save()
        print('Saved subj_df to file: {}'.format(subj_df_file))                
    
    return subj_df, lfp_raw, lfp_proc, session_spikes, fr_df, clus_to_chan, chan_to_clus
    
def process_lfp(subj_sess, 
                subj_df, 
                chans=None, # str channels starting at '1'
                sampling_rate=2000,
                resampling_rate=0,
                notch_freqs=[60, 120, 180],
                high_pass=None,
                low_pass=None,
                interpolate=False, 
                session_spikes=None, 
                ms_before=2, 
                ms_after=4):
    """Notch filter the raw LFP data and linearly interpolate around spikes.
    
    Returns
    -------
    lfp_raw : ptsa.data.timeseries.TimeSeries
        An n_channels x n_timepoints TimeSeries of the raw LFP data.
    lfp_proc : ptsa.data.timeseries.TimeSeries
        An n_channels x n_timepoints TimeSeries of the processed LFP data. 
    """
    if sampling_rate == resampling_rate:
        resampling_rate = 0
    if resampling_rate > sampling_rate:
        print('CANNOT UPSAMPLE')
        assert False
    if interpolate and resampling_rate > 0:
        print('FIX SPIKE INTERPOLATION OF RESAMPLED DATA!')
        assert False
        
    df = subj_df.query("(subj_sess=='{}')".format(subj_sess))
    if chans is None:
        chans = df.chan.tolist()
    raw_files = df.query("(chan=={})".format(chans)).raw_lfp_file.tolist()
    lfp_dat = np.array([np.fromfile(f, dtype='float32') for f in raw_files])
    lfp_raw = TimeSeries(lfp_dat, name=subj_sess, 
                         dims=['channel', 'time'],
                         coords={'channel': chans,
                                 'time': np.arange(lfp_dat.shape[1]),
                                 'samplerate': sampling_rate})
    lfp_proc = lfp_raw.copy()
    process_tag = ''
    
    # Resample.
    if resampling_rate > 0:
        lfp_proc = resample(lfp_proc, resampling_rate)
        sampling_rate = resampling_rate
        process_tag += 'resample-{}Hz_'.format(int(sampling_rate))
    
    # Notch filter using an FIR filter.
    if notch_freqs:
        dat = mne.filter.notch_filter(np.float64(lfp_proc.copy().data), 
                                      Fs=sampling_rate, 
                                      freqs=notch_freqs,
                                      phase='zero',
                                      verbose=False)
        lfp_proc = TimeSeries(dat, name=subj_sess, 
                              dims=['channel', 'time'],
                              coords={'channel': chans,
                                      'time': lfp_proc.time.data,
                                      'samplerate': sampling_rate})
    for freq in notch_freqs:                              
        process_tag += 'notch{}Hz_'.format(freq)

    # Low pass filter the data.
    if low_pass:
        print('Low-pass feature not added yet!')
        #process_tag += 'lowpass{}Hz_'.format(low_pass)
        
    # High pass filter the data.
    if high_pass:
        print('High-pass feature not added yet!')
        #mne.filter.filter_data(np.float64(lfp_proc.data), sfreq=2000, l_freq=58, h_freq=62)
        #process_tag += 'highpass{}Hz_'.format(high_pass)
    
    # Linearly interpolate LFP around spikes, for channels with clusters.
    if interpolate:
        interp_chans = session_spikes.keys()
        lfp_proc_dat = lfp_proc.data
        for chan in interp_chans:
            interp_mask = session_spikes[chan]['interp_mask']
            keep_inds = np.where(interp_mask==0)[0]
            fill_inds = np.where(interp_mask==1)[0]
            f = interp1d(keep_inds, lfp_proc_dat[chans.index(chan), keep_inds],
                         kind='linear', fill_value='extrapolate')
            
            def apply_interp(arr, inds, f):
                arr[inds] = f(inds)
                return arr
            
            lfp_proc_dat[chans.index(chan), :] = apply_interp(
                lfp_proc_dat[chans.index(chan), :], 
                fill_inds, 
                f)
        process_tag += 'spikeinterpolation-{}to{}ms_'.format(ms_before, ms_after)
         
    if process_tag:
        process_tag = process_tag[:-1]
    else:
        process_tag = 'same as lfp_raw'
        
    lfp_proc.attrs={'process_tag': process_tag}
        
    return lfp_raw, lfp_proc
    
    
def process_spikes(subj_sess, subj_df, steps_before, steps_after,
                   sampling_rate=2000.0, fr_half_width=500.0):
    """Find and process channels in a session with spike clusters.
    
    Each cluster contains the spike times of one neuron/unit over
    the course of the recording session. Most channels have 0-3 clusters.

    From lists of spike times for each cluster (aligned to the raw LFP data)
    create a spike train and smoothed firing rate, and get the LFP waveform
    surrounding each spike. Aggregate this data into the session_spikes dict.
    
    """
    # Get channel indices for the recording session.
    df_inds = subj_df.query("(subj_sess=='{}')".format(subj_sess)).index.values

    # Get a Gaussian window for calculating firing rate.
    v = np.zeros(2001)
    v[1000] = 1
    g_std = (((fr_half_width * sampling_rate) / 1000) 
             / (2 * np.sqrt(2 * np.log(2))))
    g_win = gaussian_filter(v, g_std)

    # Iterate over channels in the recording session.
    session_spikes = OrderedDict()
    raw_lfp_file = subj_df.at[df_inds[0], 'raw_lfp_file']
    n_timepoints = len(np.fromfile(raw_lfp_file, dtype='float32'))
    for i, df_ind in enumerate(df_inds):
        chan = subj_df.at[df_ind, 'chan']

        # Load the spike times file and figure out how many clusters there are.
        # spikeTimes : array of arrays
        #     The parent array has each cluster for the channel, as determined 
        #     by the wave_clus package. The child array gives spike times for 
        #     the cluster aligned to the LFP data (e.g. a spike time of 60 would 
        #     indicate a spike at the 60th LFP time point).
        spike_times_file = subj_df.at[df_ind, 'spike_times_file']
        spikes_mat = sio.loadmat(spike_times_file)

        # Some roundabout code to get around the problem that
        # spikeTimes sometimes has empty vectors for a cluster. 
        if len(spikes_mat['spikeTimes']) > 0:
            arr_in = spikes_mat['spikeTimes'][0]
            spike_times = OrderedDict()
            clus = 0
            for m in range(len(arr_in)):
                if len(arr_in[m]) > 1:
                    spike_times[clus] = arr_in[m].squeeze()
                    clus += 1
            n_clusters = len(spike_times)
        else:
            n_clusters = 0

        # Add channel location, cluster number, and session duration to subj_df.
        location = spikes_mat['anatlabel'][0]
        subj_df.at[df_ind, 'location'] = location
        subj_df.at[df_ind, 'n_clusters'] = n_clusters
        subj_df.at[df_ind, 'sess_duration'] = n_timepoints / sampling_rate

        # Process the spike data for each cluster on the channel 
        # (if there are spikes).
        if n_clusters > 0:
            spikes = np.zeros([len(spike_times), n_timepoints], dtype='bool_')
            fr = []
            spike_lfp = []
            interp_mask = np.zeros(n_timepoints, dtype='bool_')
            for clus in range(len(spike_times)):
                # Make a Boolean spike array aligned to lfp_dat, 
                # with shape (n_clusters, n_lfp_timepoints).
                spikes[clus, spike_times[clus]] = 1

                # Calculate firing rate at each time point by convolving 
                # a Gaussian kernel with the spike array.
                fr.append(np.convolve(spikes[clus, :], g_win, mode='same'))

                # Get the LFP waveform from -2 to 8 ms 
                # surrounding each spike for each cluster.
                raw_lfp_file = subj_df.at[df_ind, 'raw_lfp_file']
                lfp_dat = np.fromfile(raw_lfp_file, dtype='float32')
                spike_lfp.append([])
                for spike_time in spike_times[clus]:
                    spike_time = int(spike_time)
                    if (spike_time > steps_before) and (spike_time < (n_timepoints - steps_after)):
                        spike_lfp[clus].append(lfp_dat[spike_time-steps_before:spike_time+steps_after])
                        interp_mask[spike_time-steps_before:spike_time+steps_after] = 1
                spike_lfp[clus] = np.array(spike_lfp[clus])
            fr = np.array(fr)
            pct_interp = np.sum(interp_mask) / len(interp_mask)
            
            session_spikes[chan] = OrderedDict()
            session_spikes[chan]['location'] = location # location of the channel
            session_spikes[chan]['pct_interp'] = pct_interp # percent of timepoints in the LFP with interpolated values
            session_spikes[chan]['spike_times'] = spike_times # n_cluster x n_spikes timepoint of each spike
            session_spikes[chan]['spikes'] = spikes # n_cluster x n_timepoints Boolean spike vector
            session_spikes[chan]['fr'] = fr # n_cluster x n_timepoints firing rate
            session_spikes[chan]['spike_lfp'] = spike_lfp # 10 ms LFP around each spike, by cluster
            session_spikes[chan]['interp_mask'] = interp_mask # n_timepoints interpolation mask across clusters
    
    # Create a DataFrame with n_clusters rows to hold smoothed firing rate data
    # (array of firing rates at each timepoint) for each cluster
    fr_df, clus_to_chan, chan_to_clus = get_fr_df(subj_sess, session_spikes)
    
    return subj_df, session_spikes, fr_df, clus_to_chan, chan_to_clus
    
def resample(lfp, new_sampling_rate, events=None):
    """Resample LFP data using MNE.
    
    Parameters
    ----------
    lfp : ptsa.data.timeseries.TimeSeries
        TimeSeries object containing an LFP data array.
    new_sampling_rate : int or float
        The sampling rate that the data is resampled to.
    events : np.ndarray
        1-d or 2-d array of event time indices at the
        old sampling rate.
    
    Returns
    -------
    lfp : ptsa.data.timeseries.TimeSeries
        TimeSeries object containing the resampled LFP data 
        array.
    events : np.ndarray
        1-d or 2-d array of event time indices at the
        new sampling rate.   
    """
    chans = lfp.channel.values.tolist()
    old_sampling_rate = lfp.samplerate.data.tolist()
    info = mne.create_info(ch_names=chans, 
                           sfreq=old_sampling_rate, 
                           ch_types=['seeg']*len(lfp.channel.data))
    lfp_mne = mne.io.RawArray(lfp.copy().data, info, verbose=False)
    lfp_mne = lfp_mne.resample(new_sampling_rate)
    
    lfp = TimeSeries(lfp_mne.get_data(), 
                     name=lfp.name,
                     dims=['channel', 'time'],
                     coords={'channel': chans,
                             'time': np.arange(lfp_mne.get_data().shape[1]),
                             'samplerate': new_sampling_rate})
    
    if events is not None:
        ratio = new_sampling_rate / old_sampling_rate
        events = np.round(events * ratio).astype(int)
        return lfp, events
    else:
        return lfp
        
def run_morlet(timeseries, 
               freqs=None, 
               width=5, 
               output=['power', 'phase'],
               log_power=False, 
               z_power=False, 
               z_power_acrossfreq=False, 
               overwrite=False,
               savedir='/data3/scratch/dscho/frLfp/data/lfp/morlet',
               power_file=None,
               phase_file=None,
               verbose=False):
    """Apply Morlet wavelet transform to a timeseries to calculate
    power and phase spectra for one or more frequencies.
    
    Serves as a wrapper for PTSA's MorletWaveletFilter. Can log 
    transform and/or Z-score power across time and can save the 
    returned power and phase timeseries objects as hdf5 files.
    
    Parameters
    ----------
    timeseries : ptsa.data.timeseries.TimeSeries
        The timeseries data to be transformed.
    freqs : numpy.ndarray or list
        A list of frequencies to apply wavelet decomposition over.
    width : int
        Number of waves for each frequency.
    output : str or list or numpy.ndarray
        ['power', 'phase'], ['power'], or ['phase'] depending on
        what output is desired.
    log_power : bool
        If True, power values are log10 transformed.
    z_power : bool
        If True, power values are Z-scored across the time dimension.
        Requires timeseries to have a dimension called 'time'.
        z_power and z_power_acrossfreq can't both be True.
    z_power_acrossfreq : bool
        If True, power values are Z-scored across frequencies and
        time for a given channel. Requires timeseries to have
        a dimension called 'time'. z_power and z_power_acrossfreq 
        can't both be True.
    overwrite : bool
        If True, existing files will be overwritten.
    savedir : str
        Directory where the output files (power and phase timeseries
        objects saved in hdf5 format) will be saved. No files are
        saved if savedir is None.
    verbose : bool
        If verbose is False, print statements are suppressed.

    Returns
    -------
    power : ptsa.data.timeseries.TimeSeries
        Power spectra with optional log and/or Z transforms applied.
        Has the same shape as timeseries. 
    phase : ptsa.data.timeseries.TimeSeries
        Phase spectra with optional log and/or Z transforms applied.
        Has the same shape as timeseries.
    """
    dims = timeseries.dims
    dim1 = dims[0]
    assert len(dims) == 2
    assert dims[1] == 'time'
    assert not np.all([z_power, z_power_acrossfreq])
    
    if type(output) == str:
        output = [output]
    assert 'power' in output or 'phase' in output
    
    if freqs is None:
        freqs = np.logspace(np.log10(2), np.log10(200), 50, base=10)
        
    fstr = ('_width{}_{:.0f}-{:.0f}Hz-{}log10steps'
            .format(width, min(freqs), max(freqs), len(freqs)))
            
    powfstr = ''
    if log_power:
        powfstr += '_log10'
    if z_power:
        powfstr += '_Z-withinfreq'
    if z_power_acrossfreq:
        powfstr += '_Z-acrossfreq'
    
    # If power and phase already exist and aren't supposed to be overwritten,
    # load and return them from disk space.
    if savedir:
        if power_file is None:
            fname = ('{}_ch{}_power{}{}.hdf'
                     .format(timeseries.name, timeseries[dim1].data[0], fstr, powfstr))
            power_file = os.path.join(savedir, fname)
        if phase_file is None:
            fname = ('{}_ch{}_phase{}.hdf'
                     .format(timeseries.name,timeseries[dim1].data[0], fstr))
            phase_file = os.path.join(savedir, fname)
            
        if len(output) == 2:
            files_exist = os.path.exists(power_file) and os.path.exists(phase_file)
        elif 'power' in output:
            files_exist = os.path.exists(power_file)
        else:
            files_exist = os.path.exists(phase_file)
            
        if files_exist and not overwrite:
            if len(output) == 2:
                if verbose:
                    print('Loading power and phase:\n\t{}\n\t{}'
                          .format(power_file, phase_file))
                power = TimeSeries.from_hdf(power_file)
                phase = TimeSeries.from_hdf(phase_file)
                return power, phase
            elif 'power' in output:
                if verbose:
                    print('Loading power:\n\t{}'
                          .format(power_file))
                power = TimeSeries.from_hdf(power_file)
                return power
            else:
                if verbose:
                    print('Loading phase:\n\t{}'
                          .format(phase_file))
                phase = TimeSeries.from_hdf(phase_file)
                return phase
    
    # Get power and phase.
    if len(output) == 2:
        if verbose:
            print('Calculating power and phase.')
        power, phase = MorletWaveletFilter(timeseries,
                                           freqs=freqs,
                                           width=width,
                                           output=['power', 'phase']).filter()
    elif 'power' in output:
        if verbose:
            print('Calculating power.')
        power = MorletWaveletFilter(timeseries,
                                    freqs=freqs,
                                    width=width,
                                    output=['power']).filter()
    else:
        if verbose:
            print('Calculating phase.')
        phase = MorletWaveletFilter(timeseries,
                                    freqs=freqs,
                                    width=width,
                                    output=['phase']).filter()                  
        
        
    if 'power' in output:
        power = TimeSeries(power.data, dims=['frequency', dim1, 'time'], 
                           name=timeseries.name, 
                           coords={'frequency': power.frequency.data,
                                   dim1: power[dim1].data,
                                   'time': power.time.data,
                                   'samplerate': power.samplerate.data},
                           attrs={'morlet_width': width})
                           
         # Log transform every power value.
        if log_power:
            if verbose: 
                print('Log-transforming power values.')
            power.data = np.log10(power)
            
        # Z-score power over time for each channel, frequency vector
        if z_power:
            if verbose:
                print('Z-scoring power across time, within each frequency.')
            power.data = (power - power.mean(dim='time')) / power.std(dim='time')
        
        # Z-score power across frequencies and time, for each channel
        if z_power_acrossfreq:
            if verbose:
                print('Z-scoring power across time and frequency.')
            power.data = ((power - power.mean(dim=['frequency', 'time'])) 
                          / power.std(dim=['frequency', 'time']))
    
    if 'phase' in output:                         
        phase = TimeSeries(phase.data, dims=['frequency', dim1, 'time'], 
                           name=timeseries.name, 
                           coords={'frequency': phase.frequency.data,
                                   dim1: phase[dim1].data,
                                   'time': phase.time.data,
                                   'samplerate': phase.samplerate.data},
                           attrs={'morlet_width': width})     
    
    # Return log-transformed power and phase.
    if savedir:
        if verbose:
            print('Saving power:\n\t{}'.format(power_file))
        power.to_hdf(power_file)
#         phase.to_hdf(phase_file)
    
    if len(output) == 2:
        return power, phase
    elif 'power' in output:
        return power
    else:
        return phase
    
def setup_session():
    """Load/create the subjects DataFrame for the Manning replication project.
    
    Loads the subjects DataFrame if it exists. Otherwise creates it by reading
    the subj_info csv file and looking up input files (raw LFP for each channel
    and aligned spike times) in /data/continuous.
    
    Returns
    -------
    dirs : dict
        Directory paths for the Manning replication project.
    subj_info : dict
        Values for each subj in the Manning replication project.
    config : dict
        Configuration parameters for the Manning replication project.
    subj_df : pandas.core.frame.DataFrame
        Subjects DataFrame for the Manning replication project. Each row has 
        info for one channel, and the total length is the sum of all channels
        from all sessions from all subjects in the study.
    subj_df_file : str
        File path to the subj_df.
    """
    # Load project directory paths, subject metadata, and configuration params.
    dirs = manning_utils.get_dirs()
    subj_info = manning_utils.get_subjs()
    config = manning_utils.get_config()

    # Load or create the DataFrame that contains metadata about each channel in
    # the study (each row = 1 channel).
    subj_df_file = os.path.join(os.path.join(dirs['data'], 'subj_df.xlsx'))
    if os.path.exists(subj_df_file):
        # Load the subjects DataFrame.
        print('Loading subjects DataFrame: {}\n'.format(subj_df_file))
        subj_df = pd.read_excel(subj_df_file)
        subj_df['location'] = subj_df['location'].astype(str)
        subj_df['chan'] = subj_df['chan'].astype(str)
    else:
        # Get a list of all channels in the study, and find input files...
        # For each session, do a glob search for LFP channel files, which each contain the
        # 2000 Hz (downsampled) LFP data from one microwire channel for the recording session. 
        # Also find the spike times file, which gives LFP-aligned data points where 
        # spikes occurred, for each wave-clus identified cluster for the channel. 
        print('Collecting channel/session info.\n')
        n_chans = []
        subj_col = []
        sess_col = []
        chan_col = []
        raw_lfp_col = []
        spike_times_col = []
        input_files_exist_col = []
        proc_lfp_col = []
        session_spikes_col = []
        for subj in sorted(subj_info.keys()):
            for sess in subj_info[subj]['sess']:
                subj_sess = '{}_{}'.format(subj, sess)
                raw_lfp_files = sorted(glob.glob(os.path.join(dirs['raw_data'], subj, sess, 'lfp', 'chan.*')))
                n_chans.append(len(raw_lfp_files))
                for raw_lfp_file in raw_lfp_files:
                    basef = os.path.basename(raw_lfp_file)
                    chan = str(int(basef[basef.find('chan.')+5:]))
                    subj_col.append(subj)
                    sess_col.append(sess)
                    chan_col.append(chan)   
                    spike_times_file = os.path.join(dirs['data'], 'raw', 
                                                    subj, sess, '{}-{}-{}.mat'
                                                    .format(subj, sess, chan))
                    raw_lfp_col.append(raw_lfp_file)
                    spike_times_col.append(spike_times_file)
                    input_files_exist_col.append(np.all((os.path.exists(raw_lfp_file), 
                                                         os.path.exists(spike_times_file))))
                    proc_lfp_file = os.path.join(dirs['data'], 'lfp', 
                                                 '{}_notch-filtered_spikes-interpolated.hdf'.format(subj_sess))
                    session_spikes_file = os.path.join(dirs['data'], 'spikes', 
                                                       '{}_session_spikes.pkl'.format(subj_sess))
                    proc_lfp_col.append(proc_lfp_file)
                    session_spikes_col.append(session_spikes_file)

        # Create an empty DataFrame to store subject metadata. 
        # Each row is a unique subj-session-channel.        
        col_names = ['subj_sess', 'subj', 'sess', 'chan', 'location', 'sess_duration',
                     'raw_lfp_file', 'spike_times_file', 'input_files_exist',
                     'process_chan', 'n_clusters', 'proc_lfp_file', 'session_spikes_file']
        subj_df = pd.DataFrame(index=np.arange(sum(n_chans)), columns=col_names)
        arr = np.vstack((subj_col, sess_col, chan_col, raw_lfp_col, spike_times_col, 
                         input_files_exist_col, proc_lfp_col, session_spikes_col)).T
        subj_df[['subj', 'sess', 'chan', 'raw_lfp_file', 'spike_times_file', 
                 'input_files_exist', 'proc_lfp_file', 'session_spikes_file']] = arr
        subj_df['subj_sess'] = subj_df['subj'] + '_' + subj_df['sess']
        subj_df['input_files_exist'] = subj_df['input_files_exist'].map({'True': 1, 'False': 0})

        # Get a list of sessions to process.
        # Only process sessions where all input files exist.
        grp = subj_df.groupby('subj_sess')
        grp_vals = np.vstack((list(grp.groups.keys()),
                              grp['chan'].count().values, 
                              grp['input_files_exist'].sum().values)).T
        mapping = dict(zip(grp_vals[:, 0], [1 if val==True else 0 
                                            for val in grp_vals[:, 1] == grp_vals[:, 2]]))
        subj_df['process_chan'] = subj_df['subj_sess'].map(mapping)

    print('There are {} subjects, {} sessions, and {} channels in the study.\n'
          .format(len(subj_df.groupby('subj')), len(subj_df.groupby('subj_sess')), len(subj_df)))
    process_sessions = list(subj_df.query("(process_chan==1)")
                            .groupby('subj_sess').groups.keys())
    missing_sessions = list(subj_df.query("(process_chan==0)")
                            .groupby('subj_sess').groups.keys())
    print('{} sessions will be processed: {}'.format(len(process_sessions), process_sessions))
    print('{} sessions have missing input files and will not be processed.\n'.format(len(missing_sessions)))
    
    return dirs, subj_info, config, subj_df, subj_df_file
    
################################################################################

################################################################################

################################################################################

################################################################################

################################################################################

def epoch_fr_power_DEPRECATED(subj_sess,
                              lfp_proc, 
                              fr_df,
                              chan_to_clus,
                              chans=None,
                              power_file_suffix='_width5_log_Zacross',
                              freq_params={'low': 2, 'high': 150, 'num': 50}, 
                              epoch_size=1000, 
                              epoch_cut=3,
                              data_dir='/data3/scratch/dscho/frLfp/data',
                              verbose=True,
                              load_existing=True,
                              save_files=False):
    """Divide each channel into epochs, and get the mean firing rate and power
    within each epoch.
    
    Also calculate narrowband and broadband power for each epoch, using a
    robust regression model for the latter.
    
    Parameters
    ----------
    subj_sess : str
        e.g. 'U367_env2'
    lfp_proc : ptsa.data.timeseries.TimeSeries
        An n_channels x n_timepoints TimeSeries of the processed LFP data.
    fr_df : pandas.core.frame.DataFrame
        A DataFrame version of session_spikes, of sorts. n_clusters long and
        stores spikes, fr, and interp_mask for each cluster across all channels 
        in the session, along with some metadata.
    chan_to_clus : collections.OrderedDict
        Mapping between each unique channel in the recording session and a list
        of corresponding (across channel) cluster numbers (only has keys for
        channels with spikes!).
    chans : list or numpy.ndarray
        A list of channel names to process. Default is to use 
        all channels in lfp_proc.
    power_file_suffix : str
        The suffix to use in selecting power files to load from
        data_dir/lfp/morlet.
    freq_params : dict
        Keys must be 'low', 'high', and 'num'. Gives the frequency
        range and number of frequencies used in calculating power
        from LFP data through Morlet decomposition.
    epoch_size : int
        The number of timepoints that will comprise each epoch.
    epoch_cut : int
        The number of epochs to cut from the beginning and end
        of the recording session (to remove edge effects).
    data_dir : str
        Parent directory within which input and output data are stored.
    verbose : bool
        If True, print statements report what the function is doing
        in several places.
    load_existing : bool
        If True and output files exist, load and return them
        rather than reprocessing epochs.
    save_files : bool
        If True, output TimeSeries objects are saved as hdf5 files.
        If files already exist, load_existing is False, and save_files
        is True, then existing files are overwritten!
        
    Returns
    -------
    epoch_fr : ptsa.data.timeseries.TimeSeries
        Mean firing rate for each epoch, for each cluster in the
        session. Dims are cluster x epoch.
    epoch_power : ptsa.data.timeseries.TimeSeries
        Mean power for each frequency, for each epoch, for each
        channel in the session. Dims are channel x epoch x freq.
    epoch_band_power : ptsa.data.timeseries.TimeSeries
        Mean power for frequencies within delta, theta, alpha, beta,
        and gamma bands; along with the slope and intercept of a robust
        robust regression model predicting power at each frequency for
        a given channel/epoch. Broadband power is also reported as the
        mean predicted value from the regression fit. Dims are 
        channel x epoch x freq.
    """
    # Load the epoched data if it exists.
    # -----------------------------------
    epoch_dir = os.path.join(data_dir, 'epoch')
    epoch_fr_file = os.path.join(epoch_dir, '{}_epoch_fr.hdf'.format(subj_sess))
    epoch_power_file = os.path.join(epoch_dir, '{}_epoch_power{}.hdf'.format(subj_sess, power_file_suffix))
    epoch_band_power_file = os.path.join(epoch_dir, '{}_epoch_band_power{}.hdf'.format(subj_sess, power_file_suffix))
    files_exist = (os.path.exists(epoch_fr_file) 
                   and os.path.exists(epoch_power_file) 
                   and os.path.exists(epoch_band_power_file))
    if files_exist and load_existing:
        if verbose:
            print('Loading epoched spike and power data.')
        epoch_fr = TimeSeries.from_hdf(epoch_fr_file)
        epoch_power = TimeSeries.from_hdf(epoch_power_file)
        epoch_band_power = TimeSeries.from_hdf(epoch_band_power_file)
        return epoch_fr, epoch_power, epoch_band_power
    
    # Process data into epochs.
    # -------------------------
    start_time = time()
    if verbose:
        print('Processing spike and power data into epochs.')
    
    # Get channels and corresponding clusters to process.
    if chans is None:
        chans = lfp_proc.channel.data
    clusters = []
    for chan in chans:
        clusters += chan_to_clus[chan]
    
    # Get time bins to divide the data into 500ms epochs, cutting
    # out the first and last (+remainder) 3000 timepoints (1500ms).
    n_timepoints = lfp_proc.shape[1]    
    time_bins = manning_utils.get_epochs(np.arange(n_timepoints), 
                                         epoch_size=epoch_size, 
                                         cut=epoch_cut)

    # Get wavelet frequencies and assign them to bands of interest.
    freqs, freq_bands = manning_utils.get_freqs(low=freq_params['low'], 
                                                high=freq_params['high'], 
                                                num=freq_params['num'])
    # X is the independent variable for the robust regression.
    X = sm.add_constant(np.log10(freqs))
    
    # Get the mean firing rate for each epoch, for each cluster.
    # ----------------------------------------------------------
    epoch_fr = []
    for clus in clusters:
        # Get the mean firing rate (in Hz) for each epoch.
        clus_fr = fr_df.loc[fr_df.clus==clus, 'fr'].iat[0]
        epoch_fr_row = []
        for epoch_start, epoch_stop in time_bins:
            epoch_fr_row.append(2 * np.sum(clus_fr[epoch_start:epoch_stop]))
        epoch_fr.append(epoch_fr_row)

    epoch_fr = TimeSeries(np.array(epoch_fr), name=lfp_proc.name,
                          dims=['cluster', 'epoch'],
                          coords={'cluster': clusters,
                                  'epoch': np.arange(len(time_bins)),        
                                  'samplerate': lfp_proc.samplerate.data / 1000})
    
    # Get narrowband and broadband power for each epoch, for each channel.
    # --------------------------------------------------------------------
    power_dir = os.path.join(data_dir, 'lfp', 'morlet')
    epoch_power = []
    epoch_band_power = []
    for i, chan in enumerate(chans):
        if i % 8 == 0 and verbose:
            print('Loading power data for {} channel {}.'.format(subj_sess, chan))
        f = os.path.join(power_dir, '{}_ch{}_power{}.hdf'.format(subj_sess, chan, power_file_suffix))
        chan_power = TimeSeries.from_hdf(f).data.squeeze() # frequency x time
        epoch_power_row = []
        epoch_band_power_row = []
        for epoch_start, epoch_stop in time_bins:
            # Get the mean power at each frequency.
            epoch_power_row.append(np.mean(chan_power[:, epoch_start:epoch_stop], axis=1))

            # Get mean delta, theta, alpha, beta, and gamma power.                
            delta_power = np.mean(chan_power[0:8, epoch_start:epoch_stop])
            theta_power = np.mean(chan_power[8:16, epoch_start:epoch_stop])
            alpha_power = np.mean(chan_power[16:21, epoch_start:epoch_stop])                
            beta_power = np.mean(chan_power[21:31, epoch_start:epoch_stop]) 
            gamma_power = np.mean(chan_power[31:50, epoch_start:epoch_stop]) 

            # Fit a robust linear regression to estimate the intercept 
            # and slope (broadband tilt) of the power spectrum. Broadband
            # power is the mean of the predicted values.
            y = epoch_power_row[-1]
            huber_t = sm.RLM(y, X, M=sm.robust.norms.HuberT())
            hub_results = huber_t.fit()
            intercept, slope = hub_results.params
            epoch_band_power_row.append(
                    [delta_power, theta_power, alpha_power, beta_power, gamma_power, 
                     intercept, slope, np.mean(hub_results.predict())]
                )

        epoch_power.append(epoch_power_row) # n_epochs x n_freqs
        epoch_band_power.append(epoch_band_power_row) # n_epochs x (n_freq_bands + 3)

    epoch_power = TimeSeries(np.array(epoch_power), name=lfp_proc.name, 
                             dims=['channel', 'epoch', 'freq'],
                             coords={'channel': chans,
                                     'epoch': np.arange(len(time_bins)),        
                                     'freq': freqs,
                                     'samplerate': lfp_proc.samplerate.data / 1000})
    epoch_band_power = TimeSeries(np.array(epoch_band_power), name=lfp_proc.name, 
                                  dims=['channel', 'epoch', 'freq'],
                                  coords={'channel': chans,
                                          'epoch': np.arange(len(time_bins)),        
                                          'freq': (list(freq_bands.keys()) 
                                                   + ['intercept', 'bband_tilt', 'bband_power']),
                                          'samplerate': lfp_proc.samplerate.data / 1000})
    
    if save_files:
        epoch_fr.to_hdf(epoch_fr_file)
        epoch_power.to_hdf(epoch_power_file)
        epoch_band_power.to_hdf(epoch_band_power_file)
        if verbose:
            print('Saved epoched data to files:\n\t{}\n\t{}\n\t{}.'
                  .format(epoch_fr, epoch_power, epoch_band_power))

    duration = time() - start_time
    if verbose:
        print('Done in {} secs.'.format(int(duration)))
        
    return epoch_fr, epoch_power, epoch_band_power

def preprocess_session_DEPRECATED(subj_sess, subj_df, overwrite=False):
    """Process the raw (2000 Hz) LFP data for each cha  nnel from the session.
    
    Notch filters are applied at 60, 120, and 180 Hz. Spikes are then
    removed from the LFP by linear interpolation. Processed LFP is
    saved as an n_channels by n_timepoints 1) MNE RawArray and 
    2) PTSA TimeSeries.
    
    Spike time data are loaded and processed (see Returns:'session_spikes').
    
    Parameters
    ----------
    subj_sess : str
        e.g. 'U367_env2'
    subj_df : pandas.core.frame.DataFrame
        The DataFrame of subject metadata for each channel.
    overwrite : bool
        If True and output files already exist, the data are reprocessed 
        and the 3 output files are saved over (lfp_tsx, sesssion_spikes, 
        and subj_df). If False and output files already exist, these files
        are loaded in memory and returned.
        
    Returns
    -------
    subj_df : pandas.core.frame.DataFrame
        Subjects DataFrame for the Manning replication project. Each row has 
        info for one channel, and the total length is the sum of all channels
        from all sessions from all subjects in the study.
    chan_to_ind : collections.OrderedDict
        A dictionary that maps channel indices ('picks') in lfp_mne to the
        subj_df row that pertains to that channel. 
    lfp_mne : mne.io.array.array.RawArray
        An n_channels x n_timepoints RawArray of the processed LFP data.
    lfp_tsx : ptsa.data.timeseries.TimeSeries
        An n_channels x n_timepoints TimeSeries of the processed LFP data.
    session_spikes : collections.OrderedDict[collections.OrderedDict]
        Dictionary keys are each channel in the session, and each value is a
        dictionary with four keys:
            'spike_times' : dict[numpy.ndarray]
                Time points of each spike, by cluster. Dict keys are the
                integer indices of each cluster in the arbitrary order that
                they appeared in the original wave_clus output file. Length
                of the ndarray = the number of spikes observed for that cluster
                during the session.
            'spikes' : numpy.ndarray
                n_clusters x n_timepoints Boolean array of spikes (i.e. spike 
                was present or absent for a given cluster at a given timepoint).
            'fr' : numpy.ndarray
                n_clusters x n_timepoints array of firing rates, calculated by
                convolving the 'spikes' array with a 500ms half-width Gaussian 
                kernel.
            'spike_lfp' : list[numpy.ndarray] 
                The list has n_cluster elements, in the same order as for 
                'spike_times', 'spikes', and 'fr'. The list elements are 
                n_spikes x 21 arrays that contain the raw LFP trace from -2
                to 8 ms after each spike event for a given cluster.
    fr_df : pandas.core.frame.DataFrame
        n_clusters long DataFrame to store smoothed firing rate arrays for each
        cluster across channels in the session, along with metadata on 
        channel name, subj_sess name, location, (within channel) cluster 
        number, (across channel) cluster number, and mean firing rate (in Hz). 
    clus_to_chan : collections.OrderedDict
        Mapping between each unique cluster in the recording session and its
        corresponding channel.
    chan_to_clus : collections.OrderedDict
        Mapping between each unique channel in the recording session and a list
        of corresponding (across channel) cluster numbers.
    """
    dirs = manning_utils.get_dirs()
    config = manning_utils.get_config()
    df_inds = subj_df.query("(subj_sess=='{}')".format(subj_sess)).index.values
    proc_lfp_file = subj_df.at[df_inds[0], 'proc_lfp_file']
    session_spikes_file = subj_df.at[df_inds[0], 'session_spikes_file']
    proc_files_exist = np.all((os.path.exists(proc_lfp_file), 
                               os.path.exists(session_spikes_file)))
    if proc_files_exist and not overwrite:
        print('Loading processed LFP and spike data for {}.\n'.format(subj_sess))
        # Load processed LFP (notch-filtered and spike iterpolated)
        lfp_tsx = TimeSeries.from_hdf(proc_lfp_file)
        info = mne.create_info(ch_names=lfp_tsx.channel.values.tolist(), 
                               sfreq=lfp_tsx.samplerate.values.tolist(), 
                               ch_types=['seeg']*len(lfp_tsx.channel.values))
        lfp_mne = mne.io.RawArray(lfp_tsx.data, info) 
        
        # Load session_spikes
        with open(session_spikes_file, 'rb') as f:
            session_spikes = pickle.load(f)    
    else:
        process_chans = []
        lfp_dat = []
        session_spikes = OrderedDict()
        config = manning_utils.get_config()
        steps_before = int(2 * config['ms_before']) # 2 ms
        steps_after = int((2 * config['ms_after']) + 1) # 8 ms
        
        # Iterate over channels in the session.
        print('Processing data for {}:\n'.format(subj_sess))
        for i, df_ind in enumerate(df_inds):
            chan = subj_df.at[df_ind, 'chan']
            raw_lfp_file = subj_df.at[df_ind, 'raw_lfp_file']
            spike_times_file = subj_df.at[df_ind, 'spike_times_file']
            if i % 10 == 0:
                print('    Loading data for channel {}/{}'.format(chan, len(df_inds)))

            # Load the spike times file and figure out how many clusters there are.
            # spikeTimes : array of arrays
            #     The parent array has each cluster for the channel, as determined 
            #     by the wave_clus package. The child array gives spike times for 
            #     the cluster aligned to the LFP data (e.g. a spike time of 60 would 
            #     indicate a spike at the 60th LFP time point).
            spikes_mat = sio.loadmat(spike_times_file)
            if len(spikes_mat['spikeTimes']) > 0:
                arr_in = spikes_mat['spikeTimes'][0]
                spike_times = OrderedDict()
                clus = 0
                for m in range(len(arr_in)):
                    if len(arr_in[m]) > 1:
                        spike_times[clus] = arr_in[m].squeeze()
                        clus += 1
                n_clusters = len(spike_times)
            else:
                n_clusters = 0

            # Get electrode position.
            location = spikes_mat['anatlabel'][0]
            subj_df.at[df_ind, 'location'] = location
            
            # Only process channels where spikes were recorded.
            subj_df.at[df_ind, 'n_clusters'] = n_clusters
                
            if n_clusters > 0:
                process_chans.append(chan)

                # Load the raw LFP data.
                lfp_dat.append(np.fromfile(raw_lfp_file, dtype='float32'))
                n_lfp_samples = len(lfp_dat[-1])
                subj_df.at[df_ind, 'sess_duration'] = n_lfp_samples / config['samplingRate']

                # Process the spike times data.
                session_spikes[chan] = OrderedDict()
                spikes_dat = np.zeros([len(spike_times), n_lfp_samples], dtype='bool_')
                fr_dat = []
                spike_lfp = []
                for clus in range(len(spike_times)):
                    # Make a Boolean spike array aligned to lfp_dat, 
                    # with shape (n_clusters, n_lfp_timepoints).
                    spikes_dat[clus, spike_times[clus]] = 1

                    # Calculate firing rate at each time point by convolving a Gaussian 
                    # kernel (half-width=500ms) with the spike array.
                    vec = np.zeros(2001)
                    vec[1000] = 1
                    g_std = (((config['halfWidth'] * config['samplingRate'])/1000) 
                             / (2 * np.sqrt(2 * np.log(2))))
                    g_win = gaussian_filter(vec, g_std)
                    fr_dat.append(np.convolve(spikes_dat[clus, :], g_win, mode='same'))

                    # Get the LFP waveform from -2 to 8 ms 
                    # surrounding each spike for each cluster.
                    spike_lfp.append([])
                    for spike_time in spike_times[clus]:
                        spike_time = int(spike_time)
                        if (spike_time > steps_before) and (spike_time < (len(lfp_dat[-1]) - steps_after)):
                            spike_lfp[clus].append(lfp_dat[-1][spike_time-steps_before:spike_time+steps_after])
                    spike_lfp[clus] = np.array(spike_lfp[clus])
                
                session_spikes[chan]['location'] = location
                session_spikes[chan]['spike_times'] = spike_times # time points of each spike, by cluster
                session_spikes[chan]['spikes'] = spikes_dat # Boolean spike vector, by cluster
                session_spikes[chan]['fr'] = np.array(fr_dat) # firing rate, by cluster
                session_spikes[chan]['spike_lfp'] = spike_lfp # 10 ms LFP around each spike, by cluster
            
        # Load LFP data an m x n MNE RawArray object, where m = the number
        # of channels and n = the number of time points in the session.
        lfp_dat = np.array(lfp_dat)
        info = mne.create_info(ch_names=process_chans, sfreq=config['samplingRate'], 
                               ch_types=['seeg']*len(process_chans))
        lfp_mne = mne.io.RawArray(lfp_dat, info) 

        # Apply a notch filter to remove line noise.    
        lfp_mne.notch_filter(np.array([60, 120, 180]))
        #lfp_mne.filter(l_freq=4, h_freq=8)

        # Linearly interpolate the LFP signal around spikes (-2 to 8 ms).
        for i, chan in enumerate(process_chans):
            if i % 10 == 0:
                print('    Interpolating channel {}/{}'.format(i+1, len(process_chans)))
            lfp_dat = lfp_mne.get_data(picks=[i]).squeeze()
            interp_mask = np.zeros(len(lfp_dat), dtype='bool_')
            spike_times = session_spikes[chan]['spike_times']
            clus_interp_mask = np.zeros([len(spike_times), len(lfp_dat)], dtype='bool_')
            for clus in range(len(spike_times)):
                for spike_time in spike_times[clus]:
                    spike_time = int(spike_time)
                    if (spike_time > steps_before) and (spike_time < (len(lfp_dat) - steps_after)):
                        interp_mask[spike_time-steps_before:spike_time+steps_after] = 1
                        clus_interp_mask[clus, spike_time-steps_before:spike_time+steps_after] = 1
            
            pct_interp = np.sum(interp_mask) / len(interp_mask)
            session_spikes[chan]['interp_mask'] = clus_interp_mask
            session_spikes[chan]['pct_interp'] = pct_interp
            keep_inds = np.where(interp_mask==0)[0]
            fill_inds = np.where(interp_mask==1)[0]
            f = interp1d(keep_inds, lfp_dat[keep_inds], kind='linear', fill_value='extrapolate')
            def apply_interp(arr, inds, f):
                arr[inds] = f(inds)
                return arr
            lfp_mne.apply_function(apply_interp, picks=[i], inds=fill_inds, f=f)

        # Convert processed LFP data (notch-filtered and spike iterpolated)
        # from MNE RawArray to PTSA TimeSeries.
        lfp_tsx = TimeSeries(lfp_mne._data, dims=['channel', 'time'], name=subj_sess,
                          coords={'channel': lfp_mne.ch_names,
                                  'time': np.arange(lfp_mne.n_times),
                                  'samplerate': lfp_mne.info['sfreq']})
        
        # Save LFP data as an hdf file.
        lfp_tsx.to_hdf(proc_lfp_file)
        print('Saved LFP data to file: {}'.format(proc_lfp_file))

        # Save session_spikes as a pickle file.
        with open(session_spikes_file, 'wb') as f:
            pickle.dump(session_spikes, f, pickle.HIGHEST_PROTOCOL)
        print('Saved spikes data to file: {}'.format(session_spikes_file))

        # Save the subj DataFrame
        subj_df_file = os.path.join(os.path.join(dirs['data'], 'subj_df.xlsx'))
        writer = pd.ExcelWriter(subj_df_file)
        subj_df.to_excel(writer, index=True)
        writer.save()
        print('Saved subj_df to file: {}'.format(subj_df_file))
    
    # Create a DataFrame with n_clusters rows to hold smoothed firing rate data
    # (array of firing rates at each timepoint) for each cluster
    session_spikes = OrderedDict(session_spikes)
    fr_dat = []
    clus = 0
    for chan in session_spikes.keys():
        for chan_clus in range(len(session_spikes[chan]['fr'])):
            fr_dat.append([subj_sess, chan, 
                           session_spikes[chan]['location'], 
                           session_spikes[chan]['pct_interp'], 
                           clus, chan_clus,
                           session_spikes[chan]['fr'][chan_clus], 
                           session_spikes[chan]['spikes'][chan_clus],
                           session_spikes[chan]['interp_mask'][chan_clus]])
            clus += 1
            
    col_names = ['subj_sess', 'chan', 'location', 'chan_pct_interp', 
                 'clus', 'chan_clus', 'fr', 'spikes', 'interp_mask']
    fr_df = pd.DataFrame(fr_dat, columns=col_names)
    fr_df.insert(6, 'mean_fr', np.vstack(fr_df['spikes'])
                 .mean(axis=1) * config['samplingRate']) 
    clus_to_chan = OrderedDict(fr_df[['clus', 'chan']].values)
    chan_to_clus = OrderedDict(fr_df.groupby('chan', sort=False)['clus']
                                             .apply(list))                   
    
    # Create a dict that maps the MNE channel indices for lfp_mne 
    # to the subject data frame index for that channel.
    df = subj_df[subj_df.subj_sess==subj_sess]
    chan_to_ind = OrderedDict(zip(lfp_mne.ch_names, 
                                  df[df.chan.isin(lfp_mne.ch_names)].index.values))
    
    return (subj_df, chan_to_ind, lfp_mne, lfp_tsx, session_spikes, 
            fr_df, clus_to_chan, chan_to_clus)
            
def process_lfp_DEPRECATED(subj_sess, subj_df, session_spikes, 
                steps_before, steps_after, sampling_rate=2000.0):
    """Notch filter the raw LFP data and linearly interpolate around spikes."""
    # Get channel indices for the recording session.
    df_inds = subj_df.query("(subj_sess=='{}')".format(subj_sess)).index.values

    # Iterate over channels in the recording session.
    lfp_dat = []
    lfp_chans = []
    for i, df_ind in enumerate(df_inds):
        lfp_chans.append(subj_df.at[df_ind, 'chan'])
        raw_lfp_file = subj_df.at[df_ind, 'raw_lfp_file']
        lfp_dat.append(np.fromfile(raw_lfp_file, dtype='float32'))

    lfp_dat = np.array(lfp_dat)
    n_timepoints = lfp_dat.shape[1]
    lfp_raw = TimeSeries(lfp_dat, name=subj_sess, 
                         dims=['channel', 'time'],
                         coords={'channel': lfp_chans,
                                 'time': np.arange(n_timepoints),
                                 'samplerate': sampling_rate})

    # Notch filter at 60, 120, 180 Hz.
    lfp_proc = ButterworthFilter(lfp_raw.copy(), freq_range=[58, 62], 
                                 filt_type='stop', order=4).filter()
    lfp_proc = ButterworthFilter(lfp_proc, freq_range=[118, 122], 
                                 filt_type='stop', order=4).filter()
    lfp_proc = ButterworthFilter(lfp_proc, freq_range=[178, 182], 
                                 filt_type='stop', order=4).filter()                                                          
    lfp_proc.name = subj_sess
    lfp_proc_dat = lfp_proc.copy().data

    # Linearly interpolate LFP around spikes, for channels with clusters.
    interp_chans = session_spikes.keys()
    for chan in interp_chans:
        interp_mask = session_spikes[chan]['interp_mask']
        keep_inds = np.where(interp_mask==0)[0]
        fill_inds = np.where(interp_mask==1)[0]

        f = interp1d(keep_inds, lfp_proc_dat[lfp_chans.index(chan), keep_inds], 
                     kind='linear', fill_value='extrapolate')
        def apply_interp(arr, inds, f):
            arr[inds] = f(inds)
            return arr
        lfp_proc.data[lfp_chans.index(chan), :] = apply_interp(
                lfp_proc_dat[lfp_chans.index(chan), :], 
                fill_inds, 
                f
            )

    return lfp_raw, lfp_proc
    
def run_morlet_DEPRECATED(timeseries, 
               freqs=None, 
               width=5, 
               log_power=False, 
               z_power=False, 
               z_power_acrossfreq=False, 
               overwrite=False,
               savedir='/data3/scratch/dscho/frLfp/data/lfp/morlet',
               power_file=None,
               phase_file=None,
               log_dir='/data3/scratch/dscho/frLfp/logs'):
    """Apply Morlet wavelet transform to a timeseries to calculate
    power and phase spectra for one or more frequencies.
    
    Serves as a wrapper for PTSA's MorletWaveletFilter. Can log 
    transform and/or Z-score power across time and can save the 
    returned power and phase timeseries objects as hdf5 files.
    
    Parameters
    ----------
    timeseries : ptsa.data.timeseries.TimeSeries
        The timeseries data to be transformed.
    freqs : numpy.ndarray or list
        A list of frequencies to apply wavelet decomposition over.
    width : int
        Number of waves for each frequency.
    log_power : bool
        If True, power values are log10 transformed.
    z_power : bool
        If True, power values are Z-scored across the time dimension.
        Requires timeseries to have a dimension called 'time'.
        z_power and z_power_acrossfreq can't both be True.
    z_power_acrossfreq : bool
        If True, power values are Z-scored across frequencies and
        time for a given channel. Requires timeseries to have
        a dimension called 'time'. z_power and z_power_acrossfreq 
        can't both be True.
    overwrite : bool
        If True, existing files will be overwritten.
    savedir : str
        Directory where the output files (power and phase timeseries
        objects saved in hdf5 format) will be saved. No files are
        saved if savedir is None.
    log_dir : str
        Directory where the log file is saved.
    
    Returns
    -------
    power : ptsa.data.timeseries.TimeSeries
        Power spectra with optional log and/or Z transforms applied.
        Has the same shape as timeseries. 
    phase : ptsa.data.timeseries.TimeSeries
        Phase spectra with optional log and/or Z transforms applied.
        Has the same shape as timeseries.
    """
    # Setup logging.
    logger = logging.getLogger(sys._getframe().f_code.co_name)
    logger.handlers = []
    log_f = os.path.join(log_dir, '{}_ch{}_{}_{}.log'.format(timeseries.name, 
                                                             timeseries.channel.data[0],
                                                             sys._getframe().f_code.co_name, 
                                                             strftime('%m-%d-%Y-%H-%M-%S')))
    handler = logging.FileHandler(log_f)
    handler.setLevel(logging.DEBUG)
    formatting = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                   datefmt='%m-%d-%Y %H:%M:%S')
    handler.setFormatter(formatting)
    logger.addHandler(handler)
        
    assert timeseries.dims == ('channel', 'time')
    assert not np.all([z_power, z_power_acrossfreq])
    
    if freqs is None:
        freqs = np.logspace(np.log10(2), np.log10(200), 50, base=10)
        
    fstr = ('_width{}_{:.0f}-{:.0f}Hz-{}log10steps'
            .format(width, min(freqs), max(freqs), len(freqs)))
            
    powfstr = ''
    if log_power:
        powfstr += '_log10'
    if z_power:
        powfstr += '_Z-withinfreq'
    if z_power_acrossfreq:
        powfstr += '_Z-acrossfreq'
    
    # If power and phase already exist and aren't supposed to be overwritten,
    # load them from memory and return.
    if savedir:
        if power_file is None:
            fname = ('{}_ch{}_power{}{}.hdf'
                     .format(timeseries.name, timeseries.channel.data[0], fstr, powfstr))
            power_file = os.path.join(savedir, fname)
        if phase_file is None:
            fname = ('{}_ch{}_phase{}.hdf'
                     .format(timeseries.name,timeseries.channel.data[0], fstr))
            phase_file = os.path.join(savedir, fname)
        files_exist = os.path.exists(power_file) and os.path.exists(phase_file)
        if files_exist and not overwrite:
            logger.info('Loading power and phase data:\n\t{}\n\t{}'
                         .format(power_file, phase_file))
            power = TimeSeries.from_hdf(power_file)
            phase = TimeSeries.from_hdf(phase_file)
            return power, phase
    
    # Get power and phase.
    logger.info('Calculating power and phase.')
    power, phase = MorletWaveletFilter(timeseries,
                                       freqs=freqs,
                                       width=width,
                                       output=['power', 'phase']).filter()
                                       
    power = TimeSeries(power.data, dims=['frequency', 'channel', 'time'], 
                       name=timeseries.name, 
                       coords={'frequency': power.frequency.data,
                               'channel': power.channel.data,
                               'time': power.time.data,
                               'samplerate': power.samplerate.data},
                       attrs={'morlet_width': width})

    phase = TimeSeries(phase.data, dims=['frequency', 'channel', 'time'], 
                       name=timeseries.name, 
                       coords={'frequency': phase.frequency.data,
                               'channel': phase.channel.data,
                               'time': phase.time.data,
                               'samplerate': phase.samplerate.data},
                       attrs={'morlet_width': width})     
    
    # Log transform every power value.
    if log_power:
        logger.info('Log-transforming power values.')
        power.data = np.log10(power)
            
    # Z-score power over time for each channel, frequency vector
    if z_power:
        logger.info('Z-scoring power across time, within each frequency.')
        power.data = (power - power.mean(dim='time')) / power.std(dim='time')
        
    # Z-score power across frequencies and time, for each channel
    if z_power_acrossfreq:
        logger.info('Z-scoring power across time and frequency.')
        power.data = ((power - power.mean(dim=['frequency', 'time'])) 
                      / power.std(dim=['frequency', 'time']))
    
    # Return log-transformed power and phase.
    if savedir:
        logger.info('Saving power and phase:\n\t{}\n\t{}'
                     .format(power_file, phase_file))
        power.to_hdf(power_file)
        #phase.to_hdf(phase_file)
    
    return power, phase