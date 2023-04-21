"""
lfp_synchrony.py

Author:
    Daniel Schonhaut
    Computational Memory Lab
    University of Pennsylvania
    daniel.schonhaut@gmail.com
    
Description: 
    Functions for calculating phase locking.

Last Edited: 
    2/2/19
"""
import sys
import os
import pickle
import glob
import random
from collections import OrderedDict
import itertools 
from math import floor, ceil
from time import sleep

import mkl
mkl.set_num_threads(1)
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
import pandas as pd
import astropy.stats.circstats as circstats
import pycircstat
import mne
from ptsa.data.TimeSeriesX import TimeSeries 
sys.path.append('/home1/dscho/code/general')
sys.path.append('/home1/dscho/code/projects/manning_replication')
sys.path.append('/home1/dscho/code/projects/unit_activity_and_hpc_theta')
import data_io as dio
import array_operations as aop
import spectral_processing as spp
import manning_analysis
import phase_locking

def calc_mrl_qtls(phase_vec, compare_vec):
    """Return top vs. bottom quartile differences in phase-locking
    strength (MRL), sorting by the values in compare_vec.
    
    Parameters
    ----------
    phase_vec : numpy.ndarray
        A vector of spike phases from which MRLs are calculated.
    compare_vec : numpy.ndarray
        A vector of values that are used to sort spikes into top
        and bottom quartiles. Must be the same length as phase_vec.
        
    Returns
    -------
    qtl_mrls : list
        MRL of sorted spike phases in each quartile.
    mrl_diff : float
        The difference in MRL between the top and bottom quartiles.
    """
    # Cut indices from the beginning so both vectors are
    # divisible by 4.
    cut = len(phase_vec) % 4
    phase_vec = phase_vec[cut:]
    compare_vec = compare_vec[cut:]
    
    # Sort spike phases by compare_vec values. 
    xsort = compare_vec.argsort()
    phase_vec = np.split(phase_vec[xsort], 4)
    
    # Calculate the MRL for spike phases in each quartile.
    qtl_mrls = [circstats.circmoment(x)[1] for x in phase_vec]
    
    # Calculate the different in MRL between the top and 
    # bottom quartiles.
    mrl_diff = qtl_mrls[-1] - qtl_mrls[0]
    
    return qtl_mrls, mrl_diff
    
def calc_plv(arr1, arr2, axis=-1):
    """Return the phase-locking value(s) between two phase arrays.
    
    PLV is the mean resultant length of the circular distances between
    corresponding arr1 and arr2 elements.
    
    Parameters
    ----------
    arr1 : np.ndarray
        Phase array of any number of dimensions.
    arr2 : np.ndarray
        Phase array of equal shape as arr1.
    axis : int
        Axis to calculate the PLV over. Axis=None will calculate PLV over
        the flattened arrays and return a single number; otherwise the output
        is an array of PLVs whose shape matches the input arrays, minus the
        dimension that PLV is calculated over.
        
    Returns
    -------
    np.float64 or np.ndarray
        PLV or an array of PLVs if the input arrays have >1 dimensions 
        and axis is not None.
    """
    return circstats.circmoment(pycircstat.descriptive.cdiff(arr1, arr2), axis=axis)[1]

def get_epoch_plvs(subj_sess, 
                   unit, 
                   lfp1_chan_ind, 
                   lfp2_chan_ind, 
                   freq_ind, 
                   epoch_size,
                   input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                   sampling_rate=500,
                   remove_secs=5,
                   save_outputs=True,
                   output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/lfp_plvs/plvs'):
    """Return the phase-locking value between two phase vectors, for each epoch.

    Returns
    -------
    epoch_plvs : np.ndarray
        Vector of phase-locking values surrounding each spike.
    """
    sampling_rate = int(sampling_rate)
    
    # Load inputs.
    cut_inds = int(sampling_rate * remove_secs)
    
    lfp_fname = 'phase-{}-iChan{}-iFreq{}-{}Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_16.0Hz.pkl'
    lfp1_phase = dio.open_pickle(os.path.join(input_dir, 'wavelet_phase', lfp_fname.format(subj_sess, lfp1_chan_ind, freq_ind, sampling_rate)))
    lfp2_phase = dio.open_pickle(os.path.join(input_dir, 'wavelet_phase', lfp_fname.format(subj_sess, lfp2_chan_ind, freq_ind, sampling_rate)))
    
    spike_inds, n_timepoints = dio.open_pickle(os.path.join(input_dir, 'spike_inds', 'spike_inds-{}Hz-{}-unit{}.pkl'
                                                            .format(sampling_rate, subj_sess, unit)))
    spike_inds = spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))]
    lfp1_epochs = make_epochs(lfp1_phase, spike_inds, epoch_size)
    lfp2_epochs = make_epochs(lfp2_phase, spike_inds, epoch_size)
    epoch_plvs = calc_plv(lfp1_epochs, lfp2_epochs, axis=-1)
    
    if save_outputs:
        output_f = os.path.join(output_dir, 'epoch_plvs-{}-unit{}-lfp1_chan_ind_{}-lfp2_chan_ind_{}-freq_ind_{}.pkl'
                                            .format(subj_sess, unit, lfp1_chan_ind, lfp2_chan_ind, freq_ind))
        dio.save_pickle(epoch_plvs, output_f, verbose=False)
    
    return epoch_plvs
     
def get_epoch_plvs2(info_fpath,
                    freqs=np.logspace(np.log10(0.5), np.log10(16), 16),
                    sampling_rate=500,
                    n_cyles=5,
                    n_bootstraps=500,
                    save_outputs=True,
                    input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data',
                    sleep_max=1800):
    """Calculate the phase-locking value around each spike, between each LFP-LFP pair.
    
    Performed for a single channel-to-channel comparison. Bootstrap distributions
    are generated by spike train shuffling.
    
    NOTE: This function takes a long time to run (~6-8 hours) due to the 
    bootstrap procedure...
    """
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # Load info DataFrame.
    info = dio.open_pickle(info_fpath)
    subj_sess = info.subj_sess
    unit = info.unit
    lfp1_chan_ind = info.lfp1_chan_ind
    lfp2_chan_ind = info.lfp2_chan_ind
    
    # General params.
    wavelet_dir = os.path.join(input_dir, 'crosselec_phase_locking', 'wavelet_phase')
    bootstrap_dir = os.path.join(input_dir, 'lfp_plvs', 'bootstrap_shifts')
    spikes_dir = os.path.join(input_dir, 'crosselec_phase_locking', 'spike_inds')
    output_dir = os.path.join(input_dir, 'lfp_plvs', 'plvs')
    lfp_fname = 'phase-{}-iChan{}-iFreq{}-{}Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_16.0Hz.pkl'
    sampling_rate = int(sampling_rate)
    cut_inds = int(sampling_rate * n_cyles)
    epoch_sizes = [int(sampling_rate * (1/freq) * n_cyles) for freq in freqs]
    
    # Load spike inds and remove n_cyles secs from the beggining and end of the session,
    # then get bootstrap time-shifted spike inds for generating null distributions.
    spike_inds, n_timepoints = dio.open_pickle(os.path.join(spikes_dir, 'spike_inds-{}Hz-{}-unit{}.pkl'.format(sampling_rate, subj_sess, unit)))
    spike_inds = spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))]
    half_win = int(epoch_sizes[0]/2) + 1
    bs_steps = dio.open_pickle(os.path.join(bootstrap_dir, '{}_{}bootstraps.pkl'.format(subj_sess, int(n_bootstraps))))
    bs_spike_inds = []
    for step in bs_steps:
        spike_inds_shifted = phase_locking.shift_spike_inds(spike_inds, n_timepoints - half_win, step)
        spike_inds_shifted[spike_inds_shifted<=half_win] = spike_inds_shifted[spike_inds_shifted<=half_win] + half_win
        bs_spike_inds.append(spike_inds_shifted)
        
    # Get phase-locking value at each frequency, and generate
    # null distributions from time-shifted spike trains.
    plvs_arr = []
    bs_plvs_arr = []
    for iFreq in range(len(freqs)):
        epoch_size = epoch_sizes[iFreq]
        lfp1_phase = dio.open_pickle(os.path.join(wavelet_dir, lfp_fname.format(subj_sess, lfp1_chan_ind, iFreq, sampling_rate)))
        lfp2_phase = dio.open_pickle(os.path.join(wavelet_dir, lfp_fname.format(subj_sess, lfp2_chan_ind, iFreq, sampling_rate)))
        lfp1_epochs = make_epochs(lfp1_phase, spike_inds, epoch_size)
        lfp2_epochs = make_epochs(lfp2_phase, spike_inds, epoch_size)
        plvs_arr.append(calc_plv(lfp1_epochs, lfp2_epochs, axis=-1))

        bs_plvs_arr_ = []
        for iStep in range(n_bootstraps):
            lfp1_epochs = make_epochs(lfp1_phase, bs_spike_inds[iStep], epoch_size)
            lfp2_epochs = make_epochs(lfp2_phase, bs_spike_inds[iStep], epoch_size)
            bs_plvs_arr_.append(calc_plv(lfp1_epochs, lfp2_epochs, axis=-1))
        bs_plvs_arr.append(bs_plvs_arr_)
    
    info['plvs'] = np.array(plvs_arr) # frequency x spike_event
    info['bs_plvs'] = np.array(bs_plvs_arr) # frequency x bootstrap_ind x spike_event
    
    if save_outputs:
        fpath = os.path.join(output_dir,
                             'lfp_plvs-{}-unit_{}-chan_ind_{}-to-chan_ind_{}-{}Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_16.0Hz.pkl'
                             .format(subj_sess, unit, lfp1_chan_ind, lfp2_chan_ind, sampling_rate))
        dio.save_pickle(info, fpath, verbose=False)
        
    return info
    
def get_epoch_plvs3(info_fpath,
                    freqs=np.logspace(np.log10(0.5), np.log10(16), 16),
                    sampling_rate=500,
                    n_cyles=5,
                    n_bootstraps=500,
                    save_outputs=True,
                    input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data',
                    sleep_max=300):
    """Calculate the phase-locking value around each spike, between each LFP-LFP pair.
    
    Performed for a single neuron to region comparison.
    
    PLVs are calculated for all possible pairs between channels in the unit's microwire bundle
    and channels in the region of interest, and the mean PLV across channel pairs and spike
    events is then obtained for each frequency. These values are also Z-scored against a 
    null distribution of random timepoints throughout the recording session.
    """
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # Load info DataFrame.
    info = dio.open_pickle(info_fpath)
    subj_sess = info.subj_sess.iat[0]
    unit = info.unit.iat[0]
    
    # General params.
    wavelet_dir = os.path.join(input_dir, 'crosselec_phase_locking', 'wavelet_phase')
    bootstrap_dir = os.path.join(input_dir, 'lfp_plvs', 'bootstrap_shifts')
    spikes_dir = os.path.join(input_dir, 'crosselec_phase_locking', 'spike_inds')
    output_dir = os.path.join(input_dir, 'lfp_plvs', 'plvs')
    lfp_fname = 'phase-{}-iChan{}-iFreq{}-{}Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_16.0Hz.pkl'
    sampling_rate = int(sampling_rate)
    cut_inds = int(sampling_rate * n_cyles)
    epoch_sizes = [int(sampling_rate * (1/freq) * n_cyles) for freq in freqs]
    
    # Load spike inds and remove n_cyles secs from the beggining and end of the session,
    # then get bootstrap time-shifted spike inds for generating null distributions.
    spike_inds, n_timepoints = dio.open_pickle(os.path.join(spikes_dir, 'spike_inds-{}Hz-{}-unit{}.pkl'.format(sampling_rate, subj_sess, unit)))
    spike_inds = spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))]
    bs_inds = np.sort(dio.open_pickle(os.path.join(bootstrap_dir, '{}_{}bootstrap_timepoints.pkl'.format(subj_sess, int(n_bootstraps)))))
        
    # Get phase-locking value at each frequency, and generate
    # null distributions by calculating PLVs for random timepoints.
    plvs_arr = []
    bs_plvs_arr = []
    for index, row in info.iterrows():
        lfp1_chan_ind = row.lfp1_chan_ind
        lfp2_chan_ind = row.lfp2_chan_ind
        plvs_arr_ = []
        bs_plvs_arr_ = []
        for iFreq in range(len(freqs)):
            epoch_size = epoch_sizes[iFreq]
            lfp1_phase = dio.open_pickle(os.path.join(wavelet_dir, lfp_fname.format(subj_sess, lfp1_chan_ind, iFreq, sampling_rate)))
            lfp2_phase = dio.open_pickle(os.path.join(wavelet_dir, lfp_fname.format(subj_sess, lfp2_chan_ind, iFreq, sampling_rate)))
            lfp1_epochs = make_epochs(lfp1_phase, spike_inds, epoch_size)
            lfp2_epochs = make_epochs(lfp2_phase, spike_inds, epoch_size)
            plvs_arr_.append(calc_plv(lfp1_epochs, lfp2_epochs, axis=-1))
            
            lfp1_epochs = make_epochs(lfp1_phase, bs_inds, epoch_size)
            lfp2_epochs = make_epochs(lfp2_phase, bs_inds, epoch_size)
            bs_plvs_arr_.append(calc_plv(lfp1_epochs, lfp2_epochs, axis=-1))
        
        plvs_arr.append(plvs_arr_)
        bs_plvs_arr.append(bs_plvs_arr_)
    
    plvs_arr = np.array(plvs_arr) # channel_pair x frequency x spike_event
    bs_plvs_arr = np.array(bs_plvs_arr) # channel_pair x frequency x random_timepoint
    
    # Z-score PLVs against their respective channel
    # and frequency from the bootstrap distribution.
    bs_means = np.expand_dims(np.mean(bs_plvs_arr, axis=-1), axis=-1)
    bs_stds = np.expand_dims(np.std(bs_plvs_arr, axis=-1), axis=-1)
    bs_plvs_arr_z = (bs_plvs_arr - bs_means) / bs_stds
    plvs_arr_z = (plvs_arr - bs_means) / bs_stds
    
    # Get phase-locking stats and append to the info series.    
    output = pd.Series({'subj_sess': subj_sess,
                        'unit': unit,
                        'lfp1_roi': info.lfp1_hemroi.iat[0],
                        'lfp2_roi': info.lfp2_hemroi.iat[0],
                        'plvs_arr': plvs_arr, # channel_pair x frequency x spike_event array of PLVs
                        'bs_plvs_arr': bs_plvs_arr, # channel_pair x frequency x random_timepoint array of PLVs
                        'plvs_arr_z': plvs_arr_z, # channel_pair x frequency x spike_event array of Z-PLVs
                        'plvs': np.mean(plvs_arr, axis=(0, 2)), # PLV by frequency (mean across channel and timepoint)
                        'plvs_z': np.mean(plvs_arr_z, axis=(0, 2))}, # Z-PLV by frequency (mean across channel and timepoint)
                        index=['subj_sess', 'unit', 'lfp1_roi', 'lfp2_roi',
                               'plvs_arr', 'bs_plvs_arr', 'plvs_arr_z', 'plvs', 'plvs_z'])
    
    if save_outputs:
        fpath = os.path.join(output_dir,
                             'lfp_plvs-{}-unit_{}-to-region_{}-{}Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_16.0Hz.pkl'
                             .format(output.subj_sess, output.unit, output.lfp2_roi, sampling_rate))
        dio.save_pickle(output, fpath, verbose=False)
        
    return output
    
def get_epoch_plvs_sta(info_fpath,
                       freqs=np.logspace(np.log10(0.5), np.log10(16), 16),
                       sampling_rate=500,
                       n_cyles=5,
                       n_bootstraps=500,
                       save_outputs=True,
                       input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data',
                       sleep_max=300):
    """Calculate the average phase-locking value at each timepoint w.r.t. spiking, between each LFP-LFP pair.
    
    (Akin to a spike-triggered average for PLV.) Performed for a single neuron to region comparison.
    
    PLVs are calculated for all possible pairs between channels in the unit's microwire bundle
    and channels in the region of interest, and the mean PLV across channel pairs and time_to_spike
    values is then obtained for each frequency. These values are also Z-scored against a 
    null distribution of random timepoints throughout the recording session.
    """
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # Load info DataFrame.
    info = dio.open_pickle(info_fpath)
    subj_sess = info.subj_sess.iat[0]
    unit = info.unit.iat[0]
    
    # General params.
    wavelet_dir = os.path.join(input_dir, 'crosselec_phase_locking', 'wavelet_phase')
    bootstrap_dir = os.path.join(input_dir, 'lfp_plvs', 'bootstrap_shifts')
    spikes_dir = os.path.join(input_dir, 'crosselec_phase_locking', 'spike_inds')
    output_dir = os.path.join(input_dir, 'lfp_plvs', 'plvs', 'sta')
    lfp_fname = 'phase-{}-iChan{}-iFreq{}-{}Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_16.0Hz.pkl'
    sampling_rate = int(sampling_rate)
    cut_inds = int(sampling_rate * n_cyles)
    epoch_sizes = [int(sampling_rate * (1/freq) * n_cyles) for freq in freqs]
    
    # Load spike inds and remove n_cyles secs from the beggining and end of the session,
    # then get bootstrap time-shifted spike inds for generating null distributions.
    spike_inds, n_timepoints = dio.open_pickle(os.path.join(spikes_dir, 'spike_inds-{}Hz-{}-unit{}.pkl'.format(sampling_rate, subj_sess, unit)))
    spike_inds = spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))]
    bs_inds = np.sort(dio.open_pickle(os.path.join(bootstrap_dir, '{}_{}bootstrap_timepoints.pkl'.format(subj_sess, int(n_bootstraps)))))
        
    # Get phase-locking value at each frequency, and generate
    # null distributions by calculating PLVs for random timepoints.
    plvs_arr = []
    bs_plvs_arr = []
    for index, row in info.iterrows():
        lfp1_chan_ind = row.lfp1_chan_ind
        lfp2_chan_ind = row.lfp2_chan_ind
        plvs_arr_ = []
        bs_plvs_arr_ = []
        for iFreq in range(len(freqs)):
            epoch_size = epoch_sizes[iFreq]
            lfp1_phase = dio.open_pickle(os.path.join(wavelet_dir, lfp_fname.format(subj_sess, lfp1_chan_ind, iFreq, sampling_rate)))
            lfp2_phase = dio.open_pickle(os.path.join(wavelet_dir, lfp_fname.format(subj_sess, lfp2_chan_ind, iFreq, sampling_rate)))
            lfp1_epochs = make_epochs(lfp1_phase, spike_inds, epoch_size)
            lfp2_epochs = make_epochs(lfp2_phase, spike_inds, epoch_size)
            plvs_arr_.append([np.mean(x) for x in np.array_split(calc_plv(lfp1_epochs, lfp2_epochs, axis=0), 15)]) # calculated at each timepoint, across spikes
            
            lfp1_epochs = make_epochs(lfp1_phase, bs_inds, epoch_size)
            lfp2_epochs = make_epochs(lfp2_phase, bs_inds, epoch_size)
            bs_plvs_arr_.append([np.mean(x) for x in np.array_split(calc_plv(lfp1_epochs, lfp2_epochs, axis=0), 15)]) # calculated at each timepoint, across spikes
        
        plvs_arr.append(plvs_arr_)
        bs_plvs_arr.append(bs_plvs_arr_)
    
    plvs_arr = np.array(plvs_arr) # channel_pair x frequency x time_to_spike
    bs_plvs_arr = np.array(bs_plvs_arr) # channel_pair x frequency x time_to_random_timepoint
        
    # Z-score PLVs against their respective channel
    # and frequency from the bootstrap distribution.
    bs_means = np.expand_dims(np.mean(bs_plvs_arr, axis=-1), axis=-1)
    bs_stds = np.expand_dims(np.std(bs_plvs_arr, axis=-1), axis=-1)
    plvs_arr_z = (plvs_arr - bs_means) / bs_stds
    
#     # Average the time domain into 15 values (3 values per cycle if n_cycles=5)
#     plvs_arr = np.moveaxis([np.mean(x, axis=-1) for x in np.array_split(plvs_arr, 15, axis=-1)], 0, -1)
#     bs_plvs_arr = np.moveaxis([np.mean(x, axis=-1) for x in np.array_split(bs_plvs_arr, 15, axis=-1)], 0, -1)
#     plvs_arr_z = np.moveaxis([np.mean(x, axis=-1) for x in np.array_split(plvs_arr_z, 15, axis=-1)], 0, -1)
    
    # Get phase-locking stats and append to the info series.    
    output = pd.Series({'subj_sess': subj_sess,
                        'unit': unit,
                        'lfp1_roi': info.lfp1_hemroi.iat[0],
                        'lfp2_roi': info.lfp2_hemroi.iat[0],
                        'plvs': np.mean(plvs_arr, axis=0), # PLV by frequency, time_to_spike (mean across channel pairs)
                        'plvs_z': np.mean(plvs_arr_z, axis=0)}, # Z-PLV by frequency, time_to_spike (mean across channel pairs)
                        index=['subj_sess', 'unit', 'lfp1_roi', 'lfp2_roi', 'plvs', 'plvs_z'])
    
    if save_outputs:
        fpath = os.path.join(output_dir,
                             'STA-lfp_plvs-{}-unit_{}-to-region_{}-{}Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_16.0Hz.pkl'
                             .format(output.subj_sess, output.unit, output.lfp2_roi, sampling_rate))
        dio.save_pickle(output, fpath, verbose=False)
        
    return output
    
def get_mrl_plv_diffs(info_fpath,
                      freqs=np.logspace(np.log10(0.5), np.log10(16), 16),
                      sampling_rate=500,
                      n_cyles=5,
                      n_bootstraps=500,
                      save_outputs=True,
                      input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data',
                      sleep_max=300):
    """Calculate the MRL of spike phases by quartile, sorted by the phase-locking value around each spike.
        
    Performed for a single neuron to channel comparison. Also returns the difference
    in mean resultant length between spike phases in the top vs. bottom quartiles.
    Values are compared against a null distribution of time-shifted spike trains.
    
    Parameters
    ----------
    info_fpath : str
         File path to a pandas Series object containing one row
         of data from the get_plv_df (contains all possible channel
         pairs for a given recording session).
    freqs : numpy.ndarray
        Array of wavelet frequencies, in Hz, from which phase values
        are to be taken.
    sampling_rate : int
        Sampling rate of the data, in Hz.
    n_cycles : int
        Number of cycles, for a given frequency, from which to draw
        phase-locking values that are centered around each spike.
    n_bootstraps : int
        Number of bootstrap time shifts from which to draw a null
        distribution for each frequency.
    save_outputs : bool
        If True, a pandas Series object is saved with the output info.
    input_dir : str
        Project directory from which input and output paths are generated.
    sleep_max : int
        Maximum number of seconds to randomly sleep before running this
        function. 
        
    Returns
    -------
    info : pandas.core.series.Series
        The input info Series plus output columns shown at the bottom
        of this function.
    """
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # Load info DataFrame.
    info = dio.open_pickle(info_fpath)
    subj_sess = info.subj_sess
    unit = info.unit
    lfp1_chan_ind = info.lfp1_chan_ind
    lfp2_chan_ind = info.lfp2_chan_ind
    
    # General params.
    wavelet_dir = os.path.join(input_dir, 'crosselec_phase_locking', 'wavelet_phase')
    bootstrap_dir = os.path.join(input_dir, 'lfp_plvs', 'bootstrap_shifts')
    spikes_dir = os.path.join(input_dir, 'crosselec_phase_locking', 'spike_inds')
    output_dir = os.path.join(input_dir, 'lfp_plvs', 'plvs', 'mrl_diffs')
    lfp_fname = 'phase-{}-iChan{}-iFreq{}-{}Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_16.0Hz.pkl'
    sampling_rate = int(sampling_rate)
    cut_inds = int(sampling_rate * n_cyles)
    epoch_sizes = [int(sampling_rate * (1/freq) * n_cyles) for freq in freqs]
    
    # Load spike inds and remove n_cyles secs from the beggining and end of the session,
    # then get bootstrap time-shifted spike inds for generating null distributions.
    spike_inds, n_timepoints = dio.open_pickle(os.path.join(spikes_dir, 'spike_inds-{}Hz-{}-unit{}.pkl'.format(sampling_rate, subj_sess, unit)))
    spike_inds = spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))]
    half_win = int(epoch_sizes[0]/2) + 1
    bs_steps = dio.open_pickle(os.path.join(bootstrap_dir, '{}_{}bootstraps.pkl'.format(subj_sess, int(n_bootstraps))))
    bs_spike_inds = []
    for step in bs_steps:
        spike_inds_shifted = phase_locking.shift_spike_inds(spike_inds, n_timepoints - half_win, step)
        spike_inds_shifted[spike_inds_shifted<=half_win] = spike_inds_shifted[spike_inds_shifted<=half_win] + half_win
        bs_spike_inds.append(spike_inds_shifted)
        
    # Get phase-locking value at each frequency, and generate
    # null distributions from time-shifted spike trains.
    plvs = []
    spike_phases = []
    bs_plvs = []
    bs_spike_phases = []
    for iFreq in range(len(freqs)):
        epoch_size = epoch_sizes[iFreq]
        lfp1_phase = dio.open_pickle(os.path.join(wavelet_dir, lfp_fname.format(subj_sess, lfp1_chan_ind, iFreq, sampling_rate)))
        lfp2_phase = dio.open_pickle(os.path.join(wavelet_dir, lfp_fname.format(subj_sess, lfp2_chan_ind, iFreq, sampling_rate)))
        lfp1_epochs = make_epochs(lfp1_phase, spike_inds, epoch_size)
        lfp2_epochs = make_epochs(lfp2_phase, spike_inds, epoch_size)
        plvs.append(calc_plv(lfp1_epochs, lfp2_epochs, axis=-1))
        spike_phases.append(lfp2_phase[spike_inds])

        bs_plvs_ = []
        bs_spike_phases_ = []
        for iStep in range(n_bootstraps):
            lfp1_epochs = make_epochs(lfp1_phase, bs_spike_inds[iStep], epoch_size)
            lfp2_epochs = make_epochs(lfp2_phase, bs_spike_inds[iStep], epoch_size)
            bs_plvs_.append(calc_plv(lfp1_epochs, lfp2_epochs, axis=-1))
            bs_spike_phases_.append(lfp2_phase[bs_spike_inds[iStep]])
        bs_plvs.append(bs_plvs_)
        bs_spike_phases.append(bs_spike_phases_)
    
    plvs = np.array(plvs) # frequency x spike_event
    spike_phases = np.array(spike_phases) # frequency x spike_event
    bs_plvs = np.array(bs_plvs) # frequency x bootstrap_ind x spike_event
    bs_spike_phases = np.array(bs_spike_phases) # frequency x bootstrap_ind x spike_event
    
    # Get MRLs by quartile and the difference in MRL between
    # top and bottom quartiles, then generate these values for
    # the null distribution.
    qtl_mrls = []
    mrl_diffs = []
    bs_qtl_mrls = []
    bs_mrl_diffs = []
    for iFreq in range(len(freqs)):
        vals = calc_mrl_qtls(spike_phases[iFreq, :], plvs[iFreq, :])
        qtl_mrls.append(vals[0])
        mrl_diffs.append(vals[1])
        
        bs_qtl_mrls_ = []
        bs_mrl_diffs_ = []
        for iStep in range(n_bootstraps):
            vals = calc_mrl_qtls(bs_spike_phases[iFreq, iStep, :], bs_plvs[iFreq, iStep, :])
            bs_qtl_mrls_.append(vals[0])
            bs_mrl_diffs_.append(vals[1])
        bs_qtl_mrls.append(bs_qtl_mrls_)
        bs_mrl_diffs.append(bs_mrl_diffs_)
    
    info['bs_qtl_mrls'] = np.array(bs_qtl_mrls) # frequency x bootstrap_ind x quartile
    info['bs_mrl_diffs'] = np.array(bs_mrl_diffs) # frequency x bootstrap_ind
    info['qtl_mrls'] = np.array(qtl_mrls) # frequency x quartile
    info['mrl_diffs'] = np.array(mrl_diffs) # frequency
    info['qtl_mrls_z'] = (info.qtl_mrls - np.mean(info.bs_qtl_mrls, axis=1)) / np.std(info.bs_qtl_mrls, axis=1) # frequency x quartile
    info['mrl_diffs_z'] = info.qtl_mrls_z[:, -1] - info.qtl_mrls_z[:, 0] # frequency
    
    if save_outputs:
        fpath = os.path.join(output_dir,
                             'mrl_plv_diffs-{}-unit_{}-to-{}-chan_ind_{}-to-chan_ind_{}-{}Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_16.0Hz.pkl'
                             .format(subj_sess, unit, info.lfp2_hemroi, lfp1_chan_ind, lfp2_chan_ind, sampling_rate))
        dio.save_pickle(info, fpath, verbose=False)
    
    return info
    
def make_epochs(vec, epoch_inds, epoch_size):
    """Epoch vec around a series of event indices.
    
    Parameters
    ----------
    vec : np.ndarray
        A vector of data that the epochs are drawn from.
    epoch_inds : np.ndarray
        Indices of vec that epochs are centered around.
    epoch_size : int
        Number of data points that comprise each epoch.
    
    Returns
    -------
    np.ndarray
        n_epochs x epoch_size array with the epoched data.
    """
    start_inds = epoch_inds - int(epoch_size/2)
    return aop.rolling_window(vec, epoch_size)[start_inds, :]