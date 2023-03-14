"""
eeg_alignment.py

Author:
    Daniel Schonhaut
    Computational Memory Lab
    University of Pennsylvania
    daniel.schonhaut@gmail.com
    
Description: 
    Functions for aligning EEG channel sync pulses to event sync pulse times.

Last Edited: 
    11/6/19
"""

# General
import sys
import os
from time import time
from collections import OrderedDict as od
from glob import glob
import itertools

# Scientific
import numpy as np
import pandas as pd
import scipy.io as sio

# Stats
import scipy.stats as stats
import statsmodels.api as sm
import random

# Personal
sys.path.append('/Volumes/rhino/home1/dscho/code/general')
import data_io as dio
import array_operations as aop

def rmse(v1, v2):
    """Return the root mean squared error
    between equal-length vectors v1 and v2.
    """
    err = v1 - v2
    return np.sqrt(np.dot(err, err)/len(v1))

def find_pulse_starts(sync_chan, 
                      pulse_thresh=200, # voltage change
                      interpulse_thresh=100, # 50ms at 2000Hz sr
                      intrapulse_thresh=10 # 5ms at 2000Hz sr
                     ): 
    """Return sync_chan indices that mark that start of each sync pulse.
    
    Note: the default arguments were defined on data that were sampled
    at 2000 Hz and might need to be adjusted if the sampling rate
    differs much from this.
    
    Algorithm
    ---------
    1) Identifies sync pulse periods by finding sync channel indices
       for which the absolute value absolute value of the trace 
       derivative exceeds pulse_thresh. 
    2) Identifies the start of each sync pulse by finding suprathreshold
       sync pulse indices for which the inter-pulse interval exceeds
       interpulse_thresh, and for which the subsequent suprathreshold 
       sync pulse occurs within a certain number of indices, defined by
       intrapulse_thresh. In other words, we are looking for dramatic
       changes in voltage that occur some time after the last dramatic
       voltage change, and that are sustained for some period of time.
    
    Parameters
    ----------
    sync_chan : numpy.ndarray
        Voltage trace from the channel that
        the sync box was plugged into
    pulse_thresh : int or float
        See algorithm description.
    interpulse_thresh : int or float
        See algorithm description.
    intrapulse_thresh : int or float
        See algorithm description.
    
    Returns
    -------
    pulse_startinds : numpy.ndarray
        Array of indices that mark the start of each sync pulse.
    """
    # Find sync pulses by looking for suprathreshold changes 
    # in the absolute value of the derivative of the sync channel
    pulse_thresh = 200
    sync_pulses = np.abs(np.pad(np.diff(sync_chan), (1, 0), 'constant'))>pulse_thresh
    pulse_inds = np.where(sync_pulses)[0]

    # Find the inter-pulse intervals
    ipis = np.insert(np.diff(pulse_inds), 0, pulse_inds[0])

    # Identify the start of each pulse by finding suprathreshold
    # inter-pulse intervals that are followed by a short IPI.
    interpulse_thresh = 200
    intrapulse_thresh = 10
    pulse_startinds = pulse_inds[[i for i in range(len(ipis)-1) 
                                  if ((ipis[i]>interpulse_thresh) 
                                      & (ipis[i+1]<intrapulse_thresh))]]
    return pulse_startinds

def align_sync_pulses(event_synctimes, # vector of event sync times
                      lfp_synctimes, # vector of LFP sync times in ms
                      good_fit_thresh=10 # 10ms at 2000Hz sr
                     ):
    """Return the slope and intercept to align event to LFP times.
    
    Algorithm
    ---------
    1) Subtracts the first sync time from all sync times, so
       both vectors start at 0.
    2) Finds the best fit between event and LFP sync times
       by comparing their inter-pulse intervals at 30 offset
       steps for the LFP sync times. An exception is raised if
       a good fit is not found.
    3) Finds the closest LFP sync time to each event sync time.
    4) Estimates the intercept and slope to align event to
       LFP sync times using robust linear regression.
       
    Parameters
    ----------
    event_synctimes : numpy.ndarray
        Vector of event sync times
    lfp_synctimes : numpy.ndarray
        Vector of LFP sync times
    good_fit_thresh : int or float
        Cutoff for the mean RMSE between event and LFP inter-pulse
        times. If we can't find a goot alignment < this threshold,
        the function will raise an exception.
        
    Returns
    -------
    sync_params : collections.OrderedDict
        Intercept and slope to align
        event timestamps to LFP timestamps
    before_stats : collections.OrderedDict
        Pearson correlation and RMSE between
        event and LFP sync times before alignment.
    after_stats : collections.OrderedDict
        Pearson correlation and RMSE between
        event and LFP sync times after alignment.
    """
    def rmse(v1, v2):
        """Return the root mean squared error
        between equal-length vectors v1 and v2.
        """
        err = v1 - v2
        return np.sqrt(np.dot(err, err)/len(v1))
    
    # Make the first pulse start at time 0.
    event_synctimes_ = np.copy(event_synctimes - event_synctimes[0])
    lfp_synctimes_ = np.copy(lfp_synctimes - lfp_synctimes[0])
    
    # Find the best starting fit between event and LFP sync times
    # by comparing the inter-pulse intervals for each, testing
    # LFP sync shifts over the first 30 labeled pulses.
    min_syncs = np.min((len(event_synctimes_), len(lfp_synctimes_)))
    ipi_fits = [rmse(np.diff(event_synctimes_[:min_syncs-30]),
                     np.diff(lfp_synctimes_[i:i+min_syncs-30]))
                for i in range(30)]
    best_fit_ind = np.argmin(ipi_fits)
    lfp_synctimes_ = lfp_synctimes_[best_fit_ind:]
    
    # If we couldn't find a good match between the inter-pulse
    # intervals there is no point in trying to align.
    if ipi_fits[best_fit_ind] > good_fit_thresh:
        msg = ('Could not find a good inter-pulse interval alignment (RMSE={:.2f})'
               .format(ipi_fits[best_fit_ind]))
        raise RuntimeError(msg)
    
    # For each event sync time, find the closest LFP sync time.
    sync_pairs = np.array([(event_synctimes_[i], 
                            lfp_synctimes_[np.abs(lfp_synctimes_ - event_synctimes_[i])
                                           .argmin()])
                           for i in range(len(event_synctimes_))])
    
    # Get a robust linear fit between the event/LFP sync pairs.
    X = sm.add_constant(sync_pairs[:, 0]) # the event sync times
    y = sync_pairs[:, 1] # the LFP channel sync times
    rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()
    intercept, slope = rlm_results.params
    
    # Add back the difference in starting times.
    intercept += (lfp_synctimes[0] - event_synctimes[0])
    
    # See how well the alignment went.
    sync_params = od([('intercept', intercept), ('slope', slope)])
    event_synctimes_aligned = intercept + (slope * event_synctimes)
    before_stats = od([('r', stats.pearsonr(event_synctimes, lfp_synctimes)[0]),
                       ('rmse', rmse(event_synctimes, lfp_synctimes))])
    after_stats = od([('r', stats.pearsonr(event_synctimes_aligned, lfp_synctimes)[0]),
                      ('rmse', rmse(event_synctimes_aligned, lfp_synctimes))])
    return sync_params, before_stats, after_stats