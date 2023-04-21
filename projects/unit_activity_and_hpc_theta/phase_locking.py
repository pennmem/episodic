"""
phase_locking.py

Author:
    Daniel Schonhaut
    Computational Memory Lab
    University of Pennsylvania
    daniel.schonhaut@gmail.com
    
Description: 
    Functions for calculating phase locking.

Last Edited: 
    4/19/21
"""
import sys
import os.path as op
import pickle
import glob
from time import sleep
import random
from collections import OrderedDict
od = OrderedDict
import itertools 
from math import floor, ceil
import scipy.stats as stats
import statsmodels.api as sm
import astropy.stats.circstats as circstats
import pycircstat
# import mkl
# mkl.set_num_threads(1)
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import mne
from ptsa.data.TimeSeriesX import TimeSeries
# from xarray import DataArray as TimeSeries
sys.path.append('/home1/dscho/code/general')
sys.path.append('/home1/dscho/code/projects/manning_replication')
import data_io as dio
import array_operations as aop
import spectral_processing as spp
import manning_analysis
from eeg_plotting import plot_trace2
import neurodsp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def calc_power_by_pl_fr_unit_to_target_region2(info,
                                               n_freqs=16,
                                               n_bootstraps=1000,
                                               sampling_rate=2000,
                                               time_win=2,
                                               input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                               save_outputs=True,
                                               overwrite=False,
                                               output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/power_by_pl_fr/lfp_target_region/sig_all_rois',
                                               sleep_max=0):
    """Obtain powers in each region and power correlations between each region pair,
    for strongly phase-locked spikes relative to randomly selected spikes.
    
    The following steps are performed in reference to the mean power
    across channels for each depth electrode. The HPC region that 
    phase relations were taken in reference to is referred to as the
    "target" region, and the neuron's region of origin is called the
    "local" region and does not include the neuron's own channel.
    
    We find the 20% of spikes that fired closest to the preferred phase 
    at the preferred phase-locked frequency, in reference to the target 
    (i.e. hippocampal) LFP.
    
    We then get:
    1) The mean (across spikes) power at each frequency, for each region. 
    2) The mean (across spikes) cross-frequency correlation between powers 
       in each pair of regions.
    
    The target region is named "target," and the neuron's local region
    (excluding power values from the neuron's own microwire) is named 
    "local."
    
    Mean power and power correlation vectors for each spike category 
    are Z-scored against a null distribution of randomly selected spikes 
    (20% of spikes for each permutation). 
    
    We return a dictionary with:
    1) The Z-scored vector of powers for each region.
    2) The Z-scored null distribution of powers for each region.
    3) The Z-scored cross-frequency power correlation for each region pair.
    4) The Z-scored null distribution of cross-frequency power correlations 
       for each region pair.
    """ 
    # Check if output file already exists.
    subj_sess = info.subj_sess
    unit = info.unit
    lfp_roi = info.lfp_hemroi
    output_fname = 'power_by_phase_locking_strength-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl'
    fpath = op.join(output_dir, output_fname.format(subj_sess, unit, lfp_roi))
    if op.exists(fpath) and not overwrite:
        output = dio.open_pickle(fpath)
        return output
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # General params.
    power_fname = op.join(input_dir, 'wavelet', 'power', 
                          'power-Z-log-{}-iChan{}-iFreq{}-2000Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    spike_fr_fname = op.join(input_dir, 'spike_frs', 'spike_frs-2000Hz-{}-unit{}.pkl')
    pl_fname = op.join(input_dir, 'phase_locking', 'unit_to_region', 
                       'phase_locking_stats-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    
    # Get phase offsets for each spike.
    target_phase_offsets = info.phase_offsets
    spike_inds, n_timepoints = dio.open_pickle(op.join(input_dir, 'spike_inds', 
                                                       'spike_inds-{}Hz-{}-unit{}.pkl'
                                                       .format(sampling_rate, subj_sess, unit)))
    cut_inds = int(sampling_rate * time_win)
    keep_spikes = np.unique(np.where(spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))])[0])
    spike_frs = dio.open_pickle(spike_fr_fname.format(subj_sess, unit))[keep_spikes]
    spike_inds = info.spike_inds
    assert len(spike_inds) == len(spike_frs) == len(target_phase_offsets)
    
    # Cut spikes from the beginning so vectors are divisible by 5.
    cut = len(spike_inds) % 5
    spike_inds = spike_inds[cut:]
    spike_frs = spike_frs[cut:]
    target_phase_offsets = target_phase_offsets[cut:]
    n_spikes = len(spike_inds)
    n_samp = int(n_spikes / 5) # 20% of all spikes
    
    # Setup the spike categories.
    sel_spikes = np.where(target_phase_offsets<=np.percentile(target_phase_offsets, 20))[0]
    
    # Select 20% of spikes at random for each bootstrap iteration.
    bs_sel_spikes = []
    for iBoot in range(n_bootstraps):
        bs_sel_spikes.append(np.random.choice(n_spikes, n_samp, replace=False))
    
    # Assess LFP power (mean across channels) at each frequency,
    # at each spike time, for channels in:
    # 1) the target region
    # 2) the unit's region (excluding the unit's channel) 
    # 3) each off-target region
    rois = info.off_target_chan_inds # an OrderedDict([(roi : str, chan_inds : list)])
    rois['local'] = info.local_lfp_chan_inds
    rois['target'] = info.lfp_chan_inds
    rois = OrderedDict(reversed(list(rois.items())))
    roi_names = list(rois.keys())
    
    # Get LFP power across frequencies at each spike time, taking the mean
    # power across channels in a region at each frequency and spike time
    power = []
    for roi, chan_inds in rois.items():
        power.append(np.mean([[dio.open_pickle(power_fname.format(subj_sess, iChan, iFreq))[spike_inds] 
                               for iFreq in range(n_freqs)]
                              for iChan in chan_inds], axis=0))
    power = np.array(power) # roi x freq x spike
    
    # Get power at each freq and time for the 20% of spikes that fired closest
    # to the preferred phase, or with the highest or lowest firing rate
    spike_powers = OrderedDict([])
    bs_spike_powers = OrderedDict([])
    for iRoi, roi in enumerate(roi_names):
        power_ = power[iRoi, :, :] # freq x spike
        spike_powers[roi] = np.mean(power_[:, sel_spikes], axis=-1) # freq; mean over spikes
    
        # Create a null distribution of powers at each frequency from random spike subsets.
        bs_powers_ = np.swapaxes([np.mean(power_[:, bs_sel_spikes[iBoot]], axis=-1) 
                                  for iBoot in range(n_bootstraps)], 0, 1) # freq x bs_index
        bs_means_ = np.expand_dims(np.mean(bs_powers_, axis=-1), axis=-1) # mean at each freq
        bs_stds_ = np.expand_dims(np.std(bs_powers_, axis=-1), axis=-1) # std at each freq
        bs_spike_powers[roi] = ((bs_powers_ - bs_means_) / bs_stds_) # freq x bs_index
        
        # Z-score powers against the null distribution.
        spike_powers[roi] = (spike_powers[roi] - np.squeeze(bs_means_)) / np.squeeze(bs_stds_) # freq
        
    # Add to output.
    output = OrderedDict([('z_power', spike_powers),
                          ('z_power_null', bs_spike_powers)])    
    
    
    # Get the cross-frequency power correlation between all possible
    # pairs of regions, for each spike, and keep track of which pair
    # compares the local region to the target region
    local_ind = roi_names.index('local')
    target_ind = roi_names.index('target')
    roi_pairs = [(x, y) for x in range(len(roi_names)) for y in range(len(roi_names)) if x<y]
    local_target_ind = roi_pairs.index((target_ind, local_ind)) # should be 0
    
    power_corrs = OrderedDict([]) # all spikes
    spike_power_corrs = OrderedDict([]) # mean over just the strongly phase-locked spikes
    bs_spike_power_corrs = OrderedDict([])
    for iRoi1, iRoi2 in roi_pairs:
        roi_pair = '{}_{}'.format(roi_names[iRoi1], roi_names[iRoi2])
        power_corrs[roi_pair] = np.array([stats.pearsonr(power[iRoi1, :, iSpike], power[iRoi2, :, iSpike])[0] 
                                          for iSpike in range(power.shape[-1])]) # spike
        
        spike_power_corrs[roi_pair] = np.mean(power_corrs[roi_pair][sel_spikes]) # scalar
    
        # Create a null distribution of power correlations at each frequency from random spike subsets.
        bs_power_corrs_ = np.array([np.mean(power_corrs[roi_pair][bs_sel_spikes[iBoot]]) 
                                    for iBoot in range(n_bootstraps)]) # bs_index
        bs_corr_mean_ = np.mean(bs_power_corrs_)
        bs_corr_std_ = np.std(bs_power_corrs_)
        bs_spike_power_corrs[roi_pair] = (bs_power_corrs_ - bs_corr_mean_) / bs_corr_std_ # bs_index
        
        # Z-score power correlation against the null distribution.
        spike_power_corrs[roi_pair] = (spike_power_corrs[roi_pair] - bs_corr_mean_) / bs_corr_std_ # scalar
        
    # Add to output.
    output['z_power_corr'] = spike_power_corrs
    output['z_power_corr_null'] = bs_spike_power_corrs

    # Save output.
    if save_outputs:
        dio.save_pickle(output, fpath, verbose=False)
        
    return output
    

def calc_xpower_corrs_by_pl_fr_unit_to_target_region(info,
                                                     n_freqs=16,
                                                     n_bootstraps=1000,
                                                     sampling_rate=2000,
                                                     time_win=2,
                                                     test_phase_offsets=True, # if False, only assess spikes by firing rate
                                                     subtract_mean=True,
                                                     input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                                     save_outputs=True,
                                                     overwrite=False,
                                                     output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/power_by_pl_fr/lfp_target_region/xpower_corrs',
                                                     sleep_max=0):
    """Obtain power values for highly phase-locked or high firing spikes.
    
    The following steps are performed in reference to the mean power
    across channels for each depth electrode. The HPC region that 
    phase relations were taken in reference to is referred to as the
    "target" region, and the neuron's region of origin is called the
    "local" region and does not include the neuron's own channel.
    
    We find the 20% of spikes that fired:
    1) Closest to the preferred phase at the preferred phase-locked 
       frequency, in reference to the target (i.e. HPC) LFP.
    2) Closest to the preferred phase in reference to the local
       LFP, but at the preferred target LFP phase-locking frequency
    3) With the highest firing rate
    
    We then get:
    1) The mean power across spikes at each frequency,
       for each of these spike categories. 
    2) The Pearson correlation of power at each frequency across spikes,
       for each of these spike categories.
    
    Mean power and power correlation vectors for each spike category 
    are Z-scored against a null distribution of randomly selected spikes 
    (20% of spikes for each permutation). 
    
    We return a dictionary with:
    1) The Z-power vector for each spike category.
    2) A p-value and some associated metrics for each spike category.
       The p-value is determined from a permutation test in which we compare
       the max(|z_power|) across frequencies against the null distribution 
       of max(|z_power|) from each permutation. This test enables us to detect 
       spike subsets that showed significantly higher or lower power at any 
       frequency, compared to the null distribution of spikes drawn at random.
    3) The Z-power correlation vector for each spike category.
    4) A p-value and some associated metrics for each spike category,
       using the same logic as in (2) but with Z-power correlations rather
       than Z-power at each frequency.
    """
    def get_xpower_corrs(roi1_power, roi2_power):
        """Take an n_freq x n_spikes matrix of power values,
        and return a vector of across-spike power correlations
        between every pair of frequencies in each region.
        """        
        xpower_corrs = [stats.pearsonr(roi1_power[iFreq1, :], roi2_power[iFreq2, :])[0]
                        for (iFreq1, iFreq2) in freq_pairs]
        return xpower_corrs
        
    # Check if output file already exists.
    subj_sess = info.subj_sess
    unit = info.unit
    lfp_roi = info.lfp_hemroi
    output_fname = 'xpower_corrs_by_phase_locking_and_fr-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl'
    fpath = op.join(output_dir, output_fname.format(subj_sess, unit, lfp_roi))
    if op.exists(fpath) and not overwrite:
        output = dio.open_pickle(fpath)
        return output
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # General params.
    power_fname = op.join(input_dir, 'wavelet', 'power', 
                          'power-Z-log-{}-iChan{}-iFreq{}-2000Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    spike_fr_fname = op.join(input_dir, 'spike_frs', 'spike_frs-2000Hz-{}-unit{}.pkl')
    pl_fname = op.join(input_dir, 'phase_locking', 'unit_to_region', 
                       'phase_locking_stats-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    
    # Get phase offsets for each spike.
    target_phase_offsets = info.phase_offsets_tl_locked_time_freq_z
    local_phase_offsets = info.local_phase_offsets
    spike_inds, n_timepoints = dio.open_pickle(op.join(input_dir, 'spike_inds', 
                                                       'spike_inds-{}Hz-{}-unit{}.pkl'
                                                       .format(sampling_rate, subj_sess, unit)))
    cut_inds = int(sampling_rate * time_win)
    keep_spikes = np.unique(np.where(spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))])[0])
    spike_frs = dio.open_pickle(spike_fr_fname.format(subj_sess, unit))[keep_spikes]
    spike_inds = info.spike_inds
    assert len(spike_inds) == len(spike_frs) == len(target_phase_offsets) == len(local_phase_offsets)
    
    # Cut spikes from the beginning so vectors are divisible by 5.
    cut = len(spike_inds) % 5
    spike_inds = spike_inds[cut:]
    spike_frs = spike_frs[cut:]
    target_phase_offsets = target_phase_offsets[cut:]
    local_phase_offsets = local_phase_offsets[cut:]    
    n_spikes = len(spike_inds)
    n_samp = int(n_spikes / 5) # 20% of all spikes
    
    # Setup the spike categories.
    spike_cat = od([('high_firing_rate', spike_frs)])
    if test_phase_offsets:
        spike_cat['target_phase_locking'] = target_phase_offsets
    
    # Select 20% of spikes at random for each bootstrap iteration.
    bs_spike_inds = []
    for iBoot in range(n_bootstraps):
        bs_spike_inds.append(np.random.choice(n_spikes, n_samp, replace=False))
    
    # Assess LFP power (mean across channels) at each frequency,
    # at each spike time, for channels in:
    # 1) the target region
    # 2) the unit's region (excluding the unit's channel) 
    # 3) each off-target region
    lfp_power_rois = info.off_target_chan_inds # an OrderedDict([(roi : str, chan_inds : list)])
    lfp_power_rois['target'] = info.lfp_chan_inds
    lfp_power_rois['local'] = info.local_lfp_chan_inds
    lfp_power_rois = OrderedDict(reversed(list(lfp_power_rois.items())))
    
    # Get LFP power across frequencies at each spike time, taking the mean
    # power across channels in a region at each frequency and spike time
    power = []
    for lfp_power_roi, lfp_chan_inds in lfp_power_rois.items():
        power.append(np.mean([[dio.open_pickle(power_fname.format(subj_sess, iChan, iFreq))[spike_inds] 
                               for iFreq in range(n_freqs)]
                              for iChan in lfp_chan_inds], axis=0))
    power = np.array(power) # roi x freq x spike
        
    # Get the cross-frequency power correlation between all possible
    # pairs of regions, for each spike, and keep track of which pair
    # compares the local region to the target region
    local_ind = list(lfp_power_rois.keys()).index('local')
    target_ind = list(lfp_power_rois.keys()).index('target')
    roi_pairs = [(x, y) for x in range(len(lfp_power_rois)) for y in range(len(lfp_power_rois)) if x<y]
    local_target_ind = roi_pairs.index((local_ind, target_ind)) # should be 0
    freq_pairs = [(x, y) for x in range(n_freqs) for y in range(n_freqs)]
    
    xpower_corrs = od()
    for key in spike_cat.keys(): 
        # Select the spike subset
        if key == 'high_firing_rate':
            sel_spikes = np.where(spike_cat[key]>=np.percentile(spike_cat[key], 80))[0]
        else:
            sel_spikes = np.where(spike_cat[key]<=np.percentile(spike_cat[key], 20))[0]
            
        # Subtract the mean correlation across region pairs,
        # for each frequency pair.
        if subtract_mean:
            xpower_corrs[key] = np.array([get_xpower_corrs(power[iRoi1, :, sel_spikes], 
                                                           power[iRoi2, :, sel_spikes])
                                          for (iRoi1, iRoi2) in roi_pairs]) # roi_pair x freq_pair
            xpower_corrs[key] = xpower_corrs[key][local_target_ind, :] - np.mean(xpower_corrs[key], axis=0) # freq_pair
        else:
            xpower_corrs[key] = np.array(get_xpower_corrs(power[local_ind, :, sel_spikes], power[target_ind, :, sel_spikes])) # freq_pair
    
    # Create the null distribution.
    bs_xpower_corrs = []
    for iBoot in range(n_bootstraps):
        # Subtract the mean correlation across region pairs,
        # for each frequency pair.
        if subtract_mean:
            bs_xpower_corrs.append(np.array([get_xpower_corrs(power[iRoi1, :, bs_spike_inds[iBoot]], 
                                                              power[iRoi2, :, bs_spike_inds[iBoot]])
                                             for (iRoi1, iRoi2) in roi_pairs]))
            bs_xpower_corrs[-1] = bs_xpower_corrs[-1][local_target_ind, :] - np.mean(bs_xpower_corrs[-1], axis=0)
        else:
            bs_xpower_corrs.append(get_xpower_corrs(power[local_ind, :, bs_spike_inds[iBoot]], power[target_ind, :, bs_spike_inds[iBoot]]))
    
    bs_xpower_corrs = np.array(bs_xpower_corrs) # bs_ind x freq_pair
    
    # Z-score the powers at each frequency and get an empirical p-value for each spike category.
    for key in spike_cat.keys():
        xpower_corrs[key] = (xpower_corrs[key] - np.mean(bs_xpower_corrs, axis=0)) / np.std(bs_xpower_corrs, axis=0)

    # Reshape correlations into an n_freq x n_freq matrix.
    # Axes are (local, target), where each value is the
    # local_freq x target_freq correlation.
    for key in spike_cat.keys():
        xpower_corrs[key] = xpower_corrs[key].reshape([n_freqs, n_freqs])
    
    # Save output.
    if save_outputs:
        dio.save_pickle(xpower_corrs, fpath, verbose=False)
        
    return xpower_corrs
    

def calc_hpcfr_by_pl_fr_unit_to_target_region(info,
                                              n_bootstraps=1000,
                                              sampling_rate=2000,
                                              time_win=2,
                                              test_phase_offsets=True, # if False, only assess spikes by firing rate
                                              input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                              save_outputs=True,
                                              overwrite=False,
                                              output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/hippocampal_fr_by_pl_fr/lfp_target_region/sig',
                                              sleep_max=0):
    """Obtain average firing rate among hippocampal neurons
    for highly phase-locked or high firing spikes among cells
    outside the hippocampus.
    
    The following steps are performed in reference to the mean firing
    rate across neurons for each depth electrode. The HPC region that 
    phase relations were taken in reference to is referred to as the
    "target" region, and the neuron's region of origin is called the
    "local" region and does not include the neuron's own channel.
    
    We find the 20% of spikes that fired:
    1) Closest to the preferred phase at the preferred phase-locked 
       frequency, in reference to the target (i.e. HPC) LFP.
    2) Closest to the preferred phase in reference to the local
       LFP, but at the preferred target LFP phase-locking frequency
    3) With the highest firing rate
    
    We then get:
    1) The mean firing rate across hippocampal single-units
       in the target region, for each spike in the spike categories
       listed above. For each neuron, firing rate is Z-scored 
       across all timepoints in the session.
    2) Same as (1) but subtracting the mean firing rate 
       across all microwire bundles from the hippocampal firing rate
       at a given timepoint.
    
    The firing rate vectors for each spike category are Z-scored 
    against a null distribution of randomly selected spikes 
    (20% of spikes for each permutation). 
    
    We return a dictionary with:
    1) The Z-fr vector for each spike category.
    2) A p-value and some associated metrics for each spike category.
       The p-value is determined from a permutation test in which we compare
       the |z_fr| against the null distribution of |z_fr| from each 
       permutation. This test enables us to detect spike subsets that showed 
       significantly higher or lower hippocampal firing rates, compared to 
       the null distribution of spikes drawn at random.
    3) The Z-fr-meanSub vector for each spike category.
    4) A p-value and some associated metrics for each spike category,
       using the same logic as in (2).
    """
    def bootstrap_p(obs, null):
        """Derive an empirical p-value.

        The maximum |Z-score| (across frequencies) is compared
        against the null distribution of maximum |Z-scores|.

        obs : n_freq vec
        null : n_freq x n_boot array
        """
        n_bootstraps = null.shape[-1]

        if len(null.shape) == 1:
            max_obs = np.abs(obs)
            max_null = np.abs(null)
            bs_ind = np.sum(max_null >= max_obs)
            pval = (1 + bs_ind) / (1 + n_bootstraps)

            return OrderedDict([('max_z', obs),
                                ('max_z_ind', 0),
                                ('bs_ind', bs_ind),
                                ('pval', pval)])
        else:
            max_obs = np.max(np.abs(obs))
            max_null = np.max(np.abs(null), axis=0)
            bs_ind = np.sum(max_null >= max_obs)
            pval = (1 + bs_ind) / (1 + n_bootstraps)

            # Get the Z-score and index of max(|Z|)
            max_z_ind = np.argmax(np.abs(obs))
            max_z = obs[max_z_ind]

            return OrderedDict([('max_z', max_z),
                                ('max_z_ind', max_z_ind),
                                ('bs_ind', bs_ind),
                                ('pval', pval)])
        
    # Check if output file already exists.
    subj_sess = info.subj_sess
    unit = info.unit
    lfp_roi = info.lfp_hemroi
    output_fname = 'hippocampal_firing_rate_by_phase_locking_and_fr-{}-unit_{}-lfp_{}-2000Hz.pkl'
    fpath = op.join(output_dir, output_fname.format(subj_sess, unit, lfp_roi))
    if op.exists(fpath) and not overwrite:
        output = dio.open_pickle(fpath)
        return output
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # General params.
    spike_fr_fname = op.join(input_dir, 'spike_frs', 'spike_frs-2000Hz-{}-unit{}.pkl')
    pl_fname = op.join(input_dir, 'phase_locking', 'unit_to_region', 
                       'phase_locking_stats-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    
    # Get phase offsets for each spike.
    target_phase_offsets = info.phase_offsets_tl_locked_time_freq_z
    local_phase_offsets = info.local_phase_offsets
    spike_inds, n_timepoints = dio.open_pickle(op.join(input_dir, 'spike_inds', 
                                                       'spike_inds-{}Hz-{}-unit{}.pkl'
                                                       .format(sampling_rate, subj_sess, unit)))
    cut_inds = int(sampling_rate * time_win)
    keep_spikes = np.unique(np.where(spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))])[0])
    spike_frs = dio.open_pickle(spike_fr_fname.format(subj_sess, unit))[keep_spikes]
    spike_inds = info.spike_inds
    assert len(spike_inds) == len(spike_frs) == len(target_phase_offsets) == len(local_phase_offsets)
    
    # Cut spikes from the beginning so vectors are divisible by 5.
    cut = len(spike_inds) % 5
    spike_inds = spike_inds[cut:]
    spike_frs = spike_frs[cut:]
    target_phase_offsets = target_phase_offsets[cut:]
    local_phase_offsets = local_phase_offsets[cut:]    
    n_spikes = len(spike_inds)
    n_samp = int(n_spikes / 5) # 20% of all spikes
    
    # Setup the spike categories.
    spike_cat = OrderedDict([('high_firing_rate', spike_frs),
                             ('low_firing_rate', spike_frs)])
    if test_phase_offsets:
        spike_cat['target_phase_locking'] = target_phase_offsets
        spike_cat['local_phase_locking'] = local_phase_offsets
    
    # Select 20% of spikes at random for each bootstrap iteration.
    bs_spike_inds = []
    for iBoot in range(n_bootstraps):
        bs_spike_inds.append(np.random.choice(n_spikes, n_samp, replace=False))
    
    # Get the mean firing rate (Z-scored over time) across
    # neurons in each region, at each spike time.
    _, fr_df, *_ = load_spikes(subj_sess)
    roi_frs = OrderedDict([])
    for iRoi, df in fr_df.groupby('location'):
        roi_frs[iRoi] = np.mean([stats.zscore(x) for x in df['fr'].tolist()], axis=0)[spike_inds]

    # Check that the target region has neurons!
    if not lfp_roi in roi_frs:
        return False

    # Get the mean firing rate in the target region at each spike time,
    # and the mean firing rate in the target region subtracting mean
    # firing rate across all regions, at each spike time
    target_fr = roi_frs[lfp_roi] # spike
    target_fr_meanSub = roi_frs[lfp_roi] - np.mean(list(roi_frs.values()), axis=0) # spike
    
    spike_frs = OrderedDict([])
    spike_frs_meanSub = OrderedDict([])
    for key in spike_cat.keys():
        # Select the spike subset
        if key == 'high_firing_rate':
            sel_spikes = np.where(spike_cat[key]>=np.percentile(spike_cat[key], 80))[0]
        else:
            sel_spikes = np.where(spike_cat[key]<=np.percentile(spike_cat[key], 20))[0]

        # Get the mean hippocampal firing rate for the 20% of spikes that fired closest
        # to the preferred phase, or with the highest firing rate
        spike_frs[key] = np.mean(target_fr[sel_spikes]) # scalar; mean over spikes

        # Get the relative mean hippocampal firing rate for the 20% of spikes that fired closest
        # to the preferred phase, or with the highest firing rate
        spike_frs_meanSub[key] = np.mean(target_fr_meanSub[sel_spikes]) # scalar; mean over spikes

    # Create null distributions of hippocampal firing rates from random spike subsets.
    bs_frs = np.array([np.mean(target_fr[bs_spike_inds[iBoot]]) for iBoot in range(n_bootstraps)]) # bs_index
    bs_mean = np.mean(bs_frs)
    bs_std = np.std(bs_frs)
    bs_frs_z = (bs_frs - bs_mean) / bs_std # freq x bs_index
    
    bs_frs_meanSub = np.array([np.mean(target_fr_meanSub[bs_spike_inds[iBoot]]) for iBoot in range(n_bootstraps)]) # bs_index
    bs_meanSub_mean = np.mean(bs_frs_meanSub)
    bs_meanSub_std = np.std(bs_frs_meanSub)
    bs_frs_meanSub_z = (bs_frs_meanSub - bs_meanSub_mean) / bs_meanSub_std # bs_index
    
    # Z-score the firing rate and relative firing rate and get an empirical p-value 
    # for each spike category.
    fr_pvals = OrderedDict([])
    fr_meanSub_pvals = OrderedDict([])
    for key in spike_cat.keys():
        spike_frs[key] = (spike_frs[key] - bs_mean) / bs_std # scalar
        fr_pvals[key] = bootstrap_p(spike_frs[key], bs_frs_z)
        
        spike_frs_meanSub[key] = (spike_frs_meanSub[key] - bs_meanSub_mean) / bs_meanSub_std # scalar
        fr_meanSub_pvals[key] = bootstrap_p(spike_frs_meanSub[key], bs_frs_meanSub_z)
    
    # Add to output
    output = OrderedDict([('hpc_fr', spike_frs), 
                          ('hpc_fr_pvals', fr_pvals),
                          ('hpc_fr_meanSub', spike_frs_meanSub),
                          ('hpc_fr_meanSub_pvals', fr_meanSub_pvals)])

    # Save output.
    if save_outputs:
        dio.save_pickle(output, fpath, verbose=False)
        
    return output
    

def calc_power_by_pl_fr_unit_to_target_region(info,
                                              n_freqs=16,
                                              n_bootstraps=1000,
                                              sampling_rate=2000,
                                              time_win=2,
                                              test_phase_offsets=True, # if False, only assess spikes by firing rate
                                              subtract_mean=True,
                                              input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                              save_outputs=True,
                                              overwrite=False,
                                              output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/power_by_pl_fr/lfp_target_region/sig',
                                              sleep_max=0):
    """Obtain power values for highly phase-locked or high firing spikes.
    
    The following steps are performed in reference to the mean power
    across channels for each depth electrode. The HPC region that 
    phase relations were taken in reference to is referred to as the
    "target" region, and the neuron's region of origin is called the
    "local" region and does not include the neuron's own channel.
    
    We find the 20% of spikes that fired:
    1) Closest to the preferred phase at the preferred phase-locked 
       frequency, in reference to the target (i.e. HPC) LFP.
    2) Closest to the preferred phase in reference to the local
       LFP, but at the preferred target LFP phase-locking frequency
    3) With the highest firing rate
    
    We then get:
    1) The mean power across spikes at each frequency,
       for each of these spike categories. 
    2) The Pearson correlation of power at each frequency across spikes,
       for each of these spike categories.
    
    Mean power and power correlation vectors for each spike category 
    are Z-scored against a null distribution of randomly selected spikes 
    (20% of spikes for each permutation). 
    
    We return a dictionary with:
    1) The Z-power vector for each spike category.
    2) A p-value and some associated metrics for each spike category.
       The p-value is determined from a permutation test in which we compare
       the max(|z_power|) across frequencies against the null distribution 
       of max(|z_power|) from each permutation. This test enables us to detect 
       spike subsets that showed significantly higher or lower power at any 
       frequency, compared to the null distribution of spikes drawn at random.
    3) The Z-power correlation vector for each spike category.
    4) A p-value and some associated metrics for each spike category,
       using the same logic as in (2) but with Z-power correlations rather
       than Z-power at each frequency.
    """
    def bootstrap_p(obs, null):
        """Derive an empirical p-value.

        The maximum |Z-score| (across frequencies) is compared
        against the null distribution of maximum |Z-scores|.

        obs : n_freq vec
        null : n_freq x n_boot array
        """
        n_bootstraps = null.shape[-1]

        if len(null.shape) == 1:
            max_obs = np.abs(obs)
            max_null = np.abs(null)
            bs_ind = np.sum(max_null >= max_obs)
            pval = (1 + bs_ind) / (1 + n_bootstraps)

            return OrderedDict([('max_z', obs),
                                ('max_z_ind', 0),
                                ('bs_ind', bs_ind),
                                ('pval', pval)])
        else:
            max_obs = np.max(np.abs(obs))
            max_null = np.max(np.abs(null), axis=0)
            bs_ind = np.sum(max_null >= max_obs)
            pval = (1 + bs_ind) / (1 + n_bootstraps)

            # Get the Z-score and index of max(|Z|)
            max_z_ind = np.argmax(np.abs(obs))
            max_z = obs[max_z_ind]

            return OrderedDict([('max_z', max_z),
                                ('max_z_ind', max_z_ind),
                                ('bs_ind', bs_ind),
                                ('pval', pval)])
        
    # Check if output file already exists.
    subj_sess = info.subj_sess
    unit = info.unit
    lfp_roi = info.lfp_hemroi
    output_fname = 'power_by_phase_locking_and_fr-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl'
    fpath = op.join(output_dir, output_fname.format(subj_sess, unit, lfp_roi))
    if op.exists(fpath) and not overwrite:
        output = dio.open_pickle(fpath)
        return output
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # General params.
    power_fname = op.join(input_dir, 'wavelet', 'power', 
                          'power-Z-log-{}-iChan{}-iFreq{}-2000Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    spike_fr_fname = op.join(input_dir, 'spike_frs', 'spike_frs-2000Hz-{}-unit{}.pkl')
    pl_fname = op.join(input_dir, 'phase_locking', 'unit_to_region', 
                       'phase_locking_stats-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    
    # Get phase offsets for each spike.
    target_phase_offsets = info.phase_offsets_tl_locked_time_freq_z
    local_phase_offsets = info.local_phase_offsets
    spike_inds, n_timepoints = dio.open_pickle(op.join(input_dir, 'spike_inds', 'spike_inds-{}Hz-{}-unit{}.pkl'
                                                       .format(sampling_rate, subj_sess, unit)))
    cut_inds = int(sampling_rate * time_win)
    keep_spikes = np.unique(np.where(spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))])[0])
    spike_frs = dio.open_pickle(spike_fr_fname.format(subj_sess, unit))[keep_spikes]
    spike_inds = info.spike_inds
    assert len(spike_inds) == len(spike_frs) == len(target_phase_offsets) == len(local_phase_offsets)
    
    # Cut spikes from the beginning so vectors are divisible by 5.
    cut = len(spike_inds) % 5
    spike_inds = spike_inds[cut:]
    spike_frs = spike_frs[cut:]
    target_phase_offsets = target_phase_offsets[cut:]
    local_phase_offsets = local_phase_offsets[cut:]    
    n_spikes = len(spike_inds)
    n_samp = int(n_spikes / 5) # 20% of all spikes
    
    # Setup the spike categories.
    spike_cat = OrderedDict([('high_firing_rate', spike_frs),
                             ('low_firing_rate', spike_frs)])
    if test_phase_offsets:
        spike_cat['target_phase_locking'] = target_phase_offsets
        spike_cat['local_phase_locking'] = local_phase_offsets
    
    # Select 20% of spikes at random for each bootstrap iteration.
    bs_spike_inds = []
    for iBoot in range(n_bootstraps):
        bs_spike_inds.append(np.random.choice(n_spikes, n_samp, replace=False))
    
    # Assess LFP power (mean across channels) at each frequency,
    # at each spike time, for channels in:
    # 1) the target region
    # 2) the unit's region (excluding the unit's channel) 
    # 3) each off-target region
    lfp_power_rois = info.off_target_chan_inds # an OrderedDict([(roi : str, chan_inds : list)])
    lfp_power_rois['local'] = info.local_lfp_chan_inds
    lfp_power_rois['target'] = info.lfp_chan_inds
    lfp_power_rois = OrderedDict(reversed(list(lfp_power_rois.items())))
    
    # Get LFP power across frequencies at each spike time, taking the mean
    # power across channels in a region at each frequency and spike time
    power = []
    for lfp_power_roi, lfp_chan_inds in lfp_power_rois.items():
        power.append(np.mean([[dio.open_pickle(power_fname.format(subj_sess, iChan, iFreq))[spike_inds] for iFreq in range(n_freqs)]
                              for iChan in lfp_chan_inds], axis=0))
    power = np.array(power) # roi x freq x spike
    
    # Get the cross-frequency power correlation between all possible
    # pairs of regions, for each spike, and keep track of which pair
    # compares the local region to the target region
    local_ind = list(lfp_power_rois.keys()).index('local')
    target_ind = list(lfp_power_rois.keys()).index('target')
    
    iPair = 0
    power_corrs = []
    for iRoi1, iRoi2 in itertools.combinations(range(len(lfp_power_rois)), 2):
        power_corrs.append([stats.pearsonr(power[iRoi1, :, iSpike], power[iRoi2, :, iSpike])[0] for iSpike in range(power.shape[-1])])
        if (local_ind in [iRoi1, iRoi2]) & (target_ind in [iRoi1, iRoi2]):
            local_target_ind = iPair
        iPair += 1
    power_corrs = np.array(power_corrs) # region_pair x spike
    
    if subtract_mean:
        # Subtract the mean power across regions at each frequency, for each spike
        power = power[target_ind, :, :] - np.mean(power, axis=0) # freq x spike
    
        # Subtract the mean power correlation across region pairs, for each spike
        power_corrs = power_corrs[local_target_ind, :] - np.mean(power_corrs, axis=0) # spike
    else:
        power = power[target_ind, :, :] # freq x spike
        power_corrs = power_corrs[local_target_ind, :] # spike
    
    spike_powers = OrderedDict([])
    spike_power_corrs = OrderedDict([])
    for key in spike_cat.keys():
        # Select the spike subset
        if key == 'high_firing_rate':
            sel_spikes = np.where(spike_cat[key]>=np.percentile(spike_cat[key], 80))[0]
        else:
            sel_spikes = np.where(spike_cat[key]<=np.percentile(spike_cat[key], 20))[0]

        # Get power at each freq and time for the 20% of spikes that fired closest
        # to the preferred phase, or with the highest or lowest firing rate
        spike_powers[key] = np.mean(power[:, sel_spikes], axis=1) # freq; mean over spikes

        # Get the correlation between local and roi LFP power,
        # at each frequency, across selected spikes
        spike_power_corrs[key] = np.mean(power_corrs[sel_spikes]) # scalar; mean over spikes

    # Create a null distribution of powers at each frequency from random spike subsets.
    bs_powers = np.swapaxes([np.mean(power[:, bs_spike_inds[iBoot]], axis=1) for iBoot in range(n_bootstraps)], 0, 1) # freq x bs_index
    bs_means = np.expand_dims(np.mean(bs_powers, axis=-1), axis=-1) # mean at each freq
    bs_stds = np.expand_dims(np.std(bs_powers, axis=-1), axis=-1) # std at each freq
    bs_powers_z = ((bs_powers - bs_means) / bs_stds) # freq x bs_index
    
    bs_power_corrs = np.array([np.mean(power_corrs[bs_spike_inds[iBoot]]) for iBoot in range(n_bootstraps)]) # bs_index
    bs_corr_mean = np.mean(bs_power_corrs)
    bs_corr_std = np.std(bs_power_corrs)
    bs_power_corrs_z = (bs_power_corrs - bs_corr_mean) / bs_corr_std # bs_index
    
    # Z-score the powers at each frequency and get an empirical p-value for each spike category.
    power_pvals = OrderedDict([])
    power_corr_pvals = OrderedDict([])
        
    for key in spike_cat.keys():
        spike_powers[key] = (spike_powers[key] - np.squeeze(bs_means)) / np.squeeze(bs_stds) # freq
        power_pvals[key] = bootstrap_p(spike_powers[key], bs_powers_z)
        
        spike_power_corrs[key] = (spike_power_corrs[key] - bs_corr_mean) / bs_corr_std # scalar
        power_corr_pvals[key] = bootstrap_p(spike_power_corrs[key], bs_power_corrs_z)
    
    # Add to output
    output = OrderedDict([('z_power', spike_powers), 
                          ('power_pvals', power_pvals),
                          ('z_power_corrs', spike_power_corrs),
                          ('power_corr_pvals', power_corr_pvals)])

    # Save output.
    if save_outputs:
        dio.save_pickle(output, fpath, verbose=False)
        
    return output
    

def calc_power_by_pl_fr_unit_to_region_all(info,
                                           n_freqs=16,
                                           n_bootstraps=1000,
                                           sampling_rate=2000,
                                           time_win=2,
                                           test_phase_offsets=True, # if False, only assess spikes by firing rate
                                           input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                           save_outputs=True,
                                           overwrite=False,
                                           output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/power_by_pl_fr/lfp_all_regions',
                                           sleep_max=0):
    """Obtain power values for highly phase-locked or high firing spikes.
    
    The following steps are performed in reference to the mean power
    across channels for each depth electrode. The HPC region that 
    phase relations were taken in reference to is referred to as the
    "target" region, and the neuron's region of origin is called the
    "local" region and does not include the neuron's own channel.
    
    We find the 20% of spikes that fired:
    1) Closest to the preferred phase at the preferred phase-locked 
       frequency, in reference to the target (i.e. HPC) LFP.
    2) Closest to the preferred phase in reference to the local
       LFP, but at the preferred target LFP phase-locking frequency
    3) With the highest firing rate
    
    We then get:
    1) The mean power across spikes at each frequency,
       for each of these spike categories. 
    2) The Pearson correlation of power at each frequency across spikes,
       for each of these spike categories.
    
    Mean power and power correlation vectors for each spike category 
    are Z-scored against a null distribution of randomly selected spikes 
    (20% of spikes for each permutation). 
    
    We return a dictionary with:
    1) The Z-power vector for each spike category.
    2) A p-value and some associated metrics for each spike category.
       The p-value is determined from a permutation test in which we compare
       the max(|z_power|) across frequencies against the null distribution 
       of max(|z_power|) from each permutation. This test enables us to detect 
       spike subsets that showed significantly higher or lower power at any 
       frequency, compared to the null distribution of spikes drawn at random.
    3) The Z-power correlation vector for each spike category.
    4) A p-value and some associated metrics for each spike category,
       using the same logic as in (2) but with Z-power correlations rather
       than Z-power at each frequency.
    """
    def bootstrap_p(obs, null):
        """Derive an empirical p-value.

        The maximum |Z-score| (across frequencies) is compared
        against the null distribution of maximum |Z-scores|.

        obs : n_freq vec
        null : n_freq x n_boot array
        """
        max_obs = np.max(np.abs(obs))
        max_null = np.max(np.abs(null), axis=0)
        n_bootstraps = len(max_null)
        bs_ind = np.sum(max_null >= max_obs)
        pval = (1 + bs_ind) / (1 + n_bootstraps)

        # Get the Z-score and index of max(|Z|)
        max_z_ind = np.argmax(np.abs(obs))
        max_z = obs[max_z_ind]

        return OrderedDict([('max_z', max_z),
                            ('max_z_ind', max_z_ind),
                            ('bs_ind', bs_ind),
                            ('pval', pval)])
        
    # Check if output file already exists.
    subj_sess = info.subj_sess
    unit = info.unit
    lfp_roi = info.lfp_hemroi
    output_fname = 'power_by_phase_locking_and_fr-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl'
    fpath = op.join(output_dir, output_fname.format(subj_sess, unit, lfp_roi))
    if op.exists(fpath) and not overwrite:
        output = dio.open_pickle(fpath)
        return output
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # General params.
    power_fname = op.join(input_dir, 'wavelet', 'power', 
                          'power-Z-log-{}-iChan{}-iFreq{}-2000Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    spike_fr_fname = op.join(input_dir, 'spike_frs', 'spike_frs-2000Hz-{}-unit{}.pkl')
    pl_fname = op.join(input_dir, 'phase_locking', 'unit_to_region', 
                       'phase_locking_stats-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    
    # Get phase offsets for each spike.
    target_phase_offsets = info.phase_offsets
    local_phase_offsets = info.local_phase_offsets
    spike_inds, n_timepoints = dio.open_pickle(op.join(input_dir, 'spike_inds', 
                                                       'spike_inds-{}Hz-{}-unit{}.pkl'
                                                       .format(sampling_rate, subj_sess, unit)))
    cut_inds = int(sampling_rate * time_win)
    keep_spikes = np.unique(np.where(spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))])[0])
    spike_frs = dio.open_pickle(spike_fr_fname.format(subj_sess, unit))[keep_spikes]
    spike_inds = info.spike_inds
    assert len(spike_inds) == len(spike_frs) == len(target_phase_offsets) == len(local_phase_offsets)
    
    # Cut spikes from the beginning so vectors are divisible by 5.
    cut = len(spike_inds) % 5
    spike_inds = spike_inds[cut:]
    spike_frs = spike_frs[cut:]
    target_phase_offsets = target_phase_offsets[cut:]
    local_phase_offsets = local_phase_offsets[cut:]    
    n_spikes = len(spike_inds)
    n_samp = int(n_spikes / 5) # 20% of all spikes
    
    # Setup the spike categories.
    spike_cat = OrderedDict([('firing_rate', spike_frs)])
    if test_phase_offsets:
        spike_cat['target_phase_locking'] = target_phase_offsets
        spike_cat['local_phase_locking'] = local_phase_offsets
    
    # Select 20% of spikes at random for each bootstrap iteration.
    bs_spike_inds = []
    for iBoot in range(n_bootstraps):
        bs_spike_inds.append(np.random.choice(n_spikes, n_samp, replace=False))
    
    # Assess LFP power (mean across channels) at each frequency,
    # at each spike time, for channels in:
    # 1) the target region
    # 2) the unit's region (excluding the unit's channel) 
    # 3) each off-target region
    lfp_power_rois = info.off_target_chan_inds # an OrderedDict([(roi : str, chan_inds : list)])
    lfp_power_rois['local'] = info.local_lfp_chan_inds
    lfp_power_rois['target'] = info.lfp_chan_inds
    lfp_power_rois = OrderedDict(reversed(list(lfp_power_rois.items())))
    
    # Get LFP power across frequencies at each spike time
    local_power = []
    for iChan in lfp_power_rois['local']:
        local_power_ = []
        for iFreq in range(n_freqs):
            local_power_.append(dio.open_pickle(power_fname.format(subj_sess, iChan, iFreq))[spike_inds])
        local_power.append(local_power_)
    local_power = np.mean(local_power, axis=0) # freq x spike; mean over channels
    
    output = OrderedDict([])
    for lfp_power_roi, lfp_chan_inds in lfp_power_rois.items():
        # Get LFP power across frequencies at each spike time
        power = []
        for iChan in lfp_chan_inds:
            power_ = []
            for iFreq in range(n_freqs):
                power_.append(dio.open_pickle(power_fname.format(subj_sess, iChan, iFreq))[spike_inds])
            power.append(power_)
        power = np.mean(power, axis=0) # freq x spike; mean over channels

        spike_powers = OrderedDict([])
        spike_power_corrs = OrderedDict([])
        for key in spike_cat.keys():
            # Select the spike subset
            if key == 'firing_rate':
                sel_spikes = np.where(spike_cat[key]>=np.percentile(spike_cat[key], 80))[0]
            else:
                sel_spikes = np.where(spike_cat[key]<=np.percentile(spike_cat[key], 20))[0]
                
            # Get power at each freq and time for the 20% of spikes that fired closest
            # to the preferred phase, or with the highest firing rate
            spike_powers[key] = np.mean(power[:, sel_spikes], axis=1) # freq; mean over spikes
            
            # Get the correlation between local and roi LFP power,
            # at each frequency, across selected spikes
            if lfp_power_roi != 'local':
                spike_power_corrs[key] = np.array([stats.pearsonr(power[iFreq, sel_spikes], local_power[iFreq, sel_spikes])[0]
                                                   for iFreq in range(n_freqs)])
                
        # Create a null distribution of powers at each frequency from random spike subsets.
        bs_powers = []
        for iBoot in range(n_bootstraps):
            sel_spikes = bs_spike_inds[iBoot]
            bs_powers.append(np.mean(power[:, sel_spikes], axis=1))
        bs_powers = np.swapaxes(bs_powers, 0, 1) # freq x bs_index
        bs_means = np.expand_dims(np.mean(bs_powers, axis=-1), axis=-1) # mean at each freq
        bs_stds = np.expand_dims(np.std(bs_powers, axis=-1), axis=-1) # std at each freq
        bs_powers_z = ((bs_powers - bs_means) / bs_stds) # freq x bs_index
        
        # Create a null distribution of power correlations
        if lfp_power_roi != 'local':
            bs_power_corrs = []
            for iBoot in range(n_bootstraps):
                sel_spikes = bs_spike_inds[iBoot]
                bs_power_corrs.append(np.array([stats.pearsonr(power[iFreq, sel_spikes], local_power[iFreq, sel_spikes])[0] 
                                                for iFreq in range(n_freqs)]))
            bs_power_corrs = np.swapaxes(bs_power_corrs, 0, 1) # freq x bs_index
            bs_corr_means = np.expand_dims(np.mean(bs_power_corrs, axis=-1), axis=-1) # mean at each freq
            bs_corr_stds = np.expand_dims(np.std(bs_power_corrs, axis=-1), axis=-1) # std at each freq
            bs_power_corrs_z = ((bs_power_corrs - bs_corr_means) / bs_corr_stds) # freq x bs_index
        
        # Z-score the powers at each frequency and get an empirical p-value for each spike category.
        power_pvals = OrderedDict([])
        power_corr_pvals = OrderedDict([])
        for key in spike_cat.keys():
            spike_powers[key] = (spike_powers[key] - np.squeeze(bs_means)) / np.squeeze(bs_stds)
            power_pvals[key] = bootstrap_p(spike_powers[key], bs_powers_z)
            
            if lfp_power_roi != 'local':
                spike_power_corrs[key] = (spike_power_corrs[key] - np.squeeze(bs_corr_means)) / np.squeeze(bs_corr_stds)
                power_corr_pvals[key] = bootstrap_p(spike_power_corrs[key], bs_power_corrs_z)
            
        # Add to output
        output[lfp_power_roi] = OrderedDict([('z_power', spike_powers), 
                                             ('power_pvals', power_pvals)])
        if lfp_power_roi != 'local':
            output[lfp_power_roi]['z_power_corrs'] = spike_power_corrs
            output[lfp_power_roi]['power_corr_pvals'] = power_corr_pvals

    # Save output.
    if save_outputs:
        dio.save_pickle(output, fpath, verbose=False)
        
    return output
    

def calc_power_by_pl_fr_unit_to_off_target_regions(info,
                                                   n_freqs=16,
                                                   n_bootstraps=500,
                                                   sampling_rate=2000,
                                                   time_win=2,
                                                   input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                                   save_outputs=True,
                                                   overwrite=False,
                                                   output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/power_by_pl_fr/off_target_lfp',
                                                   sleep_max=0):
    """Obtain power values for highly phase-locked or high firing spikes.
    
    Here, rather than finding power for the target LFP (that the neuron is phase-locked to),
    we find the mean power at each frequency for all channels that are not in the target
    region or the neuron's own region.
    
    We find the 20% of spikes that fired:
    1) closest to the preferred phase at the HPC-locked frequency (for HPC LFP)
    2) closest to the preferred phase at the HPC-locked frequency (for local LFP)
    3) with the highest firing rate
    
    We then get the mean power across spikes at each frequency 
    for each of these 20% spike vectors. 
    
    Mean powers are Z-scored against a null distribution of randomly selected spikes
    (20% of spikes selected for each permutation).
    """
    # Check if output file already exists.
    subj_sess = info.subj_sess
    unit = info.unit
    lfp_roi = info.lfp_hemroi
    output_fname = 'power_by_phase_locking_and_fr-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl'
    fpath = op.join(output_dir, output_fname.format(subj_sess, unit, lfp_roi))
    if op.exists(fpath) and not overwrite:
        output = dio.open_pickle(fpath)
        return output
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # General params.
    power_fname = op.join(input_dir, 'wavelet', 'power', 
                               'power-Z-log-{}-iChan{}-iFreq{}-2000Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    spike_fr_fname = op.join(input_dir, 'spike_frs', 'spike_frs-2000Hz-{}-unit{}.pkl')
    pl_fname = op.join(input_dir, 'phase_locking', 'unit_to_region', 
                            'phase_locking_stats-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    lfp_chan_inds = info.off_target_chan_inds
    local_phase_offsets = info.local_phase_offsets
    
    # Get phase offsets for each spike.
    phase_offsets = info.phase_offsets
    local_phase_offsets = info.local_phase_offsets
    spike_inds, n_timepoints = dio.open_pickle(op.join(input_dir, 'spike_inds', 
                                                       'spike_inds-{}Hz-{}-unit{}.pkl'
                                                       .format(sampling_rate, subj_sess, unit)))
    cut_inds = int(sampling_rate * time_win)
    keep_spikes = np.unique(np.where(spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))])[0])
    spike_frs = dio.open_pickle(spike_fr_fname.format(subj_sess, unit))[keep_spikes]
    spike_inds = info.spike_inds
    assert len(spike_inds) == len(spike_frs) == len(phase_offsets) == len(local_phase_offsets)
    
    # Cut spikes from the beginning so both vectors are divisible by 5.
    cut = len(phase_offsets) % 5
    spike_inds = spike_inds[cut:]
    spike_frs = spike_frs[cut:]
    phase_offsets = phase_offsets[cut:]
    local_phase_offsets = local_phase_offsets[cut:]    
    
    # Setup the rest.
    offsets = OrderedDict([('phase_locking', phase_offsets),
                           ('local_phase_locking', local_phase_offsets),
                           ('firing_rate', spike_frs)])
    n_spikes = len(spike_inds)
    n_samp = int(n_spikes / 5) # 20% of all spikes
            
    # Get LFP power across frequencies at each spike time
    power = []
    for iChan in lfp_chan_inds:
        power_ = []
        for iFreq in range(n_freqs):
            power_.append(dio.open_pickle(power_fname.format(subj_sess, iChan, iFreq))[spike_inds])
        power.append(power_)
    power = np.mean(power, axis=0) # freq x spike; mean over channels
    
    # Get power at each freq and time for the 20% of spikes that fired
    # closest to the preferred phase, or with the highest firing rate
    output = OrderedDict([])
    for key in offsets.keys():
        if key == 'firing_rate':
            sel_spikes = np.where(offsets[key]>=np.percentile(offsets[key], 80))[0]
            output[key] = np.mean(power[:, sel_spikes], axis=1) # freq; mean over spikes
        else:
            sel_spikes = np.where(offsets[key]<=np.percentile(offsets[key], 20))[0]
            output[key] = np.mean(power[:, sel_spikes], axis=1) # freq; mean over spikes
    
    # Z-score against a null distribution of 20% of randomly selected spikes.
    bs_powers = []
    for iBoot in range(n_bootstraps):
        sel_spikes = np.random.choice(n_spikes, n_samp, replace=False)
        bs_powers.append(np.mean(power[:, sel_spikes], axis=1))
    bs_powers = np.array(bs_powers) # bs_index x freq
    bs_means = np.mean(bs_powers, axis=0)
    bs_stds = np.std(bs_powers, axis=0)
    
    for key in offsets.keys():
        output[key] = (output[key] - bs_means) / bs_stds

    # Save output.
    if save_outputs:
        dio.save_pickle(output, fpath, verbose=False)
        
    return output
    

def calc_power_by_pl_fr_unit_to_region(info,
                                       n_freqs=16,
                                       n_bootstraps=500,
                                       sampling_rate=2000,
                                       time_win=2,
                                       input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                       save_outputs=True,
                                       overwrite=False,
                                       output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/power_by_pl_fr/target_lfp',
                                       sleep_max=0):
    """Obtain power values for highly phase-locked or high firing spikes.
    
    We find the 20% of spikes that fired:
    1) closest to the preferred phase at the HPC-locked frequency (for HPC LFP)
    2) closest to the preferred phase at the HPC-locked frequency (for local LFP)
    3) with the highest firing rate
    
    We then get the mean power across spikes at each frequency 
    for each of these 20% spike vectors. 
    
    Mean powers are Z-scored against a null distribution of randomly selected spikes
    (20% of spikes selected for each permutation).
    """
    # Check if output file already exists.
    subj_sess = info.subj_sess
    unit = info.unit
    lfp_roi = info.lfp_hemroi
    output_fname = 'power_by_phase_locking_and_fr-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl'
    fpath = op.join(output_dir, output_fname.format(subj_sess, unit, lfp_roi))
    if op.exists(fpath) and not overwrite:
        output = dio.open_pickle(fpath)
        return output
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # General params.
    power_fname = op.join(input_dir, 'wavelet', 'power', 
                          'power-Z-log-{}-iChan{}-iFreq{}-2000Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    spike_fr_fname = op.join(input_dir, 'spike_frs', 'spike_frs-2000Hz-{}-unit{}.pkl')
    pl_fname = op.join(input_dir, 'phase_locking', 'unit_to_region', 
                       'phase_locking_stats-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    lfp_chan_inds = info.lfp_chan_inds
    local_phase_offsets = info.local_phase_offsets
    
    # Get phase offsets for each spike.
    phase_offsets = info.phase_offsets
    local_phase_offsets = info.local_phase_offsets
    spike_inds, n_timepoints = dio.open_pickle(op.join(input_dir, 'spike_inds', 
                                                       'spike_inds-{}Hz-{}-unit{}.pkl'
                                                       .format(sampling_rate, subj_sess, unit)))
    cut_inds = int(sampling_rate * time_win)
    keep_spikes = np.unique(np.where(spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))])[0])
    spike_frs = dio.open_pickle(spike_fr_fname.format(subj_sess, unit))[keep_spikes]
    spike_inds = info.spike_inds
    assert len(spike_inds) == len(spike_frs) == len(phase_offsets) == len(local_phase_offsets)
    
    # Cut spikes from the beginning so both vectors are divisible by 5.
    cut = len(phase_offsets) % 5
    spike_inds = spike_inds[cut:]
    spike_frs = spike_frs[cut:]
    phase_offsets = phase_offsets[cut:]
    local_phase_offsets = local_phase_offsets[cut:]    
    
    # Setup the rest.
    offsets = OrderedDict([('phase_locking', phase_offsets),
                           ('local_phase_locking', local_phase_offsets),
                           ('firing_rate', spike_frs)])
    n_spikes = len(spike_inds)
    n_samp = int(n_spikes / 5) # 20% of all spikes
            
    # Get LFP power across frequencies at each spike time
    power = []
    for iChan in lfp_chan_inds:
        power_ = []
        for iFreq in range(n_freqs):
            power_.append(dio.open_pickle(power_fname.format(subj_sess, iChan, iFreq))[spike_inds])
        power.append(power_)
    power = np.mean(power, axis=0) # freq x spike; mean over channels
    
    # Get power at each freq and time for the 20% of spikes that fired
    # closest to the preferred phase, or with the highest firing rate
    output = OrderedDict([])
    for key in offsets.keys():
        if key == 'firing_rate':
            sel_spikes = np.where(offsets[key]>=np.percentile(offsets[key], 80))[0]
            output[key] = np.mean(power[:, sel_spikes], axis=1) # freq; mean over spikes
        else:
            sel_spikes = np.where(offsets[key]<=np.percentile(offsets[key], 20))[0]
            output[key] = np.mean(power[:, sel_spikes], axis=1) # freq; mean over spikes
    
    # Z-score against a null distribution of 20% of randomly selected spikes.
    bs_powers = []
    for iBoot in range(n_bootstraps):
        sel_spikes = np.random.choice(n_spikes, n_samp, replace=False)
        bs_powers.append(np.mean(power[:, sel_spikes], axis=1))
    bs_powers = np.array(bs_powers) # bs_index x freq
    bs_means = np.mean(bs_powers, axis=0)
    bs_stds = np.std(bs_powers, axis=0)
    
    for key in offsets.keys():
        output[key] = (output[key] - bs_means) / bs_stds

    # Save output.
    if save_outputs:
        dio.save_pickle(output, fpath, verbose=False)
        
    return output
    

def phase_offsets_by_spike(info,
                           iFreq,
                           iTimeLag=0, # number of steps to shift the spike train
                           sampling_rate=2000,
                           time_win=2,
                           input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                           phase_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/wavelet/phase',
                           phase_fname='phase-{}-iChan{}-iFreq{}-{}Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_90.5Hz.pkl'):
    """Calculate the preferred phase and phase offsets by spike for a given unit-to-region comparison."""
        
    # General params.
    subj_sess, unit, lfp_chans = info.subj_sess, info.unit, info.lfp_chan_inds
    sampling_rate = int(sampling_rate)
    cut_inds = int(sampling_rate * time_win)
    
    # Load spike inds and remove time_win secs from the beggining and end of the session.
    spike_inds, n_timepoints = dio.open_pickle(op.join(input_dir, 'spike_inds', 
                                                       'spike_inds-{}Hz-{}-unit{}.pkl'
                                                       .format(sampling_rate, subj_sess, unit)))
    spike_inds = np.unique(spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))])
    if iTimeLag != 0:
        spike_inds = shift_spike_inds(spike_inds, n_timepoints, iTimeLag)
    
    # Get the circular distance between each spike phase (at the chosen frequency) 
    # and the mean preferred phase for each channel. Then take the average circular distance
    # across channels.    
    spike_phases = []
    phase_offsets = []
    for iChan in lfp_chans:
        phase = dio.open_pickle(op.join(phase_dir, phase_fname.format(subj_sess, iChan, iFreq, sampling_rate)))
        spike_phases_ = phase[spike_inds]
        pref_phase_ = circstats.circmoment(spike_phases_)[0]
        spike_phases.append(spike_phases_)
        phase_offsets.append(np.abs(pycircstat.descriptive.cdiff(spike_phases_, np.repeat(pref_phase_, len(spike_phases_)))))
    pref_phase = circstats.circmoment(np.array(spike_phases).flatten())[0] # preferred phase taking all spike phases across channels
    phase_offsets = np.mean(phase_offsets, axis=0).astype(np.float32) # circular distance between each spike and the preferred phase for each channel, averaged across channels
    
    return pref_phase, phase_offsets
    

def update_pl_freq_time(info,
                        t0=200,
                        thresh=2,
                        time_win=2,
                        sampling_rate=2000):
    """Return the new phase-locking frequency and time.
    
    (Given the cross-correlated phase-locking strength matrix.)
    
    Finds the frequency and time that correspond to the maximum 
    phase-locking strength within the "phase-locking region,"
    defined as all phase-locking strengths>thresh that are
    continuous with the zero offset time.
    
    Returns
    -------
    new_pl_params : OrderedDict
        Contains the new phase-locking strength, frequency, and time,
        along with the duration and number of cycles that the neuron
        was phase-locked (contiguous Z>thresh).
    updated : bool
        Whether or not new phase-locking values were chosen
        vs. the existing ones.
    """
    # Get starting params.
    updated = False
    time_steps = np.arange(-time_win*sampling_rate, time_win*sampling_rate+1, sampling_rate*0.01, dtype=int)
    time_steps_ms = (time_steps / sampling_rate) * 1000
    freqs = np.array([2**((i/2) - 1) for i in range(16)])
    tl_mrls_z = info['tl_mrls_z']
    locked_mrl = info['pl_strength']
    locked_freq = info['pl_freq']
    locked_time = info['pl_time_shift']
    locked_duration = 0
    locked_cycles = 0
    
    # Find all phase-locking strengths > thresh,
    # and iterate over frequencies for which
    # the zero offset shift is > thresh.
    ma = tl_mrls_z > thresh
    ma_inds = np.array(list(np.where(ma))).T
    for iFreq in ma_inds[ma_inds[:, 1]==t0][:, 0]: 
        ma_inds_ = list(ma_inds[ma_inds[:, 0]==iFreq, 1])
        zero_ind = ma_inds_.index(t0)
        keep_inds = [t0]
        
        # Negative shifts.
        offset = 1
        while offset < len(ma_inds_)-1:
            if t0 - offset not in ma_inds_:
                break
            else:
                keep_inds.append(t0 - offset)
            offset += 1
        
        # Positive shifts.
        offset = 1
        while offset < len(ma_inds_)-1:
            if t0 + offset not in ma_inds_:
                break
            else:
                keep_inds.append(t0 + offset)
            offset += 1
        
        max_mrl = tl_mrls_z[iFreq, keep_inds].max()
        max_freq = iFreq
        max_time = keep_inds[tl_mrls_z[iFreq, keep_inds].argmax()]
        max_duration = len(keep_inds) * 10 # converting time-shift steps to ms
        max_cycles = freqs[iFreq] * (max_duration/1000.0)
        
        if max_mrl > locked_mrl:
            locked_mrl = max_mrl
            locked_freq = max_freq
            locked_time = time_steps_ms[max_time]
            locked_duration = max_duration
            locked_cycles = max_cycles
            updated = True
        elif max_freq == locked_freq:
            locked_duration = max_duration
            locked_cycles = max_cycles
        
    new_pl_params = OrderedDict([('pl_strength', locked_mrl), 
                                 ('pl_freq', locked_freq), 
                                 ('pl_time_shift', locked_time),
                                 ('pl_duration', locked_duration),
                                 ('pl_cycles', locked_cycles)])
    
    return new_pl_params, updated
    

def load_pl_df(input_files=op.join('/home1/dscho/projects/unit_activity_and_hpc_theta/data/crosselec_phase_locking/phase_locking/unit_to_region',
                                   'all_phase_locking_stats-14026_unit_to_region_pairs-2000Hz-notch60_120Hz-5cycles-16log10freqs_0.5_to_90.5Hz.pkl'),
               drop_repeat_connections=True,
               keep_only_ctx_hpc=False):
    """Return the unit-to-region phase-locking DataFrame."""
    
    # Load the phase-locking files.
    if isinstance(input_files, str):
        pl_df = dio.open_pickle(input_files)
    else:
        pl_df = pd.DataFrame(dio.open_pickle(input_files[0])).T
        for f in input_files[1:]:
            pl_df = pl_df.append(dio.open_pickle(f))
        pl_df.reset_index(drop=True, inplace=True)

    # Ensure all columns are stored as the correct data type.
    map_dtypes = {'unit_nspikes': np.uint32,
                  'unit_fr': np.float32,
                  'lfp_is_hpc': np.bool,
                  'same_chan': np.bool,
                  'same_hemroi': np.bool,
                  'same_hem': np.bool,
                  'same_roi': np.bool,
                  'both_hpc': np.bool,
                  'same_roi2': np.bool,
                  'locked_freq_ind_z': np.uint8,
                  'locked_mrl_z': np.float64,
                  'bs_ind_z': np.uint16,
                  'bs_pval_z': np.float64,
                  'sig_z': np.bool,
                  'tl_locked_freq_z': np.uint8,
                  'tl_locked_time_z': np.int32,
                  'tl_locked_mrl_z': np.float64,
                  'pref_phase': np.float64,
                  'pref_phase_tl_locked_time_freq_z': np.float64}
    for col, dtype in map_dtypes.items():
        pl_df[col] = pl_df[col].astype(dtype)
        
    # Drop edges other than ctx-hpc.
    if keep_only_ctx_hpc:
        pl_df = pl_df.loc[pl_df['edge']=='ctx-hpc']
        
    # Add some columns to the phase-locking dataframe.
    def get_session_number(x):
        d_ = {'U380_ses1a': 1,
              'U393_ses2': 1,
              'U394_ses3': 1,
              'U396_ses2': 1,
              'U396_ses3': 2}
        if x in d_.keys():
            return d_[x]
        else:
            return int(x[-1])
    pl_df.insert(0, 'subj', pl_df.subj_sess.apply(lambda x: x.split('_')[0]))
    pl_df.insert(1, 'sess', pl_df['subj_sess'].apply(lambda x: get_session_number(x)))
    pl_df.insert(4, 'subj_unit_chan', pl_df.apply(lambda x: x['subj'] + '_' + str(x['unit_chan_ind'] + 1), axis=1))
    pl_df['unit_roi3'] = ''
    pl_df.loc[pl_df['unit_roi2'] == 'hpc', 'unit_roi3'] = 'hpc'
    pl_df.loc[pl_df['unit_roi2'] == 'ec', 'unit_roi3'] = 'ec'
    pl_df.loc[pl_df['unit_roi2'] == 'amy', 'unit_roi3'] = 'amy'
    pl_df.loc[pl_df['unit_roi3'] == '', 'unit_roi3'] = 'ctx'
    pl_df['roi'] = pl_df['unit_roi3']
    pl_df.loc[pl_df['same_hem']==False, 'roi'] = 'contra'
    roi_cats = [roi for roi in ['hpc', 'ec', 'amy', 'ctx', 'contra'] if roi in pl_df['roi'].unique()]
    pl_df['roi'] = pl_df['roi'].astype('category').cat.reorder_categories(roi_cats, ordered=True)
    pl_df['roi_unit_to_lfp'] = pl_df.apply(lambda x: x['unit_roi3'] + '_ipsi' if x['same_hem'] else x['unit_roi3'] + '_cont', axis=1)
    
    time_win = 2
    sampling_rate = 2000
    time_steps = np.arange(-time_win*sampling_rate, time_win*sampling_rate+1, sampling_rate*0.01, dtype=int)
    time_steps_ms = (time_steps / sampling_rate) * 1000
    pl_df['pl_freq'] = pl_df['locked_freq_ind_z']
    pl_df['pl_strength'] = pl_df.apply(lambda x: np.max(x['tl_mrls_z'][x['pl_freq'], :]), axis=1)
    pl_df['pl_time_shift'] = pl_df.apply(lambda x: time_steps_ms[np.argmax(x['tl_mrls_z'][x['pl_freq'], :])], axis=1)
    pl_df['pl_latency'] = np.abs(pl_df['pl_time_shift'])
    
    # Remove bad HPC electrodes (U387 sessions RAH; U394_ses3 RAH).
    pl_df.drop(index=pl_df.query("(subj_sess==['U394_ses3', 'U387_ses1', 'U387_ses2', 'U387_ses3']) & (lfp_hemroi=='RAH')").index, inplace=True)
    
    # Ensure that each neuron is compared at most once to hippocampal LFPs from
    # each hemisphere. In cases with multiple comparisons to a single
    # hemisphere, we remove all but the most posterior connection.
    def axe_connections(x):
        """Return an empty string or a list of regions to remove."""
        removal_order = ['A', 'M', 'P']

        # Get a list of left and right ROIs, respectively.
        counts = OrderedDict([('left', [x_ for x_ in x if x_[0]=='L']),
                              ('right', [x_ for x_ in x if x_[0]=='R'])])

        # Return an empty string if neither hemisphere has more
        # than one ROI
        if np.all(np.array([len(x_) for x_ in counts.values()])<2):  
            return ''

        # For each hemisphere with more than one ROI,
        # remove all but the most posterior ROI.
        to_remove = []
        for key in counts.keys():
            vals = counts[key]
            if len(vals) > 1:
                iKeep = np.argmax([removal_order.index(val[1]) for val in vals])
                to_remove += [val for iVal, val in enumerate(vals) if iVal != iKeep]
        return to_remove
    
    if drop_repeat_connections:
        df = (pl_df
              .query("(edge=='ctx-hpc')")
              .reset_index()
              .groupby('subj_sess_unit')
              .agg({'unit': len, 
                    'lfp_hemroi': lambda x: list(x)}) 
              .query("(unit>=2)"))

        df['remove_rois'] = df['lfp_hemroi'].apply(lambda x: axe_connections(x))

        to_remove = []
        for subj_sess_unit, row in df.iterrows():
            remove_rois = row['remove_rois']
            if len(remove_rois):
                for roi in remove_rois:
                    to_remove.append((subj_sess_unit, roi))

        remove_inds = []
        for subj_sess_unit, lfp_hemroi in to_remove:
            remove_inds.append(pl_df.query("(subj_sess_unit=='{}') & (lfp_hemroi=='{}')".format(subj_sess_unit, lfp_hemroi)).iloc[0].name)

        pl_df.drop(index=remove_inds, inplace=True)
        pl_df.reset_index(drop=True, inplace=True)
    
    # Perform FDR correction (separately for ipsilateral and contralateral
    # comparisons within each edge type).
    alpha = 0.05
    pl_df['sig_z'] = pl_df['bs_pval_z'] < alpha
    pl_df['sig_z_fdr'] = False
    for edge_type in np.unique(pl_df.edge):
        for same_hem in [True, False]:
            pvals_in = np.array(pl_df.loc[(pl_df.edge==edge_type) & (pl_df.same_hem==same_hem)].bs_pval_z.tolist())
            if len(pvals_in) > 0:
                output = sm.stats.multipletests(pvals_in, alpha=0.05, method='fdr_tsbky', is_sorted=False, returnsorted=False)
                sig_out = list(output[0])
                pl_df.loc[(pl_df.edge==edge_type) & (pl_df.same_hem==same_hem), 'sig_z_fdr'] = sig_out

    return pl_df


def calc_power_across_spikes_unit_to_region(info,
                                            n_freqs=16,
                                            n_bootstraps=500,
                                            sampling_rate=2000,
                                            time_win=2,
                                            input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                            save_outputs=True,
                                            overwrite=False,
                                            output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/power_by_pl_fr/mean_power_across_spikes',
                                            sleep_max=0):
    """Obtain mean power values across spikes.
    
    We get a freq x time matrix centered around each of these
    spikes, extending 2s before and after the spike itself.
    
    Power has been log-transformed and then Z-scored across time (for the
    whole session).
    """
    # Check if output file already exists.
    subj_sess = info.subj_sess
    unit = info.unit
    lfp_roi = info.lfp_hemroi
    output_fname = 'mean_Z_log_power_across_spikes-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl'
    fpath = op.join(output_dir, output_fname.format(subj_sess, unit, lfp_roi))
    if op.exists(fpath) and not overwrite:
        power = dio.open_pickle(fpath)
        return power
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # General params.
    power_fname = op.join(input_dir, 'wavelet', 'power', 
                               'power-Z-log-{}-iChan{}-iFreq{}-2000Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    spike_fname = op.join(input_dir, 'spike_inds', 'spike_inds-2000Hz-{}-unit{}.pkl')
    pl_fname = op.join(input_dir, 'phase_locking', 'unit_to_region', 
                            'phase_locking_stats-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    cut_inds = int(sampling_rate * time_win)
    lfp_chan_inds = info.lfp_chan_inds
    
    # Get LFP power. 
    power_ = []
    for iChan in lfp_chan_inds:
        power__ = []
        for iFreq in range(n_freqs):
            power__.append(dio.open_pickle(power_fname.format(subj_sess, iChan, iFreq)))
        power_.append(power__)
    power_ = np.mean(power_, axis=0) # freq x time    
    
    # Get 2 sec of power surrounding each spike.
    power = []
    for spike_ind in info.spike_inds:
        power.append(power_[:, spike_ind-cut_inds:spike_ind+cut_inds+1])
    power = np.mean(power, axis=0) # freq x time; mean across spikes
    
    # Save output.
    if save_outputs:
        dio.save_pickle(power, fpath, verbose=False)
        
    return power
    

def calc_power_by_pl_unit_to_region(info,
                                    n_freqs=16,
                                    n_bootstraps=500,
                                    sampling_rate=2000,
                                    time_win=2,
                                    input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                    save_outputs=True,
                                    overwrite=False,
                                    output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/power_by_pl_fr',
                                    sleep_max=0):
    """Obtain power values for highly phase-locked spikes.
    
    We find the 20% of spikes that fired closest to the preferred phase at the
    locked frequency, and get a freq x time matrix centered around each of these
    spikes, extending 2s before and after the spike itself.
    
    This matrix is Z-scored against power matrices from randomly selected spikes
    (20% of spikes for each permutation). 
    
    These steps are performed for spike phase offsets relative to the target 
    region LFP and to the local LFP.
    """
    # Check if output file already exists.
    subj_sess = info.subj_sess
    unit = info.unit
    lfp_roi = info.lfp_hemroi
    output_fname = 'power_by_phase_locking-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl'
    fpath = op.join(output_dir, output_fname.format(subj_sess, unit, lfp_roi))
    if op.exists(fpath) and not overwrite:
        output = dio.open_pickle(fpath)
        return output
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # General params.
    power_fname = op.join(input_dir, 'wavelet', 'power', 
                               'power-Z-log-{}-iChan{}-iFreq{}-2000Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    spike_fname = op.join(input_dir, 'spike_inds', 'spike_inds-2000Hz-{}-unit{}.pkl')
    pl_fname = op.join(input_dir, 'phase_locking', 'unit_to_region', 
                            'phase_locking_stats-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    cut_inds = int(sampling_rate * time_win)
    lfp_chan_inds = info.lfp_chan_inds
    local_lfp_chan_inds = info.local_lfp_chan_inds
    local_phase_offsets = info.local_phase_offsets
    
    # Get LFP power. 
    power_ = []
    for iChan in lfp_chan_inds:
        power__ = []
        for iFreq in range(n_freqs):
            power__.append(dio.open_pickle(power_fname.format(subj_sess, iChan, iFreq)))
        power_.append(power__)
    power_ = np.mean(power_, axis=0) # freq x time
    
    # Get phase offsets for each spike.
    phase_offsets = info.phase_offsets
    local_phase_offsets = info.local_phase_offsets
    spike_inds, n_timepoints = dio.open_pickle(spike_fname.format(subj_sess, unit))
    spike_inds = info.spike_inds
    assert len(spike_inds) == len(phase_offsets) == len(local_phase_offsets)
    
    # Cut spikes from the beginning so both vectors are divisible by 5.
    cut = len(phase_offsets) % 5
    spike_inds = spike_inds[cut:]
    phase_offsets = phase_offsets[cut:]
    local_phase_offsets = local_phase_offsets[cut:]    
    
    # Setup the rest.
    offsets = OrderedDict([('phase_locking', phase_offsets),
                           ('local_phase_locking', local_phase_offsets)])
    n_spikes = len(spike_inds)
    n_samp = int(n_spikes / 5) # 20% of all spikes
    
    # Get 2 sec of power surrounding each spike.
    power = []
    for spike_ind in spike_inds:
        power.append(power_[:, spike_ind-cut_inds:spike_ind+cut_inds+1])
    power = np.array(power) # spike x freq x time
    del power_
    
    # Get power at each freq and time for the 20% of spikes that fired
    # closest to the preferred phase.
    output = OrderedDict([])
    for key in offsets.keys():
        sel_spikes = np.where(offsets[key]<=np.percentile(offsets[key], 20))[0]
        output[key] = np.mean(power[sel_spikes, :, :], axis=0) # freq x time; mean over spikes
                
    # Z-score against a null distribution of 20% of randomly selected spikes.
    bs_powers = []
    for iBoot in range(n_bootstraps):
        sel_spikes = np.random.choice(n_spikes, n_samp, replace=False)
        bs_powers.append(np.mean(power[sel_spikes, :, :], axis=0))
    bs_powers = np.array(bs_powers) # bs_index x freq x time
    bs_means = np.mean(bs_powers, axis=0)
    bs_stds = np.std(bs_powers, axis=0)
    
    for key in offsets.keys():
        output[key] = (output[key] - bs_means) / bs_stds

    # Save output.
    if save_outputs:
        dio.save_pickle(output, fpath, verbose=False)
        
    return output


def calc_power_by_pl_unit_to_region_v2(info,
                                    n_freqs=16,
                                    sampling_rate=2000,
                                    time_win=2,
                                    input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                    save_outputs=True,
                                    overwrite=False,
                                    output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/power_by_pl_fr',
                                    sleep_max=0):
    """Analyze Z-scored log-power values around each spike time.
    
    We get an freq x time matrix centered around each spike,
    extending 2s before and after the spike itself.
    
    Spikes are divided into top and bottom quartiles and analyzed according to:
    1) Simultaneous phase-locking strength, defined as the phase offset from the
       mean preferred spike phase with regard to each channel in the *target* LFP region.
    2) Simultaneous phase-locking strength, defined as the phase offset from the
       mean preferred spike phase with regard to each channel in the *local* LFP,
       at the same frequency as in (1).
    
    Where the mean freq x time power matrix is calculated for each quartile,
    across LFP channels and spikes in the quartile.
    """
    # Check if output file already exists.
    subj_sess = info.subj_sess
    unit = info.unit
    lfp_roi = info.lfp_hemroi
    output_fname = 'power_by_phase_locking-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl'
    fpath = op.join(output_dir, output_fname.format(subj_sess, unit, lfp_roi))
    if op.exists(fpath) and not overwrite:
        output = dio.open_pickle(fpath)
        return output
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # General params.
    power_fname = op.join(input_dir, 'wavelet', 'power', 
                               'power-Z-log-{}-iChan{}-iFreq{}-2000Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    spike_fname = op.join(input_dir, 'spike_inds', 'spike_inds-2000Hz-{}-unit{}.pkl')
    pl_fname = op.join(input_dir, 'phase_locking', 'unit_to_region', 
                            'phase_locking_stats-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
    #fr_fname = op.join(input_dir, 'spike_frs', 'spike_frs-2000Hz-{}-unit{}.pkl')
    cut_inds = int(sampling_rate * time_win)
    lfp_chan_inds = info.lfp_chan_inds
    local_lfp_chan_inds = info.local_lfp_chan_inds
    local_phase_offsets = info.local_phase_offsets
    
    # Get LFP power. 
    power_ = []
    for iChan in lfp_chan_inds:
        power__ = []
        for iFreq in range(n_freqs):
            power__.append(dio.open_pickle(power_fname.format(subj_sess, iChan, iFreq)))
        power_.append(power__)
    power_ = np.mean(power_, axis=0) # freq x time
    
    # Get phase offsets for each spike.
    phase_offsets = info.phase_offsets
    local_phase_offsets = info.local_phase_offsets
    spike_inds, n_timepoints = dio.open_pickle(spike_fname.format(subj_sess, unit))
    #firing_rates = dio.open_pickle(fr_fname.format(subj_sess, unit))
    #firing_rates = firing_rates[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))]
    spike_inds = info.spike_inds
    assert len(spike_inds) == len(phase_offsets) == len(local_phase_offsets)# == len(firing_rates)

    # Cut spikes from the beginning so both vectors are divisible by 4.
    cut = len(phase_offsets) % 4
    spike_inds = spike_inds[cut:]
    phase_offsets = phase_offsets[cut:]
    local_phase_offsets = local_phase_offsets[cut:]
    #firing_rates = firing_rates[cut:]

    # Get 2 sec of power surrounding each spike.
    power = []
    for spike_ind in spike_inds:
        power.append(power_[:, spike_ind-cut_inds:spike_ind+cut_inds+1])
    power = stats.zscore(power, axis=0) # spike x freq x time; Z-scored over spike events
    del power_
    
    # Sort spikes by phase offset.
    xsort = OrderedDict()
    xsort['phase_locking'] = phase_offsets.argsort()[::-1] # low to low offset from the preferred phase
    xsort['local_phase_locking'] = local_phase_offsets.argsort()[::-1]
    #xsort['firing_rate'] = firing_rates.argsort()

    # Split power into quartiles based on phase_locking offset.
    output = OrderedDict()
    for key in xsort.keys():
        split_power = np.split(power[xsort[key], :, :], 4, axis=0) # low to high phase-locking
        output[key] = OrderedDict([('q1', np.mean(split_power[0], axis=0).astype(np.float32)),
                                   ('q2', np.mean(split_power[1], axis=0).astype(np.float32)),
                                   ('q3', np.mean(split_power[2], axis=0).astype(np.float32)),
                                   ('q4', np.mean(split_power[3], axis=0).astype(np.float32))])
        output[key]['q4-q1'] = output[key]['q4'] - output[key]['q1']

    # Save output.
    if save_outputs:
        dio.save_pickle(output, fpath, verbose=False)
        
    return output
    
# def calc_power_by_pl_fr_unit_to_region_v1(subj_sess,
#                                           unit,
#                                           lfp_roi,
#                                           n_freqs=16,
#                                           sampling_rate=2000,
#                                           time_win=2,
#                                           input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
#                                           save_outputs=True,
#                                           overwrite=False,
#                                           output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/power_by_pl_fr',
#                                           sleep_max=0):
#     """Analyze Z-scored log-power values around each spike time.
#     
#     We get an freq x time matrix centered around each spike,
#     extending 2s before and after the spike itself.
#     
#     Spikes are divided into top and bottom quartiles and analyzed according to:
#     1) Simultaneous phase-locking strength, defined as the phase offset from
#        the mean preferred spike phase with regard to each channel in the LFP region
#     2) Simultaneous firing rate
#     
#     Where the mean freq x time power matrix is calculated for each quartile,
#     across LFP channels and spikes in the quartile.
#     """
#     # Check if output file already exists.
#     output_fname = 'power_by_phase_locking_and_firing_rate-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl'
#     fpath = op.join(output_dir, output_fname.format(subj_sess, unit, lfp_roi))
#     if op.exists(fpath) and not overwrite:
#         output = dio.open_pickle(fpath)
#         return output
#     
#     # Take a nap before running.
#     if sleep_max > 0:
#         sleep(int(sleep_max * np.random.rand()))
#         
#     # General params.
#     power_fname = op.join(input_dir, 'wavelet', 'power', 
#                                'power-Z-log-{}-iChan{}-iFreq{}-2000Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
#     spike_fname = op.join(input_dir, 'spike_inds', 'spike_inds-2000Hz-{}-unit{}.pkl')
#     fr_fname = op.join(input_dir, 'spike_frs', 'spike_frs-2000Hz-{}-unit{}.pkl')
#     pl_fname = op.join(input_dir, 'phase_locking', 'unit_to_region', 
#                             'phase_locking_stats-{}-unit_{}-lfp_{}-2000Hz-notch60_120Hz5cycles-16log10freqs_0.5_to_90.5Hz.pkl')
#     cut_inds = int(sampling_rate * time_win)
#     info = dio.open_pickle(pl_fname.format(subj_sess, unit, lfp_roi))
#     lfp_chan_inds = info.lfp_chan_inds
# 
#     # Get LFP power. 
#     power_ = []
#     for iChan in lfp_chan_inds:
#         power__ = []
#         for iFreq in range(n_freqs):
#             power__.append(dio.open_pickle(power_fname.format(subj_sess, iChan, iFreq)))
#         power_.append(power__)
#     power_ = np.mean(power_, axis=0) # freq x time
#     
#     # Get phase offsets and firing rates for each spike.
#     phase_offsets = info.phase_offsets
#     phase_offsets2 = info.phase_offsets_tl_locked_time_freq_z
#     spike_inds, n_timepoints = dio.open_pickle(spike_fname.format(subj_sess, unit))
#     firing_rates = dio.open_pickle(fr_fname.format(subj_sess, unit))
#     firing_rates = firing_rates[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))]
#     spike_inds = info.spike_inds
#     assert len(spike_inds) == len(phase_offsets) == len(phase_offsets2) == len(firing_rates)
# 
#     # Cut spikes from the beginning so both vectors are divisible by 4.
#     cut = len(phase_offsets) % 4
#     spike_inds = spike_inds[cut:]
#     phase_offsets = phase_offsets[cut:]
#     phase_offsets2 = phase_offsets2[cut:]
#     firing_rates = firing_rates[cut:]
# 
#     # Get 2 sec of power surrounding each spike.
#     power = []
#     for spike_ind in spike_inds:
#         power.append(power_[:, spike_ind-cut_inds:spike_ind+cut_inds+1])
#     power = stats.zscore(power, axis=0) # spike x freq x time; Z-scored over spike events
#     del power_
#     
#     # Sort spikes by phase offset and firing rate.
#     xsort = OrderedDict()
#     xsort['phase_locking'] = phase_offsets.argsort()[::-1] # low to high offset from the preferred phase
#     xsort['phase_locking2'] = phase_offsets2.argsort()[::-1]
#     xsort['firing_rate'] = firing_rates.argsort()
# 
#     # Split power into quartiles based on phase_locking offset or firing rate.
#     output = OrderedDict()
#     for key in xsort.keys():
#         split_power = np.split(power[xsort[key], :, :], 4, axis=0) # low to high phase-locking
#         output[key] = OrderedDict([('low', np.mean(split_power[0], axis=0).astype(np.float32)),
#                                    ('high', np.mean(split_power[-1], axis=0).astype(np.float32))])
#         output[key]['high-low'] = output[key]['high'] - output[key]['low']
# 
#     # Save output.
#     if save_outputs:
#         dio.save_pickle(output, fpath, verbose=False)
#         
#     return output


def calc_phase_locking_mrl_morlet_unit_to_region2(info,
                                                  n_freqs=16,
                                                  sampling_rate=2000,
                                                  time_win=2,
                                                  n_bootstraps=1000,
                                                  remove_hfa=False,
                                                  hfa_zthresh=3,
                                                  hfa_win=2.5,
                                                  max_hfa_overlap=0.2,
                                                  save_outputs=True,
                                                  input_dir='/home1/dscho/projects/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                                  phase_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/wavelet/phase',
                                                  phase_fname='phase-{}-iChan{}-iFreq{}-{}Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_90.5Hz.pkl',
                                                  output_dir='/home1/dscho/projects/unit_activity_and_hpc_theta/data/crosselec_phase_locking/phase_locking/unit_to_region',
                                                  output_fname='phase_locking_stats-{}-unit_{}-lfp_{}-{}Hz-notch60_120Hz{}5cycles-16log10freqs_0.5_to_90.5Hz.pkl',
                                                  sleep_max=0):
    """Calculate time-lag and bootstrap phase-locking."""
    
    def bootstrap_p(x):
        """Return a p-value.
    
        For each connection, the maximum observed Z-score (across frequencies) 
        is compared against the surrogate distribution of maximum Z-scores.
        """
        obs = x.mrls_z # n_freq vector
        null = x.bs_mrls_z # n_freq x n_boot matrix
        max_obs = np.max(obs)
        max_null = np.max(null, axis=0)
        n_bootstraps = len(max_null)
        bs_ind = np.sum(max_null >= max_obs)
        pval = (1 + bs_ind) / (1 + n_bootstraps)
        return bs_ind, pval
        
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # General params.
    subj_sess, unit, lfp_chans, lfp_roi = info.subj_sess, info.unit, info.lfp_chan_inds, info.lfp_hemroi
    sampling_rate = int(sampling_rate)
    cut_inds = int(sampling_rate * time_win)
    time_steps = np.arange(-time_win*sampling_rate, time_win*sampling_rate+1, sampling_rate*0.01, dtype=int)
    time_steps_ms = (time_steps / sampling_rate) * 1000
    t0 = np.where(time_steps_ms==0)[0][0]
    bs_steps = dio.open_pickle(op.join(input_dir, 'bootstrap_shifts', 
                                       '{}_{}bootstraps_{}Hz_{}timewin.pkl'
                                       .format(subj_sess, n_bootstraps, sampling_rate, time_win)))
    hfa_dir = op.join(input_dir, 'hfa_inds')
    alpha = 0.05
    
    # Load spike inds and remove time_win secs from the beggining and end of the session.
    spike_inds, n_timepoints = dio.open_pickle(op.join(input_dir, 'spike_inds', 'spike_inds-{}Hz-{}-unit{}.pkl'.format(sampling_rate, subj_sess, unit)))
    spike_inds = np.unique(spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))])
    
    # Get phase-locking strength at each time lag for each channel and frequency,
    # and generate a null distribution from time-shifted spike trains.
    mrls_arr = []
    bs_mrls_arr = []
    n_samp_spikes = []
    keep_chans = []
    for iChan in lfp_chans:
        hfa_filename = op.join(hfa_dir, 
                               'hfa_inds-{}Hz-zthresh{}-win{}ms-{}-iChan{}.pkl'
                               .format(sampling_rate, hfa_zthresh, hfa_win, subj_sess, iChan))
        if remove_hfa and op.exists(hfa_filename):
            hfa_inds = dio.open_pickle(hfa_filename)
            
            # Retain spike times that don't overlap with HFA on the LFP channel.
            time_spike_inds = []
            for step in time_steps:
                spike_inds_shifted = shift_spike_inds(spike_inds, n_timepoints, step)
                ma = np.isin(spike_inds_shifted, hfa_inds, assume_unique=True, invert=True)
                time_spike_inds.append(spike_inds_shifted[ma])

            bs_spike_inds = []
            for step in bs_steps:
                spike_inds_shifted = shift_spike_inds(spike_inds, n_timepoints, step)
                ma = np.isin(spike_inds_shifted, hfa_inds, assume_unique=True, invert=True)
                bs_spike_inds.append(spike_inds_shifted[ma])

            # Sample an equal number of uncontaminated spikes.
            n_samp_spikes_ = np.min([len(x) for x in (time_spike_inds + bs_spike_inds)])
            time_spike_inds = np.array([np.sort(np.random.permutation(x)[:n_samp_spikes_]) for x in time_spike_inds])
            bs_spike_inds = np.array([np.sort(np.random.permutation(x)[:n_samp_spikes_]) for x in bs_spike_inds])
            n_samp_spikes.append(n_samp_spikes_)
            
            # If spikes and LFP HFA overlap too much, move to the next chanenl 
            # without processing the current one.
            hfa_overlap = 1 - (n_samp_spikes_ / len(spike_inds))
            if hfa_overlap > max_hfa_overlap:
                keep_chans.append(False)
                continue

        mrls_arr_ = []
        bs_mrls_arr_ = []
        for iFreq in range(n_freqs):
            phase = dio.open_pickle(op.join(phase_dir, phase_fname.format(subj_sess, iChan, iFreq, sampling_rate)))
            mrls_arr__ = []
            bs_mrls_arr__ = []
            if remove_hfa and op.exists(hfa_filename):
                for iStep, step in enumerate(time_steps):
                    mrls_arr__.append(circstats.circmoment(phase[time_spike_inds[iStep, :]])[1])

                for iStep, step in enumerate(bs_steps):
                    bs_mrls_arr__.append(circstats.circmoment(phase[bs_spike_inds[iStep, :]])[1])
            else:
                for step in time_steps: # calculated from LFP preceding spikes to spikes preceding LFP
                    spike_inds_shifted = shift_spike_inds(spike_inds, n_timepoints, step)
                    mrls_arr__.append(circstats.circmoment(phase[spike_inds_shifted])[1])

                for step in bs_steps:
                    spike_inds_shifted = shift_spike_inds(spike_inds, n_timepoints, step)
                    bs_mrls_arr__.append(circstats.circmoment(phase[spike_inds_shifted])[1])
            
            mrls_arr_.append(mrls_arr__)
            bs_mrls_arr_.append(bs_mrls_arr__)
        mrls_arr.append(mrls_arr_)
        bs_mrls_arr.append(bs_mrls_arr_)
        keep_chans.append(True)

    mrls_arr = np.array(mrls_arr) # channel x frequency x time shift
    bs_mrls_arr = np.array(bs_mrls_arr) # channel x frequency x permutation
    if not remove_hfa:
        n_samp_spikes = len(spike_inds)

    # Check that phase-locking was quantified for at least one LFP channel.
    if np.sum(keep_chans) == 0:
        print('No channels could be processed after removing HFA overlap.')
        return None
        
    # Add phase-locking stats to info.
    info['unit_nspikes'] = len(spike_inds) # number of spikes recorded for the unit
    info['unit_nsamp_spikes'] = n_samp_spikes # number of spikes that were used to assess phase-locking (mean across channels)
    info['keep_chans'] = keep_chans # channels that were used for phase-locking comparison
    info['spike_inds'] = spike_inds
    info['bs_mrls'] = np.mean(bs_mrls_arr, axis=0).astype(np.float32) # mean across channels (leaving freq x bs_index)
    info['tl_mrls'] = np.mean(mrls_arr, axis=0).astype(np.float32) # mean across channels (leaving freq x time_shift)
    

    bs_means = np.expand_dims(np.mean(info['bs_mrls'], axis=-1), axis=-1) # mean at each freq
    bs_stds = np.expand_dims(np.std(info['bs_mrls'], axis=-1), axis=-1) # std at each freq
    info['bs_mrls_z'] = ((info['bs_mrls'] - bs_means) / bs_stds).astype(np.float32) # freq x bs_index
    # bs_means = np.expand_dims(np.mean(bs_mrls_arr, axis=-1), axis=-1) # mean at each chan x freq
    # bs_stds = np.expand_dims(np.std(bs_mrls_arr, axis=-1), axis=-1) # std at each chan x freq
    # info['bs_mrls_z'] = np.mean((bs_mrls_arr - bs_means) / bs_stds, axis=0).astype(np.float32) # freq x bs_index
    # Note: we flip tl_mrls_z in time so we go from spikes predicting LFPs to LFPs predicting spikes
    info['tl_mrls_z'] = np.flip((info['tl_mrls'] - bs_means) / bs_stds, axis=-1).astype(np.float32) # Z-scores after collapsing across channels; freq x time_shift
    # info['tl_mrls_z'] = np.flip(np.mean((mrls_arr - bs_means) / bs_stds, axis=0), 
    #                             axis=-1).astype(np.float32) # Z-score at each channel, freq, time_shift, then a mean across channels; freq x time_shift
    

    info['mrls_z'] = info['tl_mrls_z'][:, t0] # Z-MRLs across freqs at 0 time shift (t=0)
    info['locked_freq_ind_z'] = info['mrls_z'].argmax() # freq index of the max Z=MRL at t=0
    info['locked_mrl_z'] = info['mrls_z'][info['locked_freq_ind_z']] # max Z-MRL across freqs at t=0
    bs_ind_z, bs_pval_z = bootstrap_p(info)
    info['bs_ind_z'] = bs_ind_z # number of bootstrap max(Z-MRLs) >= max(Z-MRL) at t=0 (taking the max across frequencies)
    info['bs_pval_z'] = bs_pval_z # empirical p-value of the max Z-MRL at t=0
    info['sig_z'] = info['bs_pval_z'] < alpha
    tl_locked_freq_ind_z, tl_locked_time_ind_z = np.unravel_index(info['tl_mrls_z'].argmax(), info['tl_mrls_z'].shape)
    info['tl_locked_freq_z'] = tl_locked_freq_ind_z # freq index of the max Z-MRL in the tl_mrls_z matrix
    info['tl_locked_time_z'] = time_steps_ms[tl_locked_time_ind_z] # time (in ms) of the max Z-MRL in the tl_mrls_z matrix (<0 means spikes precede LFP phase)
    info['tl_locked_mrl_z'] = np.max(info['tl_mrls_z']) # max Z-MRL in the tl_mrls_z matrix
    
    # Get the circular distance between each spike phase (at the locked frequency) 
    # and the mean preferred phase for each channel. Then take the average circular distance
    # across channels. This is done for t=0 time shift and at the maximum locked time shift.
    step = time_steps[-tl_locked_time_ind_z] # negative since we flipped tl_mrls_z above
    spike_inds_shifted = shift_spike_inds(spike_inds, n_timepoints, step)
    test_spike_inds = [spike_inds, spike_inds_shifted]
    test_freqs = [info['locked_freq_ind_z'], info['tl_locked_freq_z']]
    for i in range(len(test_spike_inds)):
        spike_inds_ = test_spike_inds[i]
        iFreq = test_freqs[i]
        spike_phases = []
        phase_offsets = []
        for iChan in lfp_chans:
            phase = dio.open_pickle(op.join(phase_dir, phase_fname.format(subj_sess, iChan, iFreq, sampling_rate)))
            spike_phases_ = phase[spike_inds_]
            pref_phase_ = circstats.circmoment(spike_phases_)[0]
            spike_phases.append(spike_phases_)
            phase_offsets.append(np.abs(pycircstat.descriptive.cdiff(spike_phases_, np.repeat(pref_phase_, len(spike_phases_)))))
        pref_phase = circstats.circmoment(np.array(spike_phases).flatten())[0] # preferred phase taking all spike phases across channels
        phase_offsets = np.mean(phase_offsets, axis=0).astype(np.float32) # circular distance between each spike and the preferred phase for each channel, averaged across channels
        if i == 0:
            info['pref_phase'] = pref_phase
            info['phase_offsets'] = phase_offsets 
        else:
            info['pref_phase_tl_locked_time_freq_z'] = pref_phase
            info['phase_offsets_tl_locked_time_freq_z'] = phase_offsets 
    
    if save_outputs:
        hfa_str = '-remove_hfa_z{}_{}ms-'.format(hfa_zthresh, hfa_win) if remove_hfa else '-'
        fpath = op.join(output_dir, output_fname.format(subj_sess, unit, lfp_roi, sampling_rate, hfa_str))
        dio.save_pickle(info, fpath, verbose=False)
        
    return info


def calc_phase_locking_mrl_morlet_unit_to_region(info,
                                                 n_freqs=16,
                                                 sampling_rate=2000,
                                                 time_win=2,
                                                 n_bootstraps=1000,
                                                 remove_cospikes=False,
                                                 cospike_steps=4,
                                                 save_outputs=True,
                                                 input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                                 phase_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/wavelet/phase',
                                                 phase_fname='phase-{}-iChan{}-iFreq{}-{}Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_90.5Hz.pkl',
                                                 output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/phase_locking/unit_to_region',
                                                 output_fname='phase_locking_stats-{}-unit_{}-lfp_{}-{}Hz-notch60_120Hz{}5cycles-16log10freqs_0.5_to_90.5Hz.pkl',
                                                 sleep_max=0):
    """Calculate time-lag and bootstrap phase-locking."""
    
    def bootstrap_p(x):
        """Return a p-value.
    
        For each connection, the maximum observed Z-score (across frequencies) 
        is compared against the surrogate distribution of maximum Z-scores.
        """
        obs = x.mrls_z # n_freq vec
        null = x.bs_mrls_z # n_freq x n_boot vec
        max_obs = np.max(obs)
        max_null = np.max(null, axis=0)
        n_bootstraps = len(max_null)
        bs_ind = np.sum(max_null >= max_obs)
        pval = (1 + bs_ind) / (1 + n_bootstraps)
        return bs_ind, pval
        
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    # General params.
    subj_sess, unit, lfp_chans, lfp_roi = info.subj_sess, info.unit, info.lfp_chan_inds, info.lfp_hemroi
    sampling_rate = int(sampling_rate)
    cut_inds = int(sampling_rate * time_win)
    time_steps = np.arange(-time_win*sampling_rate, time_win*sampling_rate+1, sampling_rate*0.01, dtype=int)
    time_steps_ms = (time_steps / sampling_rate) * 1000
    t0 = np.where(time_steps_ms==0)[0][0]
    bs_steps = dio.open_pickle(op.join(input_dir, 'bootstrap_shifts', '{}_{}bootstraps_{}Hz_{}timewin.pkl'.format(subj_sess, int(n_bootstraps), int(sampling_rate), int(time_win))))
    alpha = 0.05
    mua_dir = '/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/spike_inds_mua'
    
    # Load spike inds and remove time_win secs from the beggining and end of the session.
    spike_inds, n_timepoints = dio.open_pickle(op.join(input_dir, 'spike_inds', 'spike_inds-{}Hz-{}-unit{}.pkl'.format(sampling_rate, subj_sess, unit)))
    spike_inds = np.unique(spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))])
    
    # Get phase-locking strength at each time lag for each channel and frequency,
    # and generate a null distribution from time-shifted spike trains.
    mrls_arr = []
    bs_mrls_arr = []
    n_samp_spikes = []
    for iChan in lfp_chans:
        lfp_mua_f = op.join(mua_dir, 'mua_spike_inds-{}Hz-{}-iChan{}.pkl'.format(sampling_rate, subj_sess, iChan))
        if remove_cospikes and op.exists(lfp_mua_f):
            # Get a window around each spike on the LFP channel
            mua_spike_inds_ = dio.open_pickle(lfp_mua_f)
            mua_spike_inds = np.array([])
            for offset in range(-cospike_steps, cospike_steps+1):
                mua_spike_inds = np.append(mua_spike_inds, mua_spike_inds_ + offset)
            mua_spike_inds = np.unique(mua_spike_inds).astype(int)
        
            # Retain spike times that don't overlap with MUA on the LFP channel.
            time_spike_inds = []
            for step in time_steps:
                spike_inds_shifted = shift_spike_inds(spike_inds, n_timepoints, step)
                ma = np.isin(spike_inds_shifted, mua_spike_inds, assume_unique=True, invert=True)
                time_spike_inds.append(spike_inds_shifted[ma])

            bs_spike_inds = []
            for step in bs_steps:
                spike_inds_shifted = shift_spike_inds(spike_inds, n_timepoints, step)
                ma = np.isin(spike_inds_shifted, mua_spike_inds, assume_unique=True, invert=True)
                bs_spike_inds.append(spike_inds_shifted[ma])

            # Sample an equal number of uncontaminated spikes. 
            n_samp_spikes_ = np.min([len(x) for x in (time_spike_inds + bs_spike_inds)])
            time_spike_inds = np.array([np.sort(np.random.permutation(x)[:n_samp_spikes_]) for x in time_spike_inds])
            bs_spike_inds = np.array([np.sort(np.random.permutation(x)[:n_samp_spikes_]) for x in bs_spike_inds])
            n_samp_spikes.append(n_samp_spikes_)
        
        mrls_arr_ = []
        bs_mrls_arr_ = []
        for iFreq in range(n_freqs):
            phase = dio.open_pickle(op.join(phase_dir, phase_fname.format(subj_sess, iChan, iFreq, sampling_rate)))
            mrls_arr__ = []
            bs_mrls_arr__ = []
            if remove_cospikes and op.exists(lfp_mua_f):
                for iStep, step in enumerate(time_steps):
                    mrls_arr__.append(circstats.circmoment(phase[time_spike_inds[iStep, :]])[1])

                for iStep, step in enumerate(bs_steps):
                    bs_mrls_arr__.append(circstats.circmoment(phase[bs_spike_inds[iStep, :]])[1])
            else:
                for step in time_steps: # calculated from LFP preceding spikes to spikes preceding LFP
                    spike_inds_shifted = shift_spike_inds(spike_inds, n_timepoints, step)
                    mrls_arr__.append(circstats.circmoment(phase[spike_inds_shifted])[1])

                for step in bs_steps:
                    spike_inds_shifted = shift_spike_inds(spike_inds, n_timepoints, step)
                    bs_mrls_arr__.append(circstats.circmoment(phase[spike_inds_shifted])[1])
            
            mrls_arr_.append(mrls_arr__)
            bs_mrls_arr_.append(bs_mrls_arr__)
        mrls_arr.append(mrls_arr_)
        bs_mrls_arr.append(bs_mrls_arr_)
    
    mrls_arr = np.array(mrls_arr) # channel x frequency x time shift
    bs_mrls_arr = np.array(bs_mrls_arr)
    if remove_cospikes and len(n_samp_spikes)>0:
        n_samp_spikes = np.mean(n_samp_spikes, dtype=int)
    else:
        n_samp_spikes = len(spike_inds)
        
    # Get phase-locking stats and append to the info series.
    info['unit_nspikes'] = len(spike_inds) # number of spikes recorded for the unit
    info['unit_nsamp_spikes'] = n_samp_spikes # number of spikes that were used to assess phase-locking (mean across channels)
    info['spike_inds'] = spike_inds
    info['bs_mrls'] = np.mean(bs_mrls_arr, axis=0).astype(np.float32) # mean across channels (leaving freq x bs_index)
    info['tl_mrls'] = np.mean(mrls_arr, axis=0).astype(np.float32) # mean across channels (leaving freq x time_shift)
    bs_means = np.expand_dims(np.mean(info['bs_mrls'], axis=-1), axis=-1) # mean at each freq
    bs_stds = np.expand_dims(np.std(info['bs_mrls'], axis=-1), axis=-1) # std at each freq
    info['bs_mrls_z'] = ((info['bs_mrls'] - bs_means) / bs_stds).astype(np.float32) # freq x bs_index
    # Note: we flip tl_mrls_z in time so we go from spikes predicting LFPs to LFPs predicting spikes
    info['tl_mrls_z'] = np.flip((info['tl_mrls'] - bs_means) / bs_stds, axis=-1).astype(np.float32) # Z-scores after collapsing across channels; freq x time_shift
    info['mrls_z'] = info['tl_mrls_z'][:, t0] # Z-MRLs across freqs at 0 time shift (t=0)
    info['locked_freq_ind_z'] = info['mrls_z'].argmax() # freq index of the max Z=MRL at t=0
    info['locked_mrl_z'] = info['mrls_z'][info['locked_freq_ind_z']] # max Z-MRL across freqs at t=0
    bs_ind_z, bs_pval_z = bootstrap_p(info)
    info['bs_ind_z'] = bs_ind_z # number of bootstrap max(Z-MRLs) >= max(Z-MRL) at t=0 (taking the max across frequencies)
    info['bs_pval_z'] = bs_pval_z # empirical p-value of the max Z-MRL at t=0
    info['sig_z'] = info['bs_pval_z'] < alpha
    tl_locked_freq_ind_z, tl_locked_time_ind_z = np.unravel_index(info['tl_mrls_z'].argmax(), info['tl_mrls_z'].shape)
    info['tl_locked_freq_z'] = tl_locked_freq_ind_z # freq index of the max Z-MRL in the tl_mrls_z matrix
    info['tl_locked_time_z'] = time_steps_ms[tl_locked_time_ind_z] # time (in ms) of the max Z-MRL in the tl_mrls_z matrix (<0 means spikes precede LFP phase)
    info['tl_locked_mrl_z'] = np.max(info['tl_mrls_z']) # max Z-MRL in the tl_mrls_z matrix
    
    # Get the circular distance between each spike phase (at the locked frequency) 
    # and the mean preferred phase for each channel. Then take the average circular distance
    # across channels. This is done for t=0 time shift and at the maximum locked time shift.
    step = time_steps[-tl_locked_time_ind_z] # negative since we flipped tl_mrls_z above
    spike_inds_shifted = shift_spike_inds(spike_inds, n_timepoints, step)
    test_spike_inds = [spike_inds, spike_inds_shifted]
    test_freqs = [info['locked_freq_ind_z'], info['tl_locked_freq_z']]
    for i in range(len(test_spike_inds)):
        spike_inds_ = test_spike_inds[i]
        iFreq = test_freqs[i]
        spike_phases = []
        phase_offsets = []
        for iChan in lfp_chans:
            phase = dio.open_pickle(op.join(phase_dir, phase_fname.format(subj_sess, iChan, iFreq, sampling_rate)))
            spike_phases_ = phase[spike_inds_]
            pref_phase_ = circstats.circmoment(spike_phases_)[0]
            spike_phases.append(spike_phases_)
            phase_offsets.append(np.abs(pycircstat.descriptive.cdiff(spike_phases_, np.repeat(pref_phase_, len(spike_phases_)))))
        pref_phase = circstats.circmoment(np.array(spike_phases).flatten())[0] # preferred phase taking all spike phases across channels
        phase_offsets = np.mean(phase_offsets, axis=0).astype(np.float32) # circular distance between each spike and the preferred phase for each channel, averaged across channels
        if i == 0:
            info['pref_phase'] = pref_phase
            info['phase_offsets'] = phase_offsets 
        else:
            info['pref_phase_tl_locked_time_freq_z'] = pref_phase
            info['phase_offsets_tl_locked_time_freq_z'] = phase_offsets 
    
    if save_outputs:
        cospikes_str = '-remove_cospikes{}-'.format(cospike_steps) if remove_cospikes else '-'
        fpath = op.join(output_dir, output_fname.format(subj_sess, unit, lfp_roi, sampling_rate, cospikes_str))
        dio.save_pickle(info, fpath, verbose=False)
        
    return info
    

def calc_phase_locking_mrl_morlet_unit_to_region_OLD(info,
                                                     n_freqs=16,
                                                     sampling_rate=500,
                                                     time_win=2,
                                                     n_bootstraps=1000,
                                                     remove_cospikes=True,
                                                     save_outputs=True,
                                                     input_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                                     phase_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/wavelet_phase',
                                                     phase_fname='phase-{}-iChan{}-iFreq{}-{}Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_16.0Hz.pkl',
                                                     output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/phase_locking/unit_to_region',
                                                     output_fname='phase_locking_stats-{}-unit_{}-lfp_{}-{}Hz-notch60_120Hz-nospikeinterp-5cycles-16log10freqs_0.5_to_16.0Hz.pkl',
                                                     sleep_max=0):
    """Calculate time-lag and bootstrap phase-locking."""
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
    
    # General params.
    subj_sess, unit, lfp_chans, lfp_roi = info.subj_sess, info.unit, info.lfp_chan_inds, info.lfp_hemroi
    sampling_rate = int(sampling_rate)
    cut_inds = int(sampling_rate * time_win)
    t0 = sampling_rate
    time_steps = np.arange(-time_win*sampling_rate, time_win*sampling_rate+1, sampling_rate*0.01, dtype=int)
    time_steps_ms = (time_steps / sampling_rate) * 1000
    bs_steps = dio.open_pickle(op.join(input_dir, 'bootstrap_shifts', '{}_{}bootstraps_{}Hz_{}timewin.pkl'.format(subj_sess, int(n_bootstraps), int(sampling_rate), int(time_win))))
    alpha = 0.05 / n_freqs
    mua_dir = '/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/spike_inds_mua'
    
    # Load spike inds and remove time_win secs from the beggining and end of the session.
    spike_inds, n_timepoints = dio.open_pickle(op.join(input_dir, 'spike_inds', 'spike_inds-{}Hz-{}-unit{}.pkl'
                                                            .format(sampling_rate, subj_sess, unit)))
    spike_inds = spike_inds[(spike_inds>cut_inds) & (spike_inds<(n_timepoints-cut_inds))]
    spike_inds = np.unique(spike_inds)
    
    # Get phase-locking strength at each time lag for each channel and frequency,
    # and generate a null distribution from time-shifted spike trains.
    mrls_arr = []
    bs_mrls_arr = []
    for iChan in lfp_chans:
        lfp_mua_f = op.join(mua_dir, 'mua_spike_inds-{}Hz-{}-iChan{}.pkl'.format(sampling_rate, subj_sess, iChan))
        if remove_cospikes and op.exists(lfp_mua_f):
            mua_spike_inds = dio.open_pickle(lfp_mua_f)
            mua_spike_inds = np.unique(np.concatenate((mua_spike_inds, (mua_spike_inds - 1), (mua_spike_inds + 1))))
        mrls_arr_ = []
        bs_mrls_arr_ = []
        for iFreq in range(n_freqs):
            phase = dio.open_pickle(op.join(phase_dir, phase_fname.format(subj_sess, iChan, iFreq, sampling_rate)))
            mrls_arr__ = []
            bs_mrls_arr__ = []
            if remove_cospikes and op.exists(lfp_mua_f):
                for step in time_steps:
                    spike_inds_shifted = shift_spike_inds(spike_inds, n_timepoints, step)
                    ma = np.isin(spike_inds_shifted, mua_spike_inds, assume_unique=True, invert=True)
                    spike_inds_shifted = spike_inds_shifted[ma]
                    mrls_arr__.append(circstats.circmoment(phase[spike_inds_shifted])[1])
                
                for step in bs_steps:
                    spike_inds_shifted = shift_spike_inds(spike_inds, n_timepoints, step)
                    ma = np.isin(spike_inds_shifted, mua_spike_inds, assume_unique=True, invert=True)
                    spike_inds_shifted = spike_inds_shifted[ma]
                    bs_mrls_arr__.append(circstats.circmoment(phase[spike_inds_shifted])[1])
            else:
                for step in time_steps:
                    spike_inds_shifted = shift_spike_inds(spike_inds, n_timepoints, step)
                    mrls_arr__.append(circstats.circmoment(phase[spike_inds_shifted])[1])
                
                for step in bs_steps:
                    spike_inds_shifted = shift_spike_inds(spike_inds, n_timepoints, step)
                    bs_mrls_arr__.append(circstats.circmoment(phase[spike_inds_shifted])[1])
                
            mrls_arr_.append(mrls_arr__)
            bs_mrls_arr_.append(bs_mrls_arr__)
        mrls_arr.append(mrls_arr_)
        bs_mrls_arr.append(bs_mrls_arr_)
    mrls_arr = np.array(mrls_arr) # channel x frequency x time shift
    bs_mrls_arr = np.array(bs_mrls_arr)
    
#    # Z-score time-shifted MRLs against their respective channel
#    # and frequency from the bootstrap distribution.
#    bs_means = np.expand_dims(np.mean(bs_mrls_arr, axis=-1), axis=-1)
#    bs_stds = np.expand_dims(np.std(bs_mrls_arr, axis=-1), axis=-1)
#    bs_mrls_arr_z = (bs_mrls_arr - bs_means) / bs_stds
#    mrls_arr_z = (mrls_arr - bs_means) / bs_stds
    
    # Get phase-locking stats and append to the info series.
    info['bs_mrls'] = np.mean(bs_mrls_arr, axis=0) # mean across channels (leaving freq x bs_index)
    info['tl_mrls'] = np.mean(mrls_arr, axis=0) # mean across channels (leaving freq x time_step)
    info['mrls'] = info['tl_mrls'][:, t0] # MRLs across freqs at 0 time shift (t=0)
    info['locked_freq_ind'] = info['mrls'].argmax() # freq index of the max MRL at t=0
    info['locked_mrl'] = info['mrls'][info['locked_freq_ind']] # max MRL across freqs at t=0
    info['bs_ind'] = np.sum(info['bs_mrls'][info['locked_freq_ind'], :] >= info['locked_mrl']) # number of bootstrap MRLs >= the max MRL at t=0 (at that freq)
    info['bs_pval'] = (1 + info['bs_ind']) / (1 + n_bootstraps) # p-value of the max MRL at t=0
    info['sig'] = info['bs_pval'] < alpha # significance of the t=0 max MRL (vs. alpha Bonferonni-corrected for the number of freqs)
    info['tl_locked_time_ind'] = info['tl_mrls'][info['locked_freq_ind'], :].argmax() # time index of the max MRL at the t=0 locked freq
    info['tl_locked_time'] = time_steps_ms[info['tl_locked_time_ind']] # time (in ms) of the max MRL at the t=0 locked freq
    info['tl_locked_time_mrl'] = info['tl_mrls'][info['locked_freq_ind'], info['tl_locked_time_ind']] # max MRL across time of the t=0 locked freq
    tl_mrl_argmax = np.argmax(info['tl_mrls'], axis=0) # freq index of the max MRL at each time shift
    tl_max_mrls = np.max(info['tl_mrls'], axis=0) # max MRL across freqs at each time shift
    info['tl_bs_ind'] = np.array([np.sum(info['bs_mrls'][tl_mrl_argmax[i], :] >= tl_max_mrls[i]) 
                                  for i in range(len(tl_max_mrls))]) # number of bootstrap MRLs >= the max MRL at each time shift (at that freq)
    info['tl_bs_pval'] = (1 + info['tl_bs_ind']) / (1 + n_bootstraps) # p-value of the max MRL at each time shift
    info['tl_bs_sig'] = info['tl_bs_pval'] < alpha # significance of max MRL at each time shfit (vs. alpha Bonferonni-corrected for the number of freqs)

    # Phase-locking stats for Z-scored MRLs.
    bs_means = np.expand_dims(np.mean(info['bs_mrls'], axis=-1), axis=-1) # mean at each freq
    bs_stds = np.expand_dims(np.std(info['bs_mrls'], axis=-1), axis=-1) # std at each freq
    bs_mrls_z = (info['bs_mrls'] - bs_means) / bs_stds 
    tl_mrls_z = (info['tl_mrls'] - bs_means) / bs_stds # Z-scores after collapsing across channels    
#    info['bs_mrls_z'] = np.mean(bs_mrls_arr_z, axis=0)
#    info['tl_mrls_z'] = np.mean(mrls_arr_z, axis=0) 
    info['mrls_z'] = tl_mrls_z[:, t0] 
    info['locked_freq_ind_z'] = info['mrls_z'].argmax()
    info['locked_mrl_z'] = info['mrls_z'][info['locked_freq_ind_z']]
    info['bs_ind_z'] = np.sum(bs_mrls_z[info['locked_freq_ind_z'], :] >= info['locked_mrl_z'])
    info['bs_pval_z'] = (1 + info['bs_ind_z']) / (1 + n_bootstraps)
    info['sig_z'] = info['bs_pval_z'] < alpha
    info['tl_locked_time_ind_z'] = tl_mrls_z[info['locked_freq_ind_z'], :].argmax()
    info['tl_locked_time_z'] = time_steps_ms[info['tl_locked_time_ind_z']]
    info['tl_locked_time_mrl_z'] = tl_mrls_z[info['locked_freq_ind_z'], info['tl_locked_time_ind_z']]
    tl_mrl_argmax_z = np.argmax(tl_mrls_z, axis=0)
    tl_max_mrls_z = np.max(tl_mrls_z, axis=0)
    info['tl_bs_ind_z'] = np.array([np.sum(bs_mrls_z[tl_mrl_argmax_z[i], :] >= tl_max_mrls_z[i])
                                    for i in range(len(tl_max_mrls_z))])
    info['tl_bs_pval_z'] = (1 + info['tl_bs_ind_z']) / (1 + n_bootstraps)
    info['tl_bs_sig_z'] = info['tl_bs_pval_z'] < alpha
    
    if save_outputs:
        fpath = op.join(output_dir, output_fname.format(subj_sess, unit, lfp_roi, sampling_rate))
        dio.save_pickle(info, fpath, verbose=False)
        
    return info


def generate_bootstrap_shifts(subj_sess,
                              sampling_rate=500,
                              time_win=5,
                              n_bootstraps=5000,
                              save_outputs=True,
                              spike_inds_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/spike_inds', 
                              output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/bootstrap_shifts'):
    """Generate integers to randomly shift spike trains by for bootstrapping.
    
    Bootstrap shifts stay at least time_win secs away (in either direction) from
    the actual data.
    """
    _, n_timepoints = dio.open_pickle(op.join(spike_inds_dir, 'spike_inds-{}Hz-{}-unit0.pkl'.format(sampling_rate, subj_sess)))
        
    if time_win == 0:
        bs_shifts = np.array([int(random.random() * n_timepoints) for _ in range(n_bootstraps)])
    else:
        cut_inds = int(sampling_rate * time_win) + 1
        bs_shifts = np.array([cut_inds + int(random.random() * (n_timepoints - (2*cut_inds-1))) 
                              for _ in range(n_bootstraps)])
    
    if save_outputs:
        dio.save_pickle(bs_shifts, op.join(output_dir, '{}_{}bootstraps_{}Hz_{}timewin.pkl'.format(subj_sess, int(n_bootstraps), int(sampling_rate), int(time_win))), verbose=False)
    
    return bs_shifts
    

def generate_bootstrap_timepoints(subj_sess,
                                  sampling_rate=500,
                                  n_bootstraps=5000,
                                  cut=2500,
                                  save_outputs=True,
                                  spike_inds_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/spike_inds', 
                                  output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/bootstrap_shifts'):
    """Generate random integers of timepoints from the recording session.
    
    Bootstrap shifts stay at least time_win secs away (in either direction) from
    the actual data.
    """
    _, n_timepoints = dio.open_pickle(op.join(spike_inds_dir, 'spike_inds-{}Hz-{}-unit0.pkl'.format(sampling_rate, subj_sess)))
    
    bs_ints = cut + np.array([int(random.random() * (n_timepoints - (2 * cut))) for _ in range(n_bootstraps)])
    
    if save_outputs:
        dio.save_pickle(bs_ints, op.join(output_dir, '{}_{}bootstrap_timepoints.pkl'.format(subj_sess, int(n_bootstraps))), verbose=False)
    
    return bs_ints
    

def shift_spike_inds(spike_inds, n_timepoints, step):
    """Return the time-shifted spike_inds array.
    
    Parameters
    ----------
    spike_inds : np.ndarray
        Array of spike time indices.
    n_timepoints : int
        Number of timepoints in the recording session.
    step : int
        Number of timepoints to shift the spike train by.
    """
    spike_inds_shifted = spike_inds + step
    
    if step < 0:
        roll_by = -len(spike_inds_shifted[spike_inds_shifted<0])
        spike_inds_shifted[spike_inds_shifted<0] = spike_inds_shifted[spike_inds_shifted<0] + n_timepoints
    else:
        roll_by = len(spike_inds_shifted[spike_inds_shifted>=n_timepoints])
        spike_inds_shifted[spike_inds_shifted>=n_timepoints] = spike_inds_shifted[spike_inds_shifted>=n_timepoints] - n_timepoints

    spike_inds_shifted = np.roll(spike_inds_shifted, roll_by)

    return spike_inds_shifted


def save_phase_vectors(subj_sess,
                       chans=None, # list of str channels with indexing starting at '1'
                       sampling_rate=2000,
                       resampling_rate=0,
                       notch_freqs=[60, 120],
                       freqs=np.array([2**((i/2) - 1) for i in range(16)]),
                       morlet_width=5,
                       save_outputs=True,
                       output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/wavelet/phase',
                       overwrite=False,
                       sleep_secs=0):
    """Calculate wavelet phase and save each phase vector as a file.
    
    Returns the number of files saved. Each vector represents wavelet phase over
    time for a given channel and frequency.
    """
    # Wait before executing this function because of stupid disk usage errors.
    if sleep_secs > 0:
        sleep(sleep_secs)
    
    # Get session info.
    subj_df = get_subj_df()
    
    # Load the raw LFP.
    lfp_raw, lfp_preproc = manning_analysis.process_lfp(subj_sess, 
                                                        subj_df=subj_df, 
                                                        sampling_rate=sampling_rate,
                                                        resampling_rate=resampling_rate,
                                                        notch_freqs=notch_freqs)
    del lfp_raw
    
    # Keep all channels or just process a subset of them.
    if chans is None:
        iChans = np.arange(lfp_preproc.shape[0])
    else:
        lfp_preproc = lfp_preproc.sel(channel=chans)
        iChans = np.array([int(chan)-1 for chan in chans])
        
    if resampling_rate > 0:
        sampling_rate = resampling_rate
    
    # Get phase.                                           
    phase = manning_analysis.run_morlet(lfp_preproc, 
                                        freqs=freqs, 
                                        width=morlet_width, 
                                        output=['phase'],
                                        savedir=False)
    del lfp_preproc
    phase = phase.data.astype(np.float32) # freq x chan x time
    
    # Save phase vectors.
    if save_outputs:
        files_saved = 0
        for iiChan, iChan in enumerate(iChans):
            for iFreq in range(phase.shape[0]):
                process_str = 'phase'
                process_str += '-{}'.format(subj_sess)
                process_str += '-iChan{}'.format(iChan)
                process_str += '-iFreq{}'.format(iFreq)
                process_str += '-{}Hz'.format(int(sampling_rate))
                process_str += '-notch' + '_'.join(str(i) for i in notch_freqs) + 'Hz' if notch_freqs else 'nonotch'
                process_str += '-nospikeinterp'
                process_str += '-{}cycles'.format(morlet_width)
                process_str += '-{}log10freqs_{:.1f}_to_{:.1f}Hz'.format(len(freqs), freqs[0], freqs[-1])
                fpath = op.join(output_dir, '{}.pkl'.format(process_str))
                if overwrite or not op.exists(fpath):
                    dat = phase[iFreq, iiChan, :]
                    dio.save_pickle(dat, fpath, verbose=False)
                    files_saved += 1

    return phase
    
# def save_phase_vectors(subj_sess,
#                        sampling_rate=2000,
#                        resampling_rate=0,
#                        notch_freqs=[60, 120],
#                        freqs=np.array([2**((i/2) - 1) for i in range(16)]),
#                        morlet_width=5,
#                        output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/wavelet/phase',
#                        overwrite=False,
#                        sleep_secs=0):
#     """Calculate wavelet phase and save each phase vector as a file.
#     
#     Returns the number of files saved. Each vector represents wavelet phase over
#     time for a given channel and frequency.
#     """
#     # Wait before executing this function because of stupid disk usage errors.
#     if sleep_secs > 0:
#         sleep(sleep_secs)
#     
#     # Get session info.
#     subj_df = get_subj_df()
#     
#     # Load the raw LFP.
#     lfp_raw, lfp_preproc = manning_analysis.process_lfp(subj_sess, 
#                                                         subj_df=subj_df, 
#                                                         sampling_rate=sampling_rate,
#                                                         resampling_rate=resampling_rate,
#                                                         notch_freqs=notch_freqs)
#     del lfp_raw
#     
#     if resampling_rate > 0:
#         sampling_rate = resampling_rate
#     
#     # Get phase.                                           
#     phase = manning_analysis.run_morlet(lfp_preproc, 
#                                         freqs=freqs, 
#                                         width=morlet_width, 
#                                         output=['phase'],
#                                         savedir=False)
#     del lfp_preproc
#     phase = phase.squeeze().data.astype(np.float32)
#     
#     # Save phase vectors.
#     files_saved = 0
#     for iChan in range(phase.shape[1]):
#         for iFreq in range(phase.shape[0]):
#             process_str = 'phase'
#             process_str += '-{}'.format(subj_sess)
#             process_str += '-iChan{}'.format(iChan)
#             process_str += '-iFreq{}'.format(iFreq)
#             process_str += '-{}Hz'.format(int(sampling_rate))
#             process_str += '-notch' + '_'.join(str(i) for i in notch_freqs) + 'Hz' if notch_freqs else 'nonotch'
#             process_str += '-nospikeinterp'
#             process_str += '-{}cycles'.format(morlet_width)
#             process_str += '-{}log10freqs_{:.1f}_to_{:.1f}Hz'.format(len(freqs), freqs[0], freqs[-1])
#             fpath = op.join(output_dir, '{}.pkl'.format(process_str))
#             if overwrite or not op.exists(fpath):
#                 dat = phase[iFreq, iChan, :]
#                 dio.save_pickle(dat, fpath, verbose=False)
#                 files_saved += 1
# 
#     return files_saved
    

def save_power_vectors(subj_sess,
                       chans=None, # list of str channels with indexing starting at '1'
                       sampling_rate=2000,
                       resampling_rate=0,
                       notch_freqs=[60, 120],
                       freqs=np.array([2**((i/2) - 1) for i in range(16)]),
                       morlet_width=5,
                       log_power=True,
                       z_power=True,
                       save_outputs=True,
                       output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/wavelet/power',
                       overwrite=False,
                       sleep_secs=0):
    """Calculate wavelet power and save each power vector as a file.
    
    Returns the number of files saved. Each vector represents wavelet phase over
    time for a given channel and frequency.
    """
    # Wait before executing this function because of stupid disk usage errors.
    if sleep_secs > 0:
        sleep(sleep_secs)
    
    # Get session info.
    subj_df = get_subj_df()
    
    # Load the raw LFP (chan x time)
    lfp_raw, lfp_preproc = manning_analysis.process_lfp(subj_sess, 
                                                        subj_df=subj_df, 
                                                        sampling_rate=sampling_rate,
                                                        resampling_rate=resampling_rate,
                                                        notch_freqs=notch_freqs)
    del lfp_raw
    
    # Keep all channels or just process a subset of them.
    if chans is None:
        iChans = np.arange(lfp_preproc.shape[0])
    else:
        lfp_preproc = lfp_preproc.sel(channel=chans)
        iChans = np.array([int(chan)-1 for chan in chans])
    
    if resampling_rate > 0:
        sampling_rate = resampling_rate
    
    # Get power.                                           
    power = manning_analysis.run_morlet(lfp_preproc, 
                                        freqs=freqs, 
                                        width=morlet_width, 
                                        output=['power'],
                                        log_power=log_power,
                                        z_power=z_power,
                                        savedir=False)
    del lfp_preproc
    power = power.data.astype(np.float32) # freq x chan x time
    
    # Save power vectors.
    if save_outputs:
        files_saved = 0
        for iiChan, iChan in enumerate(iChans):
            for iFreq in range(power.shape[0]):
                process_str = 'power'
                process_str += '-Z' if z_power else ''
                process_str += '-log' if log_power else ''
                process_str += '-{}'.format(subj_sess)
                process_str += '-iChan{}'.format(iChan)
                process_str += '-iFreq{}'.format(iFreq)
                process_str += '-{}Hz'.format(int(sampling_rate))
                process_str += '-notch' + '_'.join(str(i) for i in notch_freqs) + 'Hz' if notch_freqs else 'nonotch'
                process_str += '-nospikeinterp'
                process_str += '-{}cycles'.format(morlet_width)
                process_str += '-{}log10freqs_{:.1f}_to_{:.1f}Hz'.format(len(freqs), freqs[0], freqs[-1])
                fpath = op.join(output_dir, '{}.pkl'.format(process_str))
                if overwrite or not op.exists(fpath):
                    dat = power[iFreq, iiChan, :]
                    dio.save_pickle(dat, fpath, verbose=False)
                    files_saved += 1

    return power
    

def calc_cross_electrode_phase_locking_mrl_morlet(subj_sess,
                                                  freqs=None,
                                                  sampling_rate=2000.0,
                                                  resampling_rate=0,
                                                  interp_spikes=False,
                                                  notch_freqs=[60, 120],
                                                  zscore_lfp=True,
                                                  morlet_width=5,
                                                  hpc_subset=False,
                                                  time_win=4,
                                                  n_bootstraps=1000,
                                                  save_outputs=True,
                                                  output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking'):
    """Calculate phase-locking between each unit and each channel LFP, 
    for a given set of frequency bands.
    
    Returns
    -------
    pl_df : pd.DataFrame
        The cross-electrode phase-locking DataFrame (all unit-to-channel pairs).
    """
    phase_type='morlet'
    if freqs is None:
        freqs = np.logspace(np.log10(0.5), np.log10(16), 16)
    
    # Get session info.
    subj_df = get_subj_df()
    
    # Load spikes.
    spikes, fr_df, _, _ = load_spikes(subj_sess)
    units = fr_df.clus.tolist()
    
    # Load the raw LFP.
    ms_before = 2
    ms_after = 4
    
    lfp_raw, lfp_preproc = manning_analysis.process_lfp(subj_sess, 
                                                        subj_df=subj_df, 
                                                        sampling_rate=sampling_rate,
                                                        resampling_rate=resampling_rate,
                                                        notch_freqs=notch_freqs,
                                                        interpolate=interp_spikes,
                                                        session_spikes=spikes,
                                                        ms_before=ms_before,
                                                        ms_after=ms_after)
    del lfp_raw, spikes
    if resampling_rate > 0:
        resampling_ratio = resampling_rate / sampling_rate
        sampling_rate = resampling_rate
    two_secs = int(sampling_rate * 2)
        
    # Get phase.
    phase = manning_analysis.run_morlet(lfp_preproc, 
                                        freqs=freqs, 
                                        width=morlet_width, 
                                        output=['phase'],
                                        savedir=False, 
                                        verbose=False)
    phase = phase.squeeze().data
    del lfp_preproc
    
    # Load the cross-electrode DataFrame.
    pl_df = dio.open_pickle(op.join('/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                           '{}_cross_electrode_unit_lfp_pairs_df.pkl'.format(subj_sess)))
    
    # Restrict analyses to a subset of pairs.
    if hpc_subset:
        # Remove inter-hemispheric pairs.
        pl_df = pl_df.loc[pl_df.same_hem==True]
        # Remove same-channel pairs.
        pl_df = pl_df.loc[pl_df.same_chan!=True]
        # Remove intra-regional pairs from different microwire bundles.
        pl_df = pl_df.query("(unit_roi2!=lfp_roi2) | ((unit_roi2==lfp_roi2) & (same_hemroi==True))")
        # Remove pairs that aren't intra-regional and where the LFP is extra-hippocampal.
        pl_df = pl_df.query("(lfp_is_hpc==True) | (unit_roi2==lfp_roi2)")
        # Remove units with <250 spikes.
        pl_df = pl_df.loc[pl_df.unit_nspikes>249]
        # Remove units that don't have connections to HPC.
        keep_units = pl_df.query("(lfp_is_hpc==True)").groupby(['subj_sess_unit']).size().index.tolist()
        pl_df = pl_df.query("(subj_sess_unit=={})".format(keep_units))
    else:
        # Remove same-channel pairs.
        pl_df = pl_df.loc[pl_df.same_chan!=True]
        # Remove units with <250 spikes.
        pl_df = pl_df.loc[pl_df.unit_nspikes>249]
        # Remove units that don't have connections to HPC.
        keep_units = pl_df.query("(lfp_is_hpc==True)").groupby(['subj_sess_unit']).size().index.tolist()
        pl_df = pl_df.query("(subj_sess_unit=={})".format(keep_units))

    pl_df.reset_index(inplace=True, drop=True)
    
    # Get phase and phase stats for each frequency.
    mrls = []
    for iunit in range(len(pl_df)):
        unit = pl_df.at[iunit, 'unit']
        lfp_chan_ind = pl_df.at[iunit, 'lfp_chan_ind']
        spike_train = fr_df.at[unit, 'spikes']
        spike_train[:two_secs] = False
        spike_train[-two_secs:] = False
        mrls_ = []
        for ifreq, freq in enumerate(freqs):
            spike_inds = np.where(spike_train)[0]
            if resampling_rate > 0:
                spike_inds = np.round(spike_inds * resampling_ratio).astype(int)
            spike_phases = phase[ifreq, lfp_chan_ind, spike_inds]
            if len(spike_phases) > 0:
                mrls_.append(circstats.circmoment(spike_phases)[1])
            else:
                mrls_.append(np.nan)
        mrls.append(np.array(mrls_))
    pl_df['mrls'] = mrls
    
    # Time lag analysis.
    # Slide from past LFP predicting future spikes to the reverse
    # (10ms steps from -4 to 4secs), getting a MRL for each step.
    steps = np.arange(-time_win*sampling_rate, time_win*sampling_rate+1, sampling_rate*0.01, dtype=int)
    tl_mrls = []
    for iunit in range(len(pl_df)):
        unit = pl_df.at[iunit, 'unit']
        lfp_chan_ind = pl_df.at[iunit, 'lfp_chan_ind']
        spike_train = fr_df.at[unit, 'spikes']
        mrls_arr = []
        for ifreq, freq in enumerate(freqs):
            mrls_arr_ = []
            for step in steps:
                spike_train_shifted = np.roll(spike_train, step)
                spike_train_shifted[:two_secs] = False
                spike_train_shifted[-two_secs:] = False
                spike_inds = np.where(spike_train_shifted)[0]
                if resampling_rate > 0:
                    spike_inds = np.round(spike_inds * resampling_ratio).astype(int)
                spike_phases = phase[ifreq, lfp_chan_ind, spike_inds]
                mrls_arr_.append(circstats.circmoment(spike_phases)[1])
            mrls_arr.append(mrls_arr_)
        tl_mrls.append(np.array(mrls_arr))
    pl_df['tl_mrls'] = tl_mrls
    
    # Bootstrapping - Randomly shift the spike train n_bootstraps times and
    # recalc MRLs. Because a unit could be phase-locked at a short time
    # lag, permutations will shift the spike train by at least 2 secs.
    # IMPORTANTLY, each permutation shifts all spike trains by the same offset. 
    bs_mrls = []
    n_timepoints = len(spike_train)
    bs_offsets = np.array([two_secs+int(random.random() * (n_timepoints-(two_secs+1))) 
                           for _ in range(n_bootstraps)])
    for iunit in range(len(pl_df)):
        unit = pl_df.at[iunit, 'unit']
        lfp_chan_ind = pl_df.at[iunit, 'lfp_chan_ind']
        spike_train = fr_df.at[unit, 'spikes']
        unit_bs_mrls = []
        for ifreq, freq in enumerate(freqs):
            unit_bs_mrls_ = []
            for iperm in range(n_bootstraps):
                spike_train_shifted = np.roll(spike_train, bs_offsets[iperm])
                spike_train_shifted[:two_secs] = False
                spike_train_shifted[-two_secs:] = False
                spike_inds = np.where(spike_train_shifted)[0]
                if resampling_rate > 0:
                    spike_inds = np.round(spike_inds * resampling_ratio).astype(int)
                spike_phases = phase[ifreq, lfp_chan_ind, spike_inds]
                unit_bs_mrls_.append(circstats.circmoment(spike_phases)[1])
            unit_bs_mrls.append(unit_bs_mrls_)
        bs_mrls.append(np.array(unit_bs_mrls))
    pl_df['bs_mrls'] = bs_mrls
    del phase
    
    # Get the mean MRL across unit->LFP pairs for unit->local
    # and unit->HPC connections, for each frequency.
    alpha = 0.05 / len(freqs)
    upl_df = (pl_df.groupby(['subj_sess_unit', 'subj_sess', 
                             'unit_roi', 'lfp_roi', 'unit_roi2', 'lfp_roi2', 'unit_hem', 'lfp_hem', 
                             'same_hem', 'same_hemroi', 'same_roi2',
                             'unit_is_hpc', 'lfp_is_hpc', 'unit_fr', 'unit_nspikes'])
                   .agg({'mrls': lambda x: tuple(np.mean(x)),
                         'tl_mrls': lambda x: tuple(np.mean(x)),
                         'bs_mrls': lambda x: tuple(np.mean(x))})
                   .reset_index())
    upl_df['mrls'] = upl_df.mrls.apply(lambda x: np.array(x))
    upl_df['tl_mrls'] = upl_df.tl_mrls.apply(lambda x: np.array(x))
    upl_df['bs_mrls'] = upl_df.bs_mrls.apply(lambda x: np.array(x))
    upl_df['mrl_argmax'] = upl_df.mrls.apply(np.argmax)
    upl_df['locked_band'] = upl_df.apply(lambda x: freqs[x['mrl_argmax']], axis=1)
    upl_df['locked_mrl'] = upl_df.mrls.apply(np.max)
    upl_df['bs_ind'] = upl_df.apply(lambda x: np.sum(x['bs_mrls'][x['mrl_argmax'], :] >= x['locked_mrl']), axis=1)
    upl_df['bs_pval'] = upl_df.apply(lambda x: (1 + x['bs_ind']) / (1 + n_bootstraps), axis=1)
    upl_df['sig'] = upl_df.bs_pval < alpha
    upl_df['tl_mrl_argmax'] = upl_df.tl_mrls.apply(lambda x: np.argmax(x, axis=0))
    upl_df['tl_locked_mrl'] = upl_df.tl_mrls.apply(lambda x: np.max(x, axis=0))
    upl_df['tl_bs_ind'] = upl_df.apply(lambda x: tuple([np.sum(x['bs_mrls'][x['tl_mrl_argmax'][i], :] >= x['tl_locked_mrl'][i]) 
                                                        for i in range(len(x['tl_locked_mrl']))]), axis=1)
    upl_df['tl_bs_ind'] = upl_df.tl_bs_ind.apply(lambda x: np.array(x))
    upl_df['tl_bs_pval'] = upl_df.tl_bs_ind.apply(lambda x: (1+x) / (1+n_bootstraps))
    upl_df['tl_sig'] = upl_df.tl_bs_pval.apply(lambda x: x < alpha)
    
    # Save the cross-electrode phase-locking DataFrame.
    process_str = ''
    process_str += '{}Hz'.format(int(sampling_rate))
    process_str += '_notch' + '-'.join(str(i) for i in notch_freqs) if notch_freqs else 'nonotch'
    process_str += '_spikeinterp-{}to{}ms'.format(ms_before, ms_after) if interp_spikes else '_nospikeinterp'
    process_str += '_phase-{}{}'.format(phase_type, morlet_width)
    process_str += '_timelag-{}to{}sec-step10ms'.format(time_win, time_win)
    process_str += '_{}bootstraps'.format(n_bootstraps)
    process_str += '_hpc-subset' if hpc_subset else ''
    process_str += '_{}freqs-{:.01f}-to-{:.01f}Hz'.format(len(freqs), freqs[0], freqs[-1])
    if save_outputs:
        fpath = op.join(output_dir, '{}_crosselec_phaselock_byunit_df_{}.pkl'.format(subj_sess, process_str))
        dio.save_pickle(upl_df, fpath)
    
    return upl_df
    

def calc_cross_electrode_phase_locking_mrl_morlet_DEPRECATED(subj_sess,
                                                  freqs=None,
                                                  interp_spikes=True,
                                                  notch_freqs=[60, 120],
                                                  zscore_lfp=True,
                                                  morlet_width=5,
                                                  power_percentile=25,
                                                  osc_length=3,
                                                  mask_type=None,
                                                  hpc_subset=False,
                                                  n_bootstraps=1000,
                                                  save_outputs=True,
                                                  output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking'):
    """Calculate phase-locking between each unit and each channel LFP, 
    for a given set of frequency bands.
    
    Returns
    -------
    pl_df : pd.DataFrame
        The cross-electrode phase-locking DataFrame (all unit-to-channel pairs).
    """
    phase_type='morlet'
    if freqs is None:
        freqs = OrderedDict([('low_delta', 0.5),
                             ('mid_delta', 1.0),
                             ('high_delta', 2.0),
                             ('low_theta', 4.0),
                             ('high_theta', 8.0),
                             ('low_beta', 16.0),
                             ('high_beta', 32.0)])
    freq_names = list(freqs.keys())
    
    # Get session info.
    subj_df = get_subj_df()
    
    # Load spikes.
    spikes, fr_df, clus_to_chan, chan_to_clus = load_spikes(subj_sess)
    units = fr_df.clus.tolist()
    
    # Load the raw LFP.
    ms_before = 2
    ms_after = 4
    lfp_raw, lfp_preproc = manning_analysis.process_lfp(subj_sess, 
                                                        subj_df=subj_df, 
                                                        notch_freqs=notch_freqs,
                                                        interpolate=interp_spikes,
                                                        session_spikes=spikes,
                                                        ms_before=ms_before,
                                                        ms_after=ms_after)
    del lfp_raw, spikes
    
    # Get phase.
    phase = manning_analysis.run_morlet(lfp_preproc, 
                                        freqs=np.array(list(freqs.values())), 
                                        width=morlet_width, 
                                        output=['phase'],
                                        savedir=False, 
                                        verbose=False)
    phase = phase.squeeze().data
    
    # Get phase mask.
    if mask_type is not None: 
        cycle_stats = dio.open_pickle(op.join('/scratch/dscho/phase_precession/data/cycle_stats',
                                              '{}_cycle_stats_{}_{}powpct_{}osclength.pkl'
                                              .format(subj_sess, phase_type, power_percentile, osc_length)))
        mask = OrderedDict()
        for freq_name in freq_names:
            mask[freq_name] = cycles_to_lfp(cycle_stats, category=mask_type, cycle_type=freq_name)
    
    # Load the cross-electrode DataFrame.
    pl_df = dio.open_pickle(op.join('/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                           '{}_cross_electrode_unit_lfp_pairs_df.pkl'.format(subj_sess)))
    
    # Restrict analyses to a subset of pairs.
    if hpc_subset:
        # Remove inter-hemispheric pairs.
        pl_df = pl_df.loc[pl_df.same_hem==True]
        # Remove same-channel pairs.
        pl_df = pl_df.loc[pl_df.same_chan!=True]
        # Remove intra-regional pairs from different microwire bundles.
        pl_df = pl_df.query("(unit_roi2!=lfp_roi2) | ((unit_roi2==lfp_roi2) & (same_hemroi==True))")
        # Remove pairs that aren't intra-regional and where the LFP is extra-hippocampal.
        pl_df = pl_df.query("(lfp_is_hpc==True) | (unit_roi2==lfp_roi2)")
        # Remove units with <250 spikes.
        pl_df = pl_df.loc[pl_df.unit_nspikes>249]
        # Remove units that don't have connections to HPC.
        keep_units = pl_df.query("(lfp_is_hpc==True)").groupby(['subj_sess_unit']).size().index.tolist()
        pl_df = pl_df.query("(subj_sess_unit=={})".format(keep_units))

    pl_df.reset_index(inplace=True, drop=True)
    
    # Get phase and phase stats for each frequency.
    mrls = []
    for iunit in range(len(pl_df)):
        unit = pl_df.at[iunit, 'unit']
        lfp_chan_ind = pl_df.at[iunit, 'lfp_chan_ind']
        spike_train = fr_df.at[unit, 'spikes']
        spike_train[:4000] = False
        spike_train[-4000:] = False
        mrls_ = []
        for ifreq, freq_name in enumerate(freq_names):
            if mask_type is None:
                spike_inds = np.where(spike_train)[0]
            else:
                spike_inds = np.where(mask[freq_name][lfp_chan_ind, :] * spike_train)[0]
            spike_phases = phase[ifreq, lfp_chan_ind, spike_inds]
            if len(spike_phases) > 0:
                mrls_.append(circstats.circmoment(spike_phases)[1])
            else:
                mrls_.append(np.nan)
        mrls.append(np.array(mrls_))
    pl_df['mrls'] = mrls
    
    # Time lag analysis.
    # Slide from past LFP predicting future spikes to the reverse
    # (10ms steps from -2 to 2secs), getting a MRL for each step.
    steps = np.arange(-2*2000, 2*2000+1, 2000*0.01, dtype=int)
    tl_mrls = []
    for iunit in range(len(pl_df)):
        unit = pl_df.at[iunit, 'unit']
        lfp_chan_ind = pl_df.at[iunit, 'lfp_chan_ind']
        spike_train = fr_df.at[unit, 'spikes']
        mrls_arr = []
        for ifreq, freq_name in enumerate(freq_names):
            mrls_arr_ = []
            for step in steps:
                spike_train_shifted = np.roll(spike_train, step)
                spike_train_shifted[:4000] = False
                spike_train_shifted[-4000:] = False
                if mask_type is None:
                    spike_inds = np.where(spike_train_shifted)[0]
                else:
                    spike_inds = np.where(mask[freq_name][lfp_chan_ind, :] * spike_train_shifted)[0]
                spike_phases = phase[ifreq, lfp_chan_ind, spike_inds]
                mrls_arr_.append(circstats.circmoment(spike_phases)[1])
            mrls_arr.append(mrls_arr_)
        tl_mrls.append(np.array(mrls_arr))
    pl_df['tl_mrls'] = tl_mrls
    
    # Bootstrapping - Randomly shift the spike train n_bootstraps times and
    # recalc MRLs. Because a unit could be phase-locked at a short time
    # lag, permutations will shift the spike train by at least 2 secs.
    # IMPORTANTLY, each permutation shifts all spike trains by the same offset. 
    bs_mrls = []
    n_timepoints = len(spike_train)
    bs_offsets = np.array([4000+int(random.random() * (n_timepoints-4001)) 
                           for _ in range(n_bootstraps)])
    for iunit in range(len(pl_df)):
        unit = pl_df.at[iunit, 'unit']
        lfp_chan_ind = pl_df.at[iunit, 'lfp_chan_ind']
        spike_train = fr_df.at[unit, 'spikes']
        unit_bs_mrls = []
        for ifreq, freq_name in enumerate(freq_names):
            unit_bs_mrls_ = []
            for iperm in range(n_bootstraps):
                spike_train_shifted = np.roll(spike_train, bs_offsets[iperm])
                spike_train_shifted[:4000] = False
                spike_train_shifted[-4000:] = False
                if mask_type is None:
                    spike_inds = np.where(spike_train_shifted)[0]
                else:
                    spike_inds = np.where(mask[freq_name][lfp_chan_ind, :] * spike_train_shifted)[0]
                spike_phases = phase[ifreq, lfp_chan_ind, spike_inds]
                unit_bs_mrls_.append(circstats.circmoment(spike_phases)[1])
            unit_bs_mrls.append(unit_bs_mrls_)
        bs_mrls.append(np.array(unit_bs_mrls))
    pl_df['bs_mrls'] = bs_mrls
    del phase
    
    # Get the mean MRL across unit->LFP pairs for unit->local
    # and unit->HPC connections, for each frequency.
    alpha = 0.05 / len(freq_names)
    upl_df = (pl_df.groupby(['subj_sess_unit', 'subj_sess', 'unit_roi2', 'lfp_roi2', 
                             'unit_hem', 'lfp_hem', 'unit_is_hpc', 'lfp_is_hpc', 
                             'unit_fr', 'unit_nspikes'])
                   .agg({'mrls': lambda x: tuple(np.mean(x)),
                         'tl_mrls': lambda x: tuple(np.mean(x)),
                         'bs_mrls': lambda x: tuple(np.mean(x))})
                   .reset_index())
    upl_df['mrls'] = upl_df.mrls.apply(lambda x: np.array(x))
    upl_df['tl_mrls'] = upl_df.tl_mrls.apply(lambda x: np.array(x))
    upl_df['bs_mrls'] = upl_df.bs_mrls.apply(lambda x: np.array(x))
    upl_df['mrl_argmax'] = upl_df.mrls.apply(np.argmax)
    upl_df['locked_band'] = upl_df.apply(lambda x: freq_names[x['mrl_argmax']], axis=1)
    upl_df['locked_mrl'] = upl_df.mrls.apply(np.max)
    upl_df['bs_ind'] = upl_df.apply(lambda x: np.sum(x['bs_mrls'][x['mrl_argmax'], :] >= x['locked_mrl']), axis=1)
    upl_df['bs_pval'] = upl_df.apply(lambda x: (1 + x['bs_ind']) / (1 + n_bootstraps), axis=1)
    upl_df['sig'] = upl_df.bs_pval < alpha
    upl_df['tl_mrl_argmax'] = upl_df.tl_mrls.apply(lambda x: np.argmax(x, axis=0))
    upl_df['tl_locked_mrl'] = upl_df.tl_mrls.apply(lambda x: np.max(x, axis=0))
    upl_df['tl_bs_ind'] = upl_df.apply(lambda x: tuple([np.sum(x['bs_mrls'][x['tl_mrl_argmax'][i], :] >= x['tl_locked_mrl'][i]) 
                                                        for i in range(len(x['tl_locked_mrl']))]), axis=1)
    upl_df['tl_bs_ind'] = upl_df.tl_bs_ind.apply(lambda x: np.array(x))
    upl_df['tl_bs_pval'] = upl_df.tl_bs_ind.apply(lambda x: (1+x) / (1+n_bootstraps))
    upl_df['tl_sig'] = upl_df.tl_bs_pval.apply(lambda x: x < alpha)
    
    # Save the cross-electrode phase-locking DataFrame.
    process_str = ''
    process_str += 'notch' + '-'.join(str(i) for i in notch_freqs) if notch_freqs else 'nonotch'
    process_str += '_spikeinterp-{}to{}ms'.format(ms_before, ms_after) if interp_spikes else '_nospikeinterp'
    process_str += '_phase-{}{}'.format(phase_type, morlet_width)
    process_str += '_{}-{}osc-{}powpct'.format(''.join(mask_type.split('_')), osc_length, power_percentile) if mask_type else '_nomask'
    process_str += '_timelag-2to2sec-step10ms'
    process_str += '_{}bootstraps2'.format(n_bootstraps)
    process_str += '_hpc-subset' if hpc_subset else ''
    process_str += '_morlet-freqs--' + '-'.join([str(val) for val in freqs.values()])
    if save_outputs:
        fpath = op.join(output_dir, '{}_crosselec_phaselock_byunit_df_{}.pkl'.format(subj_sess, process_str))
        dio.save_pickle(upl_df, fpath)
    
    return upl_df


def calc_cross_electrode_phase_locking_mrl_ctx(subj_sess,
                                               bands=None,
                                               interp_spikes=True,
                                               notch_freqs=[60, 120],
                                               zscore_lfp=True,
                                               zscore_power=True,
                                               phase_type='extrema2', # troughs, peaks, or hilbert
                                               power_percentile=25,
                                               osc_length=3,
                                               mask_type=None,
                                               ctx_subset=True,
                                               n_bootstraps=1000,
                                               save_outputs=True,
                                               output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking'):
    """Calculate phase-locking between each unit and each channel LFP, 
    for a given set of frequency bands.
    
    Returns
    -------
    pl_df : pd.DataFrame
        The cross-electrode phase-locking DataFrame (all unit-to-channel pairs).
    """
    if bands is None:
        bands = OrderedDict([('sub_delta', [0.5, 2]),
                             ('delta', [1, 4]),
                             ('low_theta', [2, 8]),
                             ('high_theta', [4, 16]),
                             ('alpha_beta', [8, 32])])
    band_names = list(bands.keys())
    
    # Get session info.
    subj_df = get_subj_df()
    
    # Load spikes.
    spikes, fr_df, clus_to_chan, chan_to_clus = load_spikes(subj_sess)
    units = fr_df.clus.tolist()
    
    # Load the raw LFP.
    ms_before = 2
    ms_after = 4
    lfp_raw, lfp_preproc = manning_analysis.process_lfp(subj_sess, 
                                                        subj_df=subj_df, 
                                                        notch_freqs=notch_freqs,
                                                        interpolate=interp_spikes,
                                                        session_spikes=spikes,
                                                        ms_before=ms_before,
                                                        ms_after=ms_after)
    del lfp_raw, spikes
    
    # Bandpass filter the LFP.
    lfp_filt = spp.filter_lfp_bands(lfp_preproc, 
                                    bands=bands, 
                                    zscore_lfp=zscore_lfp)
    del lfp_preproc
    
    # Use a derivative test to identify troughs and peaks in the filtered 
    # LFP, then linearly interpolate phase between cycles.
    if phase_type == 'hilbert':
        phase_type_ = 'peaks'
    else:
        phase_type_ = phase_type
    
    phase = get_phase_bands(lfp_filt, 
                            bands=bands, 
                            find=phase_type_, 
                            lims=[-np.pi, np.pi])
        
    # Get Hilbert transform phase (instead of linearly interpolated phase).
    if phase_type == 'hilbert':
        for band_name, pass_band in bands.items():
            _, phase_hilbert = spp.get_hilbert(lfp_filt[band_name])
            phase[band_name]['phase'] = phase_hilbert
    del lfp_filt
    
    # Get phase mask.
    if mask_type is not None: 
        cycle_stats = dio.open_pickle(op.join('/scratch/dscho/phase_precession/data/cycle_stats',
                                                   '{}_cycle_stats_{}_{}powpct_{}osclength.pkl'
                                                   .format(subj_sess, phase_type_, power_percentile, osc_length)))
        mask = OrderedDict()
        for band_name, pass_band in bands.items():
            mask[band_name] = cycles_to_lfp(cycle_stats, category=mask_type, cycle_type=band_name)
    
    # Load the cross-electrode DataFrame.
    pl_df = dio.open_pickle(op.join('/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                           '{}_cross_electrode_unit_lfp_pairs_df.pkl'.format(subj_sess)))
    
    # Restrict analyses to a subset of pairs.
    if ctx_subset:
        # Remove units with <250 spikes.
        pl_df = pl_df.loc[pl_df.unit_nspikes>249]
        # Remove units that don't have connections to HPC.
        keep_units = pl_df.query("(lfp_is_hpc==True)").groupby(['subj_sess_unit']).size().index.tolist()
        pl_df = pl_df.query("(subj_sess_unit=={})".format(keep_units))
        # Remove hippocampal units.
        pl_df = pl_df.query("(unit_roi2!='hpc')")
        # Remove inter-hemispheric pairs.
        pl_df = pl_df.loc[pl_df.same_hem==True]
        # Remove same-channel pairs.
        pl_df = pl_df.loc[pl_df.same_chan!=True]
        # Remove pairs with hippocampal LFPs or that are within the same microwire bundle.
        pl_df = pl_df.query("(lfp_is_hpc==False) & (same_hemroi==False)")

    pl_df = pl_df.reset_index(drop=True).copy()
    
    # Get phase and phase stats for each band.
    mrls = []
    for iunit in range(len(pl_df)):
        unit = pl_df.at[iunit, 'unit']
        lfp_chan_ind = pl_df.at[iunit, 'lfp_chan_ind']
        spike_train = fr_df.at[unit, 'spikes']
        spike_train[:4000] = False
        spike_train[-4000:] = False
        mrls_ = []
        for band_name, pass_band in bands.items():
            if mask_type is None:
                spike_inds = np.where(spike_train)[0]
            else:
                spike_inds = np.where(mask[band_name][lfp_chan_ind, :] * spike_train)[0]
            spike_phases = phase[band_name]['phase'][lfp_chan_ind, spike_inds]
            if len(spike_phases) > 0:
                mrls_.append(circstats.circmoment(spike_phases)[1])
            else:
                mrls_.append(np.nan)
        mrls.append(np.array(mrls_))
    pl_df['mrls'] = mrls
    
    # Bootstrapping - Randomly shift the spike train n_bootstraps times and
    # recalc MRLs. Because a unit could be phase-locked at a short time
    # lag, permutations will shift the spike train by at least 2 secs.
    # IMPORTANTLY, each permutation shifts all spike trains by the same offset. 
    bs_mrls = []
    n_timepoints = len(spike_train)
    bs_offsets = np.array([4000+int(random.random() * (n_timepoints-4001)) 
                           for _ in range(n_bootstraps)])
    for iunit in range(len(pl_df)):
        unit = pl_df.at[iunit, 'unit']
        lfp_chan_ind = pl_df.at[iunit, 'lfp_chan_ind']
        spike_train = fr_df.at[unit, 'spikes']
        unit_bs_mrls = []
        for band_name, pass_band in bands.items(): 
            unit_bs_mrls_ = []
            for iperm in range(n_bootstraps):
                spike_train_shifted = np.roll(spike_train, bs_offsets[iperm])
                spike_train_shifted[:4000] = False
                spike_train_shifted[-4000:] = False
                if mask_type is None:
                    spike_inds = np.where(spike_train_shifted)[0]
                else:
                    spike_inds = np.where(mask[band_name][lfp_chan_ind, :] * spike_train_shifted)[0]
                spike_phases = phase[band_name]['phase'][lfp_chan_ind, spike_inds]
                unit_bs_mrls_.append(circstats.circmoment(spike_phases)[1])
            unit_bs_mrls.append(unit_bs_mrls_)
        bs_mrls.append(np.array(unit_bs_mrls))
    pl_df['bs_mrls'] = bs_mrls
    del phase
    
    # Get the mean MRL across unit->LFP pairs for unit->local
    # and unit->HPC connections, for each band.
    alpha = 0.05 / len(bands)
    upl_df = (pl_df.groupby(['subj_sess_unit', 'subj_sess', 'unit_roi', 'lfp_roi', 
                             'unit_hem', 'lfp_hem', 'unit_is_hpc', 'lfp_is_hpc', 
                             'unit_fr', 'unit_nspikes'])
                   .agg({'mrls': lambda x: tuple(np.mean(x)),
                         'bs_mrls': lambda x: tuple(np.mean(x))})
                   .reset_index())
    upl_df['mrls'] = upl_df.mrls.apply(lambda x: np.array(x))
    upl_df['bs_mrls'] = upl_df.bs_mrls.apply(lambda x: np.array(x))
    upl_df['mrl_argmax'] = upl_df.mrls.apply(np.argmax)
    upl_df['locked_band'] = upl_df.apply(lambda x: band_names[x['mrl_argmax']], axis=1)
    upl_df['locked_mrl'] = upl_df.mrls.apply(np.max)
    upl_df['bs_ind'] = upl_df.apply(lambda x: np.sum(x['bs_mrls'][x['mrl_argmax'], :] >= x['locked_mrl']), axis=1)
    upl_df['bs_pval'] = upl_df.apply(lambda x: (1 + x['bs_ind']) / (1 + n_bootstraps), axis=1)
    upl_df['sig'] = upl_df.bs_pval < alpha
    
    # Save the cross-electrode phase-locking DataFrame.
    process_str = ''
    process_str += 'notch' + '-'.join(str(i) for i in notch_freqs) if notch_freqs else 'nonotch'
    process_str += '_spikeinterp-{}to{}ms'.format(ms_before, ms_after) if interp_spikes else '_nospikeinterp'
    process_str += '_phase-{}'.format(phase_type)
    process_str += '_{}-{}osc-{}powpct'.format(''.join(mask_type.split('_')), osc_length, power_percentile) if mask_type else '_nomask'
    process_str += '_{}bootstraps2'.format(n_bootstraps)
    process_str += '_ctx-subset' if ctx_subset else ''
    process_str += '_bands--' + '--'.join(['{}{}-{}'.format(key, val[0], val[1]) for key, val in bands.items()])
    if save_outputs:
        fpath = op.join(output_dir, '{}_crosselec_phaselock_byunit_df_{}.pkl'.format(subj_sess, process_str))
        dio.save_pickle(upl_df, fpath)
    
    return upl_df
    

def calc_phase_locking_fr_power_differences(subj_sess,
                                            bands=None,
                                            interp_spikes=False,
                                            notch_freqs=[60, 120],
                                            zscore_lfp=True,
                                            zscore_power=True,
                                            phase_type='extrema2', 
                                            hpc_subset=True,
                                            n_bootstraps=1000,
                                            save_outputs=True,
                                            output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking'):
    """Calculate phase-locking between each unit and each channel LFP, 
    for a given set of frequency bands.
    
    Returns
    -------
    upl_df : pd.DataFrame
        The cross-electrode phase-locking DataFrame (all unit-to-LFP regional pairs).
    """
    band_names = list(bands.keys())
    
    # Get session info.
    subj_df = get_subj_df()

    # Load spikes.
    spikes, fr_df, clus_to_chan, chan_to_clus = load_spikes(subj_sess)
    units = fr_df.clus.tolist()

    # Load the raw LFP.
    ms_before = 2
    ms_after = 4
    lfp_raw, lfp_preproc = manning_analysis.process_lfp(subj_sess, 
                                                        subj_df=subj_df, 
                                                        notch_freqs=notch_freqs,
                                                        interpolate=interp_spikes,
                                                        session_spikes=spikes,
                                                        ms_before=ms_before,
                                                        ms_after=ms_after)
    del lfp_raw, spikes

    # Bandpass filter the LFP.
    lfp_filt = spp.filter_lfp_bands(lfp_preproc, 
                                    bands=bands, 
                                    zscore_lfp=zscore_lfp)

    # Use a derivative test to identify troughs and peaks in the filtered 
    # LFP, then linearly interpolate phase between cycles.
    phase = get_phase_bands(lfp_filt, 
                            bands=bands, 
                            find=phase_type, 
                            lims=[-np.pi, np.pi])

    # Get Hilbert transform phase (instead of linearly interpolated phase).
    power = OrderedDict()
    for band_name, pass_band in bands.items():
        power[band_name], _ = spp.get_hilbert(lfp_filt[band_name], 
                                              zscore_power=zscore_power)
    del _

    # Load the cross-electrode DataFrame.
    pl_df = dio.open_pickle(op.join('/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                         '{}_cross_electrode_unit_lfp_pairs_df.pkl'.format(subj_sess)))
    upl_df = dio.open_pickle(op.join('/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                          '{}_crosselec_phaselock_byunit_df_notch60-120_nospikeinterp_phase-extrema2_nomask_timelag-2to2sec-step10ms_1000bootstraps2_hpc-subset_bands--sub_delta0.5-2--delta1-4--low_theta2-8--high_theta4-16--alpha_beta8-32.pkl'.format(subj_sess)))
    pl_df = pd.merge(pl_df, upl_df[['subj_sess_unit', 'lfp_is_hpc', 'mrl_argmax', 'locked_band', 'sig']], 
                     how='inner', on=['subj_sess_unit', 'lfp_is_hpc'])
    del upl_df

    # Restrict analyses to a subset of pairs.
    if hpc_subset:
        # Remove inter-hemispheric pairs.
        pl_df = pl_df.loc[pl_df.same_hem==True]
        # Remove same-channel pairs.
        pl_df = pl_df.loc[pl_df.same_chan!=True]
        # Remove intra-regional pairs from different microwire bundles.
        pl_df = pl_df.query("(unit_roi2!=lfp_roi2) | ((unit_roi2==lfp_roi2) & (same_hemroi==True))")
        # Remove pairs that aren't intra-regional and where the LFP is extra-hippocampal.
        pl_df = pl_df.query("(lfp_is_hpc==True) | (unit_roi2==lfp_roi2)")
        # Remove units with <250 spikes.
        pl_df = pl_df.loc[pl_df.unit_nspikes>249]
        # Remove units that don't have connections to HPC.
        keep_units = pl_df.query("(lfp_is_hpc==True)").groupby(['subj_sess_unit']).size().index.tolist()
        pl_df = pl_df.query("(subj_sess_unit=={})".format(keep_units))
        # Remove insignificant units.
        pl_df = pl_df.query("(sig==True)")

    pl_df = pl_df.reset_index(drop=True).copy()

    # Get phase-locking differences between top and bottom quartiles
    # of spike phases, sorted by firing rate, power of the locked band,
    # and power of the remaining bands.
    pl_diffs = []
    bs_pl_diffs = []
    n_timepoints = len(fr_df.at[0, 'spikes'])
    bs_offsets_fr = np.array([4000+int(random.random() * (n_timepoints-4001)) 
                              for _ in range(n_bootstraps)])
    bs_offsets_pow = np.array([4000+int(random.random() * (n_timepoints-4001)) 
                               for _ in range(n_bootstraps)])
    for iunit in range(len(pl_df)):
        pl_diffs.append(get_pl_diffs_pair(iunit, pl_df, fr_df, power, phase, band_names, offset=0))

        # Get bootstrap estimates by shuffling spike trains and recalculating values.
        bs_fr_vec = []
        bs_pow_locked_vec = []
        bs_pow_nonlocked_vec = []
        for iperm in range(n_bootstraps):
            bs_fr_vec.append(get_pl_diffs_pair(iunit, pl_df, fr_df, power, phase, band_names, offset=bs_offsets_fr[iperm])['fr'])
            output = get_pl_diffs_pair(iunit, pl_df, fr_df, power, phase, band_names, offset=bs_offsets_pow[iperm])
            bs_pow_locked_vec.append(output['pow_locked'])
            bs_pow_nonlocked_vec.append(output['pow_nonlocked'])
        bs_pl_diffs.append(OrderedDict([('fr', np.array(bs_fr_vec)),
                                        ('pow_locked', np.array(bs_pow_locked_vec)),
                                        ('pow_nonlocked', np.array(bs_pow_nonlocked_vec))]))

    # Append results to the phase-locking DataFrame.
    pl_df['pl_diffs_fr'] = [x['fr'] for x in pl_diffs]
    pl_df['pl_diffs_pow_locked'] = [x['pow_locked'] for x in pl_diffs]
    pl_df['pl_diffs_pow_nonlocked'] = [x['pow_nonlocked'] for x in pl_diffs]
    pl_df['bs_pl_diffs_fr'] = [x['fr'] for x in bs_pl_diffs]
    pl_df['bs_pl_diffs_pow_locked'] = [x['pow_locked'] for x in bs_pl_diffs]
    pl_df['bs_pl_diffs_pow_nonlocked'] = [x['pow_nonlocked'] for x in bs_pl_diffs]

    # Collapse the DataFrame into unit->LFP pairs for unit->local and unit->HPC 
    # connections, and determine significance vs. bootstrap estimates.
    alpha = 0.05
    upl_df = (pl_df.groupby(['subj_sess_unit', 'subj_sess', 'unit_roi2', 'lfp_roi2', 
                             'unit_hem', 'lfp_hem', 'unit_is_hpc', 'lfp_is_hpc', 
                             'unit_fr', 'unit_nspikes'])
                   .agg({'pl_diffs_fr': np.mean,
                         'pl_diffs_pow_locked': np.mean,
                         'pl_diffs_pow_nonlocked': np.mean,
                         'bs_pl_diffs_fr': lambda x: tuple(np.mean(x)),
                         'bs_pl_diffs_pow_locked': lambda x: tuple(np.mean(x)),
                         'bs_pl_diffs_pow_nonlocked': lambda x: tuple(np.mean(x))})
                   .reset_index())
    upl_df['pl_diffs_fr_sign'] = upl_df.pl_diffs_fr.apply(np.sign)
    upl_df['pl_diffs_pow_locked_sign'] = upl_df.pl_diffs_pow_locked.apply(np.sign)
    upl_df['pl_diffs_pow_nonlocked_sign'] = upl_df.pl_diffs_pow_nonlocked.apply(np.sign)
    upl_df['bs_pl_diffs_fr'] = upl_df.bs_pl_diffs_fr.apply(lambda x: np.array(x))
    upl_df['bs_pl_diffs_pow_locked'] = upl_df.bs_pl_diffs_pow_locked.apply(lambda x: np.array(x))
    upl_df['bs_pl_diffs_pow_nonlocked'] = upl_df.bs_pl_diffs_pow_nonlocked.apply(lambda x: np.array(x))
    upl_df['pl_diffs_fr_pval'] = upl_df.apply(lambda x: (1 + np.sum(abs(x['bs_pl_diffs_fr']) >= 
                                                                    abs(x['pl_diffs_fr']))) / (1 + n_bootstraps), axis=1)
    upl_df['pl_diffs_pow_locked_pval'] = upl_df.apply(lambda x: (1 + np.sum(abs(x['bs_pl_diffs_pow_locked']) >= 
                                                                            abs(x['pl_diffs_pow_locked']))) / (1 + n_bootstraps), axis=1)
    upl_df['pl_diffs_pow_nonlocked_pval'] = upl_df.apply(lambda x: (1 + np.sum(abs(x['bs_pl_diffs_pow_nonlocked']) >= 
                                                                               abs(x['pl_diffs_pow_nonlocked']))) / (1 + n_bootstraps), axis=1)
    upl_df['pl_diffs_fr_sig'] = upl_df.pl_diffs_fr_pval < alpha
    upl_df['pl_diffs_pow_locked_sig'] = upl_df.pl_diffs_pow_locked_pval < alpha
    upl_df['pl_diffs_pow_nonlocked_sig'] = upl_df.pl_diffs_pow_nonlocked_pval < alpha
    upl_df = upl_df.loc[:, [col for col in upl_df.columns if col[:3] != 'bs_']]

    # Save the cross-electrode phase-locking DataFrame.
    process_str = ''
    process_str += 'notch' + '-'.join(str(i) for i in notch_freqs) if notch_freqs else 'nonotch'
    process_str += '_spikeinterp-{}to{}ms'.format(ms_before, ms_after) if interp_spikes else '_nospikeinterp'
    process_str += '_phase-{}'.format(phase_type)
    process_str += '_nomask'
    process_str += '_{}bootstraps2'.format(n_bootstraps)
    process_str += '_hpc-subset' if hpc_subset else ''
    process_str += '_bands--' + '--'.join(['{}{}-{}'.format(key, val[0], val[1]) for key, val in bands.items()])
    if save_outputs:
        fpath = op.join(output_dir, '{}_crosselec_phaselock_byunit_df_fr_power_diffs_{}.pkl'.format(subj_sess, process_str))
        dio.save_pickle(upl_df, fpath)

    return upl_df


def get_pl_diffs_pair(iunit, pl_df, fr_df, power, phase, band_names, offset=0):
    """Return top vs. bottom quartile differences in phase-locking
    strength (MRL) for spike phases in the phase-locked band that 
    are sorted by 1) firing rate, 2) power in the phase-locked band, 
    and 3) power in the remaining bands.
    """
    unit = pl_df.at[iunit, 'unit']
    lfp_chan_ind = pl_df.at[iunit, 'lfp_chan_ind']
    locked_band = pl_df.at[iunit, 'locked_band']
    fr = fr_df.at[unit, 'fr']
    spike_train = fr_df.at[unit, 'spikes']
    if offset:
        spike_train = np.roll(spike_train, offset)
    spike_inds = np.where(spike_train)[0]
    qtl = int(len(spike_inds) / 4)

    # Sort spikes by 1) firing rate of the neuron, and 2) power in each band. 
    xsort = OrderedDict()
    xsort['fr'] = fr[spike_inds].argsort()
    for band_name in band_names:
        xsort[band_name] = power[band_name][lfp_chan_ind, spike_inds].argsort()

    # Get spike phases for the phase-locked band, sorted by 1) firing rate, and 2) power in each band.
    spike_phases = OrderedDict()
    spike_phases_ = phase[locked_band]['phase'][lfp_chan_ind, spike_inds]
    spike_phases['fr'] = spike_phases_[xsort['fr']]
    for band_name in band_names:
        spike_phases['pow_' + band_name] = spike_phases_[xsort[band_name]]

    # Get the difference in MRL for spike phases in the top vs. bottom quartile of 1) firing rate,
    # 2) power in the locked band, and 3) power in all remaining bands (mean MRL difference).
    pl_diffs = OrderedDict()
    pl_diffs['fr'] = (circstats.circmoment(spike_phases['fr'][-qtl:])[1] - 
                      circstats.circmoment(spike_phases['fr'][:qtl])[1])
    pl_diffs['pow_locked'] = (circstats.circmoment(spike_phases['pow_' + locked_band][-qtl:])[1] - 
                              circstats.circmoment(spike_phases['pow_' + locked_band][:qtl])[1])
    nonlocked_bands = [x for x in band_names if x not in [locked_band]]
    nonlocked_pl_diffs = []
    for band_name in nonlocked_bands:
        nonlocked_pl_diffs.append((circstats.circmoment(spike_phases['pow_' + band_name][-qtl:])[1] - 
                                   circstats.circmoment(spike_phases['pow_' + band_name][:qtl])[1]))
    pl_diffs['pow_nonlocked'] = np.mean(nonlocked_pl_diffs)
    
    return pl_diffs
    

def calc_phase_locking_fr_power_bins(subj_sess,
                                     bands=None,
                                     interp_spikes=False,
                                     notch_freqs=[60, 120],
                                     zscore_lfp=True,
                                     zscore_power=True,
                                     phase_type='extrema2', 
                                     hpc_subset=True,
                                     n_bins=4,
                                     save_outputs=True,
                                     output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking'):
    """Calculate phase-locking between each unit and each channel LFP, 
    for a given set of frequency bands.
    
    Returns
    -------
    upl_df : pd.DataFrame
        The cross-electrode phase-locking DataFrame (all unit-to-LFP regional pairs).
    """
    band_names = list(bands.keys())
    
    # Get session info.
    subj_df = get_subj_df()

    # Load spikes.
    spikes, fr_df, clus_to_chan, chan_to_clus = load_spikes(subj_sess)
    units = fr_df.clus.tolist()

    # Load the raw LFP.
    ms_before = 2
    ms_after = 4
    lfp_raw, lfp_preproc = manning_analysis.process_lfp(subj_sess, 
                                                        subj_df=subj_df, 
                                                        notch_freqs=notch_freqs,
                                                        interpolate=interp_spikes,
                                                        session_spikes=spikes,
                                                        ms_before=ms_before,
                                                        ms_after=ms_after)
    del lfp_raw, spikes

    # Bandpass filter the LFP.
    lfp_filt = spp.filter_lfp_bands(lfp_preproc, 
                                    bands=bands, 
                                    zscore_lfp=zscore_lfp)

    # Use a derivative test to identify troughs and peaks in the filtered 
    # LFP, then linearly interpolate phase between cycles.
    phase = get_phase_bands(lfp_filt, 
                            bands=bands, 
                            find=phase_type, 
                            lims=[-np.pi, np.pi])

    # Get Hilbert transform phase (instead of linearly interpolated phase).
    power = OrderedDict()
    for band_name, pass_band in bands.items():
        power[band_name], _ = spp.get_hilbert(lfp_filt[band_name], 
                                              zscore_power=zscore_power)
    del _

    # Load the cross-electrode DataFrame.
    pl_df = dio.open_pickle(op.join('/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                         '{}_cross_electrode_unit_lfp_pairs_df.pkl'.format(subj_sess)))
    upl_df = dio.open_pickle(op.join('/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                          '{}_crosselec_phaselock_byunit_df_notch60-120_nospikeinterp_phase-extrema2_nomask_timelag-2to2sec-step10ms_1000bootstraps2_hpc-subset_bands--sub_delta0.5-2--delta1-4--low_theta2-8--high_theta4-16--alpha_beta8-32.pkl'.format(subj_sess)))
    pl_df = pd.merge(pl_df, upl_df[['subj_sess_unit', 'lfp_is_hpc', 'mrl_argmax', 'locked_band', 'sig']], 
                     how='inner', on=['subj_sess_unit', 'lfp_is_hpc'])
    del upl_df

    # Restrict analyses to a subset of pairs.
    if hpc_subset:
        # Remove inter-hemispheric pairs.
        pl_df = pl_df.loc[pl_df.same_hem==True]
        # Remove same-channel pairs.
        pl_df = pl_df.loc[pl_df.same_chan!=True]
        # Remove intra-regional pairs from different microwire bundles.
        pl_df = pl_df.query("(unit_roi2!=lfp_roi2) | ((unit_roi2==lfp_roi2) & (same_hemroi==True))")
        # Remove pairs that aren't intra-regional and where the LFP is extra-hippocampal.
        pl_df = pl_df.query("(lfp_is_hpc==True) | (unit_roi2==lfp_roi2)")
        # Remove units with <250 spikes.
        pl_df = pl_df.loc[pl_df.unit_nspikes>249]
        # Remove units that don't have connections to HPC.
        keep_units = pl_df.query("(lfp_is_hpc==True)").groupby(['subj_sess_unit']).size().index.tolist()
        pl_df = pl_df.query("(subj_sess_unit=={})".format(keep_units))
        # Remove insignificant units.
        pl_df = pl_df.query("(sig==True)")

    pl_df = pl_df.reset_index(drop=True).copy()

    # Get phase-locking differences between top and bottom quartiles
    # of spike phases, sorted by firing rate, power of the locked band,
    # and power of the remaining bands.
    pl_bins = []
    for iunit in range(len(pl_df)):
        pl_bins.append(get_pl_bins_pair(iunit, pl_df, fr_df, power, phase, band_names, offset=0, n_bins=n_bins))

    # Append results to the phase-locking DataFrame.
    pl_df['pl_bins_fr'] = [x['fr'] for x in pl_bins]
    pl_df['pl_bins_pow_locked'] = [x['pow_locked'] for x in pl_bins]
    pl_df['pl_bins_pow_nonlocked'] = [x['pow_nonlocked'] for x in pl_bins]

    # Collapse the DataFrame into unit->LFP pairs for unit->local and unit->HPC 
    # connections, and determine significance vs. bootstrap estimates.
    upl_df = (pl_df.groupby(['subj_sess_unit', 'subj_sess', 'unit_roi2', 'lfp_roi2', 
                             'unit_hem', 'lfp_hem', 'unit_is_hpc', 'lfp_is_hpc', 
                             'unit_fr', 'unit_nspikes'])
                   .agg({'pl_bins_fr': lambda x: tuple(np.mean(x)),
                         'pl_bins_pow_locked': lambda x: tuple(np.mean(x)),
                         'pl_bins_pow_nonlocked': lambda x: tuple(np.mean(x))})
                   .reset_index())
    upl_df['pl_bins_fr'] = upl_df.pl_bins_fr.apply(lambda x: np.array(x))
    upl_df['pl_bins_pow_locked'] = upl_df.pl_bins_pow_locked.apply(lambda x: np.array(x))
    upl_df['pl_bins_pow_nonlocked'] = upl_df.pl_bins_pow_nonlocked.apply(lambda x: np.array(x))

    # Save the cross-electrode phase-locking DataFrame.
    process_str = ''
    process_str += 'notch' + '-'.join(str(i) for i in notch_freqs) if notch_freqs else 'nonotch'
    process_str += '_spikeinterp-{}to{}ms'.format(ms_before, ms_after) if interp_spikes else '_nospikeinterp'
    process_str += '_phase-{}'.format(phase_type)
    process_str += '_nomask'
    process_str += '_{}bins'.format(n_bins)
    process_str += '_hpc-subset' if hpc_subset else ''
    process_str += '_bands--' + '--'.join(['{}{}-{}'.format(key, val[0], val[1]) for key, val in bands.items()])
    if save_outputs:
        fpath = op.join(output_dir, '{}_crosselec_phaselock_byunit_df_fr_power_bins_{}.pkl'.format(subj_sess, process_str))
        dio.save_pickle(upl_df, fpath)

    return upl_df

    
def get_pl_bins_pair(iunit, pl_df, fr_df, power, phase, band_names, offset=0, n_bins=4):
    """Return phase-locking strength (MRL) at each bin for spike phases
    in the phase-locked band that are sorted by 1) firing rate, 2) power 
    in the phase-locked band, and 3) power in the remaining bands.
    """
    unit = pl_df.at[iunit, 'unit']
    lfp_chan_ind = pl_df.at[iunit, 'lfp_chan_ind']
    locked_band = pl_df.at[iunit, 'locked_band']
    fr = fr_df.at[unit, 'fr']
    spike_train = fr_df.at[unit, 'spikes']
    if offset:
        spike_train = np.roll(spike_train, offset)
    spike_inds = np.where(spike_train)[0]
    discard = len(spike_inds) % n_bins
    start, stop = floor(discard/2), ceil(discard/2)
    if stop > 0:
        spike_inds = spike_inds[start:-stop]
    
    # Sort spikes by 1) firing rate of the neuron, and 2) power in each band. 
    xsort = OrderedDict()
    xsort['fr'] = fr[spike_inds].argsort()
    for band_name in band_names:
        xsort[band_name] = power[band_name][lfp_chan_ind, spike_inds].argsort()

    # Get spike phases for the phase-locked band, sorted by 1) firing rate, and 2) power in each band.
    spike_phases = OrderedDict()
    spike_phases_ = phase[locked_band]['phase'][lfp_chan_ind, spike_inds]
    spike_phases['fr'] = spike_phases_[xsort['fr']]
    for band_name in band_names:
        spike_phases['pow_' + band_name] = spike_phases_[xsort[band_name]]

    # Get the difference in MRL for spike phases in the top vs. bottom quartile of 1) firing rate,
    # 2) power in the locked band, and 3) power in all remaining bands (mean MRL difference).
    pl_bins = OrderedDict()
    pl_bins['fr'] = np.array([circstats.circmoment(x)[1] for x in np.split(spike_phases['fr'], n_bins)])
    pl_bins['pow_locked'] = np.array([circstats.circmoment(x)[1] for x in np.split(spike_phases['pow_' + locked_band], n_bins)])
    nonlocked_bands = [x for x in band_names if x not in [locked_band]]
    nonlocked_pl_bins = []
    for band_name in nonlocked_bands:
        nonlocked_pl_bins.append(np.array([circstats.circmoment(x)[1] for x in np.split(spike_phases['pow_' + band_name], n_bins)]))
    pl_bins['pow_nonlocked'] = np.mean(np.array(nonlocked_pl_bins), axis=0)
    
    return pl_bins

    
def calc_cross_electrode_phase_locking_mrl(subj_sess,
                                           bands=None,
                                           interp_spikes=True,
                                           notch_freqs=[60, 120],
                                           zscore_lfp=True,
                                           zscore_power=True,
                                           phase_type='peaks', # troughs, peaks, or hilbert
                                           power_percentile=25,
                                           osc_length=3,
                                           mask_type=None,
                                           hpc_subset=False,
                                           n_bootstraps=1000,
                                           save_outputs=True,
                                           output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking'):
    """Calculate phase-locking between each unit and each channel LFP, 
    for a given set of frequency bands.
    
    Returns
    -------
    pl_df : pd.DataFrame
        The cross-electrode phase-locking DataFrame (all unit-to-channel pairs).
    """
    if bands is None:
        bands = OrderedDict([('sub_delta', [0.5, 2]),
                             ('delta', [1, 4]),
                             ('low_theta', [2, 8]),
                             ('high_theta', [4, 16]),
                             ('alpha_beta', [8, 32])])
    band_names = list(bands.keys())
    
    # Get session info.
    subj_df = get_subj_df()
    
    # Load spikes.
    spikes, fr_df, clus_to_chan, chan_to_clus = load_spikes(subj_sess)
    units = fr_df.clus.tolist()
    
    # Load the raw LFP.
    ms_before = 2
    ms_after = 4
    lfp_raw, lfp_preproc = manning_analysis.process_lfp(subj_sess, 
                                                        subj_df=subj_df, 
                                                        notch_freqs=notch_freqs,
                                                        interpolate=interp_spikes,
                                                        session_spikes=spikes,
                                                        ms_before=ms_before,
                                                        ms_after=ms_after)
    del lfp_raw, spikes
    
    # Bandpass filter the LFP.
    lfp_filt = spp.filter_lfp_bands(lfp_preproc, 
                                    bands=bands, 
                                    zscore_lfp=zscore_lfp)
    del lfp_preproc
    
    # Use a derivative test to identify troughs and peaks in the filtered 
    # LFP, then linearly interpolate phase between cycles.
    if phase_type == 'hilbert':
        phase_type_ = 'peaks'
    else:
        phase_type_ = phase_type
    
    phase = get_phase_bands(lfp_filt, 
                            bands=bands, 
                            find=phase_type_, 
                            lims=[-np.pi, np.pi])
        
    # Get Hilbert transform phase (instead of linearly interpolated phase).
    if phase_type == 'hilbert':
        for band_name, pass_band in bands.items():
            _, phase_hilbert = spp.get_hilbert(lfp_filt[band_name])
            phase[band_name]['phase'] = phase_hilbert
    del lfp_filt
    
    # Get phase mask.
    if mask_type is not None: 
        cycle_stats = dio.open_pickle(op.join('/scratch/dscho/phase_precession/data/cycle_stats',
                                                   '{}_cycle_stats_{}_{}powpct_{}osclength.pkl'
                                                   .format(subj_sess, phase_type_, power_percentile, osc_length)))
        mask = OrderedDict()
        for band_name, pass_band in bands.items():
            mask[band_name] = cycles_to_lfp(cycle_stats, category=mask_type, cycle_type=band_name)
    
    # Load the cross-electrode DataFrame.
    pl_df = dio.open_pickle(op.join('/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                           '{}_cross_electrode_unit_lfp_pairs_df.pkl'.format(subj_sess)))
    
    # Restrict analyses to a subset of pairs.
    if hpc_subset:
        # Remove inter-hemispheric pairs.
        pl_df = pl_df.loc[pl_df.same_hem==True]
        # Remove same-channel pairs.
        pl_df = pl_df.loc[pl_df.same_chan!=True]
        # Remove intra-regional pairs from different microwire bundles.
        pl_df = pl_df.query("(unit_roi2!=lfp_roi2) | ((unit_roi2==lfp_roi2) & (same_hemroi==True))")
        # Remove pairs that aren't intra-regional and where the LFP is extra-hippocampal.
        pl_df = pl_df.query("(lfp_is_hpc==True) | (unit_roi2==lfp_roi2)")
        # Remove units with <250 spikes.
        pl_df = pl_df.loc[pl_df.unit_nspikes>249]
        # Remove units that don't have connections to HPC.
        keep_units = pl_df.query("(lfp_is_hpc==True)").groupby(['subj_sess_unit']).size().index.tolist()
        pl_df = pl_df.query("(subj_sess_unit=={})".format(keep_units))

    pl_df.reset_index(inplace=True, drop=True)
    
    # Get phase and phase stats for each band.
    mrls = []
    for iunit in range(len(pl_df)):
        unit = pl_df.at[iunit, 'unit']
        lfp_chan_ind = pl_df.at[iunit, 'lfp_chan_ind']
        spike_train = fr_df.at[unit, 'spikes']
        spike_train[:4000] = False
        spike_train[-4000:] = False
        mrls_ = []
        for band_name, pass_band in bands.items():
            if mask_type is None:
                spike_inds = np.where(spike_train)[0]
            else:
                spike_inds = np.where(mask[band_name][lfp_chan_ind, :] * spike_train)[0]
            spike_phases = phase[band_name]['phase'][lfp_chan_ind, spike_inds]
            if len(spike_phases) > 0:
                mrls_.append(circstats.circmoment(spike_phases)[1])
            else:
                mrls_.append(np.nan)
        mrls.append(np.array(mrls_))
    pl_df['mrls'] = mrls
    
    # Time lag analysis.
    # Slide from past LFP predicting future spikes to the reverse
    # (10ms steps from -2 to 2secs), getting a MRL for each step.
    steps = np.arange(-2*2000, 2*2000+1, 2000*0.01, dtype=int)
    tl_mrls = []
    for iunit in range(len(pl_df)):
        unit = pl_df.at[iunit, 'unit']
        lfp_chan_ind = pl_df.at[iunit, 'lfp_chan_ind']
        spike_train = fr_df.at[unit, 'spikes']
        mrls_arr = []
        for band_name, pass_band in bands.items():
            mrls_arr_ = []
            for step in steps:
                spike_train_shifted = np.roll(spike_train, step)
                spike_train_shifted[:4000] = False
                spike_train_shifted[-4000:] = False
                if mask_type is None:
                    spike_inds = np.where(spike_train_shifted)[0]
                else:
                    spike_inds = np.where(mask[band_name][lfp_chan_ind, :] * spike_train_shifted)[0]
                spike_phases = phase[band_name]['phase'][lfp_chan_ind, spike_inds]
                mrls_arr_.append(circstats.circmoment(spike_phases)[1])
            mrls_arr.append(mrls_arr_)
        tl_mrls.append(np.array(mrls_arr))
    pl_df['tl_mrls'] = tl_mrls
    
    # Bootstrapping - Randomly shift the spike train n_bootstraps times and
    # recalc MRLs. Because a unit could be phase-locked at a short time
    # lag, permutations will shift the spike train by at least 2 secs.
    # IMPORTANTLY, each permutation shifts all spike trains by the same offset. 
    bs_mrls = []
    n_timepoints = len(spike_train)
    bs_offsets = np.array([4000+int(random.random() * (n_timepoints-4001)) 
                           for _ in range(n_bootstraps)])
    for iunit in range(len(pl_df)):
        unit = pl_df.at[iunit, 'unit']
        lfp_chan_ind = pl_df.at[iunit, 'lfp_chan_ind']
        spike_train = fr_df.at[unit, 'spikes']
        unit_bs_mrls = []
        for band_name, pass_band in bands.items(): 
            unit_bs_mrls_ = []
            for iperm in range(n_bootstraps):
                spike_train_shifted = np.roll(spike_train, bs_offsets[iperm])
                spike_train_shifted[:4000] = False
                spike_train_shifted[-4000:] = False
                if mask_type is None:
                    spike_inds = np.where(spike_train_shifted)[0]
                else:
                    spike_inds = np.where(mask[band_name][lfp_chan_ind, :] * spike_train_shifted)[0]
                spike_phases = phase[band_name]['phase'][lfp_chan_ind, spike_inds]
                unit_bs_mrls_.append(circstats.circmoment(spike_phases)[1])
            unit_bs_mrls.append(unit_bs_mrls_)
        bs_mrls.append(np.array(unit_bs_mrls))
    pl_df['bs_mrls'] = bs_mrls
    del phase
    
    # Get the mean MRL across unit->LFP pairs for unit->local
    # and unit->HPC connections, for each band.
    alpha = 0.05 / len(bands)
    upl_df = (pl_df.groupby(['subj_sess_unit', 'subj_sess', 'unit_roi2', 'lfp_roi2', 
                             'unit_hem', 'lfp_hem', 'unit_is_hpc', 'lfp_is_hpc', 
                             'unit_fr', 'unit_nspikes'])
                   .agg({'mrls': lambda x: tuple(np.mean(x)),
                         'tl_mrls': lambda x: tuple(np.mean(x)),
                         'bs_mrls': lambda x: tuple(np.mean(x))})
                   .reset_index())
    upl_df['mrls'] = upl_df.mrls.apply(lambda x: np.array(x))
    upl_df['tl_mrls'] = upl_df.tl_mrls.apply(lambda x: np.array(x))
    upl_df['bs_mrls'] = upl_df.bs_mrls.apply(lambda x: np.array(x))
    upl_df['mrl_argmax'] = upl_df.mrls.apply(np.argmax)
    upl_df['locked_band'] = upl_df.apply(lambda x: band_names[x['mrl_argmax']], axis=1)
    upl_df['locked_mrl'] = upl_df.mrls.apply(np.max)
    upl_df['bs_ind'] = upl_df.apply(lambda x: np.sum(x['bs_mrls'][x['mrl_argmax'], :] >= x['locked_mrl']), axis=1)
    upl_df['bs_pval'] = upl_df.apply(lambda x: (1 + x['bs_ind']) / (1 + n_bootstraps), axis=1)
    upl_df['sig'] = upl_df.bs_pval < alpha
    upl_df['tl_mrl_argmax'] = upl_df.tl_mrls.apply(lambda x: np.argmax(x, axis=0))
    upl_df['tl_locked_mrl'] = upl_df.tl_mrls.apply(lambda x: np.max(x, axis=0))
    upl_df['tl_bs_ind'] = upl_df.apply(lambda x: tuple([np.sum(x['bs_mrls'][x['tl_mrl_argmax'][i], :] >= x['tl_locked_mrl'][i]) 
                                                        for i in range(len(x['tl_locked_mrl']))]), axis=1)
    upl_df['tl_bs_ind'] = upl_df.tl_bs_ind.apply(lambda x: np.array(x))
    upl_df['tl_bs_pval'] = upl_df.tl_bs_ind.apply(lambda x: (1+x) / (1+n_bootstraps))
    upl_df['tl_sig'] = upl_df.tl_bs_pval.apply(lambda x: x < alpha)
    
    # Save the cross-electrode phase-locking DataFrame.
    process_str = ''
    process_str += 'notch' + '-'.join(str(i) for i in notch_freqs) if notch_freqs else 'nonotch'
    process_str += '_spikeinterp-{}to{}ms'.format(ms_before, ms_after) if interp_spikes else '_nospikeinterp'
    process_str += '_phase-{}'.format(phase_type)
    process_str += '_{}-{}osc-{}powpct'.format(''.join(mask_type.split('_')), osc_length, power_percentile) if mask_type else '_nomask'
    process_str += '_timelag-2to2sec-step10ms'
    process_str += '_{}bootstraps2'.format(n_bootstraps)
    process_str += '_hpc-subset' if hpc_subset else ''
    process_str += '_bands--' + '--'.join(['{}{}-{}'.format(key, val[0], val[1]) for key, val in bands.items()])
    if save_outputs:
        fpath = op.join(output_dir, '{}_crosselec_phaselock_byunit_df_{}.pkl'.format(subj_sess, process_str))
        dio.save_pickle(upl_df, fpath)
    
    return upl_df


def calc_cross_electrode_phase_locking_mrl_timelag_DEPRECATED(subj_sess,
                                                   bands=None,
                                                   interp_spikes=True,
                                                   notch_freqs=[60, 120],
                                                   zscore_lfp=True,
                                                   zscore_power=True,
                                                   phase_type='peaks', # troughs, peaks, or hilbert
                                                   power_percentile=25,
                                                   osc_length=3,
                                                   mask_type=None,
                                                   hpc_subset=False,
                                                   save_outputs=True,
                                                   output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/time_lag'):
    """Calculate phase-locking between each unit and each channel LFP, 
    for a given set of frequency bands.
    
    Returns
    -------
    pl_df : pd.DataFrame
        The cross-electrode phase-locking DataFrame (all unit-to-channel pairs).
    """
    if bands is None:
        bands = OrderedDict([('sub_delta', [0.5, 2]),
                             ('delta', [1, 4]),
                             ('low_theta', [2, 8]),
                             ('high_theta', [4, 16]),
                             ('alpha_beta', [8, 32])])
    band_names = list(bands.keys())
    hpc_rois = ['LAH', 'LMH', 'LPH', 'RAH', 'RMH', 'RPH']
    
    # Get session info.
    subj_df = get_subj_df()
    
    # Load spikes.
    spikes, fr_df, clus_to_chan, chan_to_clus = load_spikes(subj_sess)
    units = fr_df.clus.tolist()
    
    # Load the raw LFP.
    ms_before = 2
    ms_after = 4
    lfp_raw, lfp_preproc = manning_analysis.process_lfp(subj_sess, 
                                                        subj_df=subj_df, 
                                                        notch_freqs=notch_freqs,
                                                        interpolate=interp_spikes,
                                                        session_spikes=spikes,
                                                        ms_before=ms_before,
                                                        ms_after=ms_after)
    del lfp_raw, spikes
    sampling_rate = lfp_preproc.samplerate.data.tolist()
    
    # Bandpass filter the LFP.
    lfp_filt = spp.filter_lfp_bands(lfp_preproc, 
                                    bands=bands, 
                                    zscore_lfp=zscore_lfp)
    del lfp_preproc
    
    # Use a derivative test to identify troughs and peaks in the filtered 
    # LFP, then linearly interpolate phase between cycles.
    if phase_type == 'hilbert':
        phase_type_ = 'peaks'
    else:
        phase_type_ = phase_type
    
    phase = get_phase_bands(lfp_filt, 
                            bands=bands, 
                            find=phase_type_, 
                            lims=[-np.pi, np.pi])
        
    # Get Hilbert transform phase (instead of linearly interpolated phase).
    if phase_type == 'hilbert':
        for band_name, pass_band in bands.items():
            _, phase_hilbert = spp.get_hilbert(lfp_filt[band_name])
            phase[band_name]['phase'] = phase_hilbert
    del lfp_filt
    
    # Get phase mask.
    if mask_type is not None: 
        cycle_stats = dio.open_pickle(op.join('/scratch/dscho/phase_precession/data/cycle_stats',
                                                   '{}_cycle_stats_{}_{}powpct_{}osclength.pkl'
                                                   .format(subj_sess, phase_type_, power_percentile, osc_length)))
        mask = OrderedDict()
        for band_name, pass_band in bands.items():
            mask[band_name] = cycles_to_lfp(cycle_stats, category=mask_type, cycle_type=band_name)
    
    # Load the cross-electrode DataFrame.
    pl_df = dio.open_pickle(op.join('/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                           '{}_cross_electrode_unit_lfp_pairs_df.pkl'.format(subj_sess)))
    
    # Restrict analyses to a subset of pairs.
    if hpc_subset:
        # Remove inter-hemispheric pairs.
        pl_df = pl_df.loc[pl_df.same_hem==True]
        # Remove same-channel pairs.
        pl_df = pl_df.loc[pl_df.same_chan!=True]
        # Remove intra-regional pairs from different microwire bundles.
        pl_df = pl_df.query("(unit_roi2!=lfp_roi2) | ((unit_roi2==lfp_roi2) & (same_hemroi==True))")
        # Remove pairs that aren't intra-regional and where the LFP is extra-hippocampal.
        pl_df = pl_df.query("(lfp_is_hpc==True) | (unit_roi2==lfp_roi2)")
        # Remove units with <250 spikes.
        pl_df = pl_df.loc[pl_df.unit_nspikes>249]
        # Remove units that don't have connections to HPC.
        keep_units = pl_df.query("(lfp_is_hpc==True)").groupby(['subj_sess_unit']).size().index.tolist()
        pl_df = pl_df.query("(subj_sess_unit=={})".format(keep_units))

    pl_df.reset_index(inplace=True, drop=True)
    
    # Time lag analysis.
    # Slide from past LFP predicting future spikes to the reverse
    # (10ms steps from -2 to 2secs), getting a MRL for each step.
#     steps = np.arange(-2*2000, 2*2000+20, 2000*0.01, dtype=int)
    steps = np.arange(-2*2000000, 2*2000000+20, 2000*10, dtype=int)
    tl_mrls = []
    for i in range(len(pl_df)):
        unit = pl_df.at[i, 'unit']
        lfp_chan_ind = pl_df.at[i, 'lfp_chan_ind']
        spike_train = fr_df.at[unit, 'spikes']
#         n_timepoints = len(spike_train) # REMOVE LATER!!
#         spike_train = np.roll(spike_train, int(random.random() * n_timepoints)) # REMOVE LATER!!
        mrls_arr = []
        for band_name, pass_band in bands.items():
            mrls_arr_ = []
            for step in steps:
                spike_train_ = np.roll(spike_train, step)
#                 spike_train_[:4000] = False
#                 spike_train_[-4000:] = False
#                 spike_train_[:40000] = False
#                 spike_train_[-40000:] = False
                if mask_type is None:
                    spike_inds = np.where(spike_train_)[0]
                else:
                    spike_inds = np.where(mask[band_name][lfp_chan_ind, :] * spike_train_)[0]
                spike_phases = phase[band_name]['phase'][lfp_chan_ind, spike_inds]
                mrls_arr_.append(circstats.circmoment(spike_phases)[1])
            mrls_arr.append(mrls_arr_)
        tl_mrls.append(np.array(mrls_arr))
    pl_df['tl_mrls'] = tl_mrls
    del phase
    
    # Bootstrapping - Randomly shift the spike train n_bootstraps times and
    # recalc MRLs. Because a unit could be phase-locked at a short time
    # lag, permutations will shift the spike train by at least 2 secs.
#     n_timepoints = len(spike_train)
#     bs_mrls = []
#     for iunit in range(len(pl_df)):
#         unit = pl_df.at[iunit, 'unit']
#         lfp_chan_ind = pl_df.at[iunit, 'lfp_chan_ind']
#         spike_train = fr_df.at[unit, 'spikes']
#         unit_bs_mrls = []
#         for band_name, pass_band in bands.items(): 
#             unit_bs_mrls_ = []
#             for _ in range(n_bootstraps):
#                 spike_train_shifted = np.roll(spike_train, 4000+int(random.random() * (n_timepoints-4001)))
#                 if mask_type is None:
#                     spike_inds = np.where(spike_train_shifted)[0]
#                 else:
#                     spike_inds = np.where(mask[band_name][lfp_chan_ind, :] * spike_train_shifted)[0]
#                 spike_phases = phase[band_name]['phase'][lfp_chan_ind, spike_inds]
#                 unit_bs_mrls_.append(circstats.circmoment(spike_phases)[1])
#             unit_bs_mrls.append(unit_bs_mrls_)
#         bs_mrls.append(np.array(unit_bs_mrls))
#     pl_df['bs_mrls'] = bs_mrls
#     del phase
    
    # Get the mean MRL across unit->LFP pairs for unit->local
    # and unit->HPC connections, for each band.
    upl_df = (pl_df.groupby(['subj_sess_unit', 'subj_sess', 'unit_roi2', 'lfp_roi2', 
                             'unit_hem', 'lfp_hem', 'unit_is_hpc', 'lfp_is_hpc', 
                             'unit_fr', 'unit_nspikes'])
                   .agg({'tl_mrls': lambda x: tuple(np.mean(x))})
                   .reset_index())
    upl_df['tl_mrls'] = upl_df.tl_mrls.apply(lambda x: np.array(x))
    
    # Save the cross-electrode phase-locking DataFrame.
    process_str = ''
    process_str += 'notch' + '-'.join(str(i) for i in notch_freqs) if notch_freqs else 'nonotch'
    process_str += '_spikeinterp-{}to{}ms'.format(ms_before, ms_after) if interp_spikes else '_nospikeinterp'
    process_str += '_phase-{}'.format(phase_type)
    process_str += '_{}-{}osc-{}powpct'.format(''.join(mask_type.split('_')), osc_length, power_percentile) if mask_type else '_nomask'
    process_str += '_hpc-subset' if hpc_subset else ''
#     process_str += '_timelag-2to2sec-step10ms'
#     process_str += '_timelag-20to20sec-step100ms'
    process_str += '_timelag-2000to2000sec-step10000ms' # REMOVE LATER!!
    process_str += '_bands--' + '--'.join(['{}{}-{}'.format(key, val[0], val[1]) for key, val in bands.items()])
    if save_outputs:
        fpath = op.join(output_dir, '{}_crosselec_phaselock_byunit_df_{}.pkl'.format(subj_sess, process_str))
        dio.save_pickle(upl_df, fpath)
    
    return upl_df

    
def calc_cross_electrode_phase_locking_DEPRECATED(subj_sess,
                                       bands=None,
                                       interp_spikes=True,
                                       notch_freqs=[60, 120],
                                       zscore_lfp=True,
                                       zscore_power=True,
                                       phase_type='peaks', # troughs, peaks, or hilbert
                                       power_percentile=25,
                                       osc_length=3,
                                       mask_type=None,
                                       hpc_subset=False,
                                       n_bootstraps=0,
                                       roll2=False,
                                       save_outputs=True,
                                       output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking'):
    """Calculate phase-locking between each unit and each channel LFP, 
    for a given set of frequency bands.
    
    Returns
    -------
    pl_df : pd.DataFrame
        The cross-electrode phase-locking DataFrame (all unit-to-channel pairs).
    """
    if bands is None:
        bands = OrderedDict([('low_theta', [1, 5]),
                             ('high_theta', [5, 10]),
                             ('alpha_beta', [10, 20]),
                             ('low_gamma', [30, 50]),
                             ('mid_gamma', [70, 90]),
                             ('high_gamma', [90, 110])])
    band_names = list(bands.keys())
    hpc_rois = ['LAH', 'LMH', 'LPH', 'RAH', 'RMH', 'RPH']
    
    # Get session info.
    subj_df = get_subj_df()
    
    # Load spikes.
    spikes, fr_df, clus_to_chan, chan_to_clus = load_spikes(subj_sess)
    units = fr_df.clus.tolist()
    
    # Load the raw LFP.
    ms_before = 2
    ms_after = 4
    lfp_raw, lfp_preproc = manning_analysis.process_lfp(subj_sess, 
                                                        subj_df=subj_df, 
                                                        notch_freqs=notch_freqs,
                                                        interpolate=interp_spikes,
                                                        session_spikes=spikes,
                                                        ms_before=ms_before,
                                                        ms_after=ms_after)
    del lfp_raw, spikes
    sampling_rate = lfp_preproc.samplerate.data.tolist()
    
    # Bandpass filter the LFP.
    lfp_filt = spp.filter_lfp_bands(lfp_preproc, 
                                    bands=bands, 
                                    zscore_lfp=zscore_lfp)
    del lfp_preproc
    
    # Use a derivative test to identify troughs and peaks in the filtered 
    # LFP, then linearly interpolate phase between cycles.
    if phase_type == 'hilbert':
        phase_type_ = 'peaks'
    else:
        phase_type_ = phase_type
    
    phase = get_phase_bands(lfp_filt, 
                            bands=bands, 
                            find=phase_type_, 
                            lims=[-np.pi, np.pi])
        
    # Get Hilbert transform phase (instead of linearly interpolated phase).
    if phase_type == 'hilbert':
        for band_name, pass_band in bands.items():
            _, phase_hilbert = spp.get_hilbert(lfp_filt[band_name])
            phase[band_name]['phase'] = phase_hilbert
    del lfp_filt
    
    # Get phase mask.
    if mask_type is not None: 
        cycle_stats = dio.open_pickle(op.join('/scratch/dscho/phase_precession/data/cycle_stats',
                                                   '{}_cycle_stats_{}_{}powpct_{}osclength.pkl'
                                                   .format(subj_sess, phase_type_, power_percentile, osc_length)))
        mask = OrderedDict()
        for band_name, pass_band in bands.items():
            mask[band_name] = cycles_to_lfp(cycle_stats, category=mask_type, cycle_type=band_name)
    
    # Load the cross-electrode DataFrame.
    pl_df = dio.open_pickle(op.join('/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                           '{}_cross_electrode_unit_lfp_pairs_df.pkl'.format(subj_sess)))
    
    # Restrict analyses to a subset of pairs.
    if hpc_subset:
        # Remove inter-hemispheric pairs.
        pl_df = pl_df.loc[pl_df.same_hem==True]
        # Remove same-channel pairs.
        pl_df = pl_df.loc[pl_df.same_chan!=True]
        # Remove intra-regional pairs from different microwire bundles.
        pl_df = pl_df.query("(unit_roi2!=lfp_roi2) | ((unit_roi2==lfp_roi2) & (same_hemroi==True))")
        # Remove pairs that aren't intra-regional and where the LFP is extra-hippocampal.
        pl_df = pl_df.query("(lfp_is_hpc==True) | (unit_roi2==lfp_roi2)")
        # Remove units with <250 spikes.
        pl_df = pl_df.loc[pl_df.unit_nspikes>249]
    pl_df.reset_index(inplace=True, drop=True)
    
    # Get phase and phase stats for each band.
    spike_inds = []
    spike_phases = []
    means = []
    lengths = []
    pvals = []
    for i in range(len(pl_df)):
        unit = pl_df.at[i, 'unit']
        lfp_chan_ind = pl_df.at[i, 'lfp_chan_ind']
        spike_train = fr_df.at[unit, 'spikes']
        if roll2: # shuffle spike train 10secs forward
            spike_train = np.roll(spike_train, 4000)
        spike_inds_ = []
        spike_phases_ = []
        means_ = []
        lengths_ = []
        pvals_ = []
        for band_name, pass_band in bands.items():
            if mask_type is None:
                spike_inds_.append(np.where(spike_train)[0])
            else:
                spike_inds_.append(np.where(mask[band_name][lfp_chan_ind, :] * spike_train)[0])
            spike_phases_.append(phase[band_name]['phase'][lfp_chan_ind, spike_inds_[-1]])
            if len(spike_phases_[-1]) > 0:
                mean, length = circstats.circmoment(spike_phases_[-1])
                means_.append(mean)
                lengths_.append(length)
                pvals_.append(circstats.rayleightest(spike_phases_[-1]))
            else:
                means_.append(np.nan)
                length_.append(np.nan)
                pvals_.append(np.nan)
        spike_inds.append(spike_inds_)
        spike_phases.append(spike_phases_)
        means.append(means_)
        lengths.append(lengths_)
        pvals.append(pvals_)
    
    # Add new columns to the cross-electrode dataframe.
    pl_df['spike_inds'] = spike_inds
    pl_df['spike_phases'] = spike_phases
    pl_df['means'] = means
    pl_df['lengths'] = lengths
    pl_df['pvals'] = pvals
    
    # Get the band with maximum phase-locking. 
    pl_df['pval_argmin'] = pl_df.pvals.apply(lambda x: np.argmin(x))
    locked_bands = []
    locked_phases = []
    locked_lengths = []
    locked_pvals = []
    for i in range(len(pl_df)):
        locked_bands.append(band_names[pl_df.at[i, 'pval_argmin']])
        locked_phases.append(pl_df.at[i, 'means'][pl_df.at[i, 'pval_argmin']])
        locked_lengths.append(pl_df.at[i, 'lengths'][pl_df.at[i, 'pval_argmin']])
        locked_pvals.append(pl_df.at[i, 'pvals'][pl_df.at[i, 'pval_argmin']])
    pl_df['locked_band'] = np.array(locked_bands)
    pl_df['locked_phase'] = np.array(locked_phases)
    pl_df['locked_length'] = np.array(locked_lengths)
    pl_df['locked_pval'] = np.array(locked_pvals)
    
    # Bootstrapping - Randomly shift the spike train n_bootstraps times and
    # recalc Rayleigh. Because a unit could be phase-locked at a short time
    # lag, permutations will shift the spike train by at least 2 secs.
    if n_bootstraps > 0:
        bs_pval_vecs = []
        bs_pval_inds = []
        bs_pvals = []
        n_timepoints = len(spike_train)
        for i in range(len(pl_df)):
            obs_pval = pl_df.at[i, 'locked_pval']
            # Only test if the original Rayleigh p-value was below 0.05
#             if obs_pval >= 0.05:
#                 bs_pval_vecs.append([])
#                 bs_pval_inds.append(n_bootstraps)
#                 bs_pvals.append(1.0)
#                 continue
            unit = pl_df.at[i, 'unit']
            spike_train = fr_df.at[unit, 'spikes']
            band_name = pl_df.at[i, 'locked_band']
            lfp_chan_ind = pl_df.at[i, 'lfp_chan_ind']
            bs_pval_vecs_ = []
            for ii in range(n_bootstraps):
                spike_train_shifted = np.roll(spike_train, 4000+int(random.random() * (n_timepoints-4001)))
                if mask_type is None:
                    spike_phases = phase[band_name]['phase'][lfp_chan_ind, np.where(spike_train_shifted)[0]]
                else:
                    spike_phases = phase[band_name]['phase'][lfp_chan_ind, np.where(mask[band_name][lfp_chan_ind, :] * spike_train_shifted)[0]]
                bs_pval_vecs_.append(circstats.rayleightest(spike_phases))
            bs_pval_vecs.append(np.array(bs_pval_vecs_))
            bs_pval_inds.append(np.sum(bs_pval_vecs[-1] <= obs_pval))
            bs_pvals.append((bs_pval_inds[-1] + 1) / (n_bootstraps + 1))
        pl_df['bs_pval_vec'] = bs_pval_vecs
        pl_df['bs_pval_ind'] = bs_pval_inds
        pl_df['bs_pval'] = bs_pvals
    del phase
        
    # Determine phase-locking significance, using bootstrapped
    # p-values if n_bootstraps > 0 and otherwise using the raw
    # Rayleigh p-value.
    alpha = 0.05 / len(band_names)
    pl_df['n_comparisons'] = np.array(pl_df.groupby(['subj_sess_unit', 'lfp_hemroi2']).unit.transform(len).tolist())
    pl_df['bonf_alpha'] = alpha / np.array(pl_df.groupby(['subj_sess_unit', 'lfp_hemroi2']).unit.transform(len).tolist())
    if n_bootstraps > 0:
        pl_df['sig'] = pl_df.bs_pval < pl_df.bonf_alpha
    else:
        pl_df['sig'] = pl_df.locked_pval < pl_df.bonf_alpha
    
    # Save the cross-electrode phase-locking DataFrame.
    process_str = ''
    process_str += 'notch' + '-'.join(str(i) for i in notch_freqs) if notch_freqs else 'nonotch'
    process_str += '_spikeinterp-{}to{}ms'.format(ms_before, ms_after) if interp_spikes else '_nospikeinterp'
    process_str += '_phase-{}'.format(phase_type)
    process_str += '_{}-{}osc-{}powpct'.format(''.join(mask_type.split('_')), osc_length, power_percentile) if mask_type else '_nomask'
    process_str += '_{}bootstraps'.format(n_bootstraps)
    process_str += '_hpc-subset' if hpc_subset else ''
    process_str += '_roll2' if roll2 else ''
    process_str += '_bands--' + '--'.join(['{}{}-{}'.format(key, val[0], val[1]) for key, val in bands.items()])
    if save_outputs:
        fpath = op.join(output_dir, '{}_crosselec_phaselock_df_{}.pkl'.format(subj_sess, process_str))
        dio.save_pickle(pl_df, fpath)
    
    return pl_df


def calc_cross_electrode_phase_locking_timelag_DEPRECATED(subj_sess,
                                               bands=None,
                                               interp_spikes=True,
                                               notch_freqs=[60, 120],
                                               zscore_lfp=True,
                                               zscore_power=True,
                                               phase_type='peaks',
                                               power_percentile=25,
                                               osc_length=3,
                                               mask_type=None,
                                               hpc_subset=False,
                                               save_outputs=True,
                                               output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/time_lag'):
    """Calculate phase-locking between each unit and each channel LFP, 
    for a given set of frequency bands.
    
    Returns
    -------
    pl_df : pd.DataFrame
        The cross-electrode phase-locking DataFrame (all unit-to-channel pairs).
    """
    band_names = list(bands.keys())
    hpc_rois = ['LAH', 'LMH', 'LPH', 'RAH', 'RMH', 'RPH']
    
    # Get session info.
    subj_df = get_subj_df()
    
    # Load spikes.
    spikes, fr_df, clus_to_chan, chan_to_clus = load_spikes(subj_sess)
    units = fr_df.clus.tolist()
    
    # Load the raw LFP.
    ms_before = 2
    ms_after = 4
    lfp_raw, lfp_preproc = manning_analysis.process_lfp(subj_sess, 
                                                        subj_df=subj_df, 
                                                        notch_freqs=notch_freqs,
                                                        interpolate=interp_spikes,
                                                        session_spikes=spikes,
                                                        ms_before=ms_before,
                                                        ms_after=ms_after)
    del lfp_raw, spikes
    sampling_rate = lfp_preproc.samplerate.data.tolist()
    
    # Bandpass filter the LFP.
    lfp_filt = spp.filter_lfp_bands(lfp_preproc, 
                                    bands=bands, 
                                    zscore_lfp=zscore_lfp)
    del lfp_preproc
    
    # Use a derivative test to identify troughs and peaks in the filtered 
    # LFP, then linearly interpolate phase between cycles.
    if phase_type == 'hilbert':
        phase_type_ = 'peaks'
    else:
        phase_type_ = phase_type
    
    phase = get_phase_bands(lfp_filt, 
                            bands=bands, 
                            find=phase_type_, 
                            lims=[-np.pi, np.pi])
        
    # Get Hilbert transform phase (instead of linearly interpolated phase).
    if phase_type == 'hilbert':
        for band_name, pass_band in bands.items():
            _, phase_hilbert = spp.get_hilbert(lfp_filt[band_name])
            phase[band_name]['phase'] = phase_hilbert
    del lfp_filt
    
    # Get phase mask.
    if mask_type is not None: 
        cycle_stats = dio.open_pickle(op.join('/scratch/dscho/phase_precession/data/cycle_stats',
                                                   '{}_cycle_stats_{}_{}powpct_{}osclength.pkl'
                                                   .format(subj_sess, phase_type_, power_percentile, osc_length)))
        mask = OrderedDict()
        for band_name, pass_band in bands.items():
            mask[band_name] = cycles_to_lfp(cycle_stats, category=mask_type, cycle_type=band_name)
    
    # Load the cross-electrode DataFrame.
    pl_df = dio.open_pickle(op.join('/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking',
                                           '{}_cross_electrode_unit_lfp_pairs_df.pkl'.format(subj_sess)))
    
    # Restrict analyses to a subset of pairs.
    if hpc_subset:
        # Remove inter-hemispheric pairs.
        pl_df = pl_df.loc[pl_df.same_hem==True]
        # Remove same-channel pairs.
        pl_df = pl_df.loc[pl_df.same_chan!=True]
        # Remove intra-regional pairs from different microwire bundles.
        pl_df = pl_df.query("(unit_roi2!=lfp_roi2) | ((unit_roi2==lfp_roi2) & (same_hemroi==True))")
        # Remove pairs that aren't intra-regional and where the LFP is extra-hippocampal.
        pl_df = pl_df.query("(lfp_is_hpc==True) | (unit_roi2==lfp_roi2)")
        # Remove units with <250 spikes.
        pl_df = pl_df.loc[pl_df.unit_nspikes>249]
    pl_df.reset_index(inplace=True, drop=True)
    
    # Get phase and phase stats for each band.
    spike_inds = []
    spike_phases = []
    means = []
    lengths = []
    pvals = []
    
    # Time lag analysis.
    # Slide from past LFP predicting future spikes to the reverse
    # (10ms steps from -2 to 2secs), getting a Rayleigh p-value for
    # each step.
    steps = np.arange(-2*2000, 2*2000+20, 2000*0.01, dtype=int)
    pvals = []
    for i in range(len(pl_df)):
        unit = pl_df.at[i, 'unit']
        lfp_chan_ind = pl_df.at[i, 'lfp_chan_ind']
        spike_train = fr_df.at[unit, 'spikes']
        pvals_arr = []
        for band_name, pass_band in bands.items():
            pvals_arr_ = []
            for step in steps:
                spike_train_ = np.roll(spike_train, step)
                spike_train_[:4000] = False
                spike_train_[-4000:] = False
                if mask_type is None:
                    spike_inds = np.where(spike_train_)[0]
                else:
                    spike_inds = np.where(mask[band_name][lfp_chan_ind, :] * spike_train_)[0]
                spike_phases = phase[band_name]['phase'][lfp_chan_ind, spike_inds]
                pvals_arr_.append(circstats.rayleightest(spike_phases))
            pvals_arr.append(pvals_arr_)
        pvals.append(np.array(pvals_arr))
    
    # Add new columns to the cross-electrode dataframe.
    pl_df['pvals'] = pvals
    
    # Save the cross-electrode phase-locking DataFrame.
    process_str = ''
    process_str += 'notch' + '-'.join(str(i) for i in notch_freqs) if notch_freqs else 'nonotch'
    process_str += '_spikeinterp-{}to{}ms'.format(ms_before, ms_after) if interp_spikes else '_nospikeinterp'
    process_str += '_phase-{}'.format(phase_type)
    process_str += '_{}-{}osc-{}powpct'.format(''.join(mask_type.split('_')), osc_length, power_percentile) if mask_type else '_nomask'
    process_str += '_hpc-subset' if hpc_subset else ''
    process_str += '_timelag-2to2sec-step10ms'
    process_str += '_bands--' + '--'.join(['{}{}-{}'.format(key, val[0], val[1]) for key, val in bands.items()])
    if save_outputs:
        fpath = op.join(output_dir, '{}_crosselec_phaselock_df_{}.pkl'.format(subj_sess, process_str))
        dio.save_pickle(pl_df, fpath)
    
    return pl_df

    
def get_cross_electrode_unit_lfp_pairs(subj_sess,
                                       save_outputs=True,
                                       output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking/metadata'):
    """Save DataFrames for 1) all unit-to-LFP channel pairs and
    2) all unit-to-LFP region pairs.
    
    For the second DataFrame, units are retained if they have a connection to
    the HPC and have at least 250 spikes. 
    
    Returns
    -------
    ce_df : pd.DataFrame
        The unit-to-channel DataFrame.
    unit_df : pd.DataFrame
        The unit-to-region DataFrame.
    """
    hpc_rois = ['LAH', 'LMH', 'LPH', 'RAH', 'RMH', 'RPH']
    col_names = ['subj_sess', 'subj_sess_unit', 'unit', 'unit_chan', 'unit_chan_ind',
                 'unit_hemroi', 'unit_hem', 'unit_roi', 'unit_is_hpc', 'unit_nspikes', 'unit_fr', 
                 'lfp_chan', 'lfp_chan_ind', 'lfp_hemroi', 'lfp_hem', 'lfp_roi', 'lfp_is_hpc',
                 'same_chan', 'same_hemroi', 'same_hem', 'same_roi', 'both_hpc']
    # Load subject DataFrame.
    subj_df = get_subj_df()
    df = subj_df.loc[subj_df.subj_sess==subj_sess].reset_index(drop=True)
    lfp_chans = df.chan.tolist()
    
    # Load spikes.
    spikes, fr_df, clus_to_chan, chan_to_clus = load_spikes(subj_sess)
    units = fr_df.clus.tolist()    
    
    unit_lfp_pairs = []
    for unit in units:
        unit_nspikes = np.sum(fr_df.at[unit, 'spikes'])
        unit_fr = fr_df.at[unit, 'mean_fr']
        unit_chan = fr_df.at[unit, 'chan']
        unit_chan_ind = int(unit_chan) - 1
        unit_hemroi = fr_df.at[unit, 'location']
        unit_hem = unit_hemroi[0]
        unit_roi = unit_hemroi[1:]
        unit_is_hpc = 1 if (unit_hemroi in hpc_rois) else 0
        for lfp_chan in lfp_chans:
            lfp_chan_ind = int(lfp_chan) - 1
            lfp_hemroi = df.loc[df.chan==lfp_chan, 'location'].iat[0]
            lfp_hem = lfp_hemroi[0]
            lfp_roi = lfp_hemroi[1:]
            lfp_is_hpc = 1 if (lfp_hemroi in hpc_rois) else 0
            
            same_chan = 1 if unit_chan_ind == lfp_chan_ind else 0
            same_hemroi = 1 if unit_hemroi == lfp_hemroi else 0
            same_roi = 1 if unit_roi == lfp_roi else 0
            same_hem = 1 if unit_hem == lfp_hem else 0
            both_hpc = 1 if (unit_is_hpc and lfp_is_hpc) else 0
            
            unit_lfp_pairs.append([subj_sess, '{}_{}'.format(subj_sess, unit), unit, unit_chan, unit_chan_ind, 
                                   unit_hemroi, unit_hem, unit_roi, unit_is_hpc, unit_nspikes, unit_fr,
                                   lfp_chan, lfp_chan_ind, lfp_hemroi, lfp_hem, lfp_roi, lfp_is_hpc,
                                   same_chan, same_hemroi, same_hem, same_roi, both_hpc])
    ce_df = pd.DataFrame(unit_lfp_pairs, columns=col_names)
    ce_df = ce_df.query("(unit_hemroi!='none') & (lfp_hemroi!='none')").copy()
    
    # Add higher-level ROI mappings.
    hemrois = {'lhpc': ['LAH', 'LMH', 'LPH'],
               'lamy': ['LA'],
               'lec': ['LEC'],
               'lphg': ['LPG'],
               'lac': ['LAC'],
               'lofc': ['LOF'],
               'rhpc': ['RAH', 'RMH', 'RPH'],
               'ramy': ['RA'],
               'rec': ['REC'],
               'rphg': ['RPG'],
               'rac': ['RAC'],
               'rofc': ['ROF']}
    hemrois['lctx'] = [roi for roi in sorted(set(ce_df.loc[ce_df.lfp_hem=='L'].lfp_hemroi)) 
                       if roi not in list(itertools.chain(*hemrois.values()))]
    hemrois['rctx'] = [roi for roi in sorted(set(ce_df.loc[ce_df.lfp_hem=='R'].lfp_hemroi)) 
                       if roi not in list(itertools.chain(*hemrois.values()))]
    hemrois_inv = invert_dict(hemrois)
    ce_df['unit_hemroi2'] = ce_df.unit_hemroi.agg(lambda x: hemrois_inv[x])
    ce_df['lfp_hemroi2'] = ce_df.lfp_hemroi.agg(lambda x: hemrois_inv[x])

    rois = {'hpc': ['AH', 'MH', 'PH'],
            'amy': ['A'],
            'ec': ['EC'],
            'phg': ['PG'],
            'ac': ['AC'],
            'ofc': ['OF']}
    rois['ctx'] = [roi for roi in sorted(set(ce_df.lfp_roi)) if roi not in list(itertools.chain(*rois.values()))]
    rois_inv = invert_dict(rois)
    ce_df['unit_roi2'] = ce_df.unit_roi.agg(lambda x: rois_inv[x])
    ce_df['lfp_roi2'] = ce_df.lfp_roi.agg(lambda x: rois_inv[x])
    ce_df['same_roi2'] = ce_df['unit_roi2'] == ce_df['lfp_roi2']
    
    # Collapse channel-to-channel connections into unit-to-region connections.
    # Units are kept if they have >= 250 spikes.
    cols = [c for c in ce_df.columns if c not in ['lfp_chan', 'lfp_chan_ind', 'unit_chan']]
    #keep_units = ce_df.query("(lfp_is_hpc==True) & (unit_nspikes>249)").groupby(['subj_sess_unit']).size().index.tolist()
    keep_units = ce_df.query("(unit_nspikes>249)").groupby(['subj_sess_unit']).size().index.tolist()
    unit_df = (ce_df
               .query("(unit_chan_ind!=lfp_chan_ind) & (subj_sess_unit=={})".format(keep_units))
               .groupby(cols)
               .lfp_chan_ind.apply(lambda x: list(x))
               .reset_index()
               .rename(columns={'lfp_chan_ind': 'lfp_chan_inds'}))
    unit_df['edge'] = ''
    unit_df.loc[(unit_df.unit_is_hpc==False) & (unit_df.same_hemroi==True), 'edge'] = 'ctx-local'
    unit_df.loc[(unit_df.unit_is_hpc==False) & (unit_df.lfp_is_hpc==True), 'edge'] = 'ctx-hpc'
    unit_df.loc[(unit_df.unit_is_hpc==False) & (unit_df.same_hemroi==False) & (unit_df.lfp_is_hpc==False), 'edge'] = 'ctx-ctx'
    unit_df.loc[(unit_df.unit_is_hpc==True) & (unit_df.same_hemroi==True), 'edge'] = 'hpc-local'
    unit_df.loc[(unit_df.unit_is_hpc==True) & (unit_df.same_hemroi==False) & (unit_df.lfp_is_hpc==True), 'edge'] = 'hpc-hpc'
    unit_df.loc[(unit_df.unit_is_hpc==True) & (unit_df.lfp_is_hpc==False), 'edge'] = 'hpc-ctx'
           
    if save_outputs:
        ce_fpath = op.join(output_dir, '{}_unit_to_lfp-channel_pairs_df.pkl'.format(subj_sess))
        unit_fpath = op.join(output_dir, '{}_unit_to_lfp-region_pairs_df.pkl'.format(subj_sess))
        dio.save_pickle(ce_df, ce_fpath, verbose=False)
        dio.save_pickle(unit_df, unit_fpath, verbose=False)
    
    return ce_df, unit_df


def invert_dict(d):
    """Invert a dictionary of string keys and list values."""
    if type(d) == dict:
        newd = {}
    else:
        newd = OrderedDict()
    for k, v in d.items():
        for x in v:
            newd[x] = k
    return newd
    

def get_cross_electrode_unit_lfp_pairs_DEPRECATED(subj_sess,
                                       save_outputs=True,
                                       output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/crosselec_phase_locking'):
    """Save a DataFrame of all cross-electrode unit-to-LFP channel pairs.
    
    Returns
    -------
    ce_df : pd.DataFrame
        The cross-electrode DataFrame (all unit-to-channel pairs).
    """
    hpc_rois = ['LAH', 'LMH', 'LPH', 'RAH', 'RMH', 'RPH']
    col_names = ['subj_sess', 'unit_lfpchanind', 'unit', 'unit_chan', 'unit_chan_ind',
                 'unit_hemroi', 'unit_hem', 'unit_roi', 'unit_is_hpc', 'lfp_chan', 
                 'lfp_chan_ind', 'lfp_hemroi', 'lfp_hem', 'lfp_roi', 'lfp_is_hpc',
                 'same_chan', 'same_hemroi', 'same_hem', 'same_roi', 'both_hpc']
    # Load subject DataFrame.
    subj_df = phase_locking.get_subj_df()
    df = subj_df.loc[subj_df.subj_sess==subj_sess].reset_index(drop=True)
    lfp_chans = df.chan.tolist()
    
    # Load spikes.
    spikes, fr_df, clus_to_chan, chan_to_clus = phase_locking.load_spikes(subj_sess)
    units = fr_df.clus.tolist()    
    
    unit_lfp_pairs = []
    for unit in units:
        unit_chan = fr_df.at[unit, 'chan']
        unit_chan_ind = int(unit_chan) - 1
        unit_hemroi = fr_df.at[unit, 'location']
        unit_hem = unit_hemroi[0]
        unit_roi = unit_hemroi[1:]
        unit_is_hpc = 1 if (unit_hemroi in hpc_rois) else 0
        for lfp_chan in lfp_chans:
            lfp_chan_ind = int(lfp_chan) - 1
            lfp_hemroi = df.loc[df.chan==lfp_chan, 'location'].iat[0]
            lfp_hem = lfp_hemroi[0]
            lfp_roi = lfp_hemroi[1:]
            lfp_is_hpc = 1 if (lfp_hemroi in hpc_rois) else 0
            
            same_chan = 1 if unit_chan_ind == lfp_chan_ind else 0
            same_hemroi = 1 if unit_hemroi == lfp_hemroi else 0
            same_roi = 1 if unit_roi == lfp_roi else 0
            same_hem = 1 if unit_hem == lfp_hem else 0
            both_hpc = 1 if (unit_is_hpc and lfp_is_hpc) else 0
            
            unit_lfp_pairs.append([subj_sess, '{}_{}'.format(unit, lfp_chan_ind), unit, unit_chan, unit_chan_ind, 
                                   unit_hemroi, unit_hem, unit_roi, unit_is_hpc, lfp_chan, 
                                   lfp_chan_ind, lfp_hemroi, lfp_hem, lfp_roi, lfp_is_hpc,
                                   same_chan, same_hemroi, same_hem, same_roi, both_hpc])
    ce_df = pd.DataFrame(unit_lfp_pairs, columns=col_names)
    
    if save_outputs:
        fpath = op.join(output_dir, '{}_cross_electrode_unit_lfp_pairs_df.pkl'.format(subj_sess))
        dio.save_pickle(ce_df, fpath)
    
    return ce_df
    
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def calc_cycle_stats(cycle_inds, power, pass_band, sampling_rate=2000., 
                     power_percentile=25, osc_length=3):
    """Return cycle-by-cycle statistics for a given frequency band.

    Parameters
    ----------
    cycle_inds : list 
        (x_inds, y_inds) that mark divisions between cycles in the filtered LFP.
    power : numpy.ndarray
        Power of the Hilbert-transformed filtered LFP, with dims channel x time.
    pass_band : list or tuple
        The pass-band of the filtered LFP, with (low, high) cutoffs in Hz.
        Used to determine which cycles are within the pass-band.
    sampling_rate : int or float
        Sampling rate of the LFP.
    power_percentile : int
        Percentile above which the mean power for a given cycle must exceed
        vs. all cycles in a channel (affects cycle_stats['power_mask'],
        cycle_stats['powdur_mask'], cycle_stats['osc_start_mask'], 
        and cycle_stats['osc_mask']).
    osc_length : int
        Number of consectutive cycles that must meet duration and power 
        criteria to be considered an oscillation. A cycle meets duration
        criteria if its duration is within the passband, and meets power
        criteria according to power_percentile.

    Returns
    -------
    cycle_stats : dict
        'power' : mean power for each cycle
        'power_rank' : cycle power rank, taken across time, for each channel
        'power_mask' : mask of power ranks > power_percentile
        'duration' : cycle duration, in number of timepoints
        'duration_hz' : cycle duration, in Hz
        'duration_mask' : mask of cycle durations within the passband
        'powdur_mask' : power_mask * duration_mask
        'osc_start_mask' : cycles at the start of an oscillation (of length=osc_length)
        'osc_mask' : cycles that are part of an oscillation
    """
    cycle_stats = {}
    xinds, yinds = cycle_inds
    cycle_pows = [] # mean power Z-score for each cycle
    cycle_pow_ranks = [] # rank of mean power Z-score for each cycle
    cycle_pow_ma = [] # bool; whether cycle power exceeds the power percentile
    cycle_durs = [] # duration of each cycle, in number of timepoints
    cycle_durs_hz = [] # duration of each cycle, in Hz
    cycle_durs_ma = [] # bool; whether cycle length is within the passband
    cycle_powdur_ma = [] # bool; cycle_durs_ma * cycle_pow_ma
    cycle_osc_start_ma = [] # bool; cycles at the start of an oscillation (of length = osc_length)
    cycle_osc_ma = [] # bool; cycles that are part of an oscillation
    for xind in np.unique(xinds):
        yinds_ = yinds[xinds==xind]

        # Get mean Z-power and ranked power for each cycle.
        split_pows = np.split(power[xind, :], yinds_)
        cycle_pows.append(np.array([np.mean(arr) for arr in split_pows]))
        xsort = cycle_pows[-1].argsort()
        cycle_pow_ranks_ = np.empty(xsort.shape)
        cycle_pow_ranks_[xsort] = np.arange(len(cycle_pow_ranks_))
        cycle_pow_ranks.append(cycle_pow_ranks_)

        # Get duration (in timepoints and Hz) for each cycle.
        cycle_durs.append(np.array([len(arr) for arr in split_pows]))
        cycle_durs_hz.append(np.array([sampling_rate / len(arr) for arr in split_pows]))

        # Mask cycles according to power and duration criteria.
        cycle_pow_ma.append(cycle_pow_ranks_ > np.percentile(cycle_pow_ranks_, power_percentile))
        cycle_durs_ma.append((cycle_durs_hz[-1] > pass_band[0]) * (cycle_durs_hz[-1] < pass_band[1]))
        cycle_powdur_ma.append(cycle_durs_ma[-1] * cycle_pow_ma[-1])

        # Get oscillation masks.
        cycle_osc_start_ma.append(rolling_func(cycle_powdur_ma[-1], np.all, window=osc_length, right_fill=False))
        osc_cycles = np.zeros(len(cycle_osc_start_ma[-1]), dtype=np.bool_)
        osc_starts = np.where(cycle_osc_start_ma[-1])[0]
        for i in range(osc_length):
            osc_cycles[osc_starts+i] = True
        cycle_osc_ma.append(osc_cycles)

    cycle_stats = {'power': cycle_pows,
                   'power_rank': cycle_pow_ranks,
                   'power_mask': cycle_pow_ma,
                   'duration': cycle_durs,
                   'duration_hz': cycle_durs_hz,
                   'duration_mask': cycle_durs_ma,
                   'powdur_mask': cycle_powdur_ma,
                   'osc_start_mask': cycle_osc_start_ma,
                   'osc_mask': cycle_osc_ma}

    return cycle_stats

    
def calc_local_phase_locking(subj_sess,
                             freqs=np.array([f for f in np.logspace(np.log10(1), np.log10(200), 53) if (abs(60 - f) > 5) and (abs(120 - f) > 5) and (abs(180 - f) > 5)]),
                             n_bootstraps=1000,
                             interp_spikes=True,
                             pow_thresh=0, # CHANGE BACK TO 25 LATER!
                             ms_before=2,
                             ms_after=4,
                             morlet_width=5,
                             save_outputs=True,
                             output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/phase_locking'):
    """Assess phase-locking of each unit in the session to the local LFP at the selected frequencies.
    
    Uses Morlet wavelets (width=5) to estimate power and phase at each timepoint.
    
    Phase-locking p-values are assessed with the Rayleigh test for phase uniformity. Output DataFrame
    provides both the "raw" Rayleigh p-value and a value compared against bootstrapped estimates from
    randomly shifting each spike vector in time and reassessing phase-locking at each frequency.
    (Reports the number of randomly derived p-values that are <= the observed p-value.)
    
    Also returns the direction of the preferred phase (mu), length of first trigonometric moment 
    (circmom), and concentration parameter (kappa) from the estimated von Mises distribution
    for each unit and frequency tested. 
    
    Returns
    -------
    pl_df : pandas.core.frame.DataFrame
    """
#     freqs = np.array([2**(x/8) for x in range(54)]) # REMOVE THIS LATER!
#     freqs = np.array([f for f in np.logspace(np.log10(0.1), np.log10(500), 62) if (abs(60 - f) > 5) and (abs(120 - f) > 5) and (abs(180 - f) > 5)]) # REMOVE LATER!
    freqs = np.logspace(np.log10(1), np.log10(10), 20)
    col_names = ['subj_sess', 'unit', 'chan', 'hem', 'roi', 'hemroi', 'is_hpc', 
                 'phases', 'mus', 'kappas', 'circmoms', 'pvals', 'pval_inds_1000']
    hpc_rois = ['LAH', 'LMH', 'LPH', 'RAH', 'RMH', 'RPH']
    freq_iter = range(len(freqs))
    output = []
    
    files = glob.glob(op.join('/data3/scratch/dscho/frLfp/data/metadata', 'subj_df_*.xlsx'))
    subj_df = pd.read_excel(files[0], converters={'chan': str})
    for f in files[1:]:
        subj_df = subj_df.append(pd.read_excel(f, converters={'chan': str}))
    subj_df = subj_df.loc[subj_df.location!='none']
    
    # Load spikes.
    with open(op.join('/data3/scratch/dscho/frLfp/data/spikes', '{}_session_spikes.pkl'.format(subj_sess)), 'rb') as f:
        spikes = pickle.load(f)
    fr_df, clus_to_chan, chan_to_clus = manning_analysis.get_fr_df(subj_sess, spikes)
    units = fr_df.clus.tolist()
    
    # Load LFP
#     lfp_raw, lfp_proc = manning_analysis.process_lfp(subj_sess, subj_df, 
#                                                      notch_freqs=[60, 120, 180],
#                                                      interpolate=interp_spikes,
#                                                      session_spikes=spikes, ms_before=ms_before, ms_after=ms_after)         
    lfp_raw, lfp_proc = manning_analysis.process_lfp(subj_sess, subj_df, 
                                                     notch_freqs=[],
                                                     interpolate=interp_spikes,
                                                     session_spikes=spikes, ms_before=ms_before, ms_after=ms_after)                                        
        
    for unit in units:
        spike_train = fr_df.loc[unit, 'spikes']
        n_timepoints = len(spike_train)
        chan = fr_df.at[unit, 'chan']
        hemroi = fr_df.at[unit, 'location']
        hem = hemroi[0]
        roi = hemroi[1:]
        is_hpc = hemroi in hpc_rois
        lfp_chan = lfp_proc.sel(channel=[chan])
        
        # Get power and phase.
        power, phase = manning_analysis.run_morlet(lfp_chan, freqs=freqs, width=morlet_width, output=['power', 'phase'], 
                                                   log_power=False, z_power=False, savedir=False, verbose=True)
        power = power.squeeze()
        phase = phase.squeeze()
        phase_dat = phase.data
        
        # Get power values above pow_thresh% for each freq.
        if pow_thresh > 0:
            power_mask = power.data > np.expand_dims(np.percentile(power, pow_thresh, axis=1), 1)
        else:
            power_mask = power.data >= np.expand_dims(np.percentile(power, pow_thresh, axis=1), 1)
                                
        # Collect phases of spike times above power thresholds.        
        phases = []
        mus = []
        kappas = []
        pvals = []
        circmoms = []
        for ifreq in freq_iter:
            ifreq_phases = phase_dat[ifreq, np.flatnonzero(power_mask[ifreq, :] * spike_train)]
            phases.append(ifreq_phases)
            if len(ifreq_phases) > 0:
                mu, kappa = circstats.vonmisesmle(ifreq_phases)
                mus.append(mu)
                kappas.append(kappa)
                circmoms.append(circstats.circmoment(ifreq_phases)[1])
                pvals.append(circstats.rayleightest(ifreq_phases))
            else:
                mus.append(np.nan)
                kappas.append(0.0)
                circmoms.append(0.0)
                pvals.append(1.0)
        mus = np.array(mus)
        kappas = np.array(kappas)
        circmoms = np.array(circmoms)
        pvals = np.array(pvals)
        
        # Bootstrapping - Randomly shift the spike train n_bootstraps times and
        # recalc Rayleigh. Because a unit could be phase-locked at a short time
        # lag, permutations will shift the spike train by at least a second.
        bs_pvals = []
        for i in range(n_bootstraps):
            spike_train_shifted = np.roll(spike_train, 2000+int(random.random() * (n_timepoints-2001)))
            bs_pvals_ = []
            for ifreq in freq_iter:
                ifreq_phases = phase_dat[ifreq, np.flatnonzero(power_mask[ifreq, :] * spike_train_shifted)]
                if len(ifreq_phases) > 0:
                    bs_pvals_.append(circstats.rayleightest(ifreq_phases))
                else:
                    bs_pvals_.append(1.0)
            bs_pvals.append(bs_pvals_)
        bs_pvals = np.array(bs_pvals) # bootstrap x freq
        pval_inds = np.sum(bs_pvals <= pvals, axis=0)
        
        # Add values for the unit to the output list
        output.append([subj_sess, unit, chan, hem, roi, hemroi, is_hpc, 
                       phases, mus, kappas, circmoms, pvals, pval_inds])
        
    # Covert output list to DataFrame format.
    pl_df = pd.DataFrame(output, columns=col_names)

    # Save outputs.
    if save_outputs:
        interp_str = ''
        if interp_spikes:
            interp_str = '_interp-{}_{}ms'.format(ms_before, ms_after)
        fname = op.join(output_dir, 'local_phase_locking_df_nonotch_{}freq{}-{}Hz_pow{}mask{}_{}.pkl'.format(len(freqs), int(np.min(freqs)), int(np.max(freqs)), pow_thresh, interp_str, subj_sess))
        with open(fname, 'wb') as f:
            pickle.dump(pl_df, f, pickle.HIGHEST_PROTOCOL)
        
    return pl_df


def calc_phase_locking_bands(subj_sess,
                             bands=None,
                             interp_spikes=True,
                             notch_freqs=[60, 120],
                             zscore_lfp=True,
                             zscore_power=True,
                             phase_type='peaks', # troughs, peaks, or hilbert
                             power_percentile=25,
                             osc_length=3,
                             mask_type=None,
                             save_outputs=True,
                             output_dir='/scratch/dscho/phase_precession/data/phase_locking'):
    """Calculate phase-locking to bandpass filtered LFPs.
    
    Parameters
    ----------
    subj_sess : str
        e.g. 'U367_env2'
    bands : OrderedDict()
        LFP bands to filter. Default bands are:
            low_theta (1-5 Hz)
            high_theta (5-10 Hz)
            alpha_beta (10-20 Hz)
            low_gamma (30-50 Hz)
            mid_gamma (70-90 Hz)
            high_gamma (90-110 Hz)
    interp_spikes : bool
        If True, spikes for each unit are removed from the preprocessed
        LFP by linear interpolation (-2 to 4ms surrounding each spike).
    notch_freqs : list
        Frequencies at which to notch-filter the LFP data during
        preprocessing. If notch_freqs=[] then no notch filtering is done.
    zscore_lfp : bool
        If True, filtered LFP values are Z-scored across time, for each channel.
    zscore_power : bool
        If True, Hilbert transform power values for filtered LFPs are
        Z-scored across time, for each channel.
    phase_type : str
        'peaks' to return phase values that are linearly interpolated
            (0 to 2pi) between identified peaks in the filtered LFPs.
        'troughs' to return phase values that are linearly interpolated
            (0 to 2pi) between identified troughs in the filtered LFPs.
        'hilbert' to return phase from the Hilbert transform of filtered
            LFPs. NOTE if 'hilbert' is chosen then cycle_stats are
            calculated between peaks in the filtered LFPs.
    power_percentile : int
        Percentile above which the mean power for a given cycle must exceed
        vs. all cycles in a channel (affects cycle_stats['power_mask'],
        cycle_stats['powdur_mask'], cycle_stats['osc_start_mask'], 
        and cycle_stats['osc_mask']).
    osc_length : int
        Number of consectutive cycles that must meet duration and power 
        criteria to be considered an oscillation. A cycle meets duration
        criteria if its duration is within the passband, and meets power
        criteria according to power_percentile.
    mask_type : str
        cycle_stats field to use for masking out spikes from phase-locking
        calculations. 'power_mask', 'duration_mask', 'powdur_mask', 
        'osc_start_mask', or 'osc_mask'. If mask_type is None then all spikes
        are included.
    save_files : bool
        If True, pl_df is saved to the filename coded into this function.
    output_dir : bool
        Directory where pl_df is saved.
        
    Returns
    -------
    pl_df : pandas.DataFrame
        Phase-locking data for each unit in the session (including phases of
        all spikes included in the analysis for each unit).
    """
    if bands is None:
        bands = OrderedDict([('low_theta', [1, 5]),
                             ('high_theta', [5, 10]),
                             ('alpha_beta', [10, 20]),
                             ('low_gamma', [30, 50]),
                             ('mid_gamma', [70, 90]),
                             ('high_gamma', [90, 110])])
    hpc_rois = ['LAH', 'LMH', 'LPH', 'RAH', 'RMH', 'RPH']
    
    # Get session info.
    subj_df = get_subj_df()
    
    # Load spikes.
    spikes, fr_df, clus_to_chan, chan_to_clus = load_spikes(subj_sess)
    units = fr_df.clus.tolist()
    
    # Load the raw LFP.
    lfp_raw, lfp_preproc = manning_analysis.process_lfp(subj_sess, 
                                                        subj_df=subj_df, 
                                                        notch_freqs=notch_freqs,
                                                        interpolate=interp_spikes,
                                                        session_spikes=spikes,
                                                        ms_before=2,
                                                        ms_after=4)
    sampling_rate = lfp_preproc.samplerate.data.tolist()
    
    # Bandpass filter the LFP.
    lfp_filt = filter_lfp_bands(lfp_preproc, 
                                bands=bands, 
                                zscore_lfp=zscore_lfp)
    
    # Use a derivative test to identify troughs and peaks in the filtered 
    # LFP, then linearly interpolate phase between cycles.
    if phase_type == 'hilbert':
        phase_type_ = 'peaks'
    else:
        phase_type_ = phase_type
    
    phase = get_phase_bands(lfp_filt, 
                            bands=bands, 
                            find=phase_type_, 
                            lims=[-np.pi, np.pi])
        
    # Get Hilbert transform phase (instead of linearly interpolated phase).
    if phase_type == 'hilbert':
        for band_name, pass_band in bands.items():
            _, phase_hilbert = get_hilbert(lfp_filt[band_name])
            phase[band_name]['phase'] = phase_hilbert
    
    # Load cycle_stats
    cycle_stats = open_pickle(op.join('/scratch/dscho/phase_precession/data/cycle_stats',
                                           '{}_cycle_stats_{}_{}powpct_{}osclength.pkl'
                                           .format(subj_sess, phase_type_, power_percentile, osc_length)))
    
    # Get phase mask.
    if mask_type is not None:
        mask = OrderedDict()
        for band_name, pass_band in bands.items():
            mask[band_name] = cycles_to_lfp(cycle_stats, category=mask_type, cycle_type=band_name)
    
    output = []
    col_names = ['subj_sess', 'unit', 'chan', 'hem', 'roi', 'hemroi', 'is_hpc', 'spike_inds',
                 'phases', 'means', 'lengths', 'pvals']
    hpc_rois = ['LAH', 'LMH', 'LPH', 'RAH', 'RMH', 'RPH']
    for unit in units:
        spike_train = fr_df.loc[unit, 'spikes']
        chan = fr_df.at[unit, 'chan']
        chan_ind = int(chan) - 1
        hemroi = fr_df.at[unit, 'location']
        hem = hemroi[0]
        roi = hemroi[1:]
        is_hpc = hemroi in hpc_rois

        # Get phase and phase stats for each band.
        spike_phases = []
        means = []
        lengths = []
        pvals = []
        for band_name, pass_band in bands.items():
            if mask_type is None:
                spike_inds = np.where(spike_train)[0]
            else:
                spike_inds = np.where(mask[band_name][chan_ind, :] * spike_train)[0]
            spike_phases_ = phase[band_name]['phase'][chan_ind, spike_inds]
            if len(spike_phases_) > 0:
                spike_phases.append(spike_phases_)
                mean, length = circstats.circmoment(spike_phases_)
                means.append(mean)
                lengths.append(length)
                pvals.append(circstats.rayleightest(spike_phases_))       

        # Add values for the unit to the output list
        output.append([subj_sess, unit, chan, hem, roi, hemroi, is_hpc, spike_inds,
                       spike_phases, np.array(means), np.array(lengths), np.array(pvals)])

    # Covert output list to DataFrame format.
    pl_df = pd.DataFrame(output, columns=col_names)
    
    # Save the phase-locking DataFrame.
    if mask_type is None:
        mask_type = 'no_mask'
    if save_outputs:
        fpath = op.join(output_dir, '{}_phase_locking_df_{}_{}_{}powpct_{}osclength.pkl'
                             .format(subj_sess, mask_type, phase_type_, power_percentile, osc_length))
        save_pickle(pl_df, fpath)
        
    return pl_df

    
def convert_radians(vec):
    """Convert from -pi:pi to 0:2pi"""
    return (vec + (2*np.pi)) % (2*np.pi)
    

def cycles_to_lfp(cycle_stats, arr=None, channel=None, category='power', cycle_type='max'):
    """Return an array that fits a cycle stats category to the shape of the LFP."""
    if arr is not None:
        if channel is None:
            return np.array([np.repeat(arr, cycle_stats[cycle_type]['duration'][channel])
                             for channel in range(len(arr))])
        else:
            return np.repeat(arr, cycle_stats[cycle_type]['duration'][channel])
    else:
        if channel is None:
            return np.array([np.repeat(cycle_stats[cycle_type][category][channel], cycle_stats[cycle_type]['duration'][channel])
                             for channel in range(len(cycle_stats[cycle_type][category]))])
        else:
            return np.repeat(cycle_stats[cycle_type][category][channel], cycle_stats[cycle_type]['duration'][channel])


def filter_lfp(lfp, 
               l_freq=None, 
               h_freq=None):
    """Bandpass filter the LFP using MNE.

    Parameters
    ----------
    lfp : ptsa.data.timeseries.TimeSeries
        An n_channels x n_timepoints TimeSeries of LFP data.
        Coords should be ['channel', 'time', 'samplerate'].
    l_freq : int or float
        Low cut-off frequency in Hz.
    h_freq : int or float
        High cut-off frequency in Hz.

    Returns
    -------
    lfp_filt : ptsa.data.timeseries.TimeSeries
        The filtered LFP data.
    """
    dat = mne.filter.filter_data(np.float64(lfp.copy().data), 
                                 sfreq=lfp.samplerate.data.tolist(), 
                                 l_freq=l_freq, h_freq=h_freq, verbose=False)
    lfp_filt = TimeSeries(dat, name=lfp.name, 
                          dims=['channel', 'time'],
                          coords={'channel': lfp.channel.data,
                                  'time': lfp.time.data,
                                  'samplerate': lfp.samplerate.data.tolist()})
    return lfp_filt


def filter_lfp_bands(lfp, bands, zscore_lfp=False):
    """Bandpass filter the LFP for 1+ bands.

    Parameters
    ----------
    lfp : ptsa.data.timeseries.TimeSeries
        An n_channels x n_timepoints TimeSeries of LFP data.
        Coords should be ['channel', 'time', 'samplerate'].
    bands : dict or OrderedDict
        Keys are the names of each band.
        Values are a list of (low, high) cutoff frequencies in Hz.
    zscore_lfp : bool
        If True, filtered LFP values are Z-scored across time, for each channel.

    Returns
    -------
    lfp_filt : OrderedDict
        Keys are the names of each band.
        Values are TimeSeries objects containing the filtered LFP.
    """
    lfp_filt = OrderedDict()
    for band_name, pass_band in bands.items():
        lfp_filt[band_name] = filter_lfp(lfp, pass_band[0], pass_band[1])
        if zscore_lfp:
            lfp_filt[band_name] = ((lfp_filt[band_name] - lfp_filt[band_name].mean(dim='time')) / lfp_filt[band_name].std(dim='time'))
    return lfp_filt

    
def get_cycle_stats(subj_sess,
                    bands=None,
                    interp_spikes=True,
                    notch_freqs=[60, 120],
                    zscore_lfp=True,
                    zscore_power=True,
                    phase_type='peaks', # troughs, peaks, or hilbert
                    power_percentile=25,
                    osc_length=3,
                    save_outputs=True,
                    output_dir='/scratch/dscho/unit_activity_and_hpc_theta/data/cycle_stats'):
    """Calculate phase, power, and cycle-by-cycle stats for bandpass-filtered LFPs.
    
    Cycles are determined by finding peaks or troughs in the filtered LFPs 
    (positive-to-negative or negative-to-positive derivatives, respectively).
    
    Parameters
    ----------
    subj_sess : str
        e.g. 'U367_env2'
    bands : OrderedDict()
        LFP bands to filter. Default bands are:
            low_theta (1-5 Hz)
            high_theta (5-10 Hz)
            alpha_beta (10-20 Hz)
            low_gamma (30-50 Hz)
            mid_gamma (70-90 Hz)
            high_gamma (90-110 Hz)
    interp_spikes : bool
        If True, spikes for each unit are removed from the preprocessed
        LFP by linear interpolation (-2 to 4ms surrounding each spike).
    notch_freqs : list
        Frequencies at which to notch-filter the LFP data during
        preprocessing. If notch_freqs=[] then no notch filtering is done.
    zscore_lfp : bool
        If True, filtered LFP values are Z-scored across time, for each channel.
    zscore_power : bool
        If True, Hilbert transform power values for filtered LFPs are
        Z-scored across time, for each channel.
    phase_type : str
        'peaks' to return phase values that are linearly interpolated
            (0 to 2pi) between identified peaks in the filtered LFPs.
        'troughs' to return phase values that are linearly interpolated
            (0 to 2pi) between identified troughs in the filtered LFPs.
        'hilbert' to return phase from the Hilbert transform of filtered
            LFPs. NOTE if 'hilbert' is chosen then cycle_stats are
            calculated between peaks in the filtered LFPs.
    power_percentile : int
        Percentile above which the mean power for a given cycle must exceed
        vs. all cycles in a channel (affects cycle_stats['power_mask'],
        cycle_stats['powdur_mask'], cycle_stats['osc_start_mask'], 
        and cycle_stats['osc_mask']).
    osc_length : int
        Number of consectutive cycles that must meet duration and power 
        criteria to be considered an oscillation. A cycle meets duration
        criteria if its duration is within the passband, and meets power
        criteria according to power_percentile.
    save_files : bool
        If True, cycle_stats is saved to the filename coded into this function.
    output_dir : bool
        Directory where cycle_stats is saved.
        
    Returns
    -------
    lfp_filt : OrderedDict() of ptsa.data.timeseries.TimeSeries
        Keys are the a priori frequency bands.
        Values are TimeSeries objects of the channel x time
        bandpass-filtered LFPs.
    phase : OrderedDict() of OrderedDict()
        Keys are the a priori frequency bands.
        Nested keys:
            'inds' : [xinds, yinds] identifying breaks between cycles
                in the filtered LFPs, where x refers to channel and y to time.
            'phase' : channel x time phase values by the method specified
                in phase_type.
    power : OrderedDict() of numpy.ndarray
        Keys are the a priori frequency bands.
        Values are channel x time power values from Hilbert transform
        of the filtered LFPs.
    cycle_stats : OrderedDict() of dict
        Keys are the a priori frequency bands.
        Nested keys:
            'power' : mean power for each cycle
            'power_rank' : cycle power rank, taken across time, for each channel
            'power_mask' : mask of power ranks > power_percentile
            'duration' : cycle duration, in number of timepoints
            'duration_hz' : cycle duration, in Hz
            'duration_mask' : mask of cycle durations within the passband
            'powdur_mask' : power_mask * duration_mask
            'osc_start_mask' : cycles at the start of an oscillation (of length=osc_length)
            'osc_mask' : cycles that are part of an oscillation
    """
    if bands is None:
        bands = OrderedDict([('low_theta', [1, 5]),
                             ('high_theta', [5, 10]),
                             ('alpha_beta', [10, 20]),
                             ('low_gamma', [30, 50]),
                             ('mid_gamma', [70, 90]),
                             ('high_gamma', [90, 110])])
    hpc_rois = ['LAH', 'LMH', 'LPH', 'RAH', 'RMH', 'RPH']
    
    # Get session info.
    subj_df = get_subj_df()
    
    # Load spikes.
    spikes, fr_df, clus_to_chan, chan_to_clus = load_spikes(subj_sess)

    # Load the raw LFP.
    ms_before = 2
    ms_after = 4
    lfp_raw, lfp_preproc = manning_analysis.process_lfp(subj_sess, 
                                                        subj_df=subj_df, 
                                                        notch_freqs=notch_freqs,
                                                        interpolate=interp_spikes,
                                                        session_spikes=spikes,
                                                        ms_before=ms_before,
                                                        ms_after=ms_after)
    sampling_rate = lfp_preproc.samplerate.data.tolist()
    
    # Bandpass filter the LFP.
    lfp_filt = filter_lfp_bands(lfp_preproc, 
                                bands=bands, 
                                zscore_lfp=zscore_lfp)
    
    # Use a derivative test to identify troughs and peaks in the filtered 
    # LFP, then linearly interpolate phase between cycles.
    if phase_type == 'hilbert':
        phase_type_ = 'peaks'
    else:
        phase_type_ = phase_type
    
    phase = get_phase_bands(lfp_filt, 
                            bands=bands, 
                            find=phase_type_, 
                            lims=[-np.pi, np.pi])
        
    # Get Hilbert transform of the LFP and derive mean power for each cycle.
    power = OrderedDict()
    for band_name, pass_band in bands.items():
        power[band_name], phase_hilbert = get_hilbert(lfp_filt[band_name], 
                                                      zscore_power=zscore_power)
        if phase_type == 'hilbert':
            phase[band_name]['phase'] = phase_hilbert
            
    # For each channel, get the mean power and duration of each cycle.
    cycle_stats = OrderedDict()
    for band_name, pass_band in bands.items():
        cycle_stats[band_name] = calc_cycle_stats(phase[band_name]['inds'], 
                                                  power=power[band_name], 
                                                  pass_band=pass_band, 
                                                  sampling_rate=sampling_rate, 
                                                  power_percentile=power_percentile, 
                                                  osc_length=osc_length)
        
    # Save cycle_stats.
    process_str = ''
    process_str += 'notch' + '-'.join(str(i) for i in notch_freqs) if notch_freqs else 'nonotch'
    process_str += '_spikeinterp-{}to{}ms'.format(ms_before, ms_after) if interp_spikes else '_nospikeinterp'
    process_str += '_phase-{}'.format(phase_type)
    process_str += '_{}osc-{}powpct'.format(osc_length, power_percentile)
    process_str += '_bands--' + '--'.join(['{}{}-{}'.format(key, val[0], val[1]) for key, val in bands.items()])
    if save_outputs:
        fpath = op.join(output_dir, '{}_cycle_stats_{}.pkl'.format(subj_sess, process_str))
        dio.save_pickle(cycle_stats, fpath)
        
    return lfp_filt, phase, power, cycle_stats


def get_phase_bands(lfp_filt, bands, find='peaks', lims=[-np.pi, np.pi]):
    """Return linearly interpolated phase for each frequency band."""
    phase = OrderedDict()
    for band_name, pass_band in bands.items():
        phase[band_name] = OrderedDict()
        phase[band_name]['inds'], phase[band_name]['phase'] = interp_phase(lfp_filt[band_name].data, 
                                                                           find=find, 
                                                                           lims=lims, 
                                                                           return_uphase=False)
    return phase
    

def get_subj_df(input_dir='/data3/scratch/dscho/frLfp/data/metadata'):
    """Return subj_df."""
    files = glob.glob(op.join(input_dir, 'subj_df_*.xlsx'))
    subj_df = pd.read_excel(files[0], converters={'chan': str})
    for f in files[1:]:
        subj_df = subj_df.append(pd.read_excel(f, converters={'chan': str}))
    subj_df = subj_df.loc[subj_df.location!='none']
    return subj_df


def get_hilbert(lfp, zscore_power=False):
    """Return Hilbert-transformed LFP power and phase.

    Parameters
    ----------
    lfp : ptsa.data.timeseries.TimeSeries
        An n_channels x n_timepoints TimeSeries of LFP data.
        Coords should be ['channel', 'time', 'samplerate'].
    zscore_power : bool
        If True, Hilbert transform power values are Z-scored across 
        time, for each channel.

    Returns
    -------
    power : numpy.ndarray
        Power of the complex-valued signal, with dims channel x time.
    phase : numpy.ndarray
        Phase (-np.pi to np.pi) of the complex-valued signal, 
        with dims channel x time.
    """
    info = mne.create_info(ch_names=lfp.channel.values.tolist(), 
                           sfreq=lfp.samplerate.data.tolist(), 
                           ch_types=['seeg']*len(lfp.channel.data))
    lfp_mne = mne.io.RawArray(lfp.copy().data, info, verbose=False)
    lfp_mne.apply_hilbert(envelope=False)
    dat = lfp_mne.get_data()
    power, phase = np.abs(dat), np.angle(dat)

    # Z-score power across time, for each channel.
    if zscore_power:
        power = ((power.T - power.T.mean(axis=0)) / power.T.std(axis=0)).T

    return power, phase


def inspect_session(subj_sess):
    """Generate a pdf file for the session to inspect channel quality.
    
    Five figures are made for each channel:
        1) 3.6 sec LFP
        2) 36 sec LFP
        3) 360 sec LFP
        4) whole session LFP
        5) whole session PSD curve
    """
    fname = op.join('/scratch/dscho/unit_activity_and_hpc_theta/figs/chan_inspection',
                         '{}_microLFP_notch60-120_channel_inspection.pdf'.format(subj_sess))
    pdf = PdfPages(fname)
    subj_df = get_subj_df()
    lfp_raw, lfp = manning_analysis.process_lfp(subj_sess, 
                                                subj_df=subj_df, 
                                                notch_freqs=[60, 120],
                                                interpolate=False)
    for ch in lfp.channel.data:
        # Plot the channel LFP for 3.6 secs. 
        fig, ax = plot_trace2(lfp.sel(channel=[ch]), start=60, duration=3.6, nwin=6, same_yaxis=True, 
                              linewidths=[0.5], alphas=[0.8], colors=['C1'], dpi=150)
        ax[0].set_title('{} | ch{} | 3.6 secs'.format(subj_sess, ch), fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close('all')
        
        # Plot the channel LFP for 36 secs. 
        fig, ax = plot_trace2(lfp.sel(channel=[ch]), start=60, duration=36, nwin=6, same_yaxis=True, 
                              linewidths=[0.2], alphas=[0.8], colors=['C3'], dpi=150)
        ax[0].set_title('{} | ch{} | 36 secs'.format(subj_sess, ch), fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close('all')
        
        # Plot the channel LFP for 360 secs. 
        fig, ax = plot_trace2(lfp.sel(channel=[ch]), start=60, duration=360, nwin=6, same_yaxis=True, 
                              linewidths=[0.1], alphas=[0.8], colors=['C2'], dpi=150)
        ax[0].set_title('{} | ch{} | 360 secs'.format(subj_sess, ch), fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close('all')
        
        # Plot the channel LFP for the whole session. 
        fig, ax = plot_trace2(lfp.sel(channel=[ch]), nwin=6, same_yaxis=True, 
                              linewidths=[0.08], alphas=[0.8], colors=['C0'], dpi=150)
        ax[0].set_title('{} | ch{} | whole session'.format(subj_sess, ch), fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close('all')
        
        # Plot the channel PSD.
        freq_means, P_means = neurodsp.spectral.psd(lfp.sel(channel=ch), Fs=2000)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
        ax = np.ravel(ax)
        ax[0].plot(np.log10(freq_means[1:201]), np.log10(P_means[1:201]), color='C7', linewidth=1.5)
        ax[0].set_xticks(np.linspace(np.log10(1), np.log10(200), 10))
        ax[0].set_xticklabels([int(round(10**x, 0)) for x in np.linspace(np.log10(1), np.log10(200), 10)])
        ax[0].set_xlabel('Frequency (Hz)', fontsize=14)
        ax[0].set_ylabel('log10(Power)', fontsize=14)
        ax[0].tick_params(axis='both', labelsize=12)
        ax[0].set_title('{} | ch{} PSD'.format(subj_sess, ch), fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close('all')
    pdf.close()
    return fname
    

def interp_phase(dat, find='peaks', lims=[-np.pi, np.pi], return_uphase=False):
    """Linearly interpolate phase between peaks or troughs.
    
    Parameters
    ----------
    dat : numpy.ndarray
        Bandpass-filtered LFP data with dims channel x time.
    find : str
        'minima' or 'troughs' to interpolate between troughs
            (identified by derivative sign change).
        'maxima' or 'peaks' to interpolate between peaks.
            (identified by derivative sign change).
        'extrema' to interpolate (separately) peak-to-trough 
            and trough-to-peak, with peaks and troughs
            identified by derivative sign change.
        'extrema2' to interpolate (separately) peak-to-trough
            and trough-to-peak, with peaks and troughs
            identified as local maxima and minima between 
            successive zero-crossings.
    lims : list or tuple
        Phase is interpolated between lims[0] and lims[1].
    return_uphase : bool
        If True, unwrapped phase is returned.
        
    Returns
    -------
    cycle_inds : list
        [xvals, yvals] of dat that mark the dividing indices
            between each cycle (i.e. identified peaks or troughs).
        If find=='extrema', [xmins, ymins, xmaxs, ymaxs] of 
            dat that mark the indices for both troughs and peaks.
        If find=='extrema2', [xvals, yvals] of dat that mark the
            dividing indices between peaks.
    phase : numpy.ndarray
        Wrapped phase values in the same shape as dat.
    phase_unwrapped : numpy.ndarray
        Unwrapped phase values in the same shape as dat.
        Only returned if uphase==True.
    """
    assert len(dat.shape) == 2
    
    phase = -555 * np.ones(dat.shape)
    default_val = -555
      
    if find == 'extrema2':
        # Find 0-crossing indices.
        xvals, yvals = np.where(np.diff(np.signbit(dat), axis=-1))
        # Loop over channels.
        for i in range(phase.shape[0]):
            yvals_ = yvals[xvals==i]
            # Find local minima and maxima within each cycle.
            extremas = [(np.argmin(x), np.argmax(x)) for x in np.split(dat[i, :], yvals_[1::2])]
            phase_vec = np.split(phase[i, :], yvals_[1::2])
            for ii in range(len(extremas)-1):
                min_ind, max_ind = extremas[ii]
                phase_vec[ii][min_ind] = 0
                phase_vec[ii][max_ind] = np.pi
                #phase_vec[ii][max_ind+1] = -np.pi
            phase_vec = np.concatenate(phase_vec)
            phase_vec[np.where(phase_vec==np.pi)[0]+1] = -np.pi
            phase[i, :] = phase_vec
        #phase[xvals, yvals] = 0
        phase[:, 0] = 0
        phase[:, -1] = 0
        cycle_inds = np.where(phase==np.pi)
        phase = interpolate(phase, default_val)
    elif find == 'extrema':
        xmins, ymins = signal.argrelextrema(dat, np.less, axis=1)
        xmaxs, ymaxs = signal.argrelextrema(dat, np.greater, axis=1)
        cycle_inds = [xmins, ymins, xmaxs, ymaxs]
        phase[:, 0] = lims[0]
        phase[:, -1] = lims[0]
        phase[xmaxs, ymaxs] = lims[1]
        phase[xmaxs, ymaxs+1] = lims[0]
        phase[xmins, ymins] = np.mean(lims)
        phase = interpolate(phase, default_val)
    else:
        if find in ('minima', 'troughs'):
            comparator = np.less
        elif find in ('maxima', 'peaks'):
            comparator = np.greater
        xvals, yvals = signal.argrelextrema(dat, comparator, axis=1)
        cycle_inds = [xvals, yvals]
        phase[:, 0] = lims[0]
        phase[:, -1] = lims[0]
        phase[xvals, yvals] = lims[1]
        phase[xvals, yvals+1] = lims[0]
        phase = interpolate(phase, default_val)
    
    if return_uphase:
        lim_range = np.abs(lims[1] - lims[0]) / 2
        phase_unwrapped = np.unwrap(phase - lims[0], lim_range)
        return cycle_inds, phase, phase_unwrapped
    else:
        return cycle_inds, phase


def interpolate(arr, default_val=0):
    """Linearly interpolate default_val values.
    
    Parameters
    ----------
    arr : numpy.ndarray
        Any 2D array. Interpolation is done over the
        second axis.
    default_val : int or float
        The value in arr that gets interpolated.
    
    Returns
    -------
    arr : numpy.ndarray
        The interpolated array, with matching
        dimensions to the input array.
    """
    for xind in range(arr.shape[0]):
        keep_inds = np.where(arr[xind, :]!=default_val)[0]
        fill_inds = np.where(arr[xind, :]==default_val)[0]
        f = interp1d(keep_inds, arr[xind, keep_inds], 
                     kind='linear', fill_value='extrapolate')
        arr[xind, fill_inds] = f(fill_inds)
    return arr


def load_spikes(subj_sess, spike_dir='/data3/scratch/dscho/frLfp/data/spikes'):
    """Return spikes, fr_df, clus_to_chan, chan_to_clus."""
    with open(op.join(spike_dir, '{}_session_spikes.pkl'.format(subj_sess)), 'rb') as f:
        spikes = pickle.load(f)
    fr_df, clus_to_chan, chan_to_clus = manning_analysis.get_fr_df(subj_sess, spikes)
    return spikes, fr_df, clus_to_chan, chan_to_clus
    
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def rolling_func(arr, f=np.all, window=3, right_fill=None):
    """Divide an array into rolling windows along the -1 axis, 
    and apply a function over the values in each window.
    
    Parameters
    ----------
    arr : numpy.ndarray
        The input array.
    f : function
        The function to apply (must take an axis argument).
    window : int
        The number of values in each window.
        
    Returns
    -------
    numpy.ndarray in the shape of arr if right_fill!=None,
    or with the selected axis equal to its original length
    minus (window-1) if right_fill==None.
    
    Example
    -------
    For arr=np.array([0, 1, 2, 3, 4]), f=np.sum, window=3, right_fill=None:
    
    1) The array is split into windows: [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    2) We take the sum over each tuple to get the return array: [3, 6, 9]
    3) If right_fill=0 then the return array is [1, 2, 3, 0, 0]
    """
    arr = f(rolling_window(arr, window), axis=-1)
    if right_fill is not None:
        arr = np.append(arr, np.ones(list(arr.shape[:-1]) + [window-1]) * right_fill, axis=-1)
    return arr
    

def rolling_window(a, window):
    """Return a rolling window."""
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def save_pickle(obj, fpath, verbose=True):
    """Save object as a pickle file."""
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    if verbose:
        print('Saved {}'.format(fpath))
        

def open_pickle(fpath):
    """Return object."""
    with open(fpath, 'rb') as f:
        obj = pickle.load(f)
    return obj

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def get_cycle_stats_DEPRECATED(subj_sess,
                    zscore_lfp=True,
                    zscore_power=True,
                    phase_type='peaks', # troughs, peaks, or hilbert
                    power_percentile=25,
                    osc_length=3,
                    save_outputs=True,
                    data_dir='/scratch/dscho/phase_precession/data'):
    """Calculate phase, power, and cycle-by-cycle stats for bandpass-filtered LFPs.
    
    Cycles are determined by finding peaks or troughs in the filtered LFPs 
    (positive-to-negative or negative-to-positive derivatives, respectively).
    
    LFP bands are:
        low_theta (1-5 Hz)
        high_theta (5-10 Hz)
        alpha_beta (10-20 Hz)
        low_gamma (30-50 Hz)
        mid_gamma (70-90 Hz)
        high_gamma (90-110 Hz)
    
    Parameters
    ----------
    subj_sess : str
        e.g. 'U367_env2'
    zscore_lfp : bool
        If True, filtered LFP values are Z-scored
        across time, for each channel.
    zscore_power : bool
        If True, Hilbert-calculated power values for filtered
        LFPs are Z-score across time, for each channel.
    phase_type : str
        'peaks' to return phase values that are linearly interpolated
            (0 to 2pi) between identified peaks in the filtered LFPs.
        'troughs' to return phase values that are linearly interpolated
            (0 to 2pi) between identified troughs in the filtered LFPs.
        'hilbert' to return phase from the Hilbert transform of filtered
            LFPs. NOTE if 'hilbert' is chosen then cycle_stats are
            calculated between peaks in the filtered LFPs.
    power_percentile : int
        Percentile above which the mean power for a given cycle must exceed
        vs. all cycles in a channel (affects cycle_stats['power_mask'],
        cycle_stats['powdur_mask'], cycle_stats['osc_start_mask'], 
        and cycle_stats['osc_mask']).
    osc_length : int
        Number of consectutive cycles that must meet duration and power 
        criteria to be considered an oscillation. A cycle meets duration
        criteria if its duration is within the passband, and meets power
        criteria according to power_percentile.
    save_files : bool
        If True, cycle_stats is saved to the path hard-coded into this function.
    data_dir : bool
        Data directory into which files are saved (in subdirs).
        
    Returns
    -------
    lfp_filt : OrderedDict() of ptsa.data.timeseries.TimeSeries
        Keys are the a priori frequency bands.
        Values are TimeSeries objects of the channel x time
        bandpass-filtered LFPs.
    phase : OrderedDict() of OrderedDict()
        Keys are the a priori frequency bands.
        Nested keys:
            'inds' : [xinds, yinds] identifying breaks between cycles
                in the filtered LFPs, where x refers to channel and y to time.
            'phase' : channel x time phase values by the method specified
                in phase_type.
    power : OrderedDict() of numpy.ndarray
        Keys are the a priori frequency bands.
        Values are channel x time power values from Hilbert transform
        of the filtered LFPs.
    cycle_stats : OrderedDict() of dict
        Keys are the a priori frequency bands.
        Nested keys:
            'power' : mean power for each cycle
            'power_rank' : cycle power rank, taken across time, for each channel
            'power_mask' : mask of power ranks > power_percentile
            'duration' : cycle duration, in number of timepoints
            'duration_hz' : cycle duration, in Hz
            'duration_mask' : mask of cycle durations within the passband
            'powdur_mask' : power_mask * duration_mask
            'osc_start_mask' : cycles at the start of an oscillation (of length=osc_length)
            'osc_mask' : cycles that are part of an oscillation
    """
    bands = OrderedDict([('low_theta', [1, 5]),
                     ('high_theta', [5, 10]),
                     ('alpha_beta', [10, 20]),
                     ('low_gamma', [30, 50]),
                     ('mid_gamma', [70, 90]),
                     ('high_gamma', [90, 110])])
    hpc_rois = ['LAH', 'LMH', 'LPH', 'RAH', 'RMH', 'RPH']
    
    # Get session info.
    files = glob.glob(op.join('/data3/scratch/dscho/frLfp/data/metadata', 'subj_df_*.xlsx'))
    subj_df = pd.read_excel(files[0], converters={'chan': str})
    for f in files[1:]:
        subj_df = subj_df.append(pd.read_excel(f, converters={'chan': str}))
    subj_df = subj_df.loc[subj_df.location!='none']
    
    # Load spikes.
    with open(op.join('/data3/scratch/dscho/frLfp/data/spikes', '{}_session_spikes.pkl'.format(subj_sess)), 'rb') as f:
        spikes = pickle.load(f)
    fr_df, clus_to_chan, chan_to_clus = manning_analysis.get_fr_df(subj_sess, spikes)
    units = fr_df.clus.tolist()

    # Load the raw LFP.
    df = subj_df.query("(subj_sess=='{}')".format(subj_sess))
    chans = df.chan.tolist()
    interp_spikes = True
    notch_freqs = [60, 120]
    lfp_raw, lfp_preproc = manning_analysis.process_lfp(subj_sess, subj_df, 
                                                        notch_freqs=notch_freqs,
                                                        interpolate=interp_spikes,
                                                        session_spikes=spikes,
                                                        ms_before=2,
                                                        ms_after=4)
    sampling_rate = lfp_preproc.samplerate.data.tolist()
    
    # Bandpass filter the LFP. 
    lfp_filt = OrderedDict()
    for band_name, pass_band in bands.items():
        dat = mne.filter.filter_data(np.float64(lfp_preproc.copy().data), 
                                     sfreq=sampling_rate, 
                                     l_freq=pass_band[0], h_freq=pass_band[1], verbose=False)
        lfp_filt[band_name] = TimeSeries(dat, name=subj_sess, 
                                         dims=['channel', 'time'],
                                         coords={'channel': lfp_preproc.channel.data,
                                                 'time': lfp_preproc.time.data,
                                                 'samplerate': sampling_rate})
        # Z-score each channel across time.
        if zscore_lfp:
            lfp_filt[band_name] = ((lfp_filt[band_name] - lfp_filt[band_name].mean(dim='time')) 
                                   / lfp_filt[band_name].std(dim='time'))

    # Use a derivative test to identify troughs and peaks in the filtered 
    # LFP, then linearly interpolate phase between cycles.
    lims = [-np.pi, np.pi] # [-np.pi, np.pi] or [0, 2*np.pi]
    phase = OrderedDict()
    if phase_type == 'hilbert':
        phase_type_ = 'peaks'
    else:
        phase_type_ = phase_type
    for band_name, pass_band in bands.items():
        phase[band_name] = OrderedDict()
        #phase[band_name]['inds'], phase[band_name]['phase'], phase[band_name]['uphase'] = interp_phase(lfp_filt[band_name].data, phase_type_, lims=lims)
        phase[band_name]['inds'], phase[band_name]['phase'] = interp_phase(lfp_filt[band_name].data, phase_type_, lims=lims, return_uphase=False)
                
    # Get Hilbert transform of the LFP and derive mean power for each cycle
    power = OrderedDict()
    for band_name, pass_band in bands.items():
        info = mne.create_info(ch_names=lfp_filt[band_name].channel.values.tolist(), 
                               sfreq=lfp_filt[band_name].samplerate.data.tolist(), 
                               ch_types=['seeg']*len(lfp_filt[band_name].channel.data))
        lfp_mne = mne.io.RawArray(lfp_filt[band_name].copy().data, info, verbose=False)
        lfp_mne.apply_hilbert(envelope=False)
        dat = lfp_mne.get_data()
        power[band_name], phase_hilbert = np.abs(dat), np.angle(dat)
        if phase_type == 'hilbert':
            phase[band_name]['phase'] = phase_hilbert
            #phase[band_name]['uphase'] = np.unwrap(phase_hilbert)

        # Z-score power across time, for each channel.
        if zscore_power:
            power[band_name] = ((power[band_name].T - power[band_name].T.mean(axis=0)) / power[band_name].T.std(axis=0)).T
    
    # For each channel, get the mean power and duration of each cycle.
    cycle_stats = OrderedDict()
    for band_name, pass_band in bands.items():
        xinds, yinds = phase[band_name]['inds']
        cycle_pows = [] # mean power Z-score for each cycle
        cycle_pow_ranks = [] # rank of mean power Z-score for each cycle
        cycle_pow_ma = [] # bool; whether cycle power exceeds the power percentile
        cycle_durs = [] # duration of each cycle, in number of timepoints
        cycle_durs_hz = [] # duration of each cycle, in Hz
        cycle_durs_ma = [] # bool; whether cycle length is within the passband
        cycle_powdur_ma = [] # bool; cycle_durs_ma * cycle_pow_ma
        cycle_osc_start_ma = [] # bool; cycles at the start of an oscillation (of length = osc_length)
        cycle_osc_ma = [] # bool; cycles that are part of an oscillation
        for xind in np.unique(xinds):
            yinds_ = yinds[xinds==xind]

            # Get mean Z-power and ranked power for each cycle.
            split_pows = np.split(power[band_name][xind, :], yinds_)
            cycle_pows.append(np.array([np.mean(arr) for arr in split_pows]))
            xsort = cycle_pows[-1].argsort()
            cycle_pow_ranks_ = np.empty(xsort.shape)
            cycle_pow_ranks_[xsort] = np.arange(len(cycle_pow_ranks_))
            cycle_pow_ranks.append(cycle_pow_ranks_)

            # Get duration (in timepoints and Hz) for each cycle.
            cycle_durs.append(np.array([len(arr) for arr in split_pows]))
            cycle_durs_hz.append(np.array([sampling_rate / len(arr) for arr in split_pows]))

            # Mask cycles according to power and duration criteria.
            cycle_pow_ma.append(cycle_pow_ranks_ > np.percentile(cycle_pow_ranks_, power_percentile))
            cycle_durs_ma.append((cycle_durs_hz[-1] > pass_band[0]) * (cycle_durs_hz[-1] < pass_band[1]))
            cycle_powdur_ma.append(cycle_durs_ma[-1] * cycle_pow_ma[-1])

            # Get oscillation masks.
            cycle_osc_start_ma.append(rolling_func(cycle_powdur_ma[-1], np.all, window=osc_length, right_fill=False))
            osc_cycles = np.zeros(len(cycle_osc_start_ma[-1]), dtype=np.bool_)
            osc_starts = np.where(cycle_osc_start_ma[-1])[0]
            for i in range(osc_length):
                osc_cycles[osc_starts+i] = True
            cycle_osc_ma.append(osc_cycles)

        cycle_stats[band_name] = {'power': cycle_pows,
                                  'power_rank': cycle_pow_ranks,
                                  'power_mask': cycle_pow_ma,
                                  'duration': cycle_durs,
                                  'duration_hz': cycle_durs_hz,
                                  'duration_mask': cycle_durs_ma,
                                  'powdur_mask': cycle_powdur_ma,
                                  'osc_start_mask': cycle_osc_start_ma,
                                  'osc_mask': cycle_osc_ma}
    if save_outputs:
        # Save cycle_stats.
        fname = op.join(data_dir, 'cycle_stats', '{}_cycle_stats_{}_{}powpct_{}osclength.pkl'
                             .format(subj_sess, phase_type_, power_percentile, osc_length))
        with open(fname, 'wb') as f:
            pickle.dump(cycle_stats, f, pickle.HIGHEST_PROTOCOL)
        print('Saved cycle_stats as {}'.format(fname))
        
    return lfp_filt, phase, power, cycle_stats