"""
manning_utils.py

Author:
    Daniel Schonhaut
    Computational Memory Lab
    University of Pennsylvania
    daniel.schonhaut@gmail.com
    
Description: 
    A group of simple, common functions used in code for replicating/extending
    the Manning et al., J Neurosci 2009 paper.

Last Edited: 
    11/5/18
"""
import os
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

def epoch_vec(v, func=np.mean, 
              start_stop=None, n_bins=None, 
              epoch_size=None, cut=None):
    """Divide a vector into equal-size epochs, and
    perform a function over the values in each epoch.
    
    Either start_stop and num_bins must be passed,
    OR epoch_size and cut must be passed, in which case
    start_stop and num_bins will be determined from these
    inputs. The function runs faster if start_stop and
    num_bins are passed so these arguments take precedent.
    
    Parameters
    ----------
    v : numpy.ndarray
        The input vector that will be epoched.
    func : function
        A function to perform over the values in each epoch
        (e.g. np.mean, np.sum, np.max).
    start_stop : list or tuple
        (start, stop) indices such that v_out
        consists of data from v[start:stop]
    n_bins : int
        The number of epochs to divide v into
        (i.e. length of the output vector).
    epoch_size : int
        Number of values in each epoch.
    cut : int
        Number of epochs to cut from the start and end of v.
    
    Returns
    -------
    v_out : numpy.ndarray
        A vector of epoched values.
    start_stop : tuple
        (start, stop) indices such that v_out
        consists of data from v[start:stop]
    """
    if (start_stop is None) and (n_bins is None):
        v_inds = np.arange(len(v))
        epoch_borders = v_inds[v_inds % epoch_size == 0]
        start_ind = epoch_borders[cut]
        stop_ind = epoch_borders[len(epoch_borders) - (cut+1)]
        start_stop = (start_ind, stop_ind)
        n_bins = len(epoch_borders) - (2*cut+1)
    else:
        start_ind, stop_ind = start_stop
    
    v_out = np.apply_over_axes(func, 
                               np.array(np.split(v[start_ind:stop_ind], n_bins)), 
                               axes=1).squeeze()
    
    return v_out, start_stop
    
def get_config(root='/'):
    """Return a dictionary of config params for the Manning replication project.
    
    Parameters
    ----------
    root : str
        Root directory for rhino (e.g. /Volumes/rhino if rhino is mounted to macOS X).
    """
    def convert_to_num(x):
        if x.find(',') > 0:
            try:
                x_new = [float(x) for x in x.split(', ')]
            except ValueError:
                x_new = x.split(', ')
        else:
            try:
                x_new = float(x)
            except ValueError:
                x_new = x
        return x_new
        
    f = os.path.join(root, 'home1/dscho/code/projects/manning_replication/data_files',
                     'fl_config.csv')
    dat = pd.read_csv(f).set_index('field')
    dat.value = dat.value.apply(convert_to_num)
    dat = dat.value.to_dict()
    dat['freqParam'] = '{}Bands-{}-{}'.format(len(dat['freqBinLbl']), 
                                              int(dat['freQ'][0]), int(dat['freQ'][1]))
    dat['configID'] = ('{}-multiUnit{}-remSpikes{}({}:{})-{}'
                       .format(int(dat['analNum']), int(dat['multiUnitFlag']), 
                               int(dat['removeSpikesFlag']), int(-dat['ms_before']),
                               int(dat['ms_after']), dat['freqParam']))                              
    return dat
    
def get_dirs(root='/'):
    """Return a dictionary of directory paths for the Manning replication project.
    
    Parameters
    ----------
    root : str
        Root directory for rhino (e.g. /Volumes/rhino if rhino is mounted to macOS X).
    """
    dirs = {'raw_data': os.path.join(root, 'data/continuous'),
            'scratch': os.path.join(root, 'data3/scratch/dscho/frLfp')}
    dirs['analyses'] = os.path.join(dirs['scratch'], 'analyses')
    dirs['data'] = os.path.join(dirs['scratch'], 'data')
    dirs['figs'] = os.path.join(dirs['scratch'], 'figs')
    dirs['lockDir'] = os.path.join(dirs['scratch'], 'lockDir')
    dirs['Session_Spikes'] = os.path.join(dirs['scratch'], 'Session_Spikes')
    dirs['uclaSessSpikes'] = os.path.join(dirs['scratch'], 'uclaSessSpikes')
    return dirs

def get_epochs(time, epoch_size=1000, cut=3):
    """Divide a time dimension into evenly spaced epochs.
    
    Parameters
    ----------
    time : numpy.ndarray
        Array of timepoints of length n_samples.
    epoch_size : int
        Number of timepoints in each epoch.
    cut : int
        Number of epochs to cut from the start and end of time.

    Returns
    -------
    time_bins : list[list]
        A list of start and stop timepoints for each epoch.
    """
    epoch_borders = time[time % epoch_size == 0]
    time_bins = []
    time_start = cut
    time_stop = len(epoch_borders) - (cut+1)
    for i in np.arange(time_start, time_stop):
        bin_start = epoch_borders[i]
        bin_stop = epoch_borders[i+1]
        time_bins.append([int(bin_start), int(bin_stop)])
    return time_bins

def get_freqs(low=2, high=150, num=50):
    """Return an array of log-spaced frequencies and an OrderedDict of bands 
    that contain them.
    
    Parameters
    ----------
    low : int or float
        The lowest frequency.
    high : int or float
        The highest frequency.
    num : int
        The number of frequencies to return.
        
    Returns
    -------
    freqs : numpy.ndarray
        An array of log-spaced frequencies between low and high.
    freq_bands : OrderedDict
        Frequency bands that each contain subsets of freqs.
    """
    freqs = np.logspace(np.log10(low), np.log10(high), num, base=10)
    freq_bands = OrderedDict()
    freq_bands['delta'] = freqs[(freqs>low-0.01) & (freqs<4)]
    freq_bands['theta'] = freqs[(freqs>=4) & (freqs<8)]
    freq_bands['alpha'] = freqs[(freqs>=8) & (freqs<12)]
    freq_bands['beta'] = freqs[(freqs>=12) & (freqs<30)]
    freq_bands['gamma'] = freqs[(freqs>=30) & (freqs<high+0.01)]
    return freqs, freq_bands
     
def get_gwin(fwhm, sampling_rate=2000, width=5):
    """Return a Gaussian window that sums to 1.
    
    Parameters
    ----------
    fwhm : int or float
        Desired FWHM of the Gaussian window, in Hz.
    sampling_rate : int or float
        Sampling rate to size the kernel to, in Hz.
    width : int or float
        Number of standard deviations that the kernel
        extends from the mean.
    """
    g_fwhm = sampling_rate / fwhm
    g_sigma = g_fwhm / np.sqrt(8 * np.log(2))
    v = np.zeros(int((g_sigma * width * 2) + 1))
    v[int(len(v)/2)] = 1
    g_win = gaussian_filter(v, g_sigma)
    return g_win
        
def get_subjs(root='/'):
    """Return a dictionary of values for each subj in the Manning replication project.
    
    Parameters
    ----------
    root : str
        Root directory for rhino (e.g. /Volumes/rhino if rhino is mounted to macOS X).
    """ 
    root = '/'
    f = os.path.join(root, 'home1/dscho/code/projects/manning_replication/data_files',
                     'fl_subjList.csv')
    dat = pd.read_csv(f)
    dat.expType = dat.expType.apply(lambda x: x.split(', '))
    dat.sess = dat.sess.apply(lambda x: x.split(', '))
    dat = dat.set_index('subj').to_dict('index')
    return dat
    
def get_wavelet_len(freq, width):
    """Return ptsa wavelet length in ms, given frequency and wave number."""
    sigma = width / (2 * np.pi * freq)
    return 3.5 * sigma