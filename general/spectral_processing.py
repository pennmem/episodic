"""
spectral_processing.py

Author:
    Daniel Schonhaut
    Computational Memory Lab
    University of Pennsylvania
    daniel.schonhaut@gmail.com

Description: 
    Functions for processing EEG data.

Last Edited: 
    4/21/21
"""
from collections import OrderedDict
import mkl
mkl.set_num_threads(1)
import numpy as np
import mne
from ptsa.data.TimeSeriesX import TimeSeries 


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
