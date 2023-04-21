"""
pal_reinstatement.py

Author
------
Daniel Schonhaut
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com

Description
-----------
Functions for performing encoding/retrieval reinstatement analyses
in paired-associates learning.

Last Edited
-----------
8/31/20
"""
import sys
import os
from time import sleep
from collections import OrderedDict as od

import mkl
mkl.set_num_threads(1)
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
import astropy.stats.circstats as circstats
import pycircstat

from ptsa.data.TimeSeriesX import TimeSeries
from ptsa.data.filters import MorletWaveletFilter
from cmlreaders import CMLReader

sys.path.append('/home1/dscho/code/general')
import data_io as dio
import array_operations as aop
sys.path.insert(0, '/home1/esolo/notebooks/codebases/')
import CML_stim_pipeline
from loc_toolbox import update_pairs


def calc_plv(arr1, arr2, axis=-1):
    """Return the mean phase and phase-locking value between two arrays.
    
    PLV is the mean resultant length of the circular distances between
    corresponding arr1 and arr2 elements.
    
    Parameters
    ----------
    arr1 : np.ndarray
        Phase array of any number of dimensions.
    arr2 : np.ndarray
        Phase array of equal shape as arr1.
    axis : int
        Axis to calculate the PLV over. Axis=None will calculate PLV 
        over the flattened arrays and return a single number; otherwise 
        the output is an array of PLVs whose shape matches the input 
        arrays, minus the dimension that PLV is calculated over.
        
    Returns
    -------

    phase : np.float64 or np.ndarray
        Mean phase or an array of mean phases if the input arrays have
        more than one dimension and axis is not None.
    length : np.float64 or np.ndarray
        PLV or an array of PLVs if the input arrays have more than one
        dimension and axis is not None.
    """
    phase, length = circstats.circmoment(pycircstat.descriptive.cdiff(arr1, arr2), axis=axis)
    return phase, length


def desc_events(events,
                verbose=True):
    """Describe behavioral events."""
    correct_col = None
    for col in ['correct', 'iscorrect']:
        if col in events.columns:
            correct_col = col
            break
    
    output = od([])
    output['n_events'] = len(events)
    output['n_lists'] = len([i for i in pd.unique(events['list']) if (i>0)])
    output['n_study_pairs'] = len(events.query("(type=='STUDY_PAIR')"))
    output['n_test_probes'] = len(events.query("(type=='TEST_PROBE')"))
    if output['n_test_probes'] == 0:
        output['n_test_probes'] = len(events.query("(type=='PROBE_START')"))
    output['n_rec_events'] = len(events.query("(type=='REC_EVENT')"))
    if correct_col is not None:
        output['n_correct'] = np.sum(events.query("(type=='REC_EVENT')")[correct_col])
        output['pct_correct'] = 100 * (output['n_correct']/output['n_test_probes'])
    if 'n_pass' in events.columns:
        output['n_pass'] = np.sum(events.query("(type=='REC_EVENT')")['resp_pass'])
        output['pct_pass'] = 100 * (output['n_pass']/output['n_test_probes'])
    if 'intrusion' in events.columns:
        output['n_intrusions'] = len(events.query("(type=='REC_EVENT') & (intrusion!=0)"))
        output['pct_intrusions'] = 100 * (output['n_intrusions']/output['n_test_probes'])
    
    if verbose:
        for k, v in output.items():
            if k[0] == 'n':
                print('{} : {}'.format(k, v))
            elif k[:3] == 'pct':
                print('{} : {:.1f}%'.format(k, v))
            else:
                print('{} : {}'.format(k, v))
        print('')
        
        print(aop.unique(events['type'], sort=False), end='\n\n')
        
        study_list_isis(events)

    return output


def elec_pair_dist(elecs, idx1, idx2):
    """Return the Euclidean distance between two electrodes.
    
    Parameters
    ----------
    elecs : pd.DataFrame
        Each row contains info on one electrode. 'ind.x', 'ind.y',
        and 'ind.z' columns hold the electrode coordinates.
    idx1 : int
        Selects the first row from elecs using iloc
    idx2 : int
        Selects the second row from elecs using iloc
    """
    row1 = elecs.iloc[idx1]
    row2 = elecs.iloc[idx2]
    
    roi1_loc = np.array((row1['ind.x'], row1['ind.y'], row1['ind.z']))
    roi2_loc = np.array((row2['ind.x'], row2['ind.y'], row2['ind.z']))
    
    return np.linalg.norm(roi1_loc - roi2_loc)


def epoch_pows(ts,
               freqs,
               freq_bands,
               timebin_inds,
               timebin_size_samp=250, # samples
               width=6,
               log_power=True, 
               overwrite=False,
               savedir='/home1/dscho/projects/theta_reinstatement/data/morlet/power',
               output_ftag='',
               verbose=False):
    """Return epoched powers within each trial.
    
    Output array shape is trial x chan x freq x time.
    """
    # Get the output file name.
    fstr = ('-width{}-{:.0f}to{:.0f}Hz-{}freqs'
            .format(width, min(freqs), max(freqs), len(freqs)))
    if log_power:
        fstr += '-log10'
    fname = '{}-power{}-{}.pkl'.format(ts.name, fstr, output_ftag)
    power_file = os.path.join(savedir, fname)
    # Load the output file if it exists.
    if os.path.exists(power_file) and not overwrite:
        if verbose:
            print('Loading power:\n\t{}'.format(power_file))
        power = dio.open_pickle(power_file)
        return power
        
    # Wavelet filter the EEG data.
    if verbose:
        print('Calculating power...')
    power = MorletWaveletFilter(ts, freqs, width=width, output=['power'], verbose=False).filter()
    
    # Convert data type.
    power = power.data.astype(np.float32)
    
    # Log transform power values.
    if log_power:
        power = np.log10(power)
        
    # Average powers within each frequency band.
    power = np.array([np.mean(power[np.where((freqs>=v[0]) & (freqs<v[1]))[0], :, :, :], axis=0)
                      for v in freq_bands.values()])
        
    # Transpose data shape.
    power = power.transpose(1, 2, 0, 3) # trial x chan x freq x time
    
    # Calculate mean power within each time window.
    if verbose:
        print('Making power time bins...')
    power = np.mean(aop.rolling_window(power, timebin_size_samp)[:, :, :, timebin_inds, :], axis=-1) # trial x chan x freq x time_window
    
    # Save the data.
    if savedir:
        dio.save_pickle(power, power_file, verbose)
        
    return power
    

def get_elec_pairs(elecs):
    """Return DataFrame with all electrode pairs.
    
    Parameters
    ----------
    elecs : pd.DataFrame
        The input DataFrame with individual electrode 
        contacts from which electrode pairs are drawn.
    """
    n_elec = elecs.shape[0]
    idx_pairs = np.array([(x, y) for x in np.arange(n_elec) for y in np.arange(n_elec) if x<y])
    rois = idx_pair_vals(elecs, idx_pairs, 'roi', use_at=False)
    hems = idx_pair_vals(elecs, idx_pairs, 'hem', use_at=False)
    dists = np.array([elec_pair_dist(elecs, idx_pairs[iElec, 0], idx_pairs[iElec, 1])
                      for iElec in range(idx_pairs.shape[0])])
    subj = elecs.iloc[0]['subject']
    sess = elecs.iloc[0]['session']
    elec_pairs = pd.DataFrame(od([('subject', [subj for x in range(idx_pairs.shape[0])]),
                                  ('session', [sess for x in range(idx_pairs.shape[0])]),
                                  ('idx_elec1', idx_pairs[:, 0]),
                                  ('idx_elec2', idx_pairs[:, 1]),
                                  ('roi_elec1', rois[:, 0]),
                                  ('roi_elec2', rois[:, 1]),
                                  ('hem_elec1', hems[:, 0]),
                                  ('hem_elec2', hems[:, 1]),
                                  ('distance', dists)]))
    return elec_pairs


def get_epoch_inds(wins_ms,
                   wins_full_ms,
                   sr,
                   buff_ms,
                   spacing_ms=100,
                   verbose=True): 
    """Return evenly-spaced indices for epoch time windows.

    Indices capture the left-most points in each time window.

    Epochs are presumed to incude the buffer.
    """
    buff_samp = np.int(buff_ms * (sr/1000))
    spacing_samp = np.int(spacing_ms * (sr/1000))
    
    epoch_inds = od([])
    for k in wins_ms:
        win_size_samp = np.int((wins_ms[k][1] - wins_ms[k][0]) * (sr/1000))
        epoch_inds[k] = buff_samp + np.arange(0, win_size_samp, spacing_samp)
                          
    epoch_times = od([('enc', (np.arange(np.int(wins_full_ms['enc'][0] * (sr/1000)), 
                                         np.int(wins_full_ms['enc'][1] * (sr/1000)))[epoch_inds['enc']] * (1000/sr)).astype(np.int)),
                      ('rec', (np.arange(np.int(wins_full_ms['rec'][0] * (sr/1000)), 
                                         np.int(wins_full_ms['rec'][1] * (sr/1000)))[epoch_inds['rec']] * (1000/sr)).astype(np.int))])
                  
    if verbose:
        for k in epoch_inds:
            print('{}: {} epochs ({}, {}, ..., {})'
                  .format(k, len(epoch_inds[k]), epoch_inds[k][0], 
                          epoch_inds[k][1], epoch_inds[k][-1]))
                                                 
    return epoch_inds, epoch_times


def get_event_pairs(events, 
                    list_='all', 
                    pair_type='all'):
    """Return DataFrame with all event pairs for the specified list or lists.
    
    Parameters
    ----------
    events : pd.DataFrame
        The behavioral events.
    list_ : scalar
        The word list to operate on (events['list'] values). 
        If 'all', a DataFrame is returned with all within-list pairs of the
        specified pair_type for lists > 0.
    pair_type : str
        'enc-rec' pairs each encoding item with each retrieval item.
        'enc-enc' pairs each encoding item with each other encoding item.
        'rec-rec' pairs each recall item with each other recall item.
        'all' includes each of the above categories.
    
    Returns
    -------
    pd.DataFrame
    """
    if list_ == 'all':
        lists = np.array([i for i in pd.unique(events['list']) if (i>0)])
        event_pairs = pd.concat([get_event_pairs(events, list_) for list_ in lists])
        return event_pairs
    
    enc_inds = events.loc[(events['list']==list_) & (events['type']=='STUDY_PAIR')].index.tolist()
    rec_inds = events.loc[(events['list']==list_) & (events['type']=='REC_EVENT')].index.tolist()
    if (len(enc_inds) == 0) or (len(rec_inds) == 0):
        return None
    
    # Get the index pairs.
    if pair_type == 'enc-rec':
        idx_pairs = np.array([(x, y) for x in enc_inds for y in rec_inds])
        idx_labs = np.array(['enc-rec' for _ in idx_pairs])
    elif pair_type == 'enc-enc':
        idx_pairs = np.array([(x, y) for x in enc_inds for y in enc_inds if x>y])
        idx_labs = np.array(['enc-enc' for _ in idx_pairs])
    elif pair_type == 'rec-rec':
        idx_pairs = np.array([(x, y) for x in rec_inds for y in rec_inds if x>y])
        idx_labs = np.array(['rec-rec' for _ in idx_pairs])
    elif pair_type == 'all':
        idx_pairs = np.concatenate((np.array([(x, y) for x in enc_inds for y in rec_inds]),
                                    np.array([(x, y) for x in enc_inds for y in enc_inds if x<y]),
                                    np.array([(x, y) for x in rec_inds for y in rec_inds if x<y])))
        idx_labs = np.concatenate((['enc-rec' for _ in np.array([(x, y) for x in enc_inds for y in rec_inds])],
                                   ['enc-enc' for _ in np.array([(x, y) for x in enc_inds for y in enc_inds if x<y])],
                                   ['rec-rec' for _ in np.array([(x, y) for x in rec_inds for y in rec_inds if x<y])]))
    
    # Calculate the serial position lag between each (enc, rec) pair.
    # Lag == 0 means we are comparing a probe to its actual study pair.
    # Lag < 0 means we are comparing a probe to a study pair that
    #         appeared before the probe in the study list.
    # Lag > 0 means we are comparing a probe to a study pair that 
    #         appeared after the probe in the study list.
    serial_pos = idx_pair_vals(events, idx_pairs, 'serialpos')
    serial_pos_lag = serial_pos[:, 0] - serial_pos[:, 1]
    probe_pos = idx_pair_vals(events, idx_pairs, 'probepos')
    probe_pos_lag = probe_pos[:, 0] - probe_pos[:, 1]
    
    eegoffsets = idx_pair_vals(events, idx_pairs, 'eegoffset')
    correct_events = idx_pair_vals(events, idx_pairs, 'correct')
    subj = events.iloc[0]['subject']
    sess = events.iloc[0]['session']
    
    event_pairs = pd.DataFrame(od([('subject', [subj for x in range(idx_pairs.shape[0])]),
                                   ('session', [sess for x in range(idx_pairs.shape[0])]),
                                   ('list', [list_ for x in range(idx_pairs.shape[0])]),
                                   ('idx_enc', idx_pairs[:, 0]),
                                   ('idx_rec', idx_pairs[:, 1]),
                                   ('pair_type', idx_labs),
                                   ('eegoffset_enc', eegoffsets[:, 0]),
                                   ('eegoffset_rec', eegoffsets[:, 1]),
                                   ('serialpos_enc', serial_pos[:   , 0]),
                                   ('serialpos_rec', serial_pos[:, 1]),
                                   ('serialpos_lag', serial_pos_lag),
                                   ('probepos_enc', probe_pos[:, 0]),
                                   ('probepos_rec', probe_pos[:, 1]),
                                   ('probepos_lag', probe_pos_lag),
                                   ('correct_enc', correct_events[:, 0]),
                                   ('correct_rec', correct_events[:, 1]),
                                   ('intrusion_rec', idx_pair_vals(events, idx_pairs, 'intrusion')[:, 1]),
                                   ('resp_pass', idx_pair_vals(events, idx_pairs, 'resp_pass')[:, 1])]))
    
    return event_pairs
    

def get_trial_wins(enc_start=-1000, # ms
                   enc_stop=4000,
                   rec_start=-4000,
                   rec_stop=1000,
                   sr=500,
                   buff_ms=None,
                   lowest_freq=2, # Hz
                   n_cycles=6,
                   verbose=True):
    """Return EEG analysis windows for encoding and retrieval trials.

    Outputs are returned in both ms and no. of samples,
    with trial windows that do and do not include a buffer.

    lowest_freq and n_cycles are disregarded if the buffer length
    is prespecified (buff_ms).
    """
    # Define the size of the analysis window for each encoding and 
    # retrieval event.
    wins_ms = od([('enc', (enc_start, enc_stop)),
                  ('rec', (rec_start, rec_stop))])
    
    wins_samp = od([])
    for k in wins_ms:
        wins_samp[k] = np.rint(np.array(wins_ms[k]) * sr/1000).astype(np.int)

    # Add a buffer around each analysis window for doing filtering.
    # If no buffer was specified we calculate it as half the length of
    # the wavelet filter.
    if buff_ms is None:
        buff_ms = np.int((n_cycles * 1000/lowest_freq) / 2)
    buff_samp = np.int(buff_ms * (sr/1000))

    wins_full_ms = od([])
    for k, v in wins_ms.items():
        wins_full_ms[k] = (wins_ms[k][0]-buff_ms, wins_ms[k][1]+buff_ms)
    
    wins_full_samp = od([])
    for k in wins_full_ms:
        wins_full_samp[k] = np.rint(np.array(wins_full_ms[k]) * sr/1000).astype(np.int)

    # Organize the outputs.
    trial_wins = od([('wins_ms', wins_ms),
                     ('wins_full_ms', wins_full_ms),
                     ('buff_ms', buff_ms),
                     ('wins_samp', wins_samp),
                     ('wins_full_samp', wins_full_samp),
                     ('buff_samp', buff_samp)])

    if verbose:
        print('Buffer is {} ms; {} samples'
              .format(trial_wins['buff_ms'], trial_wins['buff_samp']), end='\n\n')
        print('Without buffer:')
        for k in wins_ms:
            print('{}: {} to {} ms'
                  .format(k, trial_wins['wins_ms'][k][0], trial_wins['wins_ms'][k][1]))
        print('\nWith buffer:')
        for k in wins_ms:
            print('{}: {} to {} ms'
                  .format(k, trial_wins['wins_full_ms'][k][0], trial_wins['wins_full_ms'][k][1]))
        
    return trial_wins


def idx_pair_vals(df, 
                  idx_pairs, 
                  col_name,
                  use_at=True):
    """Return column values for a list of index pairs.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame that we are getting values from.
    idx_pairs : np.ndarray
        An n x 2 array of index pairs.
    col_name : str
        The df column to grab values from.
    use_at : bool
        If True, uses df.at[] to find explicit indices.
        If False, uses df.iat[] to find positional indices.
        
    Returns
    -------
    np.ndarray
        An array in the shape of idx_pairs
        containing the corresponding col_name data.
    """
    if use_at:
        return np.array([[df.at[idx_pairs[iPair, 0], col_name], 
                          df.at[idx_pairs[iPair, 1], col_name]]
                         for iPair in range(idx_pairs.shape[0])])
    else:
        iCol = list(df.columns).index(col_name)
        return np.array([[df.iat[idx_pairs[iPair, 0], iCol], 
                          df.iat[idx_pairs[iPair, 1], iCol]]
                         for iPair in range(idx_pairs.shape[0])])


def power_sim(enc_pow, 
              rec_pow):
    """Return cos sim matrix for an encoding/recall pair.
    
    Parameters
    ----------
    enc_pow : np.array 
        chan x freq x time array of powers
        for a single encoding event.
    rec_pow : np.array 
        chan x freq x time array of powers
        for a single recall event.
        
    Returns
    -------
    enc_time x rec_time array of cosine similarities
    (calculated across chan-freq features)
    """
    # Reshape the arrays.
    enc_pow = np.transpose(enc_pow, (2, 0, 1)) # time x chan x freq
    enc_pow = enc_pow.reshape((enc_pow.shape[0], np.prod(enc_pow.shape[1:]))) # time x chan-freq feature
    rec_pow = np.transpose(rec_pow, (2, 0, 1))
    rec_pow = rec_pow.reshape((rec_pow.shape[0], np.prod(rec_pow.shape[1:]))) # time x chan-freq feature

    # Return the cosine similarities across features between
    # each pair of encoding and retrieval times.
    return cosine_similarity(enc_pow, rec_pow)
    

def reader_load_eeg(reader, 
                    pairs,
                    events=None, 
                    wins_full_ms=None, 
                    trial_type=['enc', 'rec'], 
                    list_='all', 
                    presumed_sr=None,
                    verbose=True):
    """Return epoched eeg for encoding/retrieval events.
    
    If only the reader is passed, returns eeg across all
    channels for the whole session.
    """
    if events is None:
        eeg = reader.load_eeg(scheme=pairs)
        eeg.data = np.squeeze(eeg.data)
        if verbose:
            print('{} channels'.format(eeg.data.shape[0]), 
                  '{} timepoints'.format(eeg.data.shape[1]),
                  sep='\n', end='\n\n')
            print('{} Hz sr'.format(eeg.samplerate))
        if presumed_sr:
            assert presumed_sr == eeg.samplerate
        return eeg    
    
    # Format inputs.
    if isinstance(trial_type, str):
        trial_type = [trial_type]
    if list_ == 'all':
        lists = np.array([i for i in pd.unique(events['list']) if (i>0)])
    else:
        lists = [list_]
        
    # Get the EEG.
    eeg = od([])
    if 'enc' in trial_type:
        eeg['enc'] = reader.load_eeg(events=events.loc[(events['type']=='STUDY_PAIR') & (np.isin(events['list'], lists))], 
                                     rel_start=wins_full_ms['enc'][0], rel_stop=wins_full_ms['enc'][1],
                                     scheme=pairs)
    if 'rec' in trial_type:
        eeg['rec'] = reader.load_eeg(events=events.loc[(events['type']=='REC_EVENT') & (np.isin(events['list'], lists))],
                                     rel_start=wins_full_ms['rec'][0], rel_stop=wins_full_ms['rec'][1],
                                     scheme=pairs)
    for k in eeg:
        eeg[k].data = np.squeeze(eeg[k].data)
            
    if verbose:
        for k in eeg:
            print('{}:'.format(k))
            if len(eeg[k].data.shape) == 2:
                print('{} channels'.format(eeg[k].data.shape[0]), 
                      '{} timepoints'.format(eeg[k].data.shape[1]),
                      sep='\n', end='\n\n')
            elif len(eeg[k].data.shape) == 3:
                print('{} trials'.format(eeg[k].data.shape[0]),
                      '{} channels'.format(eeg[k].data.shape[1]), 
                      '{} timepoints over {:.0f} ms'.format(eeg[k].data.shape[2], 1000 * eeg[k].data.shape[2]/eeg[k].samplerate),
                      sep='\n', end='\n\n')
        print('{} Hz sr'.format(eeg[k].samplerate))
        
    if presumed_sr:
        for k in eeg:
            assert presumed_sr == eeg[k].samplerate
            
    return eeg
       

def reader_load_events(reader, 
                       verbose=True):
    """Return the events DataFrame and event_stats for the session."""
    events = reader.load('events')
    event_stats = desc_events(events, verbose)
    
    return events, event_stats


def reader_load_pairs(reader, 
                      drop_bad=True):
    """Get the bipolar pairs DataFrame for a given subject and experiment.
    
    Also gets any MTL labels and flags bad electrodes.
    
    Returns
    -------
    pairs : pandas.core.frame.DataFrame
        Info on the bipolar electrode pairs. 
    pairs_keep_inds : list
        Electrodes flagged for removal due to epileptiform activity
        or the absence of ROI labeling.  
    """
    subj = reader.subject
    sess = reader.session
    expmt = reader.experiment
    cmlpipe = CML_stim_pipeline.cml_pipeline(subj, expmt) 
    cmlpipe.set_elecs(type='bi', flag_bad_elecs=True)
    pairs = cmlpipe.elecs
    if 'bad_elecs' not in pairs.columns: 
        pairs['bad_elecs'] = False
    if 'electrode_categories' not in pairs.columns:
        pairs['electrode_categories'] = ''

    # Load localization for MTL subregions
    pairs.insert(0, 'subject', [subj for _ in range(pairs.shape[0])])
    pairs.insert(1, 'session', [sess for _ in range(pairs.shape[0])])
    pairs['roi'] = pairs['ind.region'].astype(str)
    pairs['hem'] = pairs['ind.x'].apply(lambda x: 'L' if x<0 else 'R')
    try:
        lclz_df = cmlpipe.reader.load('localization')
        pairs = update_pairs(lclz_df, pairs)
    except FileNotFoundError:
        print('{} {} missing localization file'.format(subj, expmt))
    if ('das.region' in pairs.columns) and not ('stein.region' in pairs.columns):
        pairs.rename(columns={'das.region': 'stein.region'}, inplace=True)
    if 'stein.region' in pairs.columns:
        hpc_labels = ['CA1', 'CA2', 'CA3', 'DG', 'Sub', 'SUB']
        ec_labels = ['EC', 'ERC']
        pairs['stein.region'] = pairs['stein.region'].apply(lambda x: str(x).replace('"', ''))
        pairs.loc[pairs['stein.region'].isin(hpc_labels), 'roi'] = 'hippocampal'
        pairs.loc[pairs['stein.region'].isin(ec_labels), 'roi'] = 'entorhinal'
    else:
        print('{} {} missing MTL mapping'.format(subj, expmt))
    
    pairs_keep_inds = pairs.query("(bad_elecs==False) & (roi!={})".format(['None', 'unknown'])).index.tolist()
    
    if drop_bad:
        pairs = pairs.loc[pairs_keep_inds].copy()
        print('Keeping {}/{} bipolar pairs\n'.format(len(pairs_keep_inds), len(pairs)))            
        
    return pairs, pairs_keep_inds

       
def run_sess_power_sim(subj, 
                       sess, 
                       list_='all',
                       recall_pct_thresh=100/6, 
                       recall_n_thresh=20,
                       n_cycles=6,
                       log_power=True,
                       timebin_size_ms=500,
                       timebin_spacing_ms=100, # ms
                       verbose=True,
                       overwrite=False,
                       power_dir='/home1/dscho/projects/theta_reinstatement/data/morlet/power', # None to calc power w/o saving
                       event_pairs_dir='/home1/dscho/projects/theta_reinstatement/data/event_pairs', # None to calc event pair sims w/o saving
                       sleep_max=0):
    """Calculate power similarity between event pairs.

    Returns event_pairs DataFrame with a column storing 
    event1_time x event2_time cosine similarity matrices.
    
    Power similarity is a vector of mean powers across frequency bands 
    and channels. Powers are averaged over a defined duration 
    (timebin_size_ms) at each evenly-spaced event timepoint of interest 
    (determined by timebin_spacing_ms).

    Saves epoched powers (trial x chan x freq x time) 
    and the event_pairs DataFrame for a list or session.
    """
    # Take a nap before running.
    if sleep_max > 0:
        sleep(np.int(sleep_max * np.random.rand()))
    
    # Load event pairs if it already exists.
    subj_sess_list = '{}_ses{}_list{}'.format(subj, sess, list_)
    event_pairs_f = os.path.join(event_pairs_dir, '{}-event_pairs.pkl'.format(subj_sess_list))
    if os.path.exists(event_pairs_f) and not overwrite:
        event_pairs = dio.open_pickle(event_pairs_f)
        return event_pairs
        
    ## GENERAL PARAMS
    ## --------------
    expmt = 'PAL1'
    reader = CMLReader(subj, expmt, sess)    
    freqs = np.logspace(np.log2(2), np.log2(100), 46, endpoint=True, base=2)
    freq_bands = od([('theta', (3.5, 8)),
                     ('alpha', (8, 12)),
                     ('beta', (13, 25)),
                     ('lgamma', (30, 58)),
                     ('hgamma', (62, 100.5))])         
    if verbose:
        print('Subj: {}\nSess: {}\nExperiment: {}\n\n'.format(subj, sess, expmt))
    
    ## LOAD EVENTS
    ## -----------
    events, event_stats = reader_load_events(reader)
    
    # Restrict events to the specified list.
    if list_ != 'all':
        events = events.loc[events['list']==list_]
    
    ## CHECK INCLUSION CRITERIA
    ## ------------------------
    if recall_pct_thresh:
        if event_stats['pct_correct'] < recall_pct_thresh:
            print('\n*\n**\n***\n\nSKIPPING SESSION – RECALL PERCENT ({:.1f}%) BELOW {}% THRESHOLD\n\n***\n**\n*\n'
                  .format(event_stats['pct_correct'], recall_pct_thresh))
            return None
    if recall_n_thresh:
        if event_stats['n_correct'] < recall_n_thresh:
            print('\n*\n**\n***\n\nSKIPPING SESSION – NUMBER OF RECALLS ({}) BELOW {} THRESHOLD\n\n***\n**\n*\n'
                  .format(event_stats['n_correct'], recall_n_thresh))
            return None
            
    ## LOAD EVENT PAIRS
    ## ----------------
    event_pairs = get_event_pairs(events, list_=list_, pair_type='enc-rec')
    
    # Add columns for the relative positions of each encoding and retrieval event.
    idx_enc_pos = {v:i for i, v in enumerate(event_pairs['idx_enc'].unique())}
    idx_rec_pos = {v:i for i, v in enumerate(event_pairs['idx_rec'].unique())}
    event_pairs.insert(4, 'idx_enc_pos', event_pairs['idx_enc'].apply(lambda x: idx_enc_pos[x]))
    event_pairs.insert(6, 'idx_rec_pos', event_pairs['idx_rec'].apply(lambda x: idx_rec_pos[x]))
    
    if verbose:
        print('{} event pairs'.format(len(event_pairs)))
        
    ## LOAD BIPOLAR PAIRS
    ## ------------------
    pairs, pairs_keep_inds = reader_load_pairs(reader)
    
    ## CALC POWER FROM EEG
    ## -------------------
    trial_types = ['enc', 'rec']
    sr = np.int(reader.load_eeg(events=events.iloc[:1], rel_start=-10, rel_stop=10, scheme=pairs).samplerate)
    trial_wins = get_trial_wins(sr=sr, n_cycles=n_cycles, lowest_freq=np.min(freqs), verbose=verbose)
    
    # Get epoch inds and times relative to encoding/retrieval events.
    epoch_inds, epoch_times = get_epoch_inds(trial_wins['wins_ms'], 
                                             trial_wins['wins_full_ms'], 
                                             sr, 
                                             trial_wins['buff_ms'], 
                                             timebin_spacing_ms,
                                             verbose)
    
    power = od([])                                         
    for trial_type in trial_types:
        eeg = reader_load_eeg(reader, 
                              pairs,
                              events, 
                              trial_wins['wins_full_ms'], 
                              trial_type=trial_type, 
                              list_=list_,
                              verbose=verbose) # trial x channel x time

        # Format the EEG timeseries.
        ts = TimeSeries(eeg[trial_type].data, 
                        name='{}-{}'.format(subj_sess_list, trial_type), 
                        dims=['trial', 'channel', 'time'],
                        coords={'trial': np.arange(eeg[trial_type].shape[0]),
                                'channel': eeg[trial_type].channels,
                                'time': np.arange(eeg[trial_type].shape[-1]),
                                'samplerate': eeg[trial_type].samplerate})
                      
        # Get power timebins.
        output_ftag = '{}to{}ms-{}ms_bins-{}ms_spacing'.format(trial_wins['wins_ms'][trial_type][0], 
                                                               trial_wins['wins_ms'][trial_type][1], 
                                                               timebin_size_ms, timebin_spacing_ms)
        timebin_size_samp = np.int(timebin_size_ms * (sr/1000))
        power[trial_type] = epoch_pows(ts, 
                                       freqs,
                                       freq_bands,
                                       timebin_inds=epoch_inds[trial_type], 
                                       timebin_size_samp=timebin_size_samp, 
                                       width=n_cycles, 
                                       log_power=log_power,
                                       savedir=power_dir,
                                       output_ftag=output_ftag,
                                       overwrite=overwrite,
                                       verbose=verbose) # trial x chan x freq x time
    
        # Z-score powers across trials, separately for each channel x freq x time 
        # combination, and independently for encoding and retrieval events.
        power[trial_type] = stats.zscore(power[trial_type], axis=0)
    
    ## CALC POWER REINSTATEMENT
    ## ------------------------
    # For the two trials in each event pair, and for each pair of 
    # encoding/recall time windows, cosine similarity is calculated between 
    # EEG power vectors across all electrode channels and frequencies. 
    # A matrix of cos sim values with length enc_time x rec_time is returned.
    event_pairs['cos_sim'] = event_pairs.apply(lambda x: power_sim(power['enc'][x['idx_enc_pos'], :, :, :], 
                                                                   power['rec'][x['idx_rec_pos'], :, :, :]), axis=1)

    # Save event pairs with power similarity matrices.
    if event_pairs_dir:
        dio.save_pickle(event_pairs, event_pairs_f, verbose)
    
    return event_pairs
    

def run_sess_theta_sync(subj, 
                        sess, 
                        list_='all',
                        recall_pct_thresh=100/6, 
                        recall_n_thresh=20,
                        n_cycles=6,
                        log_power=True,
                        timebin_size_ms=500, # ms
                        timebin_spacing_ms=100, # ms
                        verbose=True,
                        overwrite=False,
                        power_dir='/home1/dscho/projects/theta_reinstatement/data/morlet/power', # None to calc power w/o saving
                        event_pairs_dir='/home1/dscho/projects/theta_reinstatement/data/event_pairs', # None to calc event pair sims w/o saving
                        sleep_max=0):
    """Calculate theta synchrony similarity between event pairs.

    Returns event_pairs DataFrame with a column storing 
    event1_time x event2_time cosine similarity matrices.
    
    Theta synchrony is a vector of phase-locking values between each
    pair of electrodes over a defined duration (timebin_size_ms) that is
    centered around each event timepoint of interest (determined by
    timebin_spacing_ms).
    
    Saves epoched theta synchronies (trial x elec_pair x time) 
    and the event_pairs DataFrame for a list or session.
    """
    # Take a nap before running.
    if sleep_max > 0:
        sleep(np.int(sleep_max * np.random.rand()))
    
    # Load event pairs if it already exists.
    subj_sess_list = '{}_ses{}_list{}'.format(subj, sess, list_)
    event_pairs_f = os.path.join(event_pairs_dir, '{}-event_pairs.pkl'.format(subj_sess_list))
    if os.path.exists(event_pairs_f) and not overwrite:
        event_pairs = dio.open_pickle(event_pairs_f)
        return event_pairs
        
    ## GENERAL PARAMS
    ## --------------
    expmt = 'PAL1'
    reader = CMLReader(subj, expmt, sess)    
    freqs = np.logspace(np.log2(2), np.log2(100), 46, endpoint=True, base=2)
    freq_bands = od([('theta', (3.5, 8)),
                     ('alpha', (8, 12)),
                     ('beta', (13, 25)),
                     ('lgamma', (30, 58)),
                     ('hgamma', (62, 100.5))])         
    if verbose:
        print('Subj: {}\nSess: {}\nExperiment: {}\n\n'.format(subj, sess, expmt))
    
    ## LOAD EVENTS
    ## -----------
    events, event_stats = reader_load_events(reader)
    
    # Restrict events to the specified list.
    if list_ != 'all':
        events = events.loc[events['list']==list_]
    
    ## CHECK INCLUSION CRITERIA
    ## ------------------------
    if recall_pct_thresh:
        if event_stats['pct_correct'] < recall_pct_thresh:
            print('\n*\n**\n***\n\nSKIPPING SESSION – RECALL PERCENT ({:.1f}%) BELOW {}% THRESHOLD\n\n***\n**\n*\n'
                  .format(event_stats['pct_correct'], recall_pct_thresh))
            return None
    if recall_n_thresh:
        if event_stats['n_correct'] < recall_n_thresh:
            print('\n*\n**\n***\n\nSKIPPING SESSION – NUMBER OF RECALLS ({}) BELOW {} THRESHOLD\n\n***\n**\n*\n'
                  .format(event_stats['n_correct'], recall_n_thresh))
            return None
            
    ## LOAD EVENT PAIRS
    ## ----------------
    event_pairs = get_event_pairs(events, list_=list_, pair_type='enc-rec')
    
    # Add columns for the relative positions of each encoding and retrieval event.
    idx_enc_pos = {v:i for i, v in enumerate(event_pairs['idx_enc'].unique())}
    idx_rec_pos = {v:i for i, v in enumerate(event_pairs['idx_rec'].unique())}
    event_pairs.insert(4, 'idx_enc_pos', event_pairs['idx_enc'].apply(lambda x: idx_enc_pos[x]))
    event_pairs.insert(6, 'idx_rec_pos', event_pairs['idx_rec'].apply(lambda x: idx_rec_pos[x]))
    
    if verbose:
        print('{} event pairs'.format(len(event_pairs)))
        
    ## LOAD BIPOLAR PAIRS
    ## ------------------
    pairs, pairs_keep_inds = reader_load_pairs(reader)
    
    ## CALC POWER FROM EEG
    ## -------------------
    trial_types = ['enc', 'rec']
    sr = np.int(reader.load_eeg(events=events.iloc[:1], rel_start=-10, rel_stop=10, scheme=pairs).samplerate)
    trial_wins = get_trial_wins(sr=sr, buff_ms=np.int(timebin_size_ms/2), verbose=verbose)
    
    # Get epoch inds and times relative to encoding/retrieval events.
    epoch_inds, epoch_times = get_epoch_inds(trial_wins['wins_ms'], 
                                             trial_wins['wins_full_ms'], 
                                             sr, 
                                             trial_wins['buff_ms'], 
                                             timebin_spacing_ms,
                                             verbose)
    
    power = od([])                                         
    for trial_type in trial_types:
        eeg = reader_load_eeg(reader, 
                              pairs,
                              events, 
                              trial_wins['wins_full_ms'], 
                              trial_type=trial_type, 
                              list_=list_,
                              verbose=verbose) # trial x channel x time

        # Format the EEG timeseries.
        ts = TimeSeries(eeg[trial_type].data, 
                        name='{}-{}'.format(subj_sess_list, trial_type), 
                        dims=['trial', 'channel', 'time'],
                        coords={'trial': np.arange(eeg[trial_type].shape[0]),
                                'channel': eeg[trial_type].channels,
                                'time': np.arange(eeg[trial_type].shape[-1]),
                                'samplerate': eeg[trial_type].samplerate})
                      
        # Get power timebins.
        output_ftag = '{}to{}ms-{}ms_bins-{}ms_spacing'.format(trial_wins['wins_ms'][trial_type][0], 
                                                               trial_wins['wins_ms'][trial_type][1], 
                                                               timebin_size_ms, timebin_spacing_ms)
        timebin_size_samp = np.int(timebin_size_ms * (sr/1000))
        power[trial_type] = epoch_pows(ts, 
                                       freqs,
                                       freq_bands,
                                       timebin_inds=epoch_inds[trial_type], 
                                       timebin_size_samp=timebin_size_samp, 
                                       width=n_cycles, 
                                       log_power=log_power,
                                       savedir=power_dir,
                                       output_ftag=output_ftag,
                                       overwrite=overwrite,
                                       verbose=verbose) # trial x chan x freq x time
    
        # Z-score powers across trials, separately for each channel x freq x time 
        # combination, and independently for encoding and retrieval events.
        power[trial_type] = stats.zscore(power[trial_type], axis=0)
    
    ## CALC POWER REINSTATEMENT
    ## ------------------------
    # For the two trials in each event pair, and for each pair of 
    # encoding/recall time windows, cosine similarity is calculated between 
    # EEG power vectors across all electrode channels and frequencies. 
    # A matrix of cos sim values with length enc_time x rec_time is returned.
    event_pairs['cos_sim'] = event_pairs.apply(lambda x: power_sim(power['enc'][x['idx_enc_pos'], :, :, :], 
                                                                   power['rec'][x['idx_rec_pos'], :, :, :]), axis=1)

    # Save event pairs with power similarity matrices.
    if event_pairs_dir:
        dio.save_pickle(event_pairs, event_pairs_f, verbose)
    
    return event_pairs


def study_list_isis(events):
    """Print mean +/- stdev inter-stimulus intervals across lists."""
    def describe(arr):
        return np.mean(arr), np.std(arr), np.min(arr), np.max(arr)
    
    lists = np.array([i for i in pd.unique(events['list']) if (i>0)])
        
    print('Encoding durations:',
          ('{:.1f} +/- {:.1f}ms (min={}, max={}) between study orient and study pair onset'
           .format(*describe(np.concatenate([np.diff(events.loc[(events['list']==list_)].query("(type==['STUDY_PAIR', 'STUDY_ORIENT'])")['mstime'])[::2] for list_ in lists])))),
          ('{:.1f} +/- {:.1f}ms (min={}, max={}) duration each study pair is shown'
           .format(*describe(np.concatenate([np.diff(events.loc[(events['list']==list_)].query("(type==['STUDY_PAIR', 'STUDY_ORIENT'])")['mstime'])[1::2] for list_ in lists])))),
          'next study orient immediately follows study pair offset',
          sep='\n', end='\n\n')

    print('Recall durations:',
          ('{:.1f} +/- {:.1f}ms (min={}, max={}) between test orient and test probe onset'
           .format(*describe([np.diff(events.loc[(events['list']==list_)].query("(type==['TEST_ORIENT', 'REC_START', 'REC_END'])")['mstime'])[::3] for list_ in lists]))),
          ('{:.1f} +/- {:.1f}ms (min={}, max={}) duration each test probe is shown'
           .format(*describe([np.diff(events.loc[(events['list']==list_)].query("(type==['TEST_ORIENT', 'REC_START', 'REC_END'])")['mstime'])[1::3] for list_ in lists]))),
          ('{:.1f} +/- {:.1f}ms (min={}, max={}) between test orient and test probe onset'
           .format(*describe([np.diff(events.loc[(events['list']==list_)].query("(type==['TEST_ORIENT', 'REC_START', 'REC_END'])")['mstime'])[2::3] for list_ in lists]))),
          sep='\n', end='\n\n')
