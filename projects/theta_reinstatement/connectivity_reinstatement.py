"""
connectivity_reinstatement.py

Author:
    Daniel Schonhaut
    Computational Memory Lab
    University of Pennsylvania
    daniel.schonhaut@gmail.com
    
Description: 
    Functions for performing free recall encoding and retrieval analyses using
    power and phase based connectivity measures.

Last Edited: 
    6/16/19
"""
import sys
import os
from time import time
from time import sleep
from collections import OrderedDict
import itertools 
import mkl
mkl.set_num_threads(1)
import numpy as np
import pandas as pd
import scipy.stats as stats
from ptsa.data.filters import ButterworthFilter
from ptsa.data.filters import MorletWaveletFilter
import cmlreaders
from cmlreaders import CMLReader
sys.path.insert(0, '/home1/esolo/notebooks/codebases/')
import CML_stim_pipeline
from loc_toolbox import get_region, update_pairs
sys.path.append('/home1/dscho/code/general')
sys.path.append('/home1/dscho/code/projects/unit_activity_and_hpc_theta')
import data_io as dio
import lfp_synchrony

def run_pipeline1(subj,
                  sess,
                  expmt='FR1',
                  lclz=0,
                  mont=0,
                  notch_freqs=[60, 120],
                  freqs=np.logspace(np.log10(3), np.log10(110), num=22),
                  enc_start=200,
                  enc_stop=1400,
                  ret_start=-1200,
                  ret_stop=0,
                  morlet_width=5,
                  buf=1000,
                  recdiff_buf=1500,
                  log_power=True,
                  z_power_across_events=True,
                  z_plv_degrees_across_events=True,
                  save_outputs=True,
                  output_dir='/scratch/dscho/connectivity_reinstatement/data',
                  sleep_max=0):
    
    # Take a nap before running.
    if sleep_max > 0:
        sleep(int(sleep_max * np.random.rand()))
        
    start_time = time()
    
    pairs, pairs_keep_inds = get_pairs(subj=subj, 
                                       expmt=expmt)
    
    enc_evs, ret_evs, _, enc_phase, _, ret_phase = get_enc_ret_wavelet_power_phase(
        subj=subj,
        expmt=expmt,
        sess=sess,
        pairs=pairs,
        lclz=lclz,
        mont=mont,
        pairs_keep_inds=pairs_keep_inds,
        notch_freqs=notch_freqs,
        freqs=freqs,
        enc_start=enc_start,
        enc_stop=enc_stop,
        ret_start=ret_start,
        ret_stop=ret_stop,
        morlet_width=morlet_width,
        buf=buf,
        recdiff_buf=recdiff_buf,
        log_power=log_power,
        z_power_across_events=z_power_across_events,
        save_outputs=save_outputs,
        output_dir=output_dir)
        
    elec_pairs, enc_plvs, ret_plvs = get_enc_ret_plvs(enc_phase=enc_phase, 
                                                      ret_phase=ret_phase)
    
    _ = get_enc_ret_plv_degrees(elec_pairs=elec_pairs,
                                enc_plvs=enc_plvs,
                                ret_plvs=ret_plvs,
                                subj=subj,
                                sess=sess,
                                z_across_events=z_plv_degrees_across_events,
                                save_outputs=save_outputs,
                                output_dir=output_dir)
        
    word_pairs = get_word_pairs(subj=subj,
                                sess=sess,
                                enc_evs=enc_evs,
                                ret_evs=ret_evs,
                                save_outputs=save_outputs,
                                output_dir=output_dir)
    
    run_time = time() - start_time
    
    return run_time
              
def get_pairs(subj,
              expmt):
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
    cmlpipe = CML_stim_pipeline.cml_pipeline(subj, expmt) 
    cmlpipe.set_elecs(type='bi', flag_bad_elecs=True)
    pairs = cmlpipe.elecs
    if 'bad_elecs' not in pairs.columns: 
        pairs['bad_elecs'] = False
    if 'electrode_categories' not in pairs.columns:
        pairs['electrode_categories'] = ''

    # Load localization for MTL subregions
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
    
    return pairs, pairs_keep_inds
    
def get_enc_ret_wavelet_power_phase(subj,
                                    expmt,
                                    sess,
                                    pairs,
                                    lclz=0,
                                    mont=0,
                                    pairs_keep_inds=None,
                                    notch_freqs=[60, 120],
                                    freqs=np.logspace(np.log10(3), np.log10(110), num=22),
                                    enc_start=200,
                                    enc_stop=1400,
                                    ret_start=-1200,
                                    ret_stop=0,
                                    morlet_width=5,
                                    buf=1000,
                                    recdiff_buf=1500,
                                    log_power=True,
                                    z_power_across_events=True,
                                    save_outputs=True,
                                    output_dir='/scratch/dscho/connectivity_reinstatement/data'):
    """Calculate wavelet power and phase for each encoding and retrieval event.
    
    Returns
    -------
    env_evs : pandas.core.frame.DataFrame
        Length is the number of encoding events.
    ret_evs : pandas.core.frame.DataFrame
        Length is the number of retrieval events, 
        minus any retrievals < recdiff_buf.
    enc_power : numpy.ndarray
        freq x event x elec (mean across time)
    enc_phase : numpy.ndarray
        freq x event x elec x time
    ret_power : numpy.ndarray
        freq x event x elec (mean across time)
    ret_phase : numpy.ndarray
        freq x event x elec x time
    """

    def rectime_diff(x):
        """Return the ms elapsed since the start of the recall
        period or the onset of the previous recalled word.
        """
        x = list(x)
        return [x[0]] + list(np.diff(x))
        
    if pairs_keep_inds is None:
        pairs_keep_inds = pairs.index.tolist()
        
    reader = CMLReader(subject=subj, 
                       experiment=expmt, 
                       session=sess,
                       localization=lclz, 
                       montage=mont)
    evs = reader.load('events')
    enc_evs = evs.query(("type=='WORD'")).copy()
    ret_evs = evs.query(("type=='REC_WORD'")).copy()
    ret_evs['recdiff'] = [item for sublist in ret_evs.groupby('list').rectime.apply(lambda x: rectime_diff(x)).tolist() for item in sublist]
    ret_evs['recpos'] = [item for sublist in ret_evs.groupby('list').item_name.apply(lambda x: list(np.arange(len(x)))).tolist() for item in sublist]
    ret_evs = ret_evs.loc[ret_evs.recdiff>=recdiff_buf] # only look at words with at least 1500ms between retrieval times
    
    enc_evs = enc_evs.reset_index(drop=True)
    ret_evs = ret_evs.reset_index(drop=True)
    
    # Get EEG
    eeg_enc = reader.load_eeg(events=enc_evs, rel_start=-buf+enc_start, rel_stop=enc_stop+buf, scheme=pairs.loc[pairs_keep_inds])
    eeg_ret = reader.load_eeg(events=ret_evs, rel_start=-buf+ret_start, rel_stop=ret_stop+buf, scheme=pairs.loc[pairs_keep_inds])
    sr = eeg_enc.samplerate
    n_buf = int(buf*sr*1e-3) # buffer size in samples

    # Remove line noise
    filt_freqs = []
    for iFreq in range(len(notch_freqs)):
        filt_freqs.append([notch_freqs[iFreq]-2, notch_freqs[iFreq]+2])
    eeg_enc_filt = eeg_enc.to_ptsa()
    eeg_ret_filt = eeg_ret.to_ptsa()
    for freq_range in filt_freqs:
        eeg_enc_filt = ButterworthFilter(timeseries=eeg_enc_filt, freq_range=freq_range, filt_type='stop', order=4).filter()
        eeg_ret_filt = ButterworthFilter(timeseries=eeg_ret_filt, freq_range=freq_range, filt_type='stop', order=4).filter()

    # Get spectral power and phase
    enc_power, enc_phase = MorletWaveletFilter(eeg_enc_filt, freqs=freqs, width=morlet_width, output=['power', 'phase']).filter()
    ret_power, ret_phase = MorletWaveletFilter(eeg_ret_filt, freqs=freqs, width=morlet_width, output=['power', 'phase']).filter()

    # Remove buffer timepoints
    enc_power = enc_power.data[:, :, :, n_buf:-n_buf] # freq x event x elec x time
    enc_phase = enc_phase.data[:, :, :, n_buf:-n_buf]
    ret_power = ret_power.data[:, :, :, n_buf:-n_buf] # freq x event x elec x time
    ret_phase = ret_phase.data[:, :, :, n_buf:-n_buf]

    # Log-transform power, take the mean over time, and Z-score across events
    if log_power:
        enc_power = np.log10(enc_power)
        ret_power = np.log10(ret_power)

    enc_power = np.mean(enc_power, axis=3) # freq x event x elec
    ret_power = np.mean(ret_power, axis=3) # freq x event x elec
    
    if z_power_across_events:
        enc_power = stats.zscore(enc_power, axis=1) # Z-scored across events
        ret_power = stats.zscore(ret_power, axis=1) # Z-scored across events
    
    if save_outputs:
        process_str = '-{}'.format(subj)
        process_str += '-ses{}'.format(sess)
        dio.save_pickle(enc_evs, os.path.join(output_dir, 'events', 'encoding_events{}.pkl'.format(process_str)), verbose=False)
        process_str += '-{}ms_recall_buffer'.format(recdiff_buf)
        dio.save_pickle(ret_evs, os.path.join(output_dir, 'events', 'retrieval_events{}.pkl'.format(process_str)), verbose=False)
        
        process_str = '-{}'.format(subj)
        process_str += '-ses{}'.format(sess)
        process_str += '-{}_to_{}_buf{}ms'.format(enc_start, enc_stop, buf)
        process_str += '-{}Hz'.format(int(sr))
        process_str += '-notch' + '_'.join(str(i) for i in notch_freqs) + 'Hz' if notch_freqs else 'nonotch'
        process_str += '-{}cycles'.format(morlet_width)
        process_str += '-{}log10freqs_{:.1f}_to_{:.1f}Hz'.format(len(freqs), freqs[0], freqs[-1])
        #dio.save_pickle(enc_phase.astype(np.float32), os.path.join(output_dir, 'wavelet', 'encoding_phase-freq_event_elec_time{}.pkl'.format(process_str)), verbose=False)
        process_str += '-log10_power' if log_power else ''
        process_str += '-z_across_events' if z_power_across_events else ''
        dio.save_pickle(enc_power.astype(np.float32), os.path.join(output_dir, 'wavelet', 'encoding_power-freq_event_elec{}.pkl'.format(process_str)), verbose=False)
        
        process_str = '-{}'.format(subj)
        process_str += '-ses{}'.format(sess)
        process_str += '-{}_to_{}_buf{}ms'.format(ret_start, ret_stop, buf)
        process_str += '-{}Hz'.format(int(sr))
        process_str += '-notch' + '_'.join(str(i) for i in notch_freqs) + 'Hz' if notch_freqs else 'nonotch'
        process_str += '-{}cycles'.format(morlet_width)
        process_str += '-{}log10freqs_{:.1f}_to_{:.1f}Hz'.format(len(freqs), freqs[0], freqs[-1])
        #dio.save_pickle(ret_phase.astype(np.float32), os.path.join(output_dir, 'wavelet', 'retrieval_phase-freq_event_elec_time{}.pkl'.format(process_str)), verbose=False)
        process_str += '-log10_power' if log_power else ''
        process_str += '-z_across_events' if z_power_across_events else ''
        dio.save_pickle(ret_power.astype(np.float32), os.path.join(output_dir, 'wavelet', 'retrieval_power-freq_event_elec{}.pkl'.format(process_str)), verbose=False)
        
    return enc_evs, ret_evs, enc_power, enc_phase, ret_power, ret_phase
        
def get_enc_ret_plvs(enc_phase,
                     ret_phase):
    """Return the phase-locking value between electrode pairs at each frequency 
    and for each encoding and retrieval event.
    
    Returns
    -------
    elec_pairs : list of tuples
        Contains electrode indices for all possible pairs.
    enc_plvs : numpy.ndarray
        freq x event x elec_pair
    ret_plvs : numpy.ndarray
        freq x event x elec_pair
    """
    n_elecs = enc_phase.shape[2]
    elec_pairs = []
    for iElec1, iElec2 in itertools.combinations(np.arange(n_elecs), 2):
        elec_pairs.append((iElec1, iElec2))

    enc_plvs = []
    for iElec1, iElec2 in elec_pairs:
        enc_plvs.append(lfp_synchrony.calc_plv(enc_phase[:, :, iElec1, :], enc_phase[:, :, iElec2, :], axis=-1))
    enc_plvs = np.moveaxis(enc_plvs, 0, 2) # freq x event x elec_pair

    ret_plvs = []
    for iElec1, iElec2 in elec_pairs:
        ret_plvs.append(lfp_synchrony.calc_plv(ret_phase[:, :, iElec1, :], ret_phase[:, :, iElec2, :], axis=-1))
    ret_plvs = np.moveaxis(ret_plvs, 0, 2) # freq x event x elec_pair
    
    return elec_pairs, enc_plvs, ret_plvs
    
def get_enc_ret_plv_degrees(elec_pairs,
                            enc_plvs,
                            ret_plvs,
                            subj,
                            sess,
                            z_across_events=True,
                            save_outputs=True,
                            output_dir='/scratch/dscho/connectivity_reinstatement/data'):
    """Return the degree of phase-locking values for each electrode,   
    at each frequency and for each encoding and retrieval event.
    
    Degree is calculated as the sum of PLVs from a given electrode to all other
    electrodes.
    
    Returns
    -------
    enc_plv_degrees : numpy.ndarray
        freq x event x elec
    ret_plv_degrees : numpy.ndarray
        freq x event x elec
    """
    # Get a list of electrode pairs involving each electrode
    n_elecs = len(np.unique(np.array(elec_pairs)))
    n_freqs = enc_plvs.shape[0]
    elec_inds = []
    for iElec in range(n_elecs):
        elec_inds.append(np.where(np.array(elec_pairs)==iElec)[0])

    # For each frequency and event, calculate the degree of each electrode
    # (sum of PLVs) and Z-score across events.
    enc_plv_degrees = []
    for iFreq in range(n_freqs):
        enc_plv_degrees_ = []
        for iEvent in range(enc_plvs.shape[1]):
            enc_plv_degrees__ = []
            for iElec in range(n_elecs):
                enc_plv_degrees__.append(np.sum(enc_plvs[iFreq, iEvent, elec_inds[iElec]]))
            enc_plv_degrees_.append(enc_plv_degrees__)
        enc_plv_degrees.append(enc_plv_degrees_)
    if z_across_events:
        enc_plv_degrees = stats.zscore(enc_plv_degrees, axis=1) 
    else:
        enc_plv_degrees = np.array(enc_plv_degrees)

    ret_plv_degrees = []
    for iFreq in range(n_freqs):
        ret_plv_degrees_ = []
        for iEvent in range(ret_plvs.shape[1]):
            ret_plv_degrees__ = []
            for iElec in range(n_elecs):
                ret_plv_degrees__.append(np.sum(ret_plvs[iFreq, iEvent, elec_inds[iElec]]))
            ret_plv_degrees_.append(ret_plv_degrees__)
        ret_plv_degrees.append(ret_plv_degrees_)
    if z_across_events:
        ret_plv_degrees = stats.zscore(ret_plv_degrees, axis=1)
    else:
        ret_plv_degrees = np.array(ret_plv_degrees)
    
    if save_outputs:
        process_str = '-{}'.format(subj)
        process_str += '-ses{}'.format(sess)
        process_str += '-z_across_events' if z_across_events else ''
        dio.save_pickle(enc_plv_degrees.astype(np.float32), os.path.join(output_dir, 'plvs', 'encoding_plv_degrees-freq_event_elec{}.pkl'.format(process_str)), verbose=False)
        dio.save_pickle(ret_plv_degrees.astype(np.float32), os.path.join(output_dir, 'plvs', 'retrieval_plv_degrees-freq_event_elec{}.pkl'.format(process_str)), verbose=False)
        
    return enc_plv_degrees, ret_plv_degrees
    
def get_word_pairs(subj,
                   sess,
                   enc_evs,
                   ret_evs,
                   save_outputs=True,
                   output_dir='/scratch/dscho/connectivity_reinstatement/data'):
    """Create a DataFrame with all within-list encoding and retrieval event 
    pairs.
    
    Returns
    -------
    word_pairs : pandas.core.frame.DataFrame
        Rows include all possible pairs of within-list encoding and retrieval 
        events.
    """
    #subj = enc_evs.subject.iat[0]
    #sess = enc_evs.session.iat[0]
    word_pairs = []
    for iList in np.unique(enc_evs.list):
        enc_inds = enc_evs.loc[enc_evs.list==iList].index.tolist()
        ret_inds = ret_evs.loc[ret_evs.list==iList].index.tolist()
    
        for enc_ind in enc_inds:
            for ret_ind in ret_inds:
                enc_word = enc_evs.loc[enc_ind, 'item_name']
                rec = enc_evs.loc[enc_ind, 'recalled']
                ret_word = ret_evs.loc[ret_ind, 'item_name']
                ret_pos = ret_evs.loc[ret_ind, 'recpos']
                word_pairs.append([subj, sess, iList, enc_ind, ret_ind, enc_word, ret_word, rec, ret_pos])
    word_pairs = pd.DataFrame(word_pairs, columns=['subject', 'session', 'list', 'enc_ind', 'ret_ind', 'enc_word', 'ret_word', 'recalled', 'recpos'])
    word_pairs['same_word'] = word_pairs['enc_word'] == word_pairs['ret_word']
    
    if save_outputs:
        process_str = '-{}'.format(subj)
        process_str += '-ses{}'.format(sess)
        dio.save_pickle(word_pairs, os.path.join(output_dir, 'events', 'encoding_retrieval_word_pairs{}.pkl'.format(process_str)), verbose=False)
    
    return word_pairs

def package_neural_dat(enc_power,
                       enc_plv_degrees,
                       ret_power,
                       ret_plv_degrees):
                       
    neural_dat = {'enc': {'power': enc_power,
                          'plv_degrees': enc_plv_degrees},
                  'ret': {'power': ret_power,
                          'plv_degrees': ret_plv_degrees}}
    return neural_dat
    
def cos_sim(v1, v2):
    """Return the cosine similarity between v1 and v2."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_enc_ret_sims(subj,
                     sess,
                     word_pairs, 
                     neural_dat,
                     sim_type='cos_sim', # cos_sim or pearson
                     save_outputs=True,
                     output_dir='/scratch/dscho/connectivity_reinstatement/data'):
    """Get the cosine similarity between encoding/retrieval word pairs.
    
    Performed separately for each frequency.
    
    Uses feature vectors from each neural type in neural_dat (e.g. power across
    electrodes or phase-locking value degree across electrodes).
    
    Returns
    -------
    neural_sims : OrderedDict
        Keys correspond to each neural type. 
        Each value is a numpy array with dims word_pair x freq
    """
    
    neural_types = list(neural_dat['enc'].keys())
    n_freqs = neural_dat['enc'][neural_types[0]].shape[0]
    
    neural_sims = OrderedDict()
    for neural_type in neural_types:
        sims = []
        for index, row in word_pairs.iterrows():
            iEnc = row.enc_ind
            iRet = row.ret_ind
            sims_ = []
            for iFreq in range(n_freqs):
                if sim_type == 'cos_sim':
                    sims_.append(cos_sim(neural_dat['enc'][neural_type][iFreq, iEnc, :],
                                         neural_dat['ret'][neural_type][iFreq, iRet, :]))
                elif sim_type == 'pearson':
                    sims_.append(stats.pearsonr(neural_dat['enc'][neural_type][iFreq, iEnc, :],
                                                neural_dat['ret'][neural_type][iFreq, iRet, :])[0])
            sims.append(sims_)
        neural_sims[neural_type] = np.array(sims) # word_pair x freq
    
    if save_outputs:
        process_str = '-{}'.format(sim_type)
        process_str += '-{}'.format(subj)
        process_str += '-ses{}'.format(sess)
        dio.save_pickle(neural_sims, os.path.join(output_dir, 'sims', 'encoding_retrieval_word_pair{}.pkl'.format(process_str)), verbose=False)
        
    return neural_sims