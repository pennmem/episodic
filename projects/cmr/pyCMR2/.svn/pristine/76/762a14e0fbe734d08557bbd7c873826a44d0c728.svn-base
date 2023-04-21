import json
import numpy as np
import scipy.stats as ss

#I added this (shai)
import os
thisdir = os.getcwd()
os.chdir("../../CMR2_Optimized")

import CMR2_pack_cyth as CMR2

os.chdir(thisdir)


from pybeh.spc import spc
from pybeh.pfr import pfr
from pybeh.pli import pli
from pybeh.crp import crp
from pybeh.make_recalls_matrix import make_recalls_matrix
#from pybeh.create_intrusions import intrusions
from crls import get_irts, R_crl, op_crl
from crls import opR_crl
from crls import firstRecallDist, nthIrtDist
from crls import avgTotalRecalls
from glob import glob
import warnings

"""
IMPORTANT: Please note that this code was developed by Jesse Pazdera for use with 
ltpFR3. This code is highly specialized for that particular study, and is not
suitable for general use. Rather, I have included it because it may serve as a
helpful reference or starting point for how to build your objective function that
will evaluate each parameter set during model fitting.
"""

global LIST_LENGTH
LIST_LENGTH = 24

def filter_by_condi(a, mods, prs, lls, dds, mod=None, pr=None, ll=None, dd=None):

    if pr == 's':
        pr = 1600
    elif pr == 'f':
        pr = 800

    ll = int(ll) if ll is not None else None
    dd = int(dd) if dd is not None else None

    ind = [i for i in range(len(a)) if ((ll is None or lls[i] == ll) and (pr is None or prs[i] == pr) and (mod is None or mods[i] == mod) and (dd is None or dds[i] == dd))]
    if len(ind) == 0:
        return np.array([])
    return np.array(a)[ind]


def pad_into_array(l, min_length=0):
    """
    Turn an array of uneven lists into a numpy matrix by padding shorter lists with zeros. Modified version of a
    function by user Divakar on Stack Overflow, here:
    http://stackoverflow.com/questions/32037893/numpy-fix-array-with-rows-of-different-lengths-by-filling-the-empty-elements-wi

    :param l: A list of lists
    :return: A numpy array made from l, where all rows have been made the same length via padding
    """
    l = np.array(l)
    # Get lengths of each row of data
    lens = np.array([len(i) for i in l])

    # If l was empty, we can simply return the empty numpy array we just created
    if len(lens) == 0:
        return lens

    # If all rows are the same length, return the original input as an array
    if lens.max() == lens.min() and lens.max() >= min_length:
        return l

    # Mask of valid places in each row
    mask = np.arange(max(lens.max(), min_length)) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=l.dtype)
    out[mask] = np.concatenate(l)

    return out


def get_data(data_files, wordpool_file, number_sessions=200):

    sources = None # we arent using sources
    
    #------Get all pres for all subjects
    data_pres = []
    
    for data_file in data_files:
        with open(data_file, 'r') as f:
            x = json.load(f)  
        fdata_pres = np.array(x['pres_words'])
        # Replace zeros with empty strings
        fdata_pres[data_pres == '0'] = ''
        data_pres.append(fdata_pres)
    data_pres = np.concatenate(data_pres)
    
    # Get PEERS word pool
    wp = [s.upper() for s in np.loadtxt(wordpool_file, dtype='U32')]
    # Convert presented words to word ID numbers
    data_pres = np.searchsorted(wp, data_pres, side='right')
    
    
    #------Randomly select 200 of the sessions
    lists_per_session = 24
    sessions_to_pick = number_sessions
    from numpy.random import default_rng
    randg = default_rng()
    sessions = randg.choice(int(data_pres.shape[0]/lists_per_session), size=sessions_to_pick, replace=False)
    selected_pres = []
    for session in sessions:
        selected_pres.append(data_pres[session:session+lists_per_session])
    sessions = np.repeat(range(0,sessions_to_pick), lists_per_session)
    data_pres = np.concatenate(selected_pres)

    return data_pres, sessions, sources


def calc_spc(recalls, sessions, return_sem=False, listLength=LIST_LENGTH):

    s = spc(recalls, subjects=sessions, listLength=listLength)
    s_start = spc(recalls, subjects=sessions, listLength=listLength, start_position=[1])
    s_l4 = spc(recalls, subjects=sessions, listLength=listLength, start_position=[9, 10, 11, 12])

    if return_sem:
        return np.nanmean(s, axis=0), ss.sem(s, axis=0, nan_policy='omit'), \
               np.nanmean(s_start, axis=0), ss.sem(s_start, axis=0, nan_policy='omit'), \
               np.nanmean(s_l4, axis=0), ss.sem(s_l4, axis=0, nan_policy='omit')
    else:
        return s.mean(axis=0), np.nanmean(s_start, axis=0), np.nanmean(s_l4, axis=0)


def calc_crp(recalls, sessions, return_sem=False, listLength=LIST_LENGTH):

    s = np.asarray(crp(recalls, subjects=sessions, listLength=listLength, lag_num=8))
    
    
    if return_sem:
        return s.mean(axis=0), ss.sem(s, axis=0, nan_policy='omit')
    else:
        return s.mean(axis=0)


def calc_pfr(recalls, sessions, return_sem=False, listLength=LIST_LENGTH):

    s = np.array(pfr(recalls, subjects=sessions, listLength=listLength))

    if return_sem:
        return s.mean(axis=0), ss.sem(s, axis=0)
    else:
        return s.mean(axis=0)


def calc_pli(intrusions, sessions, return_sem=False):

    s = np.array(pli(intrusions, subjects=sessions, per_list=True))

    if return_sem:
        return np.mean(s), ss.sem(s)
    else:
        return np.mean(s)


def pli_recency(intrusions, sessions, nmax=5, nskip=2, return_sem=False):
    
    u_sess = np.unique(sessions)
    n_sess = len(u_sess)

    result = np.zeros((n_sess, nmax))

    for i, sess in enumerate(u_sess):
        sess_intru = intrusions[sessions == sess]
        n_trials = len(sess_intru)
        pli_counts = np.zeros(n_trials-1)
        possible_counts = np.arange(n_trials-1, 0, -1)
        for trial, trial_data in enumerate(sess_intru):
            if trial < nskip:
                continue
            for item in trial_data:
                if item > 0:
                    pli_counts[item-1] += 1
        normed_counts = pli_counts / possible_counts
        result[i, :] = normed_counts[:nmax] / np.nansum(normed_counts)

    if return_sem:
        return np.nanmean(result, axis=0), ss.sem(result, axis=0, nan_policy='omit')
    else:
        return np.nanmean(result, axis=0)


def param_vec_to_dict(param_vec):
    """
    Convert parameter vector to dictionary format expected by CMR2.
    """
    dt = 100.  # Time step during accumulator (in ms)
    param_dict = {
        
        # Beta parameters
        'beta_enc': param_vec[0],
        'beta_rec': param_vec[1],
        'beta_rec_post': param_vec[9],
        'beta_source': param_vec[0],  # Default to beta encoding
        
        #shai added this:
        'beta_distract': param_vec[14],
        
        # Associative scaling parameters (No Source Coding)
        'gamma_fc': param_vec[2],
        'gamma_cf': param_vec[3],
        
        # Associative scaling parameters (With Source Coding)
        'L_FC_tftc': param_vec[2],  # Scale of items reinstating past temporal contexts; Default to gamma FC
        'L_FC_sftc': 0,  # Scale of sources reinstating past temporal contexts; Default to 0
        'L_FC_tfsc': param_vec[2],  # Scale of items reinstating past source contexts; Default to gamma FC
        'L_FC_sfsc': 0,  # Scale of sources reinstating past source contexts; Default to 0
        
        'L_CF_tctf': param_vec[3],  # Scale of temporal context cueing past items; Default to gamma CF
        'L_CF_sctf': param_vec[3],  # Scale of temporal context cueing past sources features; Default to gamma CF
        'L_CF_tcsf': 0,  # Scale of source context cueing past items; Default to 0
        'L_CF_scsf': 0,  # Scale of source context cueing past sources; Default to 0
        
        # Primacy and semantic scaling
        'phi_s': param_vec[4],
        'phi_d': param_vec[5],
        's_cf': param_vec[8],
        's_fc': 0,
        
        # Recall parameters
        'kappa': param_vec[6],
        'eta': param_vec[7],
        'omega': param_vec[10],
        'alpha': param_vec[11],
        'c_thresh': param_vec[12],
        'lamb': param_vec[13],
        
        # Timing parameters
        'rec_time_limit': 60000.,  # Duration of recall period (in ms)
        'dt': dt,
        'dt_tau': dt / 1000.,
        'sq_dt_tau': (dt / 1000.) ** .5,
        'nitems_in_accumulator': 50,  # Number of items in accumulator
        'max_recalls': 30  # Maximum recalls allowed per trial
    }

    return param_dict


def obj_func(param_vec, target_stats, data_pres, sessions, w2v, source_mat=None):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec)
    print(param_dict)
    
    # Run model with the parameters given in param_vec
    rec_nos, rts = CMR2.run_cmr2_multi_sess(param_dict, data_pres, sessions, w2v, source_mat=source_mat, mode='DFR')

    # Create recalls and intrusion matrices
    rec_nos = pad_into_array(rec_nos, min_length=1)
    cmr_recalls = make_recalls_matrix(pres_itemnos=data_pres, rec_itemnos=rec_nos)
    #cmr_intrusions = intrusions(pres_itemnos=data_pres, rec_itemnos=rec_nos, subjects=np.zeros(rec_nos.shape[0]),
    #                            sessions=sessions)
    print(cmr_recalls)
    # Get the performance stats of the model's predicted recalls
    cmr_stats = {}
    
    
    ll=24
    recalls = cmr_recalls
    sess = sessions
    cmr_stats['spc'], cmr_stats['spc_sem'], \
    cmr_stats['spc_fr1'], cmr_stats['spc_fr1_sem'],\
    cmr_stats['spc_frl4'], cmr_stats['spc_frl4_sem'] = calc_spc(recalls, sess, return_sem=True, listLength=ll)
    cmr_stats['pfr'], cmr_stats['pfr_sem'] = calc_pfr(recalls, sess, return_sem=True, listLength=ll)
    #cmr_stats['pli_perlist'], cmr_stats['pli_perlist_sem'] = calc_pli(cmr_intrusions, sessions, return_sem=True)
    
    cmr_stats['crp'], cmr_stats['crp_sem'] = calc_crp(recalls, sess, return_sem=True, listLength=ll)
    cmr_stats['crp1'], cmr_stats['crp1_sem'] = calc_crp(recalls[:, 0:2], sess, return_sem=True, listLength=ll)
    
    #irt fits
    irts = get_irts(rts, recalls)
    cmr_stats['opcrl'], cmr_stats['opcrl_sem'] = op_crl(irts, True)
    cmr_stats['rcrl'], cmr_stats['rcrl_sem'] = R_crl(irts, True)
    
    cmr_stats['opR_crl'] = opR_crl(irts, sess, ll=ll)

    cmr_stats['fRT'] = firstRecallDist(rts/1000)
    cmr_stats['irt0'] = nthIrtDist(irts, n=1)
    
    cmr_stats['avgTotalRecalls'] = avgTotalRecalls(recalls)

    # Score the model's behavioral stats as compared with the true data
    err = mean_squared_error(target_stats, cmr_stats)
    cmr_stats['err'] = err
    cmr_stats['params'] = param_vec
    

    return err, cmr_stats


def mean_squared_error(target_stats, cmr_stats):
    
    # dict_keys(['session', 'num_good_trials', 'p_rec', 'spc', 'pfr', 'crp', 'crl', 'pli_perlist', 'xli_perlist', 'rep_perlist'])
    
    y = []
    y_hat = []
    for stat in ['pfr', 'crp1']:#('spc', 'pfr', 'crp'):#, 'pli_perlist'):
        actual = np.atleast_1d(np.mean(target_stats[stat], axis=0))
        model = np.atleast_1d(cmr_stats[stat])
        if stat == 'crp1':
            actual[np.isnan(actual)] = 0
            model[np.isnan(model)] = 0
            print('crp1s:')
            print(actual)
            print(model)
        y.append(actual)
        y_hat.append(model)
    
    for stat in ['fRT', 'irt0', 'opR_crl']:#('opcrl', 'rcrl'):
        y.append(target_stats[stat])
        y_hat.append(cmr_stats[stat])
        print('%ss:' % stat)
        print(target_stats[stat], len(target_stats[stat]))
        print(cmr_stats[stat], len(cmr_stats[stat]))
    
    for stat in ['avgTotalRecalls']:
        actual = np.atleast_1d(target_stats[stat])
        model = np.atleast_1d(cmr_stats[stat])
        if stat == 'avgTotalRecalls':
            print('Avg Total Rs:')
            print(target_stats[stat])
            print(cmr_stats[stat])
        
    #for stat in ('pli_perlist', 'pli_recency'):
    #    y.append(np.atleast_1d(target_stats[stat]))
    #    y_hat.append(np.atleast_1d(cmr_stats[stat]))
    
    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)
    
    mse = np.mean((y - y_hat) ** 2)
    if np.isnan(mse):
        return np.inf
    return mse

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    basepath=''
    param_vec = np.array([ 0.60079213,  0.69040743,  0.67090956,  0.68212589,  3.99173832,
         1.01644554,  0.39983835,  0.02535611,  3.55759078,  0.86019453,
        17.1947069 ,  0.91931729,  0.04958535,  0.19153428,  0.73154344])
            
    # Load lists from participants who were not excluded in the behavioral analyses
    file_list = glob(basepath+'/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP*.json')
    for i in file_list:
        if "incomplete" in i:
            file_list.remove(i)
            
    #make it shorter for testing easy amounts of data
    #file_list = [basepath+'/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP%s.json' % SUBJ]
    
    # Set file paths for data, wordpool, and semantic similarity matrix
    wordpool_file = basepath+'/home1/shai.goldman/pyCMR2/CMR2_Optimized/wordpools/PEERS_wordpool.txt'
    w2v_file = basepath+'/home1/shai.goldman/pyCMR2/CMR2_Optimized/wordpools/PEERS_w2v.txt'
    target_stat_file = basepath+'/home1/shai.goldman/pyCMR2/IRT_Optimizations/target_stats.json'

    # Load data
    print('Loading data...')
    data_pres, sessions, sources = get_data(file_list, wordpool_file)
    sources = None  # Leave source features out of the model for the between-subjects experiment
    
    #run 5X on the data:
    #data_pres = np.concatenate(np.repeat(np.array([data_pres]), 5, axis=0))  
    #sessions = np.concatenate(np.repeat(np.array([sessions]), 5, axis=0))   
    
    
    # Load semantic similarity matrix (word2vec)
    w2v = np.loadtxt(w2v_file)

    # Load target stats from JSON file
    with open(target_stat_file, 'r') as f:
        targets = json.load(f)
    for key in targets:
        if isinstance(targets[key], list):
            targets[key] = np.array(targets[key], dtype=float)
        if isinstance(targets[key], dict):
            for subkey in targets[key]:
                if isinstance(targets[key][subkey], list):
                    targets[key][subkey] = np.array(targets[key][subkey], dtype=float)
    
    err, cmr_stats = obj_func(param_vec, targets, data_pres, sessions, w2v, source_mat=None)
    
    print(cmr_stats)