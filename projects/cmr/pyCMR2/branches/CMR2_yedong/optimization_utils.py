import json
import numpy as np
import scipy.stats as ss
import pyximport; pyximport.install()
import CMR2_pack_cyth as CMR2
import lagCRP2
from pybeh.spc import spc
from pybeh.pfr import pfr
from pybeh.mask_maker import make_clean_recalls_mask2d
import numpy.ma as ma


def recode_for_spc(data_recs, data_pres):
    ll = data_pres.shape[1]
    maxlen = ll * 2

    rec_lists = []
    for i in range(len(data_recs)):
        this_list = data_recs[i]
        pres_list = data_pres[i]

        this_list = this_list[this_list > 0]

        # get indices of first place each unique value appears
        indices = np.unique(this_list, return_index=True)[1]

        # get each unique value in array (by first appearance)
        this_list_unique = this_list[sorted(indices)]

        # get the indices of these values in the other list, and add 1
        list_recoded = np.nonzero(this_list_unique[:, None] == pres_list)[1] + 1

        # re-pad with 0's so we can reformat this as a matrix again later
        recoded_row = np.pad(list_recoded, pad_width=(
            0, maxlen - len(list_recoded)),
                             mode='constant', constant_values=0)

        # append to running list of recoded rows
        rec_lists.append(recoded_row)

    # reshape as a matrix
    recoded_lists = np.asmatrix(rec_lists)

    return recoded_lists

def lag1_all(lag_1=None):
    return True
def lag1_range_1(lag_1=None):
    return lag_1 == 1
def lag1_range_2(lag_1=None):
    return lag_1 == -1
def lag1_range_3(lag_1=None):
    return lag_1 > 3 or lag_1 < -3

def crp(recalls=None, subjects=None, listLength=None, lag_num=None, lag_range=None, skip_first_n=0):
    # Convert recalls and subjects to numpy arrays
    recalls = np.array(recalls)
    subjects = np.array(subjects)
    # Get a list of unique subjects -- we will calculate a CRP for each
    usub = np.unique(subjects)
    # Number of possible lags = (listLength - 1) * 2 + 1; e.g. a length-24 list can have lags -23 through +23
    num_lags = 2 * listLength - 1
    # Initialize array to store the CRP for each subject (or other unique identifier)
    result = np.zeros((usub.size, num_lags))
    # Initialize arrays to store transition counts
    actual = np.empty(num_lags)
    poss = np.empty(num_lags)
    

    # For each subject/unique identifier
    for i, subj in enumerate(usub):
        # Reset counts for each participant
        actual.fill(0)
        poss.fill(0)
        # Create trials x items matrix where item j, k indicates whether the kth recall on trial j was a correct recall
        clean_recalls_mask = np.array(make_clean_recalls_mask2d(recalls[subjects == subj]))
        # For each trial that matches that identifier
        for j, trial_recs in enumerate(recalls[subjects == subj]):
            seen = set()
            for k, rec in enumerate(trial_recs[:-1]):
                seen.add(rec)
                lag1_valid = False
                if(k!=0):
                    prev_rec = trial_recs[k - 1]
                    lag1_valid = lag_range(rec - prev_rec)
                # Only increment transition counts if the current and next recall are BOTH correct recalls
                if clean_recalls_mask[j, k] and clean_recalls_mask[j, k + 1] and k >= skip_first_n and lag1_valid:
                    next_rec = trial_recs[k + 1]
                    pt = np.array([trans for trans in range(1 - rec, listLength + 1 - rec) if rec + trans not in seen], dtype=int)
                    poss[pt + listLength - 1] += 1
                    trans = next_rec - rec
                    # Record the actual transition that was made
                    actual[trans + listLength - 1] += 1

        result[i, :] = actual / poss
        result[i, poss == 0] = np.nan

    result[:, listLength - 1] = np.nan

    return result[:, listLength - lag_num - 1:listLength + lag_num]

def calc_spc(recalls, sessions, listLength=24):

    s = spc(recalls, subjects=sessions, listLength=listLength)
    
    sem = ss.sem(s, axis=0, nan_policy='omit')
    
    if ma.is_masked(sem):
        sem = sem.data
    
    return np.nanmean(s, axis=0), sem


def calc_pfc(recalls, sessions, listLength=24):
    
    s = np.array(pfr(recalls, subjects=sessions, listLength=listLength))

    sem = ss.sem(s, axis=0)
    if ma.is_masked(sem):
        sem = sem.data
    
    return s.mean(axis=0), sem


def calc_crp(recalls, sessions, range_num = 0, lag_num=5, listLength=24):
    func = lag1_all
    if range_num == 1:
        func = lag1_range_1
    elif range_num == 2:
        func = lag1_range_2
    elif range_num == 3:
        func = lag1_range_3
    
    s = np.array(crp(recalls, subjects=sessions, listLength=listLength, lag_num=lag_num, lag_range = func, skip_first_n=2))
    
    sem = ss.sem(s, axis=0, nan_policy='omit')
    if ma.is_masked(sem):
        sem = sem.data
    
    return np.nanmean(s, axis=0), sem


def param_vec_to_dict(param_vec):

    dt = 10.
    # Convert parameter vector to dictionary format expected by CMR2
    param_dict = {

        'beta_enc': param_vec[0],
        'beta_rec': param_vec[1],
        'gamma_fc': param_vec[2],
        'gamma_cf': param_vec[3],
        'scale_fc': 1 - param_vec[2],
        'scale_cf': 1 - param_vec[3],

        'phi_s': param_vec[4],
        'phi_d': param_vec[5],
        'phi_s_fc': param_vec[14],
        'phi_d_fc': param_vec[15],
        'kappa': param_vec[6],

        'eta': param_vec[7],
        's_cf': param_vec[8],
        's_fc': param_vec[14],
        'beta_rec_post': param_vec[9],
        'omega': param_vec[10],
        'alpha': param_vec[11],
        'c_thresh': param_vec[12],
        'dt': dt,

        'lamb': param_vec[13],
        'rec_time_limit': 75000.,

        'dt_tau': dt / 1000.,
        'sq_dt_tau': (dt / 1000.) ** .5,
        'nlists_for_accumulator': 4,
    }

    return param_dict

# files: files to use
# path: contains all pres_nos/rec_nos files for the current focus; ex. typically "/home1/yedong/svn/pyCMR2/branches/CMR2_yedong/quart_1/"; 
def obj_func(param_vec, files, w2v):
    
    # Reformat parameter vector to the dictionary format expected by CMR2
    cur_param_dict = param_vec_to_dict(param_vec)
    
    # cur_param_dict = {
    #     'beta_enc': 2.253780040066615253e-01,
    #     'beta_rec': 4.260348999729270947e-01,
    #     'gamma_fc': 4.494940527039775757e-01,
    #     'gamma_cf': 6.694723191673451757e-01,
    #     'scale_fc': 1 - 4.494940527039775757e-01,
    #     'scale_cf': 1 - 6.694723191673451757e-01,
        
    #     'phi_s_fc': phi_s_fc,
    #     'phi_d_fc': phi_d_fc,

    #     'phi_s': 1.085769733622018007e+00,
    #     'phi_d': 3.666568790786536858e-01,
    #     'kappa': 2.716580908964218999e-01,

    #     'eta': 2.427902950429073337e-01,
    #     's_cf': 2.169717142825280831e+00,
    #     's_fc': 0.0,
    #     'beta_rec_post': 2.347880732868359299e-01,
    #     'omega': 1.032131665719187730e+01,
    #     'alpha': 9.251302062818872463e-01,
    #     'c_thresh': 3.584008911233489414e-01,
    #     'dt': 10.0,

    #     'lamb': 1.401249207934799623e-01,
    #     'rec_time_limit': 75000,

    #     'dt_tau': 0.01,
    #     'sq_dt_tau': 0.10,

    #     'nlists_for_accumulator': 4
    # }
    
    
     # Set LSA and data paths
    LSA_path = '/home1/yedong/svn/pyCMR2/branches/CMR2_yedong/w2v.txt'
    # Load the inter-item similarity matrix. No longer LSA; now w2v, but
    # didn't want to change all the variable names.
    LSA_mat = np.loadtxt(LSA_path)

    ##############################################calculate stats!##############################################
    this_recalls = []
    target_recalls = []
    sess = []
    
    for files in files:
        pres_nos = np.load(files[0])
        rec_nos = np.load(files[1])
        recalls = recode_for_spc(rec_nos, pres_nos)
        this_recalls.append(recalls)
        
        rec_nos, times = CMR2.run_CMR2(LSA_path, LSA_mat, files[0], cur_param_dict, sep_files=False)
        recalls = recode_for_spc(rec_nos, pres_nos)
        target_recalls.append(recalls)
    
        sess.append([files[1][8:-4] for i in range(len(recalls))])
        
    this_recalls = np.concatenate(this_recalls)
    target_recalls = np.concatenate(target_recalls)
    sess = np.concatenate(sess)
        
    target_spc, this_spc, target_spc_sem, target_pfc, this_pfc, target_pfc_sem, \
        target_left_crp, this_left_crp, target_left_crp_sem,\
        target_right_crp, this_right_crp, target_right_crp_sem, \
        target_left_crp1, this_left_crp1, target_left_crp_sem1, \
        target_right_crp1, this_right_crp1, target_right_crp_sem1, \
        target_left_crp2, this_left_crp2, target_left_crp_sem2, \
        target_right_crp2, this_right_crp2, target_right_crp_sem2, \
        target_left_crp3, this_left_crp3, target_left_crp_sem3, \
        target_right_crp3, this_right_crp3, target_right_crp_sem3 = get_beh(this_recalls, target_recalls, sess)
    
    ##################################################################################################################


    # get the error vectors for each type of analysis
    # fit for rando stuff for now; will specify ranges later

    # error for just part of spc values: (although there is nothing really primacy-sensitive about spc?)
    e1 = np.subtract(target_spc[:9], this_spc[:9])
    e1_norm = np.divide(e1, target_spc_sem[:9])

    # error for just part of pfr values: (since we are dealing with recency here)
    e2 = np.subtract(target_pfc[:3], this_pfc[:3])
    e2_norm = np.divide(e2, target_pfc_sem[:3])

    # error for left and right crps: 
    # now I'm only passing in top 3 I can use as it is
    e3 = np.subtract(target_left_crp, this_left_crp)
    e3_norm = np.divide(e3, target_left_crp_sem)

    e4 = np.subtract(target_right_crp, this_right_crp)
    e4_norm = np.divide(e4, target_right_crp_sem)

    # error for left and right crps1:
    e5 = np.subtract(target_left_crp1[:-1], this_left_crp1[:-1])
    e5_norm = np.divide(e5, target_left_crp_sem1[:-1])

    e6 = np.subtract(target_right_crp1, this_right_crp1)
    e6_norm = np.divide(e6, target_right_crp_sem1)

    # error for left and right crps2:
    e7 = np.subtract(target_left_crp2[1:], this_left_crp2[1:])
    e7_norm = np.divide(e7, target_left_crp_sem2[1:])

    e8 = np.subtract(target_right_crp2[1:], this_right_crp2[1:])
    e8_norm = np.divide(e8, target_right_crp_sem2[1:])

    # error for left and right crps3:
    e9 = np.subtract(target_left_crp3, this_left_crp3)
    e9_norm = np.divide(e9, target_left_crp_sem3)

    e10 = np.subtract(target_right_crp3, this_right_crp3)
    e10_norm = np.divide(e10, target_right_crp_sem3)
    
    ###
    
    numerator = (np.nansum(e1_norm ** 2)  # spc, first 9 (9) 
                    + np.nansum(e2_norm ** 2)  # PFR, first 3 (3)
                    + np.nansum(e3_norm ** 2)  # left lag-crp (3)
                    + np.nansum(e4_norm ** 2) # right lag-crp (3)
                    
                    + np.nansum(e5_norm ** 2)  # left lag-crp1 (2)
                    + np.nansum(e6_norm ** 2) # right lag-crp1 (2)
                    + np.nansum(e7_norm ** 2)  # left lag-crp2 (2)
                    + np.nansum(e8_norm ** 2) # right lag-crp2 (2)
                    + np.nansum(e9_norm ** 2)  # left lag-crp3 (3)
                    + np.nansum(e10_norm ** 2) # right lag-crp3 (3)
                   )
    denominator = (len(e1) + len(e2)
                      + len(e3) + len(e4)
                      + len(e5) + len(e6)
                      + len(e7) + len(e8)
                      + len(e9) + len(e10))


    # this one is only the lag-crp values and spc and PLI and ELI
    RMSE_normed = (numerator / denominator ) ** 0.5
 
    # keep stats for later graphing
    
    cmr_stats = dict()
    
    cmr_stats['target_spc'] = target_spc
    cmr_stats['this_spc'] = this_spc
    
    cmr_stats['target_pfc'] = target_pfc
    cmr_stats['this_pfc'] = this_pfc
    
    cmr_stats['target_left_crp'] = target_left_crp
    cmr_stats['this_left_crp'] = this_left_crp
    
    cmr_stats['target_right_crp'] = target_right_crp
    cmr_stats['this_right_crp'] = this_right_crp
    
    cmr_stats['target_left_crp1'] = target_left_crp1
    cmr_stats['this_left_crp1'] = this_left_crp1
    
    cmr_stats['target_right_crp1'] = target_right_crp1
    cmr_stats['this_right_crp1'] = this_right_crp1
    
    cmr_stats['target_left_crp2'] = target_left_crp2
    cmr_stats['this_left_crp2'] = this_left_crp2
    
    cmr_stats['target_right_crp2'] = target_right_crp2
    cmr_stats['this_right_crp2'] = this_right_crp2
    
    cmr_stats['target_left_crp3'] = target_left_crp3
    cmr_stats['this_left_crp3'] = this_left_crp3
    
    cmr_stats['target_right_crp3'] = target_right_crp3
    cmr_stats['this_right_crp3'] = this_right_crp3

    cmr_stats['err'] = RMSE_normed
    cmr_stats['params'] = cur_param_dict
    
    return RMSE_normed, cmr_stats

# inter-item similarity matrix now w2v
def get_beh(actual, predicted, sess):
    
    target_spc, target_spc_sem = calc_spc(predicted, sess)
    this_spc, _ = calc_spc(actual, sess)
    
    target_pfc, target_pfc_sem = calc_pfc(predicted, sess)
    this_pfc, _ = calc_pfc(actual, sess)
    
    
    # get Lag-CRP sections of interest
    
    num_crp = 3
    
    target_crp, target_crp_sem = calc_crp(predicted, sess, lag_num=num_crp)
    this_crp, _ = calc_crp(actual, sess, lag_num=num_crp)
    
    target_crp1, target_crp_sem1 = calc_crp(predicted, sess, range_num = 1, lag_num=num_crp)
    this_crp1, _ = calc_crp(actual, sess, range_num = 1, lag_num=num_crp)
    
    target_crp2, target_crp_sem2 = calc_crp(predicted, sess, range_num = 2, lag_num=num_crp)
    this_crp2, _ = calc_crp(actual, sess, range_num = 2, lag_num=num_crp)
    
    target_crp3, target_crp_sem3 = calc_crp(predicted, sess, range_num = 3, lag_num=num_crp)
    this_crp3, _ = calc_crp(actual, sess, range_num = 3, lag_num=num_crp)
    
    
    # set any SEM values that are equal to 0.0, equal to 1.0
    # (i.e., leave values as is)
    target_spc_sem[target_spc_sem == 0.0] = 1.0
    target_crp_sem[target_crp_sem == 0.0] = 1.0
    target_crp_sem1[target_crp_sem1 == 0.0] = 1.0
    target_crp_sem2[target_crp_sem2 == 0.0] = 1.0
    target_crp_sem3[target_crp_sem3 == 0.0] = 1.0
    target_pfc_sem[target_pfc_sem == 0.0] = 1.0

   
       
    # for all only use 3 values for now
    target_left_crp = target_crp[:num_crp]
    target_left_crp_sem = target_crp_sem[:num_crp]

    target_right_crp = target_crp[num_crp+1:]
    target_right_crp_sem = target_crp_sem[num_crp+1:]
    
    target_left_crp1 = target_crp1[:num_crp]
    target_left_crp_sem1 = target_crp_sem1[:num_crp]

    target_right_crp1 = target_crp1[num_crp+1:]
    target_right_crp_sem1 = target_crp_sem1[num_crp+1:]
    
    target_left_crp2 = target_crp2[:num_crp]
    target_left_crp_sem2 = target_crp_sem2[:num_crp]

    target_right_crp2 = target_crp2[num_crp+1:]
    target_right_crp_sem2 = target_crp_sem2[num_crp+1:]
    
    target_left_crp3 = target_crp3[:num_crp]
    target_left_crp_sem3 = target_crp_sem3[:num_crp]

    target_right_crp3 = target_crp3[num_crp+1:]
    target_right_crp_sem3 = target_crp_sem3[num_crp+1:]
    

    #######################################################################

    this_left_crp = this_crp[:num_crp]

    this_right_crp = this_crp[num_crp+1:]
    
    this_left_crp1 = this_crp1[:num_crp]

    this_right_crp1 = this_crp1[num_crp+1:]
    
    this_left_crp2 = this_crp2[:num_crp]

    this_right_crp2 = this_crp2[num_crp+1:]
    
    this_left_crp3 = this_crp3[:num_crp]

    this_right_crp3 = this_crp3[num_crp+1:]
        
    return (target_spc, this_spc, target_spc_sem, target_pfc, this_pfc, target_pfc_sem,\
            target_left_crp, this_left_crp,  target_left_crp_sem, \
            target_right_crp, this_right_crp, target_right_crp_sem, \
            target_left_crp1, this_left_crp1,  target_left_crp_sem1, \
            target_right_crp1, this_right_crp1, target_right_crp_sem1, \
            target_left_crp2, this_left_crp2,  target_left_crp_sem2, \
            target_right_crp2, this_right_crp2, target_right_crp_sem2, \
            target_left_crp3, this_left_crp3,  target_left_crp_sem3, \
            target_right_crp3, this_right_crp3, target_right_crp_sem3, \
           )