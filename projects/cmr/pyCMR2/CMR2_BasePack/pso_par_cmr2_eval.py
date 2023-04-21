import mkl
mkl.set_num_threads(1)
import numpy as np
import warnings
import os
import errno
import scipy.io
from glob import glob
import time
import sys
import pandas

import lagCRP2
import CMR2_pack_cyth as CMR2

"""
Dependencies: CMR2_pack.py, lagCRP2.py, plus all the package imports above.
              Must also have access to a data file & LSA or W2V file,
              as well as a file with the emotional valence information
              for the word list (if you want to look at emotional clustering.

This last updated on Thursday Aug 3, 2017
"""

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


def get_num_intrusions(data_recs, pres_orig_data, ndivisions, lists_per_div):
    """Return mean and sem prior-list intrusions and extra-list intrusions
    per section of lists

    A section can be determined either as a group of lists in a single session,
    or as in the case of the Kahana et al. 2002 data, as a group of lists
    belonging to a single subject, since each subject in that dataset has
    only one session of lists."""
    sum_PLIs_all = []
    sum_ELIs_all = []
    recoded_all = []
    for s in range(ndivisions):

        # get current division of data
        pres_sheet = pres_orig_data[(s*lists_per_div):(
            s*lists_per_div+lists_per_div)]
        # get responses of interest
        resp_sheet = data_recs[(s*lists_per_div):(
            s*lists_per_div+lists_per_div)]

        recoded_lists = []
        # for each list,
        for i in range(lists_per_div):

            # get this list
            resp_row = resp_sheet[i]

            # for each item presented in this list,
            recode_this_list = []
            for j in range(len(resp_row)):

                # get this item
                item = resp_row[j]

                # check each list;
                for k in range(lists_per_div):

                    # get the current list
                    test_list = pres_sheet[k]

                    # if the item is in this list, subtract
                    # the value of that list from the list number (i) that
                    # the item was presented in
                    if item in test_list:
                        recode_this_list.append(k-i)

            recoded_lists.append(recode_this_list)

        recoded_all.append(recoded_lists)

    sum_PLIs = 0
    sum_ELIs = 0

    # for each set of lists (i.e., session)
    for recoded_sheet in recoded_all:
        # for each lhist,
        for idx, each in enumerate(recoded_sheet):
            # for each item in that list,
            for item in each:
                # if item is -, item came from a preceding list
                if item < 0:
                    sum_PLIs += 1
                # if item is +, item came from the pool of remaining,
                # not-yet-presented items
                elif item > 0:
                    sum_ELIs += 1

        sum_PLIs_all.append(sum_PLIs)
        sum_ELIs_all.append(sum_ELIs)

    mn_PLIs = np.mean(sum_PLIs_all)
    mn_ELIs = np.mean(sum_ELIs_all)
    sem_PLIs = np.std(sum_PLIs_all)/(len(sum_PLIs_all) ** 0.5)
    sem_ELIs = np.std(sum_ELIs_all)/(len(sum_ELIs_all) ** 0.5)

    return mn_PLIs, mn_ELIs, sem_PLIs, sem_ELIs

def get_spc_pfc(rec_lists, ll):

    """Get spc and pfc for the recoded lists"""

    spclists = []
    pfclists = []
    for each_list in rec_lists:

        each_list = each_list[each_list > 0]

        # init. list to store whether or not an item was recalled
        spc_counts = np.zeros((1, ll))
        pfc_counts = np.zeros((1, ll))

        # get indices of where to put items in the list;
        # items start at 1, so index needs to -1
        spc_count_indices = each_list - 1
        spc_counts[0, spc_count_indices] = 1

        if each_list.shape[1] <= 0:
            continue
        else:
            # get index for first item in list
            pfc_count_index = each_list[0, 0] - 1
            pfc_counts[0, pfc_count_index] = 1

            spclists.append(np.squeeze(spc_counts))
            pfclists.append(np.squeeze(pfc_counts))

    # if no items were recalled, output a matrix of 0's
    if not spclists:
        spcmat = np.zeros((rec_lists.shape[0], ll))
    else:
        spcmat = np.array(spclists)

    if not pfclists:
        pfcmat = np.zeros((rec_lists.shape[0], ll))
    else:
        pfcmat = np.array(pfclists)

    # get mean and sem's for spc and pfc
    spc_mean = np.nanmean(spcmat, axis=0)
    spc_sem  = np.nanstd(spcmat, axis=0) / (len(spcmat) ** 0.5)

    pfc_mean = np.nanmean(pfcmat, axis=0)
    pfc_sem  = np.nanstd(pfcmat, axis=0) / (len(pfcmat) ** 0.5)

    return spc_mean, spc_sem, pfc_mean, pfc_sem


def getEmotCounts(presented_list):
    """Define helper function return the emotional valence counts
    for presented items"""

    # init no.s valenced items in this list
    sum_neg  = 0
    sum_pos  = 0
    sum_neut = 0

    # for each item in that list,
    for j in range(len(presented_list)):

        # if valid item ID,
        if presented_list[j] >= 0:

            # get index of this item in the valence key
            valence_key_index = int(presented_list[j]) - 1

            # get valence for this item
            val_j = word_valence_key.iloc[valence_key_index][1]

            # if item has valence, increment sum of valenced items
            if val_j < 4.0:
                sum_neg += 1
            elif val_j > 6.0:
                sum_pos += 1
            else:
                sum_neut += 1

    return [sum_neg, sum_pos, sum_neut]


def codeList(orig_list):
    """Define helper function to recode a given list into emotional pool IDs"""

    recoded_list = []
    for j in range(len(orig_list)):
        # if valid item ID,
        if orig_list[j] >= 0:

            # get index of this item in the valence key
            valence_key_index = int(orig_list[j]) - 1

            # get valence for this item
            val_j = word_valence_key.iloc[valence_key_index][1]

            # if item has valence, increment sum of valenced items
            if val_j < 4.0:
                recoded_list.append('Ng')
            elif val_j > 6.0:
                recoded_list.append('P')
            else:
                recoded_list.append('N')
        else:
            recoded_list.append('-1')

    return np.asarray(recoded_list)


def recode_rep_intrs(rec_list_0, pres_list_0):
    """Define helper function to recode repeats & intrusions as -1"""
    cleaned_list = np.zeros(len(rec_list_0))

    # for each item in list
    for i in range(len(rec_list_0)):

        item = rec_list_0[i]
        cleaned_list[i] = item

        # recode intrusions as -3
        if item not in pres_list_0:
            cleaned_list[i] = -3

        # recode repeats as -2
        if i > 0:
            if (item in rec_list_0[0:i - 1]):
                cleaned_list[i] = -2

    return cleaned_list


def emot_val(pres_lists, rec_lists):
    """Main method to calculate emotional valences, etc."""

    # Initialize lists to hold transition probability scores
    # for this subject, across lists
    list_probs = []

    # if no. presented lists != no. recalled lists, skip this session
    # (assumed some kind of recording error)
    if pres_lists.shape[0] != rec_lists.shape[0]:
        all_list_means = []
    else:
        for row in range(pres_lists.shape[0]):  # for each row:

            # get item nos. in row w/o 0's
            rec_row = rec_lists[row][rec_lists[row] != 0]
            pres_row = pres_lists[row]

            # if no responses, skip this list
            if len(rec_row) == 0:
                continue
            else:
                # get number of emotion words in presented list
                # Ng, P, N <-- order of counts in list
                emot_counts = getEmotCounts(pres_row)

                # if pres list doesn't have >= 1 of at least one emot. word,
                # then skip it
                if (emot_counts[0] < 1) \
                        or (emot_counts[1] < 1) or (emot_counts[2] < 1):
                    continue
                else:
                    # recode any repeats or intrusions as -1
                    # squeeze out all -1 values
                    cleaned_list = recode_rep_intrs(rec_row, pres_row)

                    # recode rec_row w/ emotional valence pool IDs
                    rec_list_emot = codeList(cleaned_list)

                    # remove all -1 values
                    rec_list_analyze = rec_list_emot[rec_list_emot != '-1']

                    # if N valid words recalled is < 2, skip list
                    if len(rec_list_analyze) < 2:
                        continue
                    else:
                        count_ng = 0
                        count_p  = 0
                        count_n  = 0

                        # iterate through the list and sum
                        # no. of val-val transitions that occur
                        for i in range(len(rec_list_analyze)):

                            # keep running count of # items of
                            # particular valence
                            this_item = rec_list_analyze[i]

                            # keep running count of # items of
                            # particular valence remaining
                            list_tot_ng = emot_counts[0]
                            list_tot_p  = emot_counts[1]
                            list_tot_n  = emot_counts[2]

                            ng_remaining = list_tot_ng - count_ng
                            p_remaining  = list_tot_p - count_p
                            n_remaining  = list_tot_n - count_n

                            # initialize transition scores
                            score_ng = 0
                            score_p  = 0
                            score_n  = 0

                            if i == 0:
                                if this_item == 'Ng':
                                    count_ng += 1
                                elif this_item == 'P':
                                    count_p += 1
                                elif this_item == 'N':
                                    count_n += 1
                            if i > 0:
                                # get previous item (base of the transition)
                                prev_item = rec_list_analyze[i - 1]

                                # get transition scores
                                if this_item == 'Ng':
                                    score_ng += 1
                                elif this_item == 'P':
                                    score_p  += 1
                                elif this_item == 'N':
                                    score_n  += 1

                                # get observed probabilities for this step
                                if ng_remaining != 0:
                                    obs_prob_ng  = score_ng / ng_remaining
                                else:
                                    obs_prob_ng  = np.nan

                                if p_remaining  != 0:
                                    obs_prob_p   = score_p / p_remaining
                                else:
                                    obs_prob_p   = np.nan

                                if n_remaining  != 0:
                                    obs_prob_n   = score_n / n_remaining
                                else:
                                    obs_prob_n   = np.nan

                                # append to appropriate base-pair list
                                if prev_item   == 'Ng':
                                    list_probs.append(
                                        [obs_prob_ng, obs_prob_p, obs_prob_n,
                                         np.nan, np.nan, np.nan,
                                         np.nan, np.nan, np.nan])
                                elif prev_item == 'P':
                                    list_probs.append(
                                        [np.nan, np.nan, np.nan,
                                         obs_prob_ng, obs_prob_p, obs_prob_n,
                                         np.nan, np.nan, np.nan])
                                elif prev_item == 'N':
                                    list_probs.append(
                                        [np.nan, np.nan, np.nan,
                                         np.nan, np.nan, np.nan,
                                         obs_prob_ng, obs_prob_p, obs_prob_n])

                                # update running item counts
                                if this_item == 'Ng':
                                    count_ng += 1
                                elif this_item == 'P':
                                    count_p += 1
                                elif this_item == 'N':
                                    count_n += 1

        # Ignore np.nanmean warning for averaging across NaN-only slices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # average down columns to get ave transition probs across lists
            all_list_means = np.nanmean(list_probs, axis=0)

            # get SEM down columns
            all_list_std = np.nanstd(list_probs, axis=0)
            all_list_sem = all_list_std/(np.count_nonzero(
                ~np.isnan(list_probs), axis=0)**0.5)

    return all_list_means, all_list_sem


def obj_func(param_vec):
    """Error function that we want to minimize"""

    # pso sometimes will try to assign eta_val = 0.0.  Do not allow this.
    if param_vec[7] > 0.0:
        eta_val = param_vec[7]
    else:
        eta_val = .001

    # desired model parameters
    param_dict = {

        'beta_enc': param_vec[0],
        'beta_rec': param_vec[1],
        'gamma_fc': param_vec[2],
        'gamma_cf': param_vec[3],
        'scale_fc': 1 - param_vec[2],
        'scale_cf': 1 - param_vec[3],

        'phi_s': param_vec[4],
        'phi_d': param_vec[5],
        'kappa': param_vec[6],

        'eta': eta_val,
        's_cf': param_vec[8],
        's_fc': 0.0,
        'beta_rec_post': param_vec[9],
        'omega':param_vec[10],
        'alpha': param_vec[11],
        'c_thresh': param_vec[12],
        'dt': 10.0,

        'lamb': param_vec[13],
        'rec_time_limit': 75000,

        'dt_tau': 0.01,
        'sq_dt_tau': 0.10,

        'nlists_for_accumulator': 2
    }

    rec_nos, times = CMR2.run_CMR2(LSA_path, LSA_mat, data_path, param_dict,
                                   sep_files=False)

    cmr_recoded_output = recode_for_spc(rec_nos, data_pres)

    # get the model's spc and pfc predictions:
    (this_spc, this_spc_sem, this_pfc,
     this_pfc_sem) = get_spc_pfc(cmr_recoded_output, ll)

    # get the model's crp predictions:
    this_crp, this_crp_sem = lagCRP2.get_crp(cmr_recoded_output, ll)

    center_val = ll - 1

    # get left crp values
    this_left_crp = this_crp[(center_val - 5):center_val]

    # get right crp values
    this_right_crp = this_crp[(center_val + 1):(center_val + 6)]

    # get metrics re: mean and sem of PLIs and ELIs
    this_PLI, this_ELI, this_PLI_sem, this_ELI_sem = get_num_intrusions(
        rec_nos, data_pres, ll, ll)

    # get eval clustering metrics
    this_eval_mean, this_eval_sem = emot_val(data_pres, rec_nos)

    # be careful not to divide by 0! some param sets may output 0 sem vec's.
    # if this happens, just leave all outputs alone.
    if np.nansum(this_spc_sem) == 0 \
            or np.nansum(this_pfc_sem) == 0 \
            or np.nansum(this_crp_sem) == 0:
        print("np.nansum equaled 0")
        this_spc_sem[range(len(this_spc_sem))] = 1
        this_pfc_sem[range(len(this_pfc_sem))] = 1
        this_crp_sem[range(len(this_crp_sem))] = 1

    # get the error vectors for each type of analysis

    # error for just part of spc values:
    e1_a = np.subtract(target_spc[:3], this_spc[:3])
    e1_a_norm = np.divide(e1_a, target_spc_sem[:3])

    e1_b = np.subtract(target_spc[22:], this_spc[22:])
    e1_b_norm = np.divide(e1_b, target_spc_sem[22:])

    e1_c = np.subtract(target_spc[11:13], this_spc[11:13])
    e1_c_norm = np.divide(e1_c, target_spc_sem[11:13])

    # error for just part of pfr values:
    e2_a = np.subtract(target_pfc[:5], this_pfc[:5])
    e2_a_norm = np.divide(e2_a, target_pfc_sem[:5])

    e2_b = np.subtract(target_pfc[19:], this_pfc[19:])
    e2_b_norm = np.divide(e2_b, target_pfc_sem[19:])

    # error for left and right crps:
    e3 = np.subtract(target_left_crp[2:], this_left_crp[2:])
    e3_norm = np.divide(e3, target_left_crp_sem[2:])

    e4 = np.subtract(target_right_crp[:3], this_right_crp[:3])
    e4_norm = np.divide(e4, target_right_crp_sem[:3])

    # error for PLI and ELI rates:
    e5 = target_PLI - this_PLI
    e5_norm = np.divide(e5, target_PLI_sem)

    e6 = target_ELI - this_ELI
    e6_norm = np.divide(e6, target_ELI_sem)

    # error for emotional valence
    e7 = target_eval_mean - this_eval_mean
    e7_norm = np.divide(e7, target_eval_sem)

    # this one is only the lag-crp values and spc and PLI and ELI
    RMSE_normed = ((np.nansum(e3_norm ** 2)  # left lag-crp (3)
                    + np.nansum(e4_norm ** 2)  # right lag-crp (3)
                    + np.nansum(e1_a_norm ** 2)  # spc, first 3 (3)
                    + np.nansum(e1_b_norm ** 2)  # spc, last 2 (2)
                    + np.nansum(e1_c_norm ** 2)  # spc, middle 2 (2)
                    + np.nansum(e5_norm ** 2)  # pli's (1)
                    + np.nansum(e6_norm ** 2)  # eli's (1)
                    + np.nansum(e7_norm ** 2))  # emotional valence (6)
                   / (len(e1_a) + len(e1_b) + len(e1_c) + len(e3)
                      + len(e4) + 2 + len(e7))) ** 0.5

    print("RMSE_normed is: ", RMSE_normed)

    # this is actually the sqrt of a chi^2 value but haven't had time to go through
    # and carefully change all the variable names.

    # If you want to recover the chi^2 value (I took the sqrt just to keep
    # things small, just square this value).

    return RMSE_normed


def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, 
        minstep=1e-8, minfunc=1e-8, debug=False):
    """
    Perform a particle swarm optimization (PSO)
   
    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
   
    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal 
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint 
        functions (Default: empty dict)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
   
    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
   
    """
   
    assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
   
    vhigh = np.abs(ub - lb)
    vlow = -vhigh
    
    # Check for constraint function(s) #########################################
    obj = lambda x: func(x, *args, **kwargs)
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            cons = lambda x: np.array([0])
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = lambda x: np.array([y(x, *args, **kwargs) for y in ieqcons])
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = lambda x: np.array(f_ieqcons(x, *args, **kwargs))
        
    def is_feasible(x):
        check = np.all(cons(x)>=0)
        return check
        
    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fp = np.zeros(S)  # best particle function values
    g = []  # best swarm position
    fg = 1e100  # artificial best swarm position starting value

    ###### Original code ######
    #fp_comp = np.zeros(S)
    #for i in range(S):
    #    fp_comp[i] = obj(p[i, :])
    ###########################
    iter0_tic = time.time()

    # os.O_CREAT --> create file if it does not exist
    # os.O_EXCL --> error if create and file exists
    # os.O_WRONLY --> open for writing only
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    obj_func_timer = time.time()
    rmse_list = []
    for idx, n in enumerate(p):

        match_file = '0tempfileb' + str(idx) + '.txt'
        try:
            # try to open the file
            fd = os.open(match_file, flags)

            # run this CMR object and get out the rmse
            rmse = func(n)
            rmse_list.append(rmse)

            # set up file contents
            file_input = str(rmse) + "," + str(idx)

            # open the empty file that accords with this
            os.write(fd, file_input.encode())
            os.close(fd)

            print("I did file: " + match_file)

        # OSError -> type of error raised for operating system errors
        except OSError as e:
            if e.errno == errno.EEXIST:     # errno.EEXIST means file exists
                continue
            else:
                raise

    print("Obj func round 0: " + str(time.time() - obj_func_timer))

    # raise ValueError("stop and check times for round 0")

    ########
    #
    #   Read in all the files and grab their RMSE values
    #
    ########

    # read in the files that start with "tempfileb" and sort numerically
    rmse_paths = sorted(glob('0tempfileb*'))

    # sanity check; make sure not too many temp files
    if len(rmse_paths) > S:
        raise ValueError("No. of temp files exceeds swarm size")

    # if length of rmse_paths is less than the swarm size (S),
    # then we are not done.  Wait 5 seconds and check again to see
    # if rmse_paths is now the right length.
    tic = time.time()
    while len(rmse_paths) < S:

        # don't check the paths more than once every 5 seconds
        time.sleep(5)

        # grab the paths again
        rmse_paths = sorted(glob('0tempfileb*'))

        #####
        #
        #   Test all files to see if they are empty
        #
        #####

        # if more than 2 minutes passes and it is not the right length,
        # then raise a value error / stop the code.
        if (time.time() - tic) > 120:
            raise ValueError(
                "Spent more than 2 mins waiting for processes to complete")

    ######
    #
    #   Check and see if any files are empty -- avoid race conditions
    #
    ######

    # check through all the paths to see if any are empty
    any_empty = True

    mini_tic = time.time()  # track time
    while any_empty:
        
        num_nonempty = 0

        # see if all paths are full
        for sub_path in rmse_paths:
            try:
                with open(
                        sub_path) as tfile:  # open file to avoid race c.
                    first_line = tfile.readline()  # read first line

                    if first_line == '':  # if first line is empty,
                        tfile.close()  # close file and break
                        break
                    else:  # if first line is not empty,
                        num_nonempty += 1  # increment count of non-empty files
                        tfile.close()
            except OSError as e:
                if e.errno == errno.EEXIST:  # as long as file exists, continue
                    continue
                else:           # if it was a different error, raise the error
                    raise

        if num_nonempty >= len(rmse_paths):
            any_empty = False

        # prevent infinite loops; run for max of 3 minutes
        if (time.time() - mini_tic) > 180:
            break

    # read in tempfilebs and get their rmse's & indices
    rmse_list0 = []
    for mini_path in rmse_paths:
        rmse_vec = np.genfromtxt(mini_path, delimiter=',', dtype=None)
        rmse_list0.append(rmse_vec.tolist())

    rmse_list0_sorted = sorted(rmse_list0, key=lambda tup: tup[1])
    fp = [tup[0] for tup in rmse_list0_sorted]

    #############
    #
    #   Initialize particle positions, velocities, & best position prior to
    #   beginning the swarm
    #
    #############

    for i in range(S):
        # Initialize the particle's position
        x[i, :] = lb + x[i, :]*(ub - lb)
   
        # Initialize the particle's best known position
        p[i, :] = x[i, :]
       
        # Calculate the objective's value at the current particle's
        # fp[i] = obj(p[i, :])
       
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        if i==0:
            g = p[0, :].copy()

        # If the current particle's position is better than the swarm's,
        # update the best swarm position
        if fp[i]<fg and is_feasible(p[i, :]):
            fg = fp[i]
            g = p[i, :].copy()
       
        # Initialize the particle's velocity
        v[i, :] = vlow + np.random.rand(D)*(vhigh - vlow)

    # if not already saved by another program / node,
    # save out the parameters' positions (x), best known positions (p),
    # and velocities (v).
    param_files = ['0xfileb.txt', '0pfileb.txt', '0vfileb.txt']
    param_entries = [x, p, v]
    for i in range(3):

        # check and see if the xfile, pfile, and vfile files have been
        # written.  If not, write them.
        try:
            # try to open the file
            os.open(param_files[i], flags)
        # OSError -> type of error raised for operating system errors
        except OSError as e:
            if e.errno == errno.EEXIST:  # errno.EEXIST means file exists
                continue
            else:
                raise

        # save out the x, p, or v parameter values, respectively
        np.savetxt(param_files[i], param_entries[i])

    toc = time.time()
    print("Iteration %i time: %f" % (0, toc - iter0_tic))

    ######
    #
    #   Swarm begins here
    #
    ######
       
    # Iterate until termination criterion met ##################################
    it = 1
    while it <= maxiter:
        # time how long this iteration took
        iter_tic = time.time()
        print("\nBeginning iteration " + str(it) + ":")

        # if the rmses file is already created for this iteration,
        # and it is non-empty (nbytes > 0),
        # then read in that file instead of re-calculating the rmse values
        this_rmsesb_file = "rmsesb_iter"+str(it)
        if (os.path.isfile(this_rmsesb_file)
            and (os.path.getsize(this_rmsesb_file) > 0.0)):
            rmse_list0 = np.loadtxt(this_rmsesb_file)

        else:
            #rp = np.random.uniform(size=(S, D))
            #rg = np.random.uniform(size=(S, D))

            # read in the noise files for this iteration
            rp = np.loadtxt('rp_iter' + str(it))
            rg = np.loadtxt('rg_iter' + str(it))

            # every 10 iterations, cleanup old temp files from the cwd
            old_rmse_paths = [] # init just in case

            # leave a buffer of the last 3 iterations' files
            # e.g., on iteration 10, we'll only clean up iter files 0-7
            if it == 10:
                old_rmse_paths = glob('[0-7]tempfileb*')
            elif it == 20:
                old_rmse_paths = glob('[8-9]tempfileb*') + glob('1[0-7]tempfileb*')
            elif it == 30:
                old_rmse_paths = glob('1*tempfileb*') + glob('2[0-7]tempfileb*')
            elif it == 40:
                old_rmse_paths = glob('2*tempfileb*') + glob('3[0-7]tempfileb*')
            elif it == 50:
                old_rmse_paths = glob('3*tempfileb*') + glob('4[0-7]tempfileb*')
            elif it == 60:
                old_rmse_paths = glob('4*tempfileb*') + glob('5[0-7]tempfileb*')
            elif it == 70:
                old_rmse_paths = glob('5*tempfileb*') + glob('6[0-7]tempfileb*')
            elif it == 80:
                old_rmse_paths = glob('6*tempfileb*') + glob('7[0-7]tempfileb*')
            elif it == 90:
                old_rmse_paths = glob('7*tempfileb*') + glob('8[0-7]tempfileb*')

            # mark cleanup points
            cleanup_points = [10, 20, 30, 40, 50, 60, 70, 80, 90]

            # if we have reached a cleanup point, clean up!
            if it in cleanup_points:
                for old_path in old_rmse_paths:
                    try:
                        # try to open the file (prevent race conditions)
                        cfile = os.open(old_path, os.O_RDONLY)

                        # if successfully opened the file, hold for a
                        # hundredth of a second with it open
                        time.sleep(.01)

                        # close the file
                        os.close(cfile)

                        # remove the file
                        os.remove(old_path)
                    except OSError as e:
                        # if can't open the file but file exists,
                        if e.errno == errno.EEXIST:
                            continue    # if file exists but is closed, move along to next file path
                        else:
                            continue    # if file does not exist, this is also okay; move along

            ###
            #   Read in the position, best, & velocity files from previous iteration
            ###
            x = []
            p = []
            v = []
            # make sure we get a full file with S entries
            no_inf_loops = time.time()
            while (len(x) < S) or (len(p) < S) or (len(v) < S):
                
                x = np.loadtxt(str(it-1) + 'xfileb.txt')
                p = np.loadtxt(str(it-1) + 'pfileb.txt')
                v = np.loadtxt(str(it-1) + 'vfileb.txt')

                # When we are getting out a full file, keep going
                if len(x) == S and len(p) == S and len(v) == S:
                    break
                else:
                    time.sleep(2)   # sleep 2 seconds before we try again

                if (time.time() - no_inf_loops) > 120:
                    raise ValueError("Incomplete entries in x, p, or v file")

            ###
            #   First update all particle positions
            ###
            for i in range(S):

                # Update the particle's velocity
                v[i, :] = omega*v[i, :] + phip*rp[i, :]*(p[i, :] - x[i, :]) + \
                          phig*rg[i, :]*(g - x[i, :])

                # Update the particle's position, correcting lower and upper bound
                # violations, then update the objective function value
                x[i, :] = x[i, :] + v[i, :]
                mark1 = x[i, :]<lb
                mark2 = x[i, :]>ub
                x[i, mark1] = lb[mark1]
                x[i, mark2] = ub[mark2]

            ###
            #  Then get the objective function for each particle
            ###

            obj_func_timer_it = time.time()
            rmse_list = []
            for idx, n in enumerate(x):

                match_file = str(it) + 'tempfileb' + str(idx) + '.txt'
                try:
                    # try to open the file
                    fd = os.open(match_file, flags)

                    # run this CMR object and get out the rmse
                    rmse = func(n)
                    rmse_list.append(rmse)

                    # set up file contents
                    file_input = str(rmse) + "," + str(idx)

                    # write the file contents
                    os.write(fd, file_input.encode())

                    # close the file
                    os.close(fd)

                    print("I did file: " + match_file)

                # OSError -> type of error raised for operating system errors
                except OSError as e:
                    if e.errno == errno.EEXIST:  # errno.EEXIST means file exists
                        continue
                    else:
                        raise

            print("Iteration " + str(it) + " timer: "
                  + str(time.time() - obj_func_timer_it))

            # read in the files that start with "tempfileb" and sort numerically
            rmse_paths = sorted(glob(str(it) + 'tempfileb*'))

            # sanity check; make sure not too many temp files
            if len(rmse_paths) > S:
                raise ValueError("No. of temp files exceeds swarm size")

            # if length of rmse_paths is less than the swarm size (S),
            # then we are not done.  Wait 5 seconds and check again to see
            # if rmse_paths is now the right length.
            tic = time.time()
            while len(rmse_paths) < S:

                # don't check the paths more than once every 5 seconds
                time.sleep(5)

                # grab the paths again
                rmse_paths = sorted(glob(str(it) + 'tempfileb*'))

                # if more than 2 minutes passes and it is not the right length,
                # then raise a value error / stop the code.
                if (time.time() - tic) > 120:
                    raise ValueError(
                        "Spent more than 2 mins waiting for processes to complete")

            ######
            #
            #   Check and see if files are empty -- avoid race conditions
            #
            ######

            # check through all the paths to see if any are empty
            any_empty = True
            mini_tic = time.time()  # track time
            while any_empty:
                
                num_nonempty = 0

                # see if all paths are full
                for sub_path in rmse_paths:
                    try:
                        with open(
                                sub_path) as tfile:  # open file to avoid race c.
                            first_line = tfile.readline()  # read first line

                            if first_line == '':  # if first line is empty,
                                tfile.close()  # close file and break
                                break
                            else:  # if first line is not empty,
                                num_nonempty += 1  # increment count of non-empty files
                                tfile.close()
                    except OSError as e:
                        if e.errno == errno.EEXIST:  # as long as file exists, continue
                            continue
                        else:
                            raise

                if num_nonempty >= len(rmse_paths):
                    any_empty = False

                # prevent infinite loops; run for max of 3 minutes
                if (time.time() - mini_tic) > 180:
                    break

            # read in tempfilebs and get their rmse's & indices
            rmse_list0 = []
            for mini_path in rmse_paths:
                rmse_vec = np.genfromtxt(mini_path, delimiter=',', dtype=None)
                rmse_list0.append(rmse_vec.tolist())

        # get all the rmse values into one array / list
        rmse_sorted = sorted(rmse_list0, key=lambda tup: tup[1])
        fx = [tup[0] for tup in rmse_sorted]

        np.savetxt('rmsesb_iter'+str(it),rmse_sorted)

        ###
        # Then compare all the particles' positions
        ###
        for i in range(S):
            
            # Compare particle's best position (if constraints are satisfied)
            if fx[i]<fp[i] and is_feasible(x[i, :]):
                p[i, :] = x[i, :].copy()
                fp[i] = fx[i]

                # Compare swarm's best position to current particle's position
                # (Can only get here if constraints are satisfied)
                if fx[i]<fg:
                    if debug:
                        print('New best for swarm at iteration {:}: {:} {:}'.format(it, x[i, :], fx))

                    tmp = x[i, :].copy()
                    stepsize = np.sqrt(np.sum((g-tmp)**2))
                    if np.abs(fg - fx[i])<=minfunc:
                        print('Stopping search: Swarm best objective change less than {:}'.format(minfunc))
                        return tmp, fx[i]
                    elif stepsize<=minstep:
                        print('Stopping search: Swarm best position change less than {:}'.format(minstep))
                        return tmp, fx[i]
                    else:
                        g = tmp.copy()
                        fg = fx[i]

        ####
        #   Save this iteration of param files so that we can start again
        ####
        param_files = [str(it)+'xfileb.txt', str(it)+'pfileb.txt',
                       str(it)+'vfileb.txt']
        param_entries = [x, p, v]
        for i in range(3):

            # check and see if the xfile, pfile, and vfile files have been
            # written.  If not, write them.
            try:
                # try to open the file
                os.open(param_files[i], flags)
            # OSError -> type of error raised for operating system errors
            except OSError as e:
                if e.errno == errno.EEXIST:  # errno.EEXIST means file exists
                    continue
                else:
                    raise

            np.savetxt(param_files[i], param_entries[i])

        toc = time.time()
        print("Iteration %i time: %f" % (it, toc-iter_tic))

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
        it += 1

    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
    
    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    return g, fg


def main():
    #########
    #
    #   Define some helpful global (yikes, I know!) variables.
    #
    #########

    global ll, data_pres, data_rec, LSA_path, data_path, LSA_mat
    global target_spc, target_spc_sem, target_pfc, target_pfc_sem
    global target_left_crp, target_left_crp_sem
    global target_right_crp, target_right_crp_sem

    # Set LSA and data paths
    on_rhino = True
    if on_rhino:
        LSA_path = 'w2v.txt'
        data_path = 'pres_nos_LTP228.txt'
        rec_path = 'rec_nos_LTP228.txt'
        valence_key_path = './wordproperties_CSV.csv'
    else:
        LSA_path = 'w2v.txt'
        data_path = 'pres_nos_LTP228.txt'
        rec_path = 'rec_nos_LTP228.txt'
        valence_key_path = "./wordproperties_CSV.csv"

    # Load the inter-item similarity matrix. No longer LSA; now w2v, but
    # didn't want to change all the variable names.
    LSA_mat = np.loadtxt(LSA_path)

    # if getting data from a text file:
    data_pres = np.loadtxt(data_path, delimiter=',')
    data_rec = np.loadtxt(rec_path, delimiter=',')

    # set list length
    ll = 24
    # set n sessions
    nsessions = 24
    # set n lists per session
    lists_per_session=24

    # recode lists for spc, pfr, and lag-CRP analyses
    recoded_lists = recode_for_spc(data_rec, data_pres)

    # get spc & pfr
    target_spc, target_spc_sem, target_pfc, target_pfc_sem = \
        get_spc_pfc(recoded_lists, ll)

    target_crp, target_crp_sem = lagCRP2.get_crp(recoded_lists, ll)

    # set any SEM values that are equal to 0.0, equal to 1.0
    # (i.e., leave values as is)
    target_spc_sem[target_spc_sem == 0.0] = 1.0
    target_crp_sem[target_crp_sem == 0.0] = 1.0
    target_pfc_sem[target_pfc_sem == 0.0] = 1.0

    # get Lag-CRP sections of interest
    center_val = ll - 1

    target_left_crp = target_crp[center_val-5:center_val]
    target_left_crp_sem = target_crp_sem[center_val-5:center_val]

    target_right_crp = target_crp[center_val+1:center_val+6]
    target_right_crp_sem = target_crp_sem[center_val+1:center_val+6]

    global target_PLI, target_PLI_sem
    global target_ELI, target_ELI_sem

    # get mean and sem for the observed data's PLI's and ELI's
    target_ELI, target_PLI, \
    target_ELI_sem, target_PLI_sem = get_num_intrusions(
        data_rec, data_pres,
        lists_per_div=lists_per_session, ndivisions=nsessions)

    # make sure we do not later divide by 0 in case the sem's are 0
    if target_ELI_sem == 0:
        target_ELI_sem = 1
    if target_PLI_sem == 0:
        target_PLI_sem = 1

    global word_valence_key
    word_valence_key = pandas.read_csv(valence_key_path)

    global target_eval_mean, target_eval_sem
    # get the emotional valence-coded responses for a single subject
    target_eval_mean, target_eval_sem = emot_val(data_pres, data_rec)

    #############
    #
    #   set lower and upper bounds
    #
    #############

    lb = [0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.01, 0.01, 0.5, .1, 5.0, .5, .001, .01]
    ub = [1.0, 1.0, 0.7, 1.0, 3.0, 1.5, 0.5, 0.5, 3.0, 1.0, 15.0, 1.0, 0.8, 0.5]

    start_time = time.time()
    xopt, fopt = pso(obj_func, lb, ub, swarmsize=100, maxiter=30, debug=False)

    print(xopt)
    print("Run time: " + str(time.time() - start_time))

    # clear out all the remaining tempfiles.
    sys.stdout.flush()
    time.sleep(5)
    tempfileb_paths = glob('*tempfileb*')
    for mini_path in tempfileb_paths:
        if os.path.isfile(mini_path):
            os.remove(mini_path)

    np.savetxt('xoptb_LTP228.txt', xopt, delimiter=',', fmt='%f')


if __name__ == "__main__": main()
