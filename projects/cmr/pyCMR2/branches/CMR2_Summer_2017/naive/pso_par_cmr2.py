import mkl
mkl.set_num_threads(1)
import numpy as np
import os
import errno
import scipy.io
from glob import glob
import time
import sys
import pandas

import lagCRP2
import CMR2_pack_cyth_LTP228 as CMR2

"""
Dependencies: CMR2_pack.py, lagCRP2.py, plus all the package imports above.
              Must also have access to a data file & LSA or W2V file.

This last updated on Sunday Jul 9, 2017
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

    sum_PLIs = 0
    sum_ELIs = 0
    for idx, each in enumerate(recoded_lists):
        for item in each:
            if item < 0:
                sum_PLIs += 1
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

# insert CMR2 objective function here, and name it obj_func

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

        'nlists_for_accumulator': 4
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
    this_left_crp = this_crp[(center_val-5):center_val]

    # get right crp values
    this_right_crp = this_crp[(center_val+1):(center_val + 6)]

    # get metrics re: mean and sem of PLIs and ELIs
    this_PLI, this_ELI, this_PLI_sem, this_ELI_sem = get_num_intrusions(
        rec_nos, data_pres, ll, ll)

    # be careful not to divide by 0! some param sets may output 0 sem vec's.
    # if this happens, just leave all outputs alone.
    if np.nansum(this_spc_sem) == 0 \
            or np.nansum(this_pfc_sem) == 0 \
            or np.nansum(this_crp_sem) == 0:
        print("np.nansum equaled 0")
        this_spc_sem[range(len(this_spc_sem))] = 1
        this_pfc_sem[range(len(this_pfc_sem))] = 1
        this_crp_sem[range(len(this_crp_sem))] = 1

    print("Target values are: ")
    print("spc:")
    print(target_spc)
    print(target_spc_sem)
    print("pfr:")
    print(target_pfc)
    print(target_pfc_sem)
    print("lag-crp:")
    print(target_left_crp)
    print(target_right_crp)
    print("sem:")
    print(target_left_crp_sem)
    print(target_right_crp_sem)
    print()

    print("Model-predicted values are: ")
    print("spc:")
    print(this_spc)
    print("pfr:")
    print(this_pfc)
    print("lag-crp:")
    print(this_left_crp)
    print(this_right_crp)

    print("Data shape:")
    print(rec_nos.shape)
    print("Data first row:")
    print(rec_nos[0])

    # get the error vectors for each type of analysis
    e1 = np.subtract(target_spc, this_spc)
    e1_norm = np.divide(e1, target_spc_sem)

    e2 = np.subtract(target_pfc, this_pfc)
    e2_norm = np.divide(e2, target_pfc_sem)

    e3 = np.subtract(target_left_crp, this_left_crp)
    e3_norm = np.divide(e3, target_left_crp_sem)

    e4 = np.subtract(target_right_crp, this_right_crp)
    e4_norm = np.divide(e4, target_right_crp_sem)

    e5 = target_PLI - this_PLI
    e5_norm = np.divide(e5, target_PLI_sem)

    e6 = target_ELI - this_ELI
    e6_norm = np.divide(e6, target_ELI_sem)
    print("Target ELI is: {}, this ELI is: {}, taget PLI is: {}, this PLI is: {}").format(target_ELI, this_ELI, target_PLI, \
                                                                                         this_PLI)
    
    print("PLI_sem is: {}, ELI_sem is: {}, PLI_error normed is {}, ELI_error normed is {}").format(target_PLI_sem, \
                                                        target_ELI_sem, e5_norm, e6_norm)
    # calculate rmse / chi^2 value after norming the error terms
    nerr_denom = (
        len(e1_norm) + len(e2_norm) + len(e3_norm)
        + len(e4_norm) + 2)

    print("error is: ")
    print(e1)
    print(e2)
    print(e3)
    print(e4)
    print(e5)
    print(e6)

    print("normed error is: ")
    print(e1_norm)
    print(e2_norm)
    print(e3_norm)
    print(e4_norm)
    print(e5_norm)
    print(e6_norm)

    print("nerr_denom is: ", nerr_denom)

    sum_squared_errors = (
        np.nansum(e1_norm ** 2) + np.nansum(e2_norm ** 2)
             + np.nansum(e3_norm ** 2) + np.nansum(e4_norm ** 2)
        + np.nansum(e5_norm ** 2) + np.nansum(e6_norm ** 2))

    print("sum squared errors is: ", sum_squared_errors)

    print("SSE / nerr_denom is: ", nerr_denom)

    RMSE_normed = (sum_squared_errors / nerr_denom) ** 0.5

    # this one is only the lag-crp values and spc and PLI and ELI
    RMSE_normed = (
                      (np.nansum(e3_norm ** 2)
                       + np.nansum(e4_norm ** 2)
                       + np.nansum(e1_norm **2)
                       + np.nansum(e5_norm **2)
                       + np.nansum(e6_norm ** 2))
                   /(len(e1) + len(e3) + len(e4) + 2)) ** 0.5

    print("RMSE_normed is: ", RMSE_normed)

    # get just a regular rmse, not normed by the sem of the data
    # nerr_raw = len(e1) + len(e2) + len(e3) + len(e4)
    # sum_squared_err_raw = (np.nansum(e1**2) + np.nansum(e2**2)
    #                        + np.nansum(e3**2) + np.nansum(e4**2))
    # RMSE = (sum_squared_err_raw / nerr_raw) ** 0.5

    return RMSE_normed
#
# def obj_func(param_vec):
#     """Error function that we want to minimize"""
#
#     # pso sometimes will try to assign eta_val = 0.0.  Do not allow this.
#     if param_vec[7] > 0.0:
#         eta_val = param_vec[7]
#     else:
#         eta_val = .001
#
#     # desired model parameters
#     param_dict = {
#
#         'beta_enc': param_vec[0],
#         'beta_rec': param_vec[1],
#         'gamma_fc': param_vec[2],
#         'gamma_cf': param_vec[3],
#         'scale_fc': 1 - param_vec[2],
#         'scale_cf': 1 - param_vec[3],
#
#         'phi_s': param_vec[4],
#         'phi_d': param_vec[5],
#         'kappa': param_vec[6],
#
#         'eta': eta_val,
#         's_cf': param_vec[8],
#         's_fc': 0.0,
#         'beta_rec_post': param_vec[9],
#         'omega':param_vec[10],
#         'alpha': param_vec[11],
#         'c_thresh': param_vec[12],
#         'dt': 10.0,
#
#         'lamb': param_vec[13],
#         'rec_time_limit': 75000,
#
#         'dt_tau': 0.01,
#         'sq_dt_tau': 0.10,
#
#         'nlists_for_accumulator': 4
#     }
#
#     rec_nos, times = CMR2.run_CMR2(LSA_path, LSA_mat, data_path, param_dict,
#                                    sep_files=False)
#
#     cmr_recoded_output = recode_for_spc(rec_nos, data_pres)
#
#     # get the model's spc and pfc predictions:
#     (this_spc, this_spc_sem, this_pfc,
# 	this_pfc_sem) = get_spc_pfc(cmr_recoded_output, ll)
#
#     # get the model's crp predictions:
#     this_crp, this_crp_sem = lagCRP2.get_crp(cmr_recoded_output, ll)
#
#     center_val = ll - 1
#
#     # get left crp values
#     this_left_crp = this_crp[(center_val-5):center_val]
#
#     # get right crp values
#     this_right_crp = this_crp[(center_val+1):(center_val + 6)]
#
#     # be careful not to divide by 0! some param sets may output 0 sem vec's.
#     # if this happens, just leave all outputs alone.
#     if np.nansum(this_spc_sem) == 0 \
#             or np.nansum(this_pfc_sem) == 0 \
#             or np.nansum(this_crp_sem) == 0:
#         print("np.nansum equaled 0")
#         this_spc_sem[range(len(this_spc_sem))] = 1
#         this_pfc_sem[range(len(this_pfc_sem))] = 1
#         this_crp_sem[range(len(this_crp_sem))] = 1
#
#     print("Target values are: ")
#     print("spc:")
#     print(target_spc)
#     print(target_spc_sem)
#     print("pfr:")
#     print(target_pfc)
#     print(target_pfc_sem)
#     print("lag-crp:")
#     print(target_left_crp)
#     print(target_right_crp)
#     print("sem:")
#     print(target_left_crp_sem)
#     print(target_right_crp_sem)
#     print()
#
#     print("Model-predicted values are: ")
#     print("spc:")
#     print(this_spc)
#     print("pfr:")
#     print(this_pfc)
#     print("lag-crp:")
#     print(this_left_crp)
#     print(this_right_crp)
#
#     print("Data shape:")
#     print(rec_nos.shape)
#     print("Data first row:")
#     print(rec_nos[0])
#
#     # get the error vectors for each type of analysis
#     e1 = np.subtract(target_spc, this_spc)
#     divisor_e1 = target_spc.copy()
#     divisor_e1[divisor_e1 == 0.0] = 1.0
#     e1_norm = np.divide(e1**2, divisor_e1)
#
#     e2 = np.subtract(target_pfc, this_pfc)
#     divisor_e2 = target_pfc.copy()
#     divisor_e2[divisor_e2 == 0.0] = 1.0
#     e2_norm = np.divide(e2**2, divisor_e2)
#
#     e3 = np.subtract(target_left_crp, this_left_crp)
#     divisor_e3 = target_left_crp.copy()
#     divisor_e3[divisor_e3 == 0.0] = 1.0
#     e3_norm = np.divide(e3**2, divisor_e3)
#
#     e4 = np.subtract(target_right_crp, this_right_crp)
#     divisor_e4 = target_left_crp.copy()
#     divisor_e4[divisor_e4 == 0.0] = 1.0
#     e4_norm = np.divide(e4**2, divisor_e4)
#
#     # calculate chi^2 value after norming the error terms
#     #nerr_denom = len(e1_norm) + len(e2_norm) + len(e3_norm) + len(e4_norm)
#
#     print("error is: ")
#     print(e1)
#     print(e2)
#     print(e3)
#     print(e4)
#
#     print("normed error is: ")
#     print(e1_norm)
#     print(e2_norm)
#     print(e3_norm)
#     print(e4_norm)
#
#     #print("nerr_denom is: ", nerr_denom)
#
#     sum_squared_errors = (np.nansum(e1_norm) + np.nansum(e2_norm)
#              + np.nansum(e3_norm) + np.nansum(e4_norm))
#
#     print("sum squared errors is: ", sum_squared_errors)
#
#     #print("SSE / nerr_denom is: ", nerr_denom)
#
#     #RMSE_normed = (sum_squared_errors / nerr_denom) ** 0.5
#
#     #print("RMSE_normed is: ", RMSE_normed)
#
#     # get just a regular rmse, not normed by the sem of the data
#     # nerr_raw = len(e1) + len(e2) + len(e3) + len(e4)
#     # sum_squared_err_raw = (np.nansum(e1**2) + np.nansum(e2**2)
#     #                        + np.nansum(e3**2) + np.nansum(e4**2))
#     # RMSE = (sum_squared_err_raw / nerr_raw) ** 0.5
#
#     return sum_squared_errors

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

        match_file = '0tempfile' + str(idx) + '.txt'
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

    # read in the files that start with "tempfile" and sort numerically
    rmse_paths = sorted(glob('0tempfile*'))

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
        rmse_paths = sorted(glob('0tempfile*'))

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

    while any_empty:
        mini_tic = time.time()  # track time
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

    # read in tempfiles and get their rmse's & indices
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
    param_files = ['0xfile.txt', '0pfile.txt', '0vfile.txt']
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

        iter_tic = time.time()
        print("\nBeginning iteration " + str(it) + ":")

        # if the rmses file is already created for this iteration,
        # and it is non-empty (nbytes > 0),
        # then read in that file instead of re-calculating the rmse values
        this_rmses_file = "rmses_iter"+str(it)
        if (os.path.isfile(this_rmses_file)
            and (os.path.getsize(this_rmses_file) > 0.0)):
            rmse_list0 = np.loadtxt(this_rmses_file)

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
                old_rmse_paths = glob('[0-7]tempfile*')
            elif it == 20:
                old_rmse_paths = glob('[8-9]tempfile*') + glob('1[0-7]tempfile*')
            elif it == 30:
                old_rmse_paths = glob('1*tempfile*') + glob('2[0-7]tempfile*')
            elif it == 40:
                old_rmse_paths = glob('2*tempfile*') + glob('3[0-7]tempfile*')
            elif it == 50:
                old_rmse_paths = glob('3*tempfile*') + glob('4[0-7]tempfile*')
            elif it == 60:
                old_rmse_paths = glob('4*tempfile*') + glob('5[0-7]tempfile*')
            elif it == 70:
                old_rmse_paths = glob('5*tempfile*') + glob('6[0-7]tempfile*')
            elif it == 80:
                old_rmse_paths = glob('6*tempfile*') + glob('7[0-7]tempfile*')
            elif it == 90:
                old_rmse_paths = glob('7*tempfile*') + glob('8[0-7]tempfile*')

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
            while (len(x) < S) or (len(p) < S) or (len(v) < S):
                no_inf_loops = time.time()
                x = np.loadtxt(str(it-1) + 'xfile.txt')
                p = np.loadtxt(str(it-1) + 'pfile.txt')
                v = np.loadtxt(str(it-1) + 'vfile.txt')

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

                match_file = str(it) + 'tempfile' + str(idx) + '.txt'
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

            # read in the files that start with "tempfile" and sort numerically
            rmse_paths = sorted(glob(str(it) + 'tempfile*'))

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
                rmse_paths = sorted(glob(str(it) + 'tempfile*'))

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

            while any_empty:
                mini_tic = time.time()  # track time
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

            # read in tempfiles and get their rmse's & indices
            rmse_list0 = []
            for mini_path in rmse_paths:
                rmse_vec = np.genfromtxt(mini_path, delimiter=',', dtype=None)
                rmse_list0.append(rmse_vec.tolist())

        # get all the rmse values into one array / list
        rmse_sorted = sorted(rmse_list0, key=lambda tup: tup[1])
        fx = [tup[0] for tup in rmse_sorted]

        np.savetxt('rmses_iter'+str(it),rmse_sorted)

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
        param_files = [str(it)+'xfile.txt', str(it)+'pfile.txt',
                       str(it)+'vfile.txt']
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

    # Set LSA and data paths -- K02 data
    on_rhino = True
    if on_rhino:
        LSA_path = 'w2v.txt'
        data_path = 'pres_nos_LTP093.txt'
        rec_path = 'rec_nos_LTP093.txt'

    else:
        LSA_path = '/Users/KahaNinja/PycharmProjects/LinearPar/CMR2_lowmem/K02_LSA.txt'
        data_path = '/Users/KahaNinja/PycharmProjects/CMR2/K02_files/K02_data.mat'


    LSA_mat = np.loadtxt(LSA_path)

    ### comment this back in if getting data from a MATLAB file
    # get data file, presented items, & recalled items
    # data_file = scipy.io.loadmat(
    #     data_path, squeeze_me=True, struct_as_record=False)
    #
    # data_pres = data_file['data'].pres_itemnos      # presented
    # data_rec = data_file['data'].rec_itemnos        # recalled

    # if getting data from a text file:
    data_pres = np.loadtxt(data_path, delimiter=',')
    data_rec = np.loadtxt(rec_path, delimiter=',')

    # set list length
    ll = 24
    # set n sessions
    nsessions = 24
    # set n lists per session
    lists_per_session=24

    # recode lists for spc, pfc, and lag-CRP analyses
    recoded_lists = recode_for_spc(data_rec, data_pres)

    # get spc & pfc
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


    #############
    #
    #   set lower and upper bounds
    #
    #############

    lb = [0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.01, 0.01, 0.5, .1, 5.0, .5, .001, .01]
    ub = [1.0, 1.0, 0.7, 1.0, 3.0, 1.5, 0.5, 0.5, 3.0, 1.0, 15.0, 1.0, 0.8, 0.5]

    start_time = time.time()
    xopt, fopt = pso(obj_func, lb, ub, swarmsize=2, maxiter=2, debug=False)

    print(xopt)
    print("Run time: " + str(time.time() - start_time))

    sys.stdout.flush()
    time.sleep(5)
    tempfile_paths = glob('*tempfile*')
    for mini_path in tempfile_paths:
        if os.path.isfile(mini_path):
            os.remove(mini_path)

    np.savetxt('xopt_LTP093.txt', xopt, delimiter=',', fmt='%f')


if __name__ == "__main__": main()
