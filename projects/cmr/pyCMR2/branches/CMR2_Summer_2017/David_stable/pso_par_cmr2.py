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

def handle_intrusions(data_rec, data_pres, lists_per_div, data_rec_times, max_back = 6):
    #TODO Realize that there are defaults in this code that should not be used for non-ltpFR2
    #For example in get_tic_for_div we default the experiment time at 75000. This is a property
    #of the recall time for ltpFR
    
    ############################################
    #Syntax:
    #data_rec: the numpy matrix of patient recordings
    #data_pres: the numpy matrix of presented words
    #lists per div: How many lists are you looking at one time???
    #data_rec_times: the times at which recordings happen
    #How far back are you looking for ppli (see abbreviations for more detail)
    
    # Abbreviations and notes:
    # tic: temporal_intrusio_curve - basically the number of intrusion in any sixth of the
    #overall recall period
    # ppli: Probability of PLI in terms of how many lists back. Like probability that a PLI comes
    # from one list back etc. If you measure up to 10 back then this code starts at the 11th list
    
    #Asides: 
    #1. Each value is calculated separately to lower debug time and increase readibility at
    #the cost of runtime. The runtime of this isn't critical for CMR2 since its not in a constantly used part
    #but you have been warned
    ############################################
    def get_ppli_for_div(pres_div, rec_div, max_back):
        #pres div: presente words for this division
        #rec div: presented words for this division
        #max_back: the maximum of how far back you are looking
        #Returns div_ppli: the ppli for this division. Initialized below
        div_ppli = np.zeros(max_back)
        
        #Get the raw intrusion count
        num_pli = np.zeros(max_back)
        #Edit A - Deciding NOT to use contingent ppli but absolute--------------------------
        total_num_words = 0.0
        #-----------------------------------------------------------------------------------
        for list_number in range(max_back, len(rec_div)):
            #------------------------------------------------------------------
            recorded_words = list(rec_div[list_number])
            while 0 in recorded_words:
                recorded_words.remove(0)
            while -1 in recorded_words:
                recorded_words.remove(-1)
            total_num_words += len(recorded_words)
            #-------------------------------------------------------------------
            #Get only the things not in the current rec list
            of_interest = [x for x in rec_div[list_number] if \
                          x not in pres_div[list_number]]
            #Clean the of_interest array of 0s. These are filler
            while 0 in of_interest:
                of_interest.remove(0)
            #Check if it was in some previous list.
            for word in of_interest:
                for earlier_list_number, earlier_list in enumerate(pres_div[0:list_number]):
                    separation = list_number - earlier_list_number
                    #Don't pay attention to lists beyond a certain amount backward
                    if separation > max_back:
                        continue
                    if word in earlier_list and word != -1:
                        num_pli[separation -1] += 1
        if np.nansum(num_pli != 0):
            #--------------------------------------------------
            div_ppli = num_pli/float(np.nansum(num_pli))
            #----------------------------------------------
        return div_ppli 
    def get_tic_for_div(pres_div, rec_div, time_div, num_tics = 6, tot_time = 75000):
        
        #num_tics: number of separations for tic curve
        #tot_time: the total amount of time the experiment runs for
        
        #UPDATES: Normalized based on the overall number of recalled words
        #this is highlighted in the comments as EDIT1 you can comment out
        #these if you dont want this
        
        #Get the end of each separation
        end_of_seps = np.linspace(0, tot_time, num_tics+1)[1:] 
        #TIC curve for each time sep
        tic_for_div = np.zeros(num_tics)
       
        #EDIT1-------
        #total number of recalls in any temporal spot
        total_num_for_div = np.zeros(num_tics)
        #---------
        for list_number in range(0, len(rec_div)):
            rec_list = rec_div[list_number]
            pres_list = pres_div[list_number]
            time_list = time_div[list_number]
            for word_num, word in enumerate(rec_list):
                #EDIT1 --------
                if word != 0 and word != -1:
                    time = time_list[word_num]
                    for possible_time in range(num_tics):
                        if time < end_of_seps[possible_time]:
                            total_num_for_div[possible_time] += 1
                            #This is a trick to get the lowest end time rather than continuing
                            break
                #-------------
                 
                if word == 0 or word in pres_list:
                    continue
                for earlier_list in pres_div[0:list_number]:
                    if word in earlier_list and word != -1:
#                         print word
#                         print pres_list
#                         print rec_list
#                         print list_number
                        time = time_list[word_num]
#                         print time
                        for possible_time in range(num_tics):
                            if time < end_of_seps[possible_time]:
                                tic_for_div[possible_time] += 1
                                break
        #EDIT1----------
        for item_num in range(len(tic_for_div)):
            if total_num_for_div[item_num] == 0:
                tic_for_div[item_num] = 0
            else:
                tic_for_div[item_num] = \
                tic_for_div[item_num]/(1.*total_num_for_div[item_num])
        #------------
        return tic_for_div
    #MAIN
    tic_holder = []
    ppli_holder = []
    #Get the starting point for each division
    for division_start in range(0, len(data_pres), lists_per_div):
         #... and the ending point
        division_end = division_start + lists_per_div
        #Focus the variables to the division we're interested in here
        pres_div = data_pres[division_start: division_end]
        rec_div = data_rec[division_start: division_end]
        time_div = data_rec_times[division_start: division_end]
        division_ppli =  get_ppli_for_div(pres_div, rec_div, max_back)
        ppli_holder.append(division_ppli)
        #if division_start == 120:
        division_time = get_tic_for_div(pres_div, rec_div, time_div)
        tic_holder.append(division_time)
    ppli_holder = np.array(ppli_holder)
    ppli_holder = ppli_holder[~np.all(ppli_holder == 0, axis=1)]
    #return ppli_holder
    mean_ppli = ppli_holder.mean(axis = 0)
    sem_ppli = np.std(ppli_holder)/(len(ppli_holder) ** 0.5)
    
    tic_holder = np.array(tic_holder)
    mean_tic = tic_holder.mean(axis = 0)
    sem_tic = np.std(tic_holder)/(len(tic_holder) ** 0.5)
    return mean_ppli, sem_ppli, mean_tic, sem_tic

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
    def mean_list(numbers):
        return float(sum(numbers)) / max(len(numbers), 1)
    def scale(vector, threshold = .15, factor = 1.5):
        for val_num in xrange(len(vector)):
            if vector[val_num] > threshold:
                vector[val_num] = vector[val_num] *1.5
        return vector
           
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
    #TODO NOTE: print('Using ll where it should not be used... coincidence of the data .. not all data is like this')
    #this_PLI, this_ELI, this_PLI_sem, this_ELI_sem = get_num_intrusions(
    #    rec_nos, data_pres, ll, ll)
    this_ppli, this_sem_ppli, this_tic, this_sem_tic = handle_intrusions(rec_nos, data_pres,  \
                                           ll, times)
    #np.savetxt('ppli_debug.txt', rec_nos)
    print 'rec nos for simul is {}'.format(rec_nos)
    #TODO Delete the 2 lines below (or comment out). Done.
    #np.savetxt('rec_nos_d.txt', rec_nos, delimiter=',', fmt='%f')
    #np.savetxt('data_pres_d.txt', data_pres, delimiter=',', fmt='%f')
    # be careful not to divide by 0! some param sets may output 0 sem vec's.
    # if this happens, just leave all outputs alone.
    if np.nansum(this_spc_sem) == 0 \
            or np.nansum(this_pfc_sem) == 0 \
            or np.nansum(this_crp_sem) == 0:
        print("np.nansum equaled 0")
        this_spc_sem[range(len(this_spc_sem))] = 1
        this_pfc_sem[range(len(this_pfc_sem))] = 1
        this_crp_sem[range(len(this_crp_sem))] = 1
    
    
    #TODO this is hard-coded for ll = 24 fix this!
    #Adjusting to focus on six values for each curve (can be averages however)
    print 'model spc = {}'.format(target_spc)
    print 'model left crp is {}'.format(target_left_crp)
    print 'model right crp is {}'.format(target_right_crp)
    print 'model pfr is {}'.format(target_pfc)
    #The -1 is to prevent me from getting confused since position one is index 0 and I wrote everything down 
    #as positions
    #I also did not want to change the global value because it would fuck up the next iteration hence the lc (local)
    #SPC
    target_spc_lc = list(mean_list(list(target_spc[x[0] - 1:x[1] - 1])) for x in [[1,2],[5,6], [9,12], [19,20], [24, 25], [1,25]])
    this_spc_lc = list(mean_list(list(this_spc[x[0] - 1:x[1] - 1])) for x in [[1,2],[5,6], [9,12], [19,20], [24, 25], [1,25]])
    #PFR
    target_pfc_lc = list(mean_list(list(target_pfc[x[0] - 1:x[1] - 1])) for x in [[1,4],[3,6], [20,23], [22,25], [9, 13], [12,16]])
    this_pfc_lc = list(mean_list(list(this_pfc[x[0] - 1:x[1] - 1])) for x in [[1,4],[3,6], [20,23], [22,25], [9, 13], [12,16]])
    #lag CRP
    target_left_crp_lc = list(mean_list(list(target_left_crp[x[0] - 1:x[1] - 1])) for x in [[1,4],[4,5], [5,6]])
    this_left_crp_lc = list(mean_list(list(this_left_crp[x[0] - 1:x[1] - 1])) for x in [[1,4],[4,5], [5,6]])
    target_right_crp_lc = list(mean_list(list(target_right_crp[x[0] - 1:x[1] - 1])) for x in [[1,2],[2,3], [3,6]])
    this_right_crp_lc = list(mean_list(list(this_right_crp[x[0] - 1:x[1] - 1])) for x in [[1,2],[2,3], [3,6]])
    
    #Sems
    target_spc_sem_lc = list(mean_list(list(target_spc_sem[x[0] - 1:x[1] - 1])) for x in [[1,2],[5,6], [9,12], [19,20], [24, 25], [1,25]])
    target_pfc_sem_lc = list(mean_list(list(target_pfc_sem[x[0] - 1:x[1] - 1])) for x in [[1,4],[3,6], [20,23], [22,25], [9, 13], [12,16]])
    target_right_crp_sem_lc = list(mean_list(list(target_right_crp_sem[x[0] - 1:x[1] - 1])) for x in [[1,2],[2,3], [3,6]])
    target_left_crp_sem_lc = list(mean_list(list(target_left_crp_sem[x[0] - 1:x[1] - 1])) for x in [[1,4],[4,5], [5,6]])
    
    e1 = np.subtract(target_spc_lc, this_spc_lc)
    e1 = scale(e1)
    e1_norm = np.divide(e1, target_spc_sem_lc)

    e2 = np.subtract(target_pfc_lc, this_pfc_lc)
    e2 = scale(e2)
    e2_norm = np.divide(e2, target_pfc_sem_lc)

    e3 = np.subtract(target_left_crp_lc, this_left_crp_lc)
    e3 = scale(e3)
    e3_norm = np.divide(e3, target_left_crp_sem_lc)

    e4 = np.subtract(target_right_crp_lc, this_right_crp_lc)
    e4 = scale(e4)
    e4_norm = np.divide(e4, target_right_crp_sem_lc)
    
    e5 = np.subtract(target_ppli, this_ppli)
    e5_norm = np.divide(e5, target_sem_ppli)
    
    e6 = np.subtract(target_tic, this_tic)
    e6_norm = np.divide(e6, target_sem_tic)
    #e5 = np.subtract(target_right_crp, this_right_crp)
    #e5_norm = np.divide(e5, target_PLI_sem)

    #e6 = target_ELI - this_ELI
    #e6_norm = np.divide(e6, target_ELI_sem)

    # calculate rmse / chi^2 value after norming the error terms
    nerr_denom = (
        len(e1_norm) + len(e2_norm) + len(e3_norm)
        + len(e4_norm) + 2)
    #print("Target ELI is: {}, this ELI is: {}, taget PLI is: {}, this PLI is: {}").format(target_ELI, this_ELI, target_PLI, \
    #                                                                                     this_PLI)
    #print("PLI_sem is: {}, ELI_sem is: {}, PLI_error normed is {}, ELI_error normed is {}").format(target_PLI_sem, \
    #                                                    target_ELI_sem, e5_norm, e6_norm)
    print 'target ppli is: {} this ppli is: {}, target_sem_ppli is {}'.format(target_ppli, this_ppli, target_sem_ppli)
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
    #Checking without the pfr
    e2_norm = 0
    RMSE_normed = (
                      (np.nansum(e3_norm ** 2)
                       + np.nansum(e4_norm ** 2)
                       + np.nansum(e1_norm **2)
                       + np.nansum(e5_norm **2)
                       + np.nansum(e6_norm **2)
                       + np.nansum(e2_norm **2)) \
                   /(len(e1) + len(e3) + len(e4) + len(e2) + len(e5) + len(e6))) ** 0.5
    #print('Only fitting for ELI/PLI')
    #RMSE_normed = (e6_norm**2 + e5_norm**2)/2**.5
    print("RMSE_normed is: ", RMSE_normed)
    # get just a regular rmse, not normed by the sem of the data
    # nerr_raw = len(e1) + len(e2) + len(e3) + len(e4)
    # sum_squared_err_raw = (np.nansum(e1**2) + np.nansum(e2**2)
    #                        + np.nansum(e3**2) + np.nansum(e4**2))
    # RMSE = (sum_squared_err_raw / nerr_raw) ** 0.5

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

def setup_txts(subj):
    data_path = '/data/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_{}.mat'.format(subj)
    data = scipy.io.loadmat(data_path, squeeze_me=True, struct_as_record=False)['data']
    np.savetxt('division_locs_ind1.txt', data.session, delimiter=',', fmt='%i')
    np.savetxt('rec_nos_{}.txt'.format(subj), data.rec_itemnos, delimiter=',', fmt='%i')
    np.savetxt('pres_nos_{}.txt'.format(subj), data.pres_itemnos, delimiter=',', fmt='%i')
    np.savetxt('rec_times_{}.txt'.format(subj), data.times, delimiter=',', fmt='%i')

def generate_subj():
    good_subjs = ['LTP093', 'LTP106', 'LTP115', 'LTP117', 'LTP122', 'LTP123', 'LTP133', 'LTP138', 'LTP207', 'LTP210','LTP228', 'LTP229', 'LTP236', 'LTP246', 'LTP249', 'LTP251', 'LTP258', 'LTP259', 'LTP260','LTP265', 'LTP269', 'LTP273', 'LTP278', 'LTP279', 'LTP280', 'LTP283', 'LTP285', 'LTP287', 'LTP293', 'LTP295', 'LTP296', 'LTP297', 'LTP299', 'LTP301', 'LTP302', 'LTP303', 'LTP304', 'LTP305', 'LTP306', 'LTP307', 'LTP309', 'LTP310', 'LTP311', 'LTP312', 'LTP314', 'LTP316', 'LTP317', 'LTP318', 'LTP320', 'LTP321', 'LTP322', 'LTP323', 'LTP324', 'LTP325', 'LTP327', 'LTP328', 'LTP330', 'LTP331', 'LTP334', 'LTP336', 'LTP338', 'LTP339', 'LTP340', 'LTP342', 'LTP343', 'LTP344', 'LTP346', 'LTP347', 'LTP348', 'LTP349', 'LTP353', 'LTP355', 'LTP357', 'LTP359', 'LTP361', 'LTP362', 'LTP364', 'LTP366']
    for good_subj in good_subjs:
        cwd = os.getcwd()
        if os.path.isfile(os.path.join(cwd, 'xopt_{}.txt'.format(good_subj))):
            continue
        else:
            return good_subj
def main():
    
    #Change tracker:
    #Updated to handle any subj and automated the creation of txt files
    #Changed phig to .9
    #Wrote working PLI ELI code
    #Wrote full ELIPLI adjustment. Put in thresholding
    #Rewrote grpahing software to be automatic. Runs with the grph.ipynb
    # Automated the process across subjects for passive run
    
    #Useful tip: if you run into a bug (like when I'm not here anymore check the TODO comments
    #They have stuff that is fishy or temporary (check both this code and the cython code)
    #########
    #
    #   Define some helpful global (yikes, I know!) variables.
    #
    #########

    global ll, data_pres, data_rec, LSA_path, data_path, LSA_mat
    global target_spc, target_spc_sem, target_pfc, target_pfc_sem
    global target_left_crp, target_left_crp_sem
    global target_right_crp, target_right_crp_sem
    subj = generate_subj()
    setup_txts(subj)
    text_file = open("Current_subject.txt", "w")
    text_file.write(subj)
    text_file.close()

    print 'on subject {}'.format(subj)

    LSA_path = 'w2v.txt'
    data_path = 'pres_nos_{}.txt'.format(subj)
    rec_path = 'rec_nos_{}.txt'.format(subj)
    rec_times_path = 'rec_times_{}.txt'.format(subj)

    LSA_mat = np.loadtxt(LSA_path)

    # if getting data from a text file:
    data_pres = np.loadtxt(data_path, delimiter=',')
    data_rec = np.loadtxt(rec_path, delimiter=',')
    data_rec_times = np.loadtxt(rec_times_path, delimiter=',')
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

    #global target_PLI, target_PLI_sem
    #global target_ELI, target_ELI_sem

    # get mean and sem for the observed data's PLI's and ELI's
    #target_PLI, target_ELI, \
    #target_PLI_sem, target_ELI_sem = get_num_intrusions(
    #    data_rec, data_pres,
    #    lists_per_div=lists_per_session, ndivisions=nsessions)
    
    global target_ppli, target_sem_ppli, target_tic, target_sem_tic
    target_ppli, target_sem_ppli, target_tic, target_sem_tic =  \
    handle_intrusions(data_rec, data_pres, lists_per_session,  data_rec_times)
    # make sure we do not later divide by 0 in case the sem's are 0
    #if target_ELI_sem == 0:
    #    target_ELI_sem = 1
    #if target_PLI_sem == 0:
    #    target_PLI_sem = 1
    if target_sem_ppli == 0:
        target_sem_ppli = 1
    if target_sem_tic == 0:
        target_sem_tic = 1

    #############
    #
    #   set lower and upper bounds
    #
    #############

    lb = [0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.01, 0.01, 0.5, .1, 5.0, .5, .001, .01]
    ub = [1.0, 1.0, 0.7, 1.0, 3.0, 1.5, 0.5, 0.5, 3.0, 1.0, 15.0, 1.0, 0.8, 0.5]

    start_time = time.time()
    xopt, fopt = pso(obj_func, lb, ub, swarmsize=90, maxiter=30, debug=False, phig = 0.65)

    print(xopt)
    print("Run time: " + str(time.time() - start_time))

    sys.stdout.flush()
    time.sleep(5)
    tempfile_paths = glob('*tempfile*')
    for mini_path in tempfile_paths:
        if os.path.isfile(mini_path):
            #The fact that i need a try except here is worrying...
            try:
                os.remove(mini_path)
            except:
                continue

    np.savetxt('xopt_{}.txt'.format(subj), xopt, delimiter=',', fmt='%f')
    #I need to save something, but that's not really the purpose of this file
    junk = [0,0]
    np.savetxt('imdone.txt',junk)
    print 'Done'

if __name__ == "__main__": main()
