#!/home1/ddiwik/anaconda2/envs/shadowfox/bin/python
#$-N testpso
#$-cwd
#$-pe python-distributed 8
import numpy as np
import time
import scipy.io
from joblib import Parallel, delayed
import sys
#sys.path.append('/home1/rivkat.cohen/PycharmProjects/pso_test2_working_6-18-17')
sys.path.append('/home1/ddiwik/_Central_Command/CMR2/Tails/et_al')
import lagCRP2
#import CMR2_pack as CMR2
sys.path.append('/home1/ddiwik/_Central_Command/CMR2/Tails/CMR2_cyth')
import CMR2_pack_cyth as CMR2
import glob
import pickle

"""
Dependencies: CMR2_pack.py, lagCRP2.py, plus all the package imports above.
              Must also have access to a data file & LSA or W2V file.
"""
#nyan nyan ~ =^.^=
#See line 312 for another reference to number of nodes
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


## insert CMR2 objective function here, and name it obj_func

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
        'rec_time_limit': 30000,

        'dt_tau': 0.01,
        'sq_dt_tau': 0.10,

        'nlists_for_accumulator': 4
    }

    rec_nos, times = CMR2.run_CMR2(LSA_path, LSA_mat, data_path, param_dict,
                                   sep_files=files_are_separate)

    cmr_recoded_output = recode_for_spc(rec_nos, data_pres)

    # get the model's spc and pfc predictions:
    (this_spc, this_spc_sem, this_pfc,
	this_pfc_sem) = get_spc_pfc(cmr_recoded_output, ll)

    # get the model's crp predictions:
    this_crp, this_crp_sem = lagCRP2.get_crp(cmr_recoded_output, ll)

    # get left crp values
    this_left_crp = this_crp[ll-6:ll-1]

    # get right crp values
    this_right_crp = this_crp[ll:ll+5]

    # be careful not to divide by 0! some param sets may output 0 sem vec's.
    # if this happens, just leave all outputs alone.
    if np.nansum(this_spc_sem) == 0 \
            or np.nansum(this_pfc_sem) == 0 \
            or np.nansum(this_crp_sem) == 0:
        this_spc_sem[range(len(this_spc_sem))] = 1
        this_pfc_sem[range(len(this_pfc_sem))] = 1
        this_crp_sem[range(len(this_crp_sem))] = 1


    # get the error vectors for each type of analysis
    e1 = np.subtract(target_spc, this_spc)
    e1_norm = np.divide(e1, target_spc_sem)

    e2 = np.subtract(target_pfc, this_pfc)
    e2_norm = np.divide(e2, target_pfc_sem)

    e3 = np.subtract(target_left_crp, this_left_crp)
    e3_norm = np.divide(e3, target_left_crp_sem)

    e4 = np.subtract(target_right_crp, this_right_crp)
    e4_norm = np.divide(e4, target_right_crp_sem)

    nerr_denom = len(e1_norm) + len(e2_norm) + len(e3_norm) + len(e4_norm)

    sum_squared_errors = (np.nansum(e1_norm ** 2) + np.nansum(e2_norm ** 2)
             + np.nansum(e3_norm ** 2) + np.nansum(e4_norm ** 2))

    RMSE = (sum_squared_errors / nerr_denom) ** 0.5

    return RMSE


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

    ################
    #
    #   to enable a distributed job here,
    #
    #   Calculate all particle fitnesses ahead of the loop updating positions
    #
    ################

    # fp = [func((n, data_stim, data_results)) for n in p]

    num_nodes = 8
    with Parallel(n_jobs=num_nodes) as parallel:
        fp = parallel(
            delayed(func)(n) for n in p)

    for i in range(S):
        # Initialize the particle's position
        x[i, :] = lb + x[i, :]*(ub - lb)
   
        # Initialize the particle's best known position
        p[i, :] = x[i, :]
       
        # Calculate the objective's value at the current particle's
        #fp[i] = obj(p[i, :])
       
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
       
    # Iterate until termination criterion met ##################################
    it = 1
    while it<=maxiter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))

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

        ##############################
        # this had to be taken outside the above loop in order to distribute it
        # fx = [obj(x[i, :]) for i in range(S)]

        with Parallel(n_jobs=num_nodes) as parallel:
            fx = parallel(
                delayed(func)(x[i, :]) for i in range(S))

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
    global files_are_separate
    
    #####################################
    #USER DEFINED VALUES: 
    print 'Please also remember that !#... is changed at the top'
    #Path to the LSA matrix
    #LSA_path = '/home1/ddiwik/_Central_Command/CMR2/Tails/et_al/K02_Data/K02_LSA.mat'
    LSA_path = '/home1/ddiwik/_Central_Command/CMR2/Tails/et_al/FR2_LSA.mat'
    #Set path to data. If the file ends in .mat it's assumed to be a data file
    #If it doesn't it is assumed to be a folder with separate files
    #data_path = '/home1/ddiwik/_Central_Command/CMR2/Tails/et_al/K02_Data/K02_data.mat'
    data_path = '/data/eeg/scalp/ltp/ltpFR2/behavioral/data/'#stat_data_LTP093.mat'
    #Set list length
    ll = 24
    #Only for when using separate files. Makes the program load the files rather
    #than concatenating. If you aren't using a lot of data, don't bother (set to False)
    load_data = True
    load_save_folder = '/home1/ddiwik/_Central_Command/CMR2/Tails/et_al/ltp_FR2_save_data/'
    
    #####################################
    
    #Get LSA mat file
    LSA_mat = scipy.io.loadmat(LSA_path, squeeze_me=True, struct_as_record=False)['LSA']
    #Check whether files are separate
    if data_path[-4:] == '.mat':
        files_are_separate = False
    else:
        files_are_separate = True
        
    # get data file, presented items, & recalled items
    if files_are_separate == False:
        data_file = scipy.io.loadmat(
        data_path, squeeze_me=True, struct_as_record=False)
        data_pres = data_file['data'].pres_itemnos      # presented
        data_rec = data_file['data'].rec_itemnos        # recalled
    else:
        if load_data == True:
            print 'Loading separate file data from data_folder'
            with open(load_save_folder + 'data_rec', 'rb') as handle:
                data_rec = pickle.load(handle)
            with open(load_save_folder + 'data_pres', 'rb') as handle:
                data_pres = pickle.load(handle)
        else:
            print 'Concatenating separate files rather than loading them' + \
            'from save folder'
            data_pres_h = []
            data_rec_h = []
            for partial_file in glob.glob('{}/stat_data*'.format(data_path)):
                partial_data = scipy.io.loadmat(
                    partial_file, squeeze_me=True, struct_as_record=False)
                try:
                    data_pres_h.append(partial_data['data'].pres_itemnos)
                    data_rec_h.append(partial_data['data'].pres_itemnos)
                except:
                    print 'Some error occured in reading file {}. Skipping it.'.format(partial_file)
                    continue
            data_pres = [np.vstack(partial_data_dat) for partial_data_dat in data_pres_h][0]
            data_rec = [np.vstack(partial_data_dat) for partial_data_dat in data_pres_h][0]
            print 'Finished reading in data files. Now saving data to file for future use.'
            with open(load_save_folder + 'data_rec', 'wb') as handle:
                pickle.dump(data_rec, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(load_save_folder + 'data_pres', 'wb') as handle:
                pickle.dump(data_pres, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # recode lists for spc, pfc, and lag-CRP analyses
    recoded_lists = recode_for_spc(data_rec, data_pres)

    # get spc & pfc
    target_spc, target_spc_sem, target_pfc, target_pfc_sem = \
        get_spc_pfc(recoded_lists, ll)

    target_crp, target_crp_sem = lagCRP2.get_crp(recoded_lists, ll)

    # get Lag-CRP sections of interest
    target_left_crp = target_crp[ll-6:ll-1]
    target_left_crp_sem = target_crp_sem[ll-6:ll-1]

    target_right_crp = target_crp[ll:ll+5]
    target_right_crp_sem = target_crp_sem[ll:ll+5]

    #############
    #
    #   set lower and upper bounds
    #
    #############

    lb = [0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.01, 0.01, 0.5, .1, 5.0, .5, .001, .01]
    ub = [1.0, 1.0, 0.7, 1.0, 3.0, 1.5, 0.5, 0.5, 3.0, 1.0, 15.0, 1.0, 0.8, 0.5]

    start_time = time.time()
    xopt, fopt = pso(obj_func, lb, ub, swarmsize=1, maxiter=1, debug=False)

    print(xopt)
    print("Run time: " + str(time.time() - start_time))

    sys.stdout.flush()


if __name__ == "__main__": main()

