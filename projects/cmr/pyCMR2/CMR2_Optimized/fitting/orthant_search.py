import mkl
mkl.set_num_threads(1)
import os
import sys
import time
import json
import errno
import numpy as np
import pickle as pkl
from glob import glob
from noise_maker_pso import make_noise
from optimization_utils import get_data, obj_func

global OUTDIR
global NOISE_DIR
OUTDIR = '/scratch/jpazdera/cmr2/orthant_search/outfiles/'
NOISE_DIR = '/scratch/jpazdera/cmr2/orthant_search/noise_files/'

"""
Dependencies: CMR2_pack_cyth plus all the package imports above.
              Must also have access to a data file & LSA or W2V file.

Developed by Jesse Pazdera for use with ltpFR3. You will need to modify
run() and the __main__ script in order to use this code with your own
projects.
"""


def orthant_search(func, lb, ub, data_pres, identifiers, w2v, sources, target_stats, n_per_orth=25):
    """
    Runs a test of parameter sets in every orthant of the search space.

    Parameters
    ==========
    func : function
        The objective function to be minimized. Usually a function that runs
        CMR2 to simulate a dataset and evaluate its fit to empirical data.
    lb : array
        The lower bounds of each dimension of the parameter space.
    ub : array
        The upper bounds of each dimension of the parameter space.
    data_pres : array
        A trials x items matrix of the ID numbers of items that will be
        presented to CMR2.
    identifiers: array
        An array of subject/session identifiers, used for separating data_pres
        into the presented item matrices for multiple sessions.
    w2v: array
        An items x items matrix of word2vec (or other) semantic similarity scores
        to be passed on to CMR2.
    sources: array
        A trials x items x feature array of source information to be passed to CMR2.
    target_stats: dictionary
        A dictionary containing the empirical recall performance stats that will
        be used by the objective function to determine the difference between
        CMR2's performance and actual human performance.

    Optional
    ========
    n_per_orth : int
        The number of parameter sets to test at random locations within each
        orthant. Note that the number of orthants is 2^len(lb).
    
    Returns
    =======
    best_params : array
        The best-fitting parameter set identified by the genetic algorithm.
    best_score : float
        The goodness-of-fit score for the best fitting parameter set
    
    """
    global OUTDIR
    global NOISE_DIR
    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'

    ##########
    #
    # Initialization
    #
    ##########

    # os.O_CREAT --> create file if it does not exist
    # os.O_EXCL --> error if create and file exists
    # os.O_WRONLY --> open for writing only
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    lb = np.array(lb)
    ub = np.array(ub)
    midlines = (ub + lb) / 2.
    D = len(lb)

    # Create representation of orthants
    orthants = np.array([[j for j in "{0:014b}".format(i)] for i in range(2 ** D)], dtype=int).astype(bool)

    # Create or load randomized parameter sets
    try:
        # Create lock file
        f = os.open(OUTDIR + 'params.txt', flags)
        os.close(f)

        # Generate random numbers for initializing parameter locations
        params = np.random.uniform(size=(len(orthants) * n_per_orth, D))

        # Convert random numbers to locations within each orthant
        for i, orth in enumerate(orthants):
            start_ind = i * n_per_orth
            stop_ind = (i + 1) * n_per_orth
            params[start_ind:stop_ind, orth] = midlines[orth] + params[start_ind:stop_ind, orth] * (ub - midlines)[orth]
            params[start_ind:stop_ind, ~orth] = lb[~orth] + params[start_ind:stop_ind, ~orth] * (midlines - lb)[~orth]

        np.savetxt(OUTDIR + 'params.txt', params)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    while True:
        params = np.loadtxt(OUTDIR + 'params.txt')
        if params.shape[0] == len(orthants) * n_per_orth:
            break
        else:
            time.sleep(2)

    ##########
    #
    # Evaluate each model
    #
    ##########

    for i in range(params.shape[0]):
        outfile = OUTDIR + 'tempfile' + str(i) + '.txt'
        try:
            # Try to open the tempfile -- if the file for this model already exists, skip to the next model
            fd = os.open(outfile, flags)

            # Run CMR2 using this parameter set and get out the fitness score
            print('Running model number %s...' % i)
            err, stats = func(params[i, :], target_stats, data_pres, identifiers, w2v, sources)
            print('Model finished with a fitness score of %s' % err)

            # Save simulated behavioral stats from the model
            with open(OUTDIR + 'data' + str(i) + '.pkl', 'wb') as f:
                pkl.dump(stats, f, 2)

            # Write the model's fitness score to the tempfile
            file_input = str(err)
            os.write(fd, file_input.encode())
            os.close(fd)

        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Model %s already complete! Skipping...' % i)
                continue
            else:
                raise

    ##########
    #
    # Save results of generation
    #
    ##########

    # Once all models have been "claimed" by parallel jobs, wait until all models have finished running. Tempfiles will
    # be empty until the associated model finishes. Proceed once none are empty.
    while True:
        for i in range(params.shape[0]):
            path = OUTDIR + 'tempfile%s.txt' % i
            if not (os.path.exists(path) and os.path.getsize(path) > 0.0):
                break
        else:
            break
        time.sleep(2)

    # Load the error values from all tempfiles, once we have confirmed they are finished
    scores = np.zeros(params.shape[0])
    for i in range(params.shape[0]):
        scores[i] = np.loadtxt(OUTDIR + 'tempfile%s.txt' % i)

    # Compile all scores into a single file
    try:
        f = os.open(OUTDIR + 'scores.txt', flags)
        os.close(f)
        np.savetxt(OUTDIR + 'scores.txt', scores)
        print('Saved results!')
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    print('Testing complete!')

    ##########
    #
    # Determine best model
    #
    ##########

    scores = np.loadtxt(OUTDIR + 'scores.txt')
    best_score = np.min(scores)
    best_params = np.loadtxt(OUTDIR + 'params.txt')[np.argmin(scores), :]

    return best_params, best_score


def run(data_pres, sessions, w2v, sources, targets):

    global OUTDIR
    global NOISE_DIR

    #    [b_e, b_r, g_fc, g_cf, p_s, p_d,  k,   e, s_cf, b_rp,   o,   a, c_t,   l]
    lb = [.30,  0., .001, .001,  0.,  0., 0., .01,   0.,   0.,  4., .01,  0.,  0.]
    ub = [.75,  1., .999, .999, 10.,  2., 1., .60,  10.,   1., 20., .99,  .5, .75]
    n_per_orth = 25
                                                                 
    print('Generating noise files...')
    n_orthants = 2 ** len(lb)
    n_models = n_orthants * n_per_orth
    make_noise(n_models, 1, lb, ub, NOISE_DIR)
                                                                 
    print('Initiating orthant search...')
    start_time = time.time()
    xopt, fopt = orthant_search(obj_func, lb, ub, data_pres, sessions, w2v, sources, targets, n_per_orth)

    print(fopt, xopt)
    print("Run time: " + str(time.time() - start_time))
    sys.stdout.flush()

    np.savetxt(OUTDIR + 'xoptb_ltpFR3.txt', xopt, delimiter=',', fmt='%f')


if __name__ == "__main__":
    N = 500  # Number of sessions to simulate

    # Load lists from participants who were not excluded in the behavioral analyses
    file_list = glob('/data/eeg/scalp/ltp/ltpFR3_MTurk/data/MTK*.json')
    wn = set(np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/WROTE_NOTES.txt', dtype=str))
    vis_subj = set(np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/EXP2_VIS.txt', dtype=str))
    vfile_list = sorted([f for f in file_list if int(f[-9:-5]) > 1308 and f[-12:-5] not in wn and f[-12:-5] in vis_subj])
    aud_subj = set(np.loadtxt('/data/eeg/scalp/ltp/ltpFR3_MTurk/EXP2_AUD.txt', dtype=str))
    afile_list = sorted([f for f in file_list if int(f[-9:-5]) > 1308 and f[-12:-5] not in wn and f[-12:-5] in aud_subj])
    # Take the first N/2 visual and N/2 auditory sessions
    file_list = vfile_list[:int(N/2)] + afile_list[:int(N/2)]
    
    # Set file paths for data, wordpool, and semantic similarity matrix
    wordpool_file = '/data/eeg/scalp/ltp/ltpFR3_MTurk/CMR2_ltpFR3/wasnorm_wordpool.txt'
    w2v_file = '/data/eeg/scalp/ltp/ltpFR3_MTurk/CMR2_ltpFR3/w2v.txt'
    target_stat_file = '/data/eeg/scalp/ltp/ltpFR3_MTurk/CMR2_ltpFR3/target_stats_sim1.json'

    # Load data
    print('Loading data...')
    data_pres, sessions, sources = get_data(file_list, wordpool_file)

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
                    
    run(file_list, modality)
