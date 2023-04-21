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
from noise_maker_ga import make_noise
from optimization_utils import get_data, obj_func

# Set where you want your noise files and output files to be saved
global OUTDIR
global NOISE_DIR
global SUBJ

SUBJ = 229

basepath = ""
try:
    os.chdir('/data')
except OSError:
    basepath = "/Users/lumdusislife/rhino_mount"

OUTDIR = basepath+'/home1/shai.goldman/pyCMR2/IRT_Optimizations/outfiles/'
NOISE_DIR = basepath+'/home1/shai.goldman/pyCMR2/IRT_Optimizations/noise_files/'

"""
Dependencies: CMR2_pack_cyth plus all the package imports above.
              Must also have access to a data file & LSA or W2V file.

Developed by Jesse Pazdera for use with ltpFR3. You will need to modify
run_ga() and the __main__ script for use with your own project.
"""


def ga(func, lb, ub, data_pres, identifiers, w2v, sources, target_stats, 
       first_gen=None, ngen=10, popsize=2000, parent_rate=.2, cross_rate=.5,
       mut_rate=1., mut_scale=.2, nsess=400):
    """
    Runs genetic algorithm optimization. Can be run using the same settings
    on every generation or can have generations grouped into "eras", each
    with different population size, mutation scaling, etc.

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
    first_gen: 2D array
        If None, the first generation's parameter sets will be randomly sampled from
        a uniform distribution. If an array, the first generation's parameter sets will
        be set equal to that array. Must be of shape (N, D) where N is the number of
        individuals in the first generation and D is the dimensionality of the search
        space.
    ngen: int or array
        If integer, the number of generations to simulate. If array, defines the
        number of generations to simulate in each "era."
    popsize: int or array
        If integer, the number of individuals to simulate in every generation.
        If array, the number of individuals to simulate in each generation
        during each "era."
    parent_rate: float or array
        If float, the fraction of best individuals that will be selected as
        parents of the next generation. If array, defines the parent rate during
        each "era."
    cross_rate: float or array
        If float, defines the fraction of parameters that will be inherited from
        parent 1 (set to .5 to have equal contribution from both parents). If
        array, defines the crossing rate during each "era."
    mut_rate: float or array
        If float, the probability of each parameter in each individual mutating.
        If array, defines the mutation rate for each "era."
    mut_scale: float or array
        If float, defines the standard deviation of mutation sizes relative to
        the range of the dimension (e.g. mut_scale=.2 means that the standard
        deviation of mutation sizes will be 20% of the parameter range in that
        dimension.) If array, defines the mutation scaling for each "era."
    nsess: int or array
        If int, defines the number of sessions to simulate. If array, defines the
        number of sessions to simulate during each "era."
    
    Returns
    =======
    best_params : array
        The best-fitting parameter set identified by the genetic algorithm.
    best_score : float
        The goodness-of-fit score for the best fitting parameter set.

    """

    global OUTDIR
    global NOISE_DIR
    global BEST_KNOWN
    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    ngen = [ngen] if not hasattr(ngen, '__iter__') else ngen
    popsize = [popsize] if not hasattr(popsize, '__iter__') else popsize
    parent_rate = [parent_rate] if not hasattr(parent_rate, '__iter__') else parent_rate
    cross_rate = [cross_rate] if not hasattr(cross_rate, '__iter__') else cross_rate
    mut_rate = [mut_rate] if not hasattr(mut_rate, '__iter__') else mut_rate
    mut_scale = [mut_scale] if not hasattr(mut_scale, '__iter__') else mut_scale
    nsess = [nsess] if not hasattr(nsess, '__iter__') else nsess
    if first_gen is not None:
        assert first_gen.shape == (popsize[0], len(lb))
    assert len(ngen) == len(popsize) == len(parent_rate) == len(cross_rate) == len(mut_rate) == len(mut_scale) == len(nsess)

    ##########
    #
    # Initialization
    #
    ##########
    
    lb = np.array(lb)
    ub = np.array(ub)

    # Create lists for hyperparameter settings at each generation
    popsizes = []
    n_parents = []
    cross_rates = []
    mut_rates = []
    mut_scales = []
    for i, n in enumerate(ngen):
        popsizes = np.concatenate((popsizes, [popsize[i] for _ in range(n)]))
        n_parents = np.concatenate((n_parents, [parent_rate[i] for _ in range(n)]))
        cross_rates = np.concatenate((cross_rates, [cross_rate[i] for _ in range(n)]))
        mut_rates = np.concatenate((mut_rates, [mut_rate[i] for _ in range(n)]))
        mut_scales = np.concatenate((mut_scales, [mut_scale[i] for _ in range(n)]))
        nsess = np.concatenate((nsess, [nsess[i] for _ in range(n)]))
    # Convert parent rates to numbers of parents (n_parents is popsize from prior generation times current parent rate)
    n_parents *= np.concatenate(([0], popsizes[:-1]))
    n_parents = n_parents.astype(int)
    popsizes = popsizes.astype(int)
    # Count up total number of generations
    ngen = np.sum(ngen)

    # os.O_CREAT --> create file if it does not exist
    # os.O_EXCL --> error if create and file exists
    # os.O_WRONLY --> open for writing only
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    ##########
    #
    # Genetic Algorithm
    #
    ##########

    # Run ngen generations
    for i, gen in enumerate(range(1, ngen + 1)):
        print('Starting generation %s...' % gen)
        
        S = popsizes[i]

        ##########
        #
        # Spawn new population
        #
        ##########

        # If this generation has already been processed, skip to the next generation
        if os.path.exists(OUTDIR + 'err_iter%s' % gen):
            print('Generation %s already complete! Skipping...' % gen)
            continue

        # If it is the first generation, initialize parameter sets using random noise file
        elif i == 0:
            if first_gen is None:
                print('Loading first generation...')
                pop = np.loadtxt(NOISE_DIR + 'rx')
            else:
                pop = first_gen

        # If it is any generation beyond the first, apply crossover and mutation on previous generation
        else:
            print('Producing generation %s...' % gen)
            # Read in the random files for this iteration
            r1 = np.loadtxt(NOISE_DIR + 'r1_iter' + str(gen), dtype=int)  # Used for pairing parents
            r2 = np.loadtxt(NOISE_DIR + 'r2_iter' + str(gen))  # Used for choosing parameters to cross
            r3 = np.loadtxt(NOISE_DIR + 'r3_iter' + str(gen))  # Used for choosing parameters to mutate
            r4 = np.loadtxt(NOISE_DIR + 'r4_iter' + str(gen))  # Used for scaling mutations
            # Read in parameters and scores from the previous generation
            while True:
                pop = np.loadtxt(OUTDIR + str(gen - 1) + 'xfile.txt')
                scores = np.loadtxt(OUTDIR + 'err_iter%s' % (gen - 1))
                if pop.shape[0] == popsizes[i-1] and scores.shape[0] == popsizes[i-1]:
                    break
                else:
                    time.sleep(2)

            # Select parents by choosing the best individuals from the previous generation
            parents = pop[np.argsort(scores)[:n_parents[i]], :].copy()
            pop = np.empty((S, len(lb)))
            # Spawn new population from parents
            for j in range(S):
                # Randomly select two parents
                parent1 = parents[r1[j, 0]]
                parent2 = parents[r1[j, 1]]
                # Cross parents
                child = np.zeros_like(lb)
                cross = cross_rates[i] >= r2[j, :]
                child[cross] = parent1[cross]
                child[~cross] = parent2[~cross]
                # Mutate with Gaussian noise, scaled by the valid range of each parameter
                mutated = mut_rates[i] >= r3[j, :]
                child[mutated] += r4[j, mutated] * mut_scales[i] * (ub - lb)
                child[child < lb] = lb[child < lb]
                child[child > ub] = ub[child > ub]
                # Add to new generation's population
                pop[j, :] = child

        ##########
        #
        # Evaluate each model
        #
        ##########
        
        # Get list of sessions to be simulated this generation
        sessions_in_gen = set()
        j = 0
        while len(sessions_in_gen) < nsess[i]:
            sessions_in_gen.add(identifiers[j])
            j += 1
        
        # Get item lists, session identifiers, and sources for sessions to be simulated
        session_mask = np.isin(identifiers, list(sessions_in_gen))
        data_pres_gen = data_pres[session_mask]
        identifiers_gen = identifiers[session_mask]
        sources_gen = sources_gen[session_mask] if sources is not None else None
        
        for j in range(S):

            outfile = OUTDIR + str(gen) + 'tempfile' + str(j) + '.txt'
            try:
                # Try to open the tempfile -- if the file for this model already exists, skip to the next individual
                fd = os.open(outfile, flags)

                # Run CMR2 using this parameter set and get out the fitness score
                print('Running model for individual %s...' % j)
                err, stats = func(pop[j, :], target_stats, data_pres_gen, identifiers_gen, w2v, sources_gen)
                print('Model finished with a fitness score of %s' % err)

                # Save simulated behavioral stats from the model
                with open(OUTDIR + str(gen) + 'data' + str(j) + '.pkl', 'wb') as f:
                    pkl.dump(stats, f, 2)

                # Write the model's fitness score to the tempfile
                file_input = str(err)
                os.write(fd, file_input.encode())
                os.close(fd)

            except OSError as e:
                if e.errno == errno.EEXIST:
                    #print('Model for individual %s already complete! Skipping...' % j)
                    continue
                else:
                    raise

        ##########
        #
        # Save results of generation
        #
        ##########

        # Once all models have been "claimed" by parallel jobs, wait until all models from the current iteration have
        # finished running. Tempfiles will be empty until the associated model finishes. Proceed once none are empty.
        while True:
            for j in range(S):
                path = OUTDIR + '%stempfile%s.txt' % (gen, j)
                if not (os.path.exists(path) and os.path.getsize(path) > 0.0):
                    break
            else:
                break
            time.sleep(2)

        # Load the error values for this iteration from all tempfiles, once we have confirmed they are finished
        scores = np.zeros(S)
        for j in range(S):
            scores[j] = np.loadtxt(OUTDIR + '%stempfile%s.txt' % (gen, j))

        # Save the results from the current generation before moving on to the next
        param_files = [OUTDIR + str(gen) + 'xfile.txt', OUTDIR + 'err_iter%s' % gen]
        param_entries = [pop, scores]
        for j in range(len(param_entries)):
            try:
                f = os.open(param_files[j], flags)
                os.close(f)
                np.savetxt(param_files[j], param_entries[j])
                print('Saved iteration results to %s!' % param_files[j])
            except OSError as e:
                if e.errno == errno.EEXIST:
                    continue
                else:
                    raise
        print('Iteration %s complete!' % gen)

    ##########
    #
    # Determine best model
    #
    ##########

    best_params = [np.nan for _ in lb]
    best_score = np.inf
    best_i_and_gen = []
    for i, gen in enumerate(range(1, ngen + 1)):
        scores = np.loadtxt(OUTDIR + 'err_iter%s' % gen)
        best_in_gen = np.min(scores)
        best_i = np.where(scores == best_in_gen)
        best_i=best_i[0][0]
        if best_in_gen < best_score:
            best_score = best_in_gen
            best_params = np.loadtxt(OUTDIR + str(gen) + 'xfile.txt')[np.argmin(scores), :]
            best_i_and_gen = [best_i, gen]
    
    print('Best i: %d from Gen %d' % (best_i_and_gen[0], best_i_and_gen[1]))
    return best_params, best_score


def run_ga(data_pres, sessions, w2v, sources, targets):

    global OUTDIR
    global NOISE_DIR
    
    #GENERATIONS = 150#150 #used to be 150
    #SESSIONS = 1 #sessions
    #POPSIZE = 1000#1000

    # GA settings: Settings are defined as lists, where the nth list item defines the settings for the nth "epoch" of generations. An epoch is defined as set of generations with identical settings, and there can be any number of generations in a given epoch. Alternatively, each generation can be given different settings by placing each in its own 1-generation epoch (as shown below).
    #ngen = [1 for _ in range(GENERATIONS)]  # Number of generations to run in each epoch
    #popsize = [POPSIZE for _ in range(GENERATIONS)]  # Population size
    #parent_rate = [.2 for _ in range(GENERATIONS)]  # The fraction of individuals that will be allowed to become parents for the next generation
    #cross_rate = [.5 for _ in range(GENERATIONS)]  # The probability of each parameter being passed from parent 1 (as opposed to parent 2)
    #mut_rate = [1. for _ in range(GENERATIONS)]  # The probability of each parameter mutating in a child
    #mut_scale = np.geomspace(.1, .005, GENERATIONS)  # The standard deviation of mutations, relative to the full range of that parameter (.1 = 10% of range)
    #nsess = np.geomspace(1, SESSIONS, GENERATIONS, dtype=int)  # The number of sessions to simulate with the model
    
    ngen = [5, 10, 15, 10, 5]
    popsize = [2000, 1600, 1200, 800, 400]
    parent_rate = [.2, .2, .2, .2, .2]
    cross_rate = [.5, .5, .5, .5, .5]
    mut_rate = [1, 1, 1, 1, 1]
    mut_scale = [.15, .1, .05, .025, .0125]
    nsess = [200, 200, 200, 200, 200] 
    
    
    # The following setting can be used to seed the first generation by setting equal to a matrix of values where each row contains the starting parameter values for one individual
    first_gen = None#np.loadtxt('/home1/shai.goldman/pyCMR2/IRT_Optimizations/model_params/top1000.txt')
    
    
    #    [b_e, b_r, g_fc, g_cf, p_s, p_d,   k,   e, s_cf, b_rp,   o,    a, c_t,  l, b_distract]
    lb = [.2,   .2, .500, .500,   0,   0,   0, .01,    0, .500,  10., .6,   0,  0, .2]
    ub = [.8,   .8, .999, .999,   4,   3,  .4, .75,    5, .999, 15., 1,  .5, .3, .8]
    # Notes: I edited o from [1,20] to [10,15] and a from [.5, .999] to [.6, 1]
    
    print('Generating noise files...')
    make_noise(popsize, ngen, lb, ub, parent_rate, NOISE_DIR)

    print('Initiating genetic algorithm...')
    start_time = time.time()
    xopt, fopt = ga(obj_func, lb, ub, data_pres, sessions, w2v, sources, targets, 
                    first_gen=first_gen, ngen=ngen, popsize=popsize, parent_rate=parent_rate,
                    cross_rate=cross_rate, mut_rate=mut_rate, mut_scale=mut_scale, nsess=nsess)

    print(fopt, xopt)
    print("Run time: " + str(time.time() - start_time))
    sys.stdout.flush()

    np.savetxt(OUTDIR + 'xopt_%sltpFR2.txt' % SUBJ, xopt, delimiter=',', fmt='%f')


if __name__ == "__main__":

    
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
    
    #irt funcs
    print(data_pres.shape)

    run_ga(data_pres, sessions, w2v, sources, targets)
