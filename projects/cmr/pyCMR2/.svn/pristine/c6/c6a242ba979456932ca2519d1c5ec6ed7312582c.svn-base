import mkl
mkl.set_num_threads(1)
import os
import sys
import time
import json
import errno
import numpy as np
import pickle as pkl
import random
from glob import glob
from noise_maker_pso import make_noise
from optimization_utils import obj_func

global OUTDIR
global NOISE_DIR

# OUTDIR = '/scratch/yedong/ltpFR2/cmr2_primacy/outfiles/'
# NOISE_DIR = '/scratch/yedong/ltpFR2/cmr2_primacy/noise_files/'

OUTDIR = '/home1/yedong/svn/pyCMR2/branches/CMR2_yedong/outfiles/'
NOISE_DIR = '/home1/yedong/svn/pyCMR2/branches/CMR2_yedong/noise_files/'

"""
Dependencies: CMR2_pack_cyth plus all the package imports above.
              Must also have access to a data file & LSA or W2V file.

Modified by Jessie Dong specifically for use with ltpFR2.
"""


def pso(func, lb, ub, data_pres, w2v, swarmsize=100, omega_min=.8, omega_max=.8, d_omega=.1, c1=2, c2=2,
        c3=0.5, c4=0.5, R=1, c2_min=.5, c2_max=2.5, hard_bounds=False, maxiter=100, algorithm='pso', optfile=None):
    """
    Perform particle swarm optimization (PSO)

    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)
    data_pres : array
        A trials x items matrix of the ID numbers of items that will be
        presented to CMR2.
    w2v: array
        An items x items matrix of word2vec similarity scores to be passed on to
        CMR2.
    target_stats: dictionary
        A dictionary containing the empirical recall performance stats that will
        be used by the objective function to determine the difference between
        CMR2's performance and actual human performance.

    Optional
    ========
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega_min : scalar
        Minimum value of the particles' inertia. Except in certain special PSO
        algorithms (e.g. chaotic particle swarm optimization), inertia will decrease
        linearly from omega_max to omega_min over the course of the optimization
        process. If omega_min == omega_max, inertia will be constant. (Default: 0.8)
    omega_max : scalar
        Maximum value of the particles' inertia. Except in certain special PSO
        algorithms (e.g. chaotic particle swarm optimization), inertia will decrease
        linearly from omega_max to omega_min over the course of the optimization
        process. If omega_min == omega_max, inertia will be constant. (Default: 0.8)
    d_omega : scalar
        The value by which to increase/decrease omega at each iteration in the APSO-VI algorithm.
    c1 : scalar
        Scaling factor to move towards the particle's best known position
        (Default: 2.0)
    c2 : scalar
        Scaling factor to move towards the swarm's best known position
        (Default: 2.0)
    c3 : scalar
        Scaling factor to move away from the particle's worst known position
        (Default: 0.5)
    c4 : scalar
        Scaling factor to move away from the swarm's worst known position
        (Default: 0.5)
    R : scalar
        Velocity will be capped such that R is the greatest fraction of a dimension
        that can be traveled over a single iteration. For example, if R=.5 the max
        velocity on each dimension is 50% of that dimension's range. (Default: 1)
    c2_min : scalar
        Minimum value of the social acceleration constant (c2) during SAPSO and DPSO.
        (Default: 0.5)
    c2_max :
        Maximum value of the social acceleration constant (c2) during SAPSO and DPSO.
        (Default: 2.5)
    hard_bounds :
        If True, particles cannot fly outside of the parameter space, and will stop upon hitting an edge. If False,
        particles will be allowed to fly outside of the parameter space, but will not be evaluated if they do.
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    algorithm : string
        Specifies which particle swarm optimization algorithm to use. Options are:
        'pso' : Particle Swarm Optimization with inertia (Kennedy & Eberhart, 1995; Shi & Eberhart, 1998)
        'pso2 : Particle Swarm Optimization with constriction (Eberhart & Shi, 2000; Clerc & Kennedy, 2002)
        'sapso' : Self-Adaptive Particle Swarm Optimization (Wu & Zhou, 2007)
        'dpso' : Dispersed Particle Swarm Optimization (Cai, Cui, Zeng, & Tan, 2008)
        'cpso' : Chaotic Particle Swarm Optimization (Chaunwen & Bompard, 2005)
        'npso' : New Particle Swarm Optimization (Selvakumar & Thanushkodi, 2007)
        'apso6' : Adaptive Particle Swarm Optimization VI (Xu, 2013)
        'awl' : Particle Swarm Optimization with Avoidance of Worst Location (Mason & Howley, 2016)
    optfile : string or None
        If a file path is provided, the best known parameter set and score will be initialized as the parameters listed
        in that file. This allows a new particle swarm to begin with knowledge of a previous swarm's best position.

    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``

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

    lb = np.array(lb)
    ub = np.array(ub)
    unused_dim = ub == lb
    algorithm = algorithm.lower()
    S = swarmsize  # Number of particles
    D = len(lb)  # Number of dimensions

    # Initialize global best and worst fitness values such that any new value will replace them
    fgb = np.inf
    fgw = -np.inf

    # Initialize particle best and worst fitness values such that any new value will replace them
    fpb = np.zeros(S)
    fpb.fill(np.inf)
    fpw = np.zeros(S)
    fpw.fill(-np.inf)

    # Initialize best and worst particle positions to NaN
    pb = np.zeros((S, D))  # Best known position of each particle
    pb.fill(np.nan)
    pw = np.zeros((S, D))  # Worst known position of each particle
    pw.fill(np.nan)

    if isinstance(optfile, str) and os.path.exists(optfile):
        old_best = np.loadtxt(optfile)
        gb = old_best[:-1]
        fgb = old_best[-1]
        print('Loaded best known parameter location from file:', gb) # in the case where it is not the first run
        print('Best known parameter RMSD:', fgb)

    # Define maximum positive and negative velocities for each dimension
    vhigh = (ub - lb) * R
    vlow = -1 * vhigh

    # os.O_CREAT --> create file if it does not exist
    # os.O_EXCL --> error if create and file exists
    # os.O_WRONLY --> open for writing only
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    ##########
    #
    # PSO
    #
    ##########

    # Run PSO for maxiter iterations
    for it in range(1, maxiter + 1):
        print('Starting PSO iteration %s...' % it)

        ##########
        #
        # Update particle positions & velocities
        #
        ##########

        # If it is the first iteration, load initial particle positions and initialize velocities to zero
        if it == 1:
            print('Initializing particle locations and velocities...')
            x = np.loadtxt(NOISE_DIR + 'rx')
            v = np.zeros((S, D))

        # If it is any iteration beyond the first, update particle positions and velocities
        else:
            print('Updating particle locations and velocities...')

            # Read in the noise files for this iteration
            r1 = np.loadtxt(NOISE_DIR + 'r1_iter' + str(it))
            r2 = np.loadtxt(NOISE_DIR + 'r2_iter' + str(it))
            r3 = np.loadtxt(NOISE_DIR + 'r3_iter' + str(it))
            r4 = np.loadtxt(NOISE_DIR + 'r4_iter' + str(it))

            # Read in the position, best/worst location, & velocity files from previous iteration
            # If the number of values in the file does not match the number of particles, it means
            # another job is in the process of writing that file. Wait for it to finish and then
            # try again after a couple seconds.
            while True:
                x = np.loadtxt(OUTDIR + str(it - 1) + 'xfileb.txt')
                v = np.loadtxt(OUTDIR + str(it - 1) + 'vfileb.txt')
                pb = np.loadtxt(OUTDIR + str(it - 1) + 'pfileb.txt')
                pw = np.loadtxt(OUTDIR + str(it - 1) + 'pwfileb.txt')
                if (len(x) == S) and (len(v) == S) and (len(pb) == S) and (len(pw) == S):
                    break
                else:
                    time.sleep(2)

            # Update particle positions based on positions, velocities, and scores from last iteration

            # Basic PSO (Shi & Eberhart, 1998)
            if algorithm == 'pso':
                # Linearly decrease inertia over iterations and update velocities
                omega = omega_max - (omega_max - omega_min) * (it - 1) / (maxiter - 1)
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega * v[i, :] + t1[i, :] + t2

            # Particle swarm with constriction
            elif algorithm == 'pso2':
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega_max * (v[i, :] + t1[i, :] + t2)

            # Self-Adaptive PSO (Wu & Zhou, 2007)
            elif algorithm == 'sapso':
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    # Calculate relational distance of current particle to best location
                    rd = (fx[i] - gb) / fx[i]

                    # Set inertia and social acceleration based on relational distance
                    # Note that these calculations were slightly modified to use max and min settings
                    omega = (omega_max - omega_min) * (1 - np.cos(.5 * np.pi * rd)) + omega_min
                    c2 = (c2_max - c2_min) * (1 - np.cos(.5 * np.pi * rd)) + c2_min

                    # Update velocities
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega * v[i, :] + t1[i, :] + t2

            # Dispersed PSO (Cai, Cui, Zeng, & Tan, 2008)
            elif algorithm == 'dpso':
                # Linearly decrease inertia over iterations
                omega = omega_max - (omega_max - omega_min) * (it - 1) / (maxiter - 1)
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    # Set social acceleration based on current particle's performance relative to others
                    # in the current iteration
                    if it == 2:
                        c2 = c2_min
                    else:
                        grade = (fx.max() - fx[i]) / (fx.max() - fx.min()) if fx.max() != fx.min() else 1
                        c2 = c2_min + (c2_max - c2_min) * grade

                    # Update velocities
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega * v[i, :] + t1[i, :] + t2

                    # On average, mutate one dimension of one particle per iteration
                    # The new velocity is some random value between vhigh and vlow (slightly modified from original)
                    for dim in range(D):
                        if r3[i, dim] < 1. / (S * D):
                            v[i, dim] = vlow[dim] + r4[i, dim] * (vhigh[dim] - vlow[dim])

            # Chaotic PSO (Chaunwen & Bompard, 2005)
            elif algorithm == 'cpso':
                # Shift inertia on every trial
                if it == 2:
                    omega = r3
                else:
                    omega = 4 * omega * (1 - omega)
                # Update velocities
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega[i, :] * v[i, :] + t1[i, :] + t2

            # New PSO (Selvakumar & Thanushkodi, 2007)
            elif algorithm == 'npso':
                # Linearly decrease inertia over iterations
                omega = omega_max - (omega_max - omega_min) * (it - 1) / (maxiter - 1)

                # Move towards best locations while moving away from worst locations
                t1 = c1 * r1 * (pb - x)
                t3 = c3 * r3 * (x - pw)
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    t4 = c4 * r4[i, :] * (x[i, :] - gw)
                    v[i, :] = omega * v[i, :] + t1[i, :] + t2 + t3[i, :] + t4

            # Adaptive PSO-VI (Xu, 2013)
            elif algorithm == 'apso6':
                if it == 2:
                    omega = omega_max
                # Get average absolute velocity on each dimension, normalize velocities by parameter ranges, and
                # then average across dimensions to get the average velocity in range [0, 1] (ignore flat dimensions)
                avg_v = np.mean(np.mean(np.abs(v), axis=0)[~unused_dim] / (ub - lb)[~unused_dim])
                # Calculate optimal velocity for current iteration
                opt_v = .5 * (1 + np.cos(np.pi * (it - 1) / (.95 * maxiter))) / 2
                # Adjust inertia to approach optimal velocity
                omega = max(omega - d_omega, omega_min) if avg_v >= opt_v else min(omega + d_omega, omega_max)

                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega * v[i, :] + t1[i, :] + t2

            # PSO AWL (Mason & Howley, 2016)
            elif algorithm == 'awl':
                # Linearly decrease inertia over iterations
                omega = omega_max - (omega_max - omega_min) * (it - 1) / (maxiter - 1)

                # Move towards best locations, and move faster if near worst locations
                t1 = c1 * r1 * (pb - x)
                t3 = c3 * r3 * t1 / (1 + np.abs(x - pw))
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    t4 = c4 * r4[i, :] * t2 / (1 + np.abs(x[i, :] - gw))
                    v[i, :] = omega * (v[i, :] + t1[i, :] + t2 + t3[i, :] + t4)

            else:
                raise ValueError('Unrecognized PSO algorithm "%s" -- see docstring '
                                 'for list of supported algorithms.' % algorithm)

            # Keep velocity within the bounds of [vlow, vhigh]
            for i in range(S):
                mask1 = v[i, :] < vlow
                mask2 = v[i, :] > vhigh
                v[i, mask1] = vlow[mask1]
                v[i, mask2] = vhigh[mask2]

            # Update all particles' positions
            x += v

            # If hard search bounds are enforced, keep the particles within bounds
            if hard_bounds:
                for i in range(S):
                    mask1 = x[i, :] < lb
                    mask2 = x[i, :] > ub
                    x[i, mask1] = lb[mask1]
                    x[i, mask2] = ub[mask2]
                    # If a particle runs into the wall, set its velocity in that dimension to 0 to prevent it from
                    # running into the wall for multiple iterations
                    v[i, mask1] = 0
                    v[i, mask2] = 0

        ##########
        #
        # Test model for each particle
        #
        ##########

        # If the rmse file for this iteration already exists, load the RMSE values from that rather than from tempfiles
        if os.path.exists(OUTDIR + 'rmsesb_iter' + str(it)):
            while True:
                fx = np.loadtxt(OUTDIR + 'rmsesb_iter' + str(it))

                # If we got all the values we needed, break out of the loop, otherwise try again shortly
                if len(fx) == S:
                    break
                else:
                    time.sleep(2)

        else:
            # For each particle, test the model with parameters corresponding to that particle's location
            for i in range(S):

                # Determine which particles (if any) have flown out of bounds
                oob = np.any((x[i, :] < lb) | (x[i, :] > ub))

                match_file = OUTDIR + str(it) + 'tempfileb' + str(i) + '.txt'
                try:
                    # Try to open the tempfile; if the file for this particle already exists, skip to the next particle
                    fd = os.open(match_file, flags)

                    # Run CMR2 using this particle's position and get out the fitness score
                    if not oob:
                        print('Running model for particle %s...' % i)
                        rmse, stats = func(x[i, :], files, w2v)
                        print('Model finished with a fitness score of %s!' % rmse)
                    else:
                        print('Skipping out-of-bounds particle %s...' % i)
                        rmse = np.nan
                        stats = {}
                    # Save simulated behavioral stats from the particle
                    with open(OUTDIR + str(it) + 'data' + str(i) + '.pkl', 'wb') as f: # save per iterper
                        pkl.dump(stats, f, 2)

                    # Write the particle's fitness score to the tempfile
                    file_input = str(rmse)
                    os.write(fd, file_input.encode())
                    os.close(fd)

                except OSError as e:
                    if e.errno == errno.EEXIST:
                        print('Model for particle %s already complete! Skipping...' % i)
                        continue
                    else:
                        raise

            # Wait until all parallel jobs have finished running the models from the current iteration
            # Check for each expected tempfile. If any are empty, wait 2 seconds before trying again. If all
            # are finished, break out of the loop and proceed.
            while True:
                for i in range(S):
                    path = OUTDIR + '%stempfileb%s.txt' % (it, i)
                    if not (os.path.exists(path) and os.path.getsize(path) > 0.0):
                        break
                else:
                    break
                # sleep 2 seconds before we try again
                time.sleep(2)

            # Load the RMSE values for this iteration from all tempfiles, once we have confirmed they are finished
            fx = np.zeros(S)
            for i in range(S):
                fx[i] = np.loadtxt(OUTDIR + '%stempfileb%s.txt' % (it, i))

        ##########
        #
        # Search for new best/worst scores
        #
        ##########

        # Once all particles have finished for this iteration, collect all fitness values from the tempfiles
        # While doing this, check whether any new best and worst positions have been found
        print('Checking for new best/worst particle positions...')
        for i in range(S):

            # Skip particles that were out of bounds, and therefore were not scored
            if np.isnan(fx[i]):
                continue

            # Check whether the particle's current position is better than its previous best location
            if fx[i] < fpb[i]:
                print('New best location for particle %s!' % i)
                pb[i, :] = x[i, :].copy()
                fpb[i] = fx[i]

                # Check whether the particle's current position is better than the previous global best location
                if fx[i] < fgb:
                    print('Particle %s found a new best global location!' % i)
                    gb = x[i, :].copy()
                    fgb = fx[i]

            # Check whether the particle's current position is worse than its previous worst location
            if fx[i] > fpw[i]:
                print('New worst location for particle %s!' % i)
                pw[i, :] = x[i, :].copy()
                fpw[i] = fx[i]

                # Check whether the particle's current position is worse than the previous global worst location
                if fx[i] > fgw:
                    print('Particle %s found a new worst global location!' % i)
                    gw = x[i, :].copy()
                    fgw = fx[i]

        ##########
        #
        # Save results of iteration
        #
        ##########

        # Save the results from the current iteration before moving on to the next
        param_files = [OUTDIR + str(it) + 'xfileb.txt', OUTDIR + str(it) + 'pfileb.txt',
                       OUTDIR + str(it) + 'pwfileb.txt', OUTDIR + str(it) + 'vfileb.txt',
                       OUTDIR + 'rmsesb_iter' + str(it)]
        param_entries = [x, pb, pw, v, fx]
        for i in range(len(param_entries)):
            try:
                f = os.open(param_files[i], flags)
                os.close(f)
                np.savetxt(param_files[i], param_entries[i])
                print('Saved iteration results to %s!' % param_files[i])
            except OSError as e:
                if e.errno == errno.EEXIST:
                    continue
                else:
                    raise
        print('Iteration %s complete!' % it)

    return gb, fgb


def run_pso(files):

    global OUTDIR

    # Set file paths for data, wordpool, and semantic similarity matrix
    w2v_file = '/data/eeg/scalp/ltp/ltpFR3_MTurk/CMR2_ltpFR3/w2v.txt'

    # Load semantic similarity matrix (word2vec)
    w2v = np.loadtxt(w2v_file)


    # Set PSO parameters
    alg = 'apso6'
    swarmsize = 100
    n_iter = 30
    omega_min = .729844 if alg in ('pso2', 'awl') else .3 if alg == 'apso6' else .4
    omega_max = .729844 if alg in ('pos2', 'awl') else .9
    d_omega = .1  # Delta omega for apso6 algorithm
    c1 = 2.05 if alg == 'pso2' else 1.845 if alg == 'awl' else 1.496180  # 2
    c2 = 2.05 if alg == 'pso2' else 1.845 if alg == 'awl' else 1.496180  # 2
    c3 = .205
    c4 = .205
    c2_min = .5
    c2_max = 2.5
    R = 1  # Maximum velocity constraint (max = R * (ub - lb))

    #    [b_e, b_r, g_fc, g_cf, p_s, p_d,  k,   e, s_cf, b_rp,   o,   a, c_t,   l, p_s_fc, p_s_cf]
    lb = [.40,  0., .001, .001,  0.,  0., 0., .01,   0.,   .5,  4., .30,  0.,   0.,  .001, .001]
    ub = [.75,  1., .999, .999, 15., 15., 1., .60,  10.,   1., 20., .99,  .5,  .75,  .999, .999]

    print('Generating noise files...')
    make_noise(swarmsize, n_iter, lb, ub)

    print('Initiating particle swarm optimization...')
    start_time = time.time()
    xopt, fopt = pso(obj_func, lb, ub, w2v, files, swarmsize=swarmsize, maxiter=n_iter,
                     omega_min=omega_min, omega_max=omega_max, d_omega=d_omega, c1=c1, c2=c2, c3=c3, c4=c4, R=R,
                     c2_min=c2_min, c2_max=c2_max, algorithm=alg, optfile=None)

    print(fopt, xopt)
    print("Run time: " + str(time.time() - start_time))#runtime will take forever I'm sure :)
    sys.stdout.flush()

    np.savetxt(OUTDIR + 'xoptb_ltpFR2.txt', xopt, delimiter=',', fmt='%f')


if __name__ == "__main__":

    quarter = "1" # or 4

    path = ("/home1/yedong/svn/pyCMR2/branches/CMR2_yedong/quart_%s/" % quarter)

    os.chdir(path)
    all_data = np.sort(glob('*'))
    grouped_data = [(path+all_data[i], path+all_data[i+int(len(all_data)/2)])for i in range(int(len(all_data)/2))]
    files = random.sample(grouped_data, 200)

    run_pso(files)