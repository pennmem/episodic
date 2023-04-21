import os
import sys
import errno
import numpy as np


def make_noise(S, ngen, lb, ub, parent_rate, path):
    """
    Make the noise matrices ahead of time for the genetic algorithm;
    This way we can keep all of the parallel instances on the
    same page, i.e., performing the same operations on each
    set of parameters.

    :param S: An integer or list indicating the population size in each block.
    :param ngen: An integer or list indicating the number of generations to be run in each block.
    :param lb:
    :param ub:
    :param D: An integer indicating the number of dimensions in the search space.
    :param parent_rate: A float or list indicating the fraction of the population to be used as parents in each block.
    """
    # os.O_CREAT --> create file if it does not exist
    # os.O_EXCL --> error if create and file exists
    # os.O_WRONLY --> open for writing only
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    # Initialize population sizes and numbers of parents
    ngen = [ngen] if not hasattr(ngen, '__iter__') else ngen
    S = [S] if not hasattr(S, '__iter__') else S
    parent_rate = [parent_rate] if not hasattr(parent_rate, '__iter__') else parent_rate
    popsizes = []
    n_parents = []
    for i, n in enumerate(ngen):
        popsizes = np.concatenate((popsizes, [S[i] for _ in range(n)]))
        n_parents = np.concatenate((n_parents, [parent_rate[i] for _ in range(n)]))
    n_parents *= np.concatenate(([0], popsizes[:-1]))
    n_parents = n_parents.astype(int)
    popsizes = popsizes.astype(int)
    ngen = np.sum(ngen)
    D = len(lb)

    # Initialize start locations
    try:
        f = os.open(path + 'rx', flags)
        os.close(f)
        rx = np.random.uniform(size=(popsizes[0], D))
        lb_mat = np.atleast_2d(lb).repeat(popsizes[0], axis=0)
        ub_mat = np.atleast_2d(ub).repeat(popsizes[0], axis=0)
        rx = lb_mat + rx * (ub_mat - lb_mat)
        np.savetxt(path + 'rx', rx)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    for gen in range(1, ngen + 1):

        # R1: Random integers for choosing parents for each child
        try:
            f = os.open(path + 'r1_iter%i' % gen, flags)
            os.close(f)
            r = np.zeros((popsizes[gen - 1], 2), dtype=int)
            for i in range(popsizes[gen - 1]):
                r[i, :] = np.random.choice(n_parents[gen - 1], size=2, replace=False) if n_parents[gen - 1] > 0 else r[i, :]
            np.savetxt(path + 'r1_iter%i' % gen, r, fmt='%i')
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # R2: Random values for choosing which parameters to cross
        try:
            f = os.open(path + 'r2_iter%i' % gen, flags)
            os.close(f)
            r = np.random.uniform(size=(popsizes[gen - 1], D))
            np.savetxt(path + 'r2_iter%i' % gen, r)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # R3: Random values for choosing which parameters to mutate
        try:
            f = os.open(path + 'r3_iter%i' % gen, flags)
            os.close(f)
            r = np.random.uniform(size=(popsizes[gen - 1], D))
            np.savetxt(path + 'r3_iter%i' % gen, r)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # R4: Random Gaussian values for scaling mutations
        try:
            f = os.open(path + 'r4_iter%i' % gen, flags)
            os.close(f)
            r = np.random.normal(size=(popsizes[gen - 1], D))
            np.savetxt(path + 'r4_iter%i' % gen, r)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
