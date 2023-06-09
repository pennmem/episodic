import os
import errno
import numpy as np


def make_noise(S, max_iter, lb, ub, path):
    """
    Make the noise matrices ahead of time for the particle swarm;
    This way we can keep all the instances of particle swarm on
    the same page, i.e., performing the same operations on each
    set of parameters.

    :param S: An integer indicating the particle swarm size.
    :param max_iter: An integer indicating the number of iterations of particle swarm to be run.
    """
    # os.O_CREAT --> create file if it does not exist
    # os.O_EXCL --> error if create and file exists
    # os.O_WRONLY --> open for writing only
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    D = len(lb)

    # Initialize particle locations
    try:
        f = os.open(path + 'rx', flags)
        os.close(f)
        rx = np.random.uniform(size=(S, D))
        lb_mat = np.atleast_2d(lb).repeat(S, axis=0)
        ub_mat = np.atleast_2d(ub).repeat(S, axis=0)
        rx = lb_mat + rx * (ub_mat - lb_mat)
        np.savetxt(path + 'rx', rx)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # Create files for r1 through r4 for each iteration
    for it in range(1, max_iter + 1):
        for i in range(1, 5):
            try:
                f = os.open(path + 'r%i_iter%i' % (i, it), flags)
                os.close(f)
                r = np.random.uniform(size=(S, D))
                np.savetxt(path + 'r%i_iter%i' % (i, it), r)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    pass
                else:
                    raise

if __name__ == "__main__":
    
    path = ''
    try:
        os.chdir('/home1')
        path = '/home1/shai.goldman/pyCMR2/IRT_Optimizations/noise_files/'
    except:
        path = '/Users/lumdusislife/Desktop/IRT/pyCMR2/IRT_Optimizations/noise_files/'
    
    swarm_size = 5
    max_iter = 5
    lb = [.30,  0., .001, .001,  0.,  0., 0., .01,   0.,   0.,  4., .01,  0.,  0.]
    ub = [.75,  1., .999, .999, 10.,  2., 1., .60,  10.,   1., 20., .99,  .5, .75]
    
    make_noise(swarm_size, max_iter, lb, ub, path)
    