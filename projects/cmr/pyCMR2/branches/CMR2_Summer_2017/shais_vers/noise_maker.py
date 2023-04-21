"""Make the noise matrices ahead of time for the particle swarm;
   This way we can keep all the instances of particle swarm on
   the same page, i.e., performing the same operations on each
   set of parameters."""
import numpy as np
import sys

def main(S, I):

    lb = [0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.01, 0.01, 0.5, .1, 5.0, .5, .001, .01]
    ub = [1.0, 1.0, 0.7, 1.0, 3.0, 1.5, 0.5, 0.5, 3.0, 1.0, 15.0, 1.0, 0.8, 0.5]
    
    # swarm size
    S = int(S)
    
    # dimensions of the parameter vectors
    D = len(lb)
    
    # max no. of iterations; enter at command line
    # e.g., python noise_maker.py 10
    max_iter = int(I)
    
    # for each iteration, save out a file that already has all the noise values
    for iter in range(max_iter):
    
        # draw & save a sheet for rp
        rp = np.random.uniform(size=(S, D))
        np.savetxt('rp_iter%i' % (iter+1), rp)
    
        # draw & save a sheet for rg
        rg = np.random.uniform(size=(S, D))
        np.savetxt('rg_iter%i' % (iter+1), rg)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])