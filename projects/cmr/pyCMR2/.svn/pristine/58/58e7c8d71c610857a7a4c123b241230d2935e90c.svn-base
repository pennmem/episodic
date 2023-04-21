import numpy as np
import glob
import matplotlib.pyplot as plt
import re

#saved_files_path = '/home1/rivkat.cohen/PycharmProjects/' \
#                   + 'testing_Jul9/ctw_method/CMR2_lowmem/noise_test_10_10/'

# for the first rmse's, in the round with SS=100, max_iter=100, and time=30k,
# enter round1_30k
# ^ best val was around 14.  same for the others.

# for the second round, using that same error function, with SS=100, max_iter50,
# and time=75k, enter round2_75k

# for the third round, using the chi^2 error function, with SS=100, max_iter50,
# and time=75k, enter round3_75k_chi

saved_files_path = './'

# for each item in the range
iter_paths = []
for i in range(101):

    # get set of paths for the info in ith iteration
    this_iter_paths = glob.glob(saved_files_path+str(i)+'tempfileb*')
    iter_paths.append(this_iter_paths)


def format_temp_files(niterations):
    """read all the temp files for j iterations and save them into just one file
    per iteration that saves all the rmse's (and their file index)
    into a single file"""

    # for each iteration's set of rmse files,
    for j in range(niterations):

        # point to the iteration's paths
        this_set_of_paths = iter_paths[j]

        # read in rmse tuples from files
        rmse_tuples = []
        for path in this_set_of_paths:
            # load (rmse, iter no.) from file
            rmse_tuple = np.loadtxt(path, delimiter=',')

            # append to list of these arrays
            rmse_tuples.append(rmse_tuple)

        # sort the array according to the col. of iteration indices
        rmse_array = np.asarray(rmse_tuples)
        sorted_rmse = rmse_array[rmse_array[:, 1].argsort()]

        # save this into just one file that we can later get these values back
        # without storing them all individually
        np.savetxt(
            saved_files_path+'rmsesb_iter'+str(j), sorted_rmse, delimiter=',')


def read_rmse_files():

    # rmse file paths are saved in this format
    rmse_stem = 'rmsesb_iter*'

    # get rmse file paths
    rmse_paths = glob.glob(saved_files_path + rmse_stem)

    # sort paths numerically / human-style!
    rmse_paths.sort(
        key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(
            r'[^0-9]|[0-9]+', var)])

    rmse_values = []
    for i in range(len(rmse_paths)):

        # read in the rmse file
        rmse_file = np.loadtxt(rmse_paths[i], delimiter=' ')

        # get rmse values from left column
        rmse_vals = rmse_file[:,0]

        # store the rmse values in overall list
        rmse_values.append(rmse_vals)

    return np.array(rmse_values)


def get_min_rmses(rmse_arr):
    """take in an rmse array, with iteration no. along each row,
    and particle ID number along each column"""

    min_rmse_per_iter = []
    mean_rmse_per_iter = []
    for iter in rmse_arr:
        # get min rmse for this iteration
        min_rmse = np.min(iter)

        # get mean rmse for this iteration
        mean_rmse = np.mean(iter)

        # append to list
        min_rmse_per_iter.append(min_rmse)
        mean_rmse_per_iter.append(mean_rmse)

    return min_rmse_per_iter, mean_rmse_per_iter


def main():

    # consolidate any temp files (comment this out if already done)
    # format_temp_files(nits)

    # read in the consolidated files
    rmse_array = read_rmse_files()

    # number of iterations, including the first (0th) iteration, pre-swarm
    nits = len(rmse_array)+1

    # get list of the minimum rmse in each iteration
    min_rmses, mean_rmses = get_min_rmses(rmse_array)

    # print min and mean rmse values for each iteration
    print("\nMin rmses: ")
    print(min_rmses)
    print("Mean rmses: ")
    print(mean_rmses)

    print("\nSmallest rmse is at iteration:")
    smallest_rmse_loc = np.where(min_rmses == min(min_rmses))[0][0]
    print(smallest_rmse_loc+1)
    print("And its value is: ")
    print(min(min_rmses))

    # plot min rmse vs. iteration number
    plt.figure()
    pso_plot, = plt.plot(
        range(nits-1), min_rmses, color='blue', label="Particle Swarm")
    vline_plot = plt.axvline(
        x=28, color='k', linestyle='--', label="Iter = 28")
    plt.title("Minimum RMSE vs. Iteration")
    plt.axis([-1, (nits-1), (min(min_rmses)-1), (max(min_rmses)+1)])
    plt.xlabel("Iteration")
    plt.ylabel("Min RMSE")
    plt.legend(handles = [pso_plot, vline_plot])
    plt.savefig('RMSE_plot.eps', format='eps', dpi=1000)

    # new plot with mean rmse vs. iteration number
    plt.figure()
    plt.plot(range(nits-1), mean_rmses)
    plt.axis([-1, (nits - 1), (min(mean_rmses) - 1), (max(mean_rmses) + 1)])
    plt.title("Mean RMSE vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Mean RMSE")

    plt.show()

if __name__=="__main__":main()

