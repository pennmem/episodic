import numpy as np
import glob
import matplotlib.pyplot as plt
import re

"""This code reads in values from a particle swarm's xfile's and rmse files,
plots the spread of parameter values across iterations, and identifies the
best-fitting parameter locations in black
(vs. the other values, which are in gray)"""


def get_best_params(iter_num):

    rmse_path = 'rmses_iter' + str(iter_num)

    # load rmse values text
    rmse_file = np.loadtxt(rmse_dir + rmse_path, delimiter=' ')

    # get rmse values from left column
    rmse_vals = rmse_file[:, 0]
    rmse_indices = rmse_file[:, 1]

    # get minimum rmse value
    min_rmse = np.min(rmse_vals)

    # get index where min rmse value was located
    min_rmse_index = int(rmse_indices[np.where(rmse_vals == min_rmse)][0])

    # find the set of parameters in the associated xfile that match
    xfile_path = str(iter_num) + 'xfile.txt'
    xfile = np.loadtxt(rmse_dir + xfile_path, delimiter=' ')

    # get out best parameter values for this iteration
    best_params = xfile[min_rmse_index]

    return best_params

#____________________________
def read_rmse_files():
    """read in the rmse files and save them in a way that is conducive
    for the functions that find the best iteration's values"""

    # rmse file paths are saved in this format
    rmse_stem = 'rmses_iter*'

    # get rmse file paths
    rmse_paths = glob.glob(rmse_dir + rmse_stem)

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


def get_min_iter_loc():
    # read in the consolidated files
    rmse_array = read_rmse_files()

    # number of iterations, including the first (0th) iteration, pre-swarm
    nits = len(rmse_array) + 1

    # get list of the minimum rmse in each iteration
    min_rmses, mean_rmses = get_min_rmses(rmse_array)

    # this is the index of the smallest rmse value across all iterations:
    smallest_rmse_loc = np.where(min_rmses == min(min_rmses))[0][0]

    return smallest_rmse_loc+1
#__________________________________________________


def main():
    import os
    os.chdir('Outs-001')
    global rmse_dir
    # set directory where the rmse files are located
    rmse_dir = './'

    # rmse file paths are saved in this format
    rmse_stem = 'rmses_iter*'

    # get rmse file paths
    rmse_paths = glob.glob(rmse_dir + rmse_stem)

    # get number of rmse paths in directory
    max_iterations = len(rmse_paths)

    # for each rmse file,
    best_params_list = []
    for r_idx in range(max_iterations):

        best_par = get_best_params(r_idx+1)
        best_params_list.append(best_par)

    # make into an array for easy indexing
    best_params_array = np.asarray(best_params_list)

    best_iter = get_min_iter_loc()
    overall_best_params = get_best_params(best_iter)

    # use lower and upper bounds to set graph limits on y axis
    lb = [0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.01, 0.01, 0.5, .1, 5.0, .5, .001, .01]
    ub = [1.0, 1.0, 0.7, 1.0, 3.0, 1.5, 0.5, 0.5, 3.0, 1.0, 15.0, 1.0, 0.8, 0.5]

    # set labels for each parameter's graph
    param_labels = [[r'$\beta_{enc}$'],
                    [r'$\beta_{rec}$'],
                    [r'$\gamma_{FC}$'],
                    [r'$\gamma_{CF}$'],
                    [r'$\phi_{s}$'],
                    [r'$\phi_{d}$'],
                    [r'$\kappa$'],
                    [r'$\eta$'],
                    [r'$s$'],
                    [r'$\beta_{post}$'],
                    [r'$\omega$'],
                    [r'$\alpha$'],
                    [r'$c_{thresh}$'],
                    [r'$\lambda$']]

    # set number of parameters we are evaluating
    num_params = len(lb)

    # plot where each parameters' values lie along their value ranges
    plt.figure(figsize=(12, 4))

    # 1 row x 2 col grid, first subplot
    for i in range(num_params):

        # make one subplot per parameter
        plt.subplot(1, num_params+1, i+1)
        plt.scatter([0]*max_iterations, best_params_array[:, i], color='.60')
        plt.scatter(0, overall_best_params[i], color='k')
        plt.ylim([lb[i], ub[i]])
        # show the values for the lower, upper, and midway bounds
        plt.yticks([lb[i], lb[i]+(ub[i]-lb[i])/2, ub[i]])
        plt.xticks([0], param_labels[i])

    plt.subplots_adjust(left = .05, right = 1.05, wspace=1.5)

    # save fig nicely
    save_figs = True
    if save_figs:
        plt.savefig('./param_diagnostics.eps', format='eps', dpi=1000)

    plt.show()


if __name__ == "__main__": main()