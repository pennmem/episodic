import numpy as np
import sys

rmse_dir = './'
rmse_path = 'rmsesb_iter' + sys.argv[1]

# load rmse values text
rmse_file = np.loadtxt(rmse_dir + rmse_path, delimiter=' ')

# get rmse values from left column
rmse_vals = rmse_file[:,0]
rmse_indices = rmse_file[:,1]

# get minimum rmse value
min_rmse = np.min(rmse_vals)

# get index where min rmse value was located
min_rmse_index = int(rmse_indices[np.where(rmse_vals == min_rmse)][0])

print("\nMinimum rmse and matching index are: ")
print("RMSE: ", min_rmse)
print("Index: ", min_rmse_index)

# find the set of parameters in the associated xfile that match
xfile_path = sys.argv[1] + 'xfileb.txt'

xfile = np.loadtxt(rmse_dir + xfile_path, delimiter=' ')

print("\nBest-fitting parameters for iteration " + sys.argv[1] + " are: ")
print(xfile[min_rmse_index])

np.savetxt('best_params_'+sys.argv[1], xfile[min_rmse_index], delimiter=',')
