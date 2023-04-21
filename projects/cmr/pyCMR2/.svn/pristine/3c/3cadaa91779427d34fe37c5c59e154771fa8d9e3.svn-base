"""This code runs parallel simulations of CMR2 predictions on the Rhino cluster"""
import mkl
mkl.set_num_threads(1)
import errno
import numpy as np
import glob
import os
import CMR2_source_dfr as CMR2
# import CMR_package as CMR

# Set directory to where the parameter files are stored
param_dir = '/home1/rivkat.cohen/PycharmProjects/CMR2_Code_Fixed/param_files/'

##########
#
#   Define helper functions
#
##########
def get_eval_param_dict_test(param_vec):
    """helper function to properly format a parameter vector"""
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

        'eta': param_vec[7],
        's_cf': param_vec[8],
        's_fc': 0.0,
        'beta_rec_post': param_vec[9],
        'omega': param_vec[10],
        'alpha': param_vec[11],
        'c_thresh': param_vec[12],
        'dt': 10.0,

        'lamb': param_vec[13],
        'beta_source': param_vec[14],
        'beta_distract': param_vec[15],
        'rec_time_limit': 75000,

        'dt_tau': 0.01,
        'sq_dt_tau': 0.10,

        'nlists_for_accumulator': 2,

        'L_CF_NW': param_vec[3],  # NW quadrant - set constant to 1.0
        'L_CF_NE': 0.0, # param_vec[16],  # NE quadrant - set to gamma_cf value
        'L_CF_SW': 0.0,  # SW quadrant - set constant to 0.0
        #'L_CF_SE': 0.0,  # SE quadrant - set constant to 0.0
        'L_CF_SE': 0.0,

        'L_FC_NW': param_vec[2],  # NW quadrant - set to gamma_fc value
        'L_FC_NE': 0.0,  # NE quadrant - set constant to 0.0
        'L_FC_SW': param_vec[2],  # SW quadrant - set to gamma_fc value
        #'L_FC_SE': 0.0  # SE quadrant - set constant to 0.0
        'L_FC_SE': 0.0
    }

    return param_dict


##########
#
#   Read in and format parameter vectors
#
##########

# Read in the list of paths of all parameter vector files.
param_paths = glob.glob(param_dir + "*xopt*")

# Use this if we just want to look at one participant's parameters (e.g., LTP321)
# param_paths = glob.glob(param_dir + "*xopt*LTP321*")

# Read in the parameter vector from each parameter file.
param_sets = []
for p_path in param_paths:

    params = np.loadtxt(p_path)
    param_sets.append(params)

##########
#
#   Run Simulation, distributed across cores on Rhino
#
##########

# set path to where simulation setup is located.
simulation_root_dir = '/home1/rivkat.cohen/PycharmProjects/Simulation_SummaryStatistics/'

# Select the simulation to run.
# This string just formats the file names across presented-items and the emotional
# valence source codes. You can replace this as long as python knows where to find your
# presented-items sheets and source code sheets.
simulation_name = 'repeat_test2_withrepeats'

# Read in the presented-items array for this simulation
pres_items = np.loadtxt(
    simulation_root_dir + 'pres_files/pres_items_' + simulation_name + '.txt',
    delimiter=',')
pres_items_path = simulation_root_dir + 'pres_files/pres_items_' \
                  + simulation_name + '.txt'

#####
#
#   Note to Joe: Emotional-valence source codes are matrices in the same shape and
#                format as the presented-item matrices.  For your purposes, just
#                create a text file in the same dimensions as the presented-item
#                sheets that has all zeros in it.  The 0's are because you're not
#                using emotion source information for your fits. To be super thorough,
#                update your function that reads in the parameters to always set
#                beta_source = 0.0
#
#####

# Read in the eval-codes file for this simulation
eval_values = np.loadtxt(
    simulation_root_dir + 'eval_files/eval_codes_' + simulation_name + '.txt',
    delimiter=',')
eval_path = simulation_root_dir + 'eval_files/eval_codes_' \
            + simulation_name + '.txt'

# Read in the inter-item similarity values for this simulation
LSA_path = 'sim_files/interitem_sims_' + simulation_name + '.txt'
LSA_mat = np.loadtxt(LSA_path, delimiter=',')

# set the output directory
output_dir = simulation_root_dir + 'output_files_' + simulation_name + '/'

# os.O_CREAT --> create file if it does not exist
# os.O_EXCL --> error if create and file exists
# os.O_WRONLY --> open for writing only
flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

# Run the CMR model; distribute this action over parameter sets.
for idx, p in enumerate(param_sets):

    # set name of temp file to check for
    match_file = simulation_name + '_tempfile' + str(idx) + '.txt'

    # try to open the file
    try:
        # try to open the file
        fd = os.open(match_file, flags)

        # format parameters into a dictionary
        params_to_test = get_eval_param_dict_test(p)

        # set parameters to the desired values for this simulation.
        params_to_test['s_cf'] = .15

        # ptsd cues simulation:
        # params_to_test['beta_source'] = .20
        # params_to_test['omega'] = 3.0

        # set path to the file that tells CMR2 how many lists
        # belong to this single subject
        subject_id_path = 'division_locs_ind1.txt'

        # run CMR2 on p and get the output
        model_rec_nos, times = CMR2.run_CMR2(LSA_path=LSA_path,
                                             LSA_mat=LSA_mat,
                                             data_path=pres_items_path,
                                             params=params_to_test,
                                             sep_files=False,
                                             source_info_path=eval_path,
                                             nsource_cells=2,
                                             subj_id_path=subject_id_path)

        # save the model-predicted recall outputs
        np.savetxt(output_dir +
            'model_recalls_' + simulation_name + 'run' + str(idx) + '.txt',
            model_rec_nos, fmt='%i')

        # close the temporary file
        os.close(fd)

    # If the temporary file already exists,
    except OSError as e:
        if e.errno == errno.EEXIST:  # then move to the next param set.
            continue
        # If it's some other error, raise the error.
        else:
            raise

