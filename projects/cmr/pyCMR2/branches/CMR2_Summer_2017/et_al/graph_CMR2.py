import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import CMR2_pack_cyth as CMR2
import pso2_par_cmr2 as pso2_cmr2
import lagCRP2
import os

# SS = 200, iterations = 10
params_Jul11_1 = {
    'beta_enc': 0.557052,
    'beta_rec': 0.335805,
    'gamma_fc': 0.467581,
    'gamma_cf': 0.629960,
    'scale_fc': 1 - 0.467581,
    'scale_cf': 1 - 0.629960,

    'phi_s': 1.243328,
    'phi_d': 1.411496,
    'kappa': 0.138064,

    'eta': 0.282556,
    's_cf': 1.389009,
    's_fc': 0.0,
    'beta_rec_post': 0.631524,
    'omega': 9.413798,
    'alpha': 1.000000,
    'c_thresh': 0.622645,
    'dt': 10.0,

    'lamb': 0.145817,
    'rec_time_limit': 30000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
}

# second full fit on the cluster
params_Jul11 = {
    'beta_enc': 0.780658,
    'beta_rec': 0.461688,
    'gamma_fc': 0.437324,
    'gamma_cf': 1.00000,
    'scale_fc': 1 - 0.437324,
    'scale_cf': 1 - 1.00000,

    'phi_s': 0.979786,
    'phi_d': 0.829000,
    'kappa': 0.180824,

    'eta': 0.484388,
    's_cf': 2.200333,
    's_fc': 0.0,
    'beta_rec_post': 0.684852,
    'omega': 14.704023,
    'alpha': 0.620958,
    'c_thresh': 0.764707,
    'dt': 10.0,

    'lamb': 0.099715,
    'rec_time_limit': 30000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
}

# first full fit on the cluster
cluster_params = {

    'beta_enc': 0.495437,
    'beta_rec': 0.439492,
    'gamma_fc': 0.383945,
    'gamma_cf': 0.604645,
    'scale_fc': 1 - 0.383945,
    'scale_cf': 1 - 0.604645,

    'phi_s': 1.705545,
    'phi_d': 1.177403,
    'kappa': 0.396722,

    'eta': 0.298908,
    's_cf': 0.873401,
    's_fc': 0.0,
    'beta_rec_post': 0.311773,
    'omega': 11.924991,
    'alpha': 0.734691,
    'c_thresh': 0.030636,
    'dt': 10.0,

    'lamb': 0.060535,
    'rec_time_limit': 30000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
}

# first full fit on local machine (27 hrs)
test_params = {

    'beta_enc': 0.680744734,
    'beta_rec': 0.361654192,
    'gamma_fc': 0.516511119,
    'gamma_cf': 1.00000000,
    'scale_fc': 1 - 0.516511119,
    'scale_cf': 1 - 1.0,

    'phi_s': 2.03215149,
    'phi_d': 1.01570302,
    'kappa': 0.0103982004,

    'eta': 0.463768758,
    's_cf': 2.83502599,
    's_fc': 0.0,
    'beta_rec_post': 0.383720613,
    'omega': 13.0579185,
    'alpha': 0.595530856,
    'c_thresh': 0.654458322,
    'dt': 10.0,

    'lamb': 0.138805728,
    'rec_time_limit': 30000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
}

# second full fit on local machine (27 hrs)
test_params2 = {

    'beta_enc': 0.726453661,
    'beta_rec': 0.447389075,
    'gamma_fc': 0.436410186,
    'gamma_cf': 1.00000000,
    'scale_fc': 1 - 0.436410186,
    'scale_cf': 1 - 1.0,

    'phi_s': 1.50013803,
    'phi_d': 0.687775003,
    'kappa': 0.0165276673,

    'eta': 0.456726229,
    's_cf': 1.28707748,
    's_fc': 0.0,
    'beta_rec_post': 0.437925031,
    'omega': 10.9426798,
    'alpha': 0.691488229,
    'c_thresh': 0.001,
    'dt': 10.0,

    'lamb': 0.120444084,
    'rec_time_limit': 30000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
}

params_Lohnas = {

    'beta_enc': 0.519769,
    'beta_rec': 0.627801,
    'gamma_fc': 0.425064,
    'gamma_cf': 0.895261,
    'scale_fc': 1 - 0.425064,
    'scale_cf': 1 - 0.895261,

    'phi_s': 1.408899,
    'phi_d': 0.989567,
    'kappa': 0.312686,

    'eta': 0.392847,
    's_cf': 1.292411,
    's_fc': 0.0,
    'beta_rec_post': 0.802543,
    'omega': 11.894106,
    'alpha': 0.678955,
    'c_thresh': 0.073708,
    'dt': 10.0,

    'lamb': 0.129620,
    'rec_time_limit': 30000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
}

params_LTP093_Jul16 = {
    'beta_enc': 0.350839,
    'beta_rec': 0.487429,
    'gamma_fc': 0.600312,
    'gamma_cf': 0.323763,
    'scale_fc': 1 - 0.600312,
    'scale_cf': 1 - 0.323763,

    'phi_s': 1.290894,
    'phi_d': 1.170226,
    'kappa': 0.361799,

    'eta': 0.208663,
    's_cf': 1.153462,
    's_fc': 0.0,
    'beta_rec_post': 0.692177,
    'omega': 8.592228,
    'alpha': 0.733212,
    'c_thresh': 0.509451,
    'dt': 10.0,

    'lamb': 0.303002,
    'rec_time_limit': 30000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
    
    }
    
    
    
    
def main():

    ###################
    #
    #   Get data & CMR2-predicted data
    #
    ###################

    # select params to test from among the options defined above
    params_to_test = params_LTP093_Jul16

    # Set LSA and data paths -- K02 data
    LSA_path = '/Users/KahaNinja/PycharmProjects/CMR2/K02_files/K02_LSA.mat'
    data_path = '/Users/KahaNinja/PycharmProjects/CMR2/K02_files/K02_data.mat'

    # read in LSA matrix
    LSA_mat = scipy.io.loadmat(
        LSA_path, squeeze_me=True, struct_as_record=False)['LSA']

    # run CMR2
    rec_nos, times = CMR2.run_CMR2(LSA_path=LSA_path, LSA_mat=LSA_mat,
                              data_path=data_path,
                              params=params_to_test, sep_files=False)

    # save the output somewhere convenient
    np.savetxt(
        './ParamTest_Outputs/resp_pso_test_K02.txt', np.asmatrix(rec_nos),
        delimiter=',', fmt='%.0d')
    np.savetxt(
        './ParamTest_Outputs/times_pso_test_K02.txt', np.asmatrix(times),
        delimiter=',', fmt='%.0d')



    # raise ValueError("stop and check")
    ####################
    #
    #   Graph results from output (lag-CRP, SPC, PFC)
    #
    ####################

    # set list length here
    ll = 10

    # decide whether to save figs out or not
    save_figs = True

    ####
    #
    #   Get data output
    #
    ####

    # get data file, presented items, & recalled items
    data_file = scipy.io.loadmat(
        data_path, squeeze_me=True, struct_as_record=False)

    data_pres = data_file['data'].pres_itemnos      # presented
    data_rec = data_file['data'].rec_itemnos        # recalled

    # recode data lists for spc, pfc, and lag-CRP analyses
    recoded_lists = pso2_cmr2.recode_for_spc(data_rec, data_pres)

    # save out the recoded lists in case you want to read this in later
    np.savetxt('recoded_lists_pso_test_K02.txt', recoded_lists,
               delimiter=',', fmt='%.0d')

    # get spc & pfc
    target_spc, target_spc_sem, target_pfc, target_pfc_sem = \
        pso2_cmr2.get_spc_pfc(recoded_lists, ll)

    target_crp, target_crp_sem = lagCRP2.get_crp(recoded_lists, ll)

    # get Lag-CRP sections of interest
    center_val = ll - 1

    target_left_crp = target_crp[center_val-1:center_val]
    target_left_crp_sem = target_crp_sem[center_val-1:center_val-]

    target_right_crp = target_crp[center_val+1:center_val+6]
    target_right_crp_sem = target_crp_sem[center_val+1:center_val+6]

    ####
    #
    #   Get CMR2 output
    #
    ####

    cmr_recoded_output = pso2_cmr2.recode_for_spc(rec_nos, data_pres)

    # get the model's spc and pfc predictions:
    (this_spc, this_spc_sem, this_pfc,
    this_pfc_sem) = pso2_cmr2.get_spc_pfc(cmr_recoded_output, ll)

    # get the model's crp predictions:
    this_crp, this_crp_sem = lagCRP2.get_crp(cmr_recoded_output, ll)

    # get left crp values
    this_left_crp = this_crp[4:9]

    # get right crp values
    this_right_crp = this_crp[10:15]

    ####
    #
    #   Plot graphs
    #
    ####

    # line width
    lw = 2

    #_______________________ plot spc
    fig1 = plt.figure()
    xvals = range(1, 11, 1)     # ticks for x-axis

    plt.plot(xvals, this_spc, lw=lw, c='k', linestyle='--', label='CMR2')
    plt.plot(xvals, target_spc, lw=lw, c='k', label='K02 Data')

    plt.ylabel('Probability of Recall', size='large')
    plt.xlabel('Serial Position', size='large')
    plt.xticks(xvals, size='large')
    plt.yticks(np.arange(0.0, 1.2, 0.2), size='large')
    plt.axis([0.5, 10.5, 0, 1], size='large')
    plt.legend(loc='upper left')
    plt.title('Serial Position Curve', size='large')

    # save fig nicely
    if save_figs:
        plt.savefig('./Figs/spc_fig.eps', format='eps', dpi=1000)

    #_______________________ plot crp
    fig2 = plt.figure()

    xvals_left = range(-5, 0, 1)
    xvals_right = range(1, 6, 1)

    # left
    plt.plot(xvals_left, this_left_crp, lw=lw, c='k',
             linestyle='--', label='CMR2')
    plt.plot(xvals_left, target_left_crp, lw=lw, c='k', label='K02 Data')

    # right
    plt.plot(xvals_right, this_right_crp, lw=lw, c='k', linestyle='--')
    plt.plot(xvals_right, target_right_crp, lw=lw, c='k')

    xticks_crp = range(-5, 6, 1)
    plt.xticks(xticks_crp, size='large')
    plt.yticks(np.arange(0.0, 0.7, 0.1), size='large')
    plt.axis([-6, 6, 0, 0.6], size='large')
    plt.xlabel('Item Lag', size='large')
    plt.ylabel('Conditional Response Probability', size='large')
    plt.title('Lag-CRP', size='large')
    plt.legend(loc='upper left')

    # save fig nicely
    if save_figs:
        plt.savefig('./Figs/lag_crp_fig.eps', format='eps', dpi=1000)

    #_______________________ plot pfc
    fig3 = plt.figure()
    plt.plot(xvals, this_pfc, lw=lw, c='k', linestyle='--', label='CMR2')
    plt.plot(xvals, target_pfc, lw=lw, c='k', label='K02 Data')

    plt.title('Probability of First Recall', size='large')
    plt.xlabel('Serial Position', size='large')
    plt.ylabel('Probability of First Recall', size='large')
    plt.xticks(xvals, size='large')
    plt.yticks(np.arange(0.0, 1.2, 0.2), size='large')
    plt.axis([0.5, 10.5, 0, 1], size='large')
    plt.legend(loc='upper left')

    # save fig nicely
    if save_figs:
        plt.savefig('./Figs/pfc_fig.eps', format='eps', dpi=1000)

    plt.show()


if __name__ == "__main__": main()


