import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import CMR2_pack_cyth_LTP228 as CMR2
import pso_par_cmr2 as pso2_cmr2
import lagCRP2
import os


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def main(subj):
    cwd = os.getcwd()
    print 'Subject is {}'.format(subj)
    ###################
    #
    #   Get data & CMR2-predicted data
    #
    ###################

    # select params to test from among the options defined above
    
    params = np.loadtxt(os.path.join(cwd, 'xopt_{}.txt'.format(subj)))
    auto_params = {

        'beta_enc': params[0],
        'beta_rec': params[1],
        'gamma_fc': params[2],
        'gamma_cf': params[3],
        'scale_fc': 1 - params[2],
        'scale_cf': 1 - params[3],

        'phi_s': params[4],
        'phi_d': params[5],
        'kappa': params[6],

        'eta': params[7],
        's_cf': params[8],
        's_fc': 0.0,
        'beta_rec_post': params[9],
        'omega': params[10],
        'alpha': params[11],
        'c_thresh': params[12],
        'dt': 10.0,

        'lamb': params[13],
        'rec_time_limit': 75000,

        'dt_tau': 0.01,
        'sq_dt_tau': 0.10,

        'nlists_for_accumulator': 4
        }
    params_to_test = auto_params
    #params_to_test = params_093_spc_crp_pli_31
    # Set LSA and data paths -- K02 data
    
    LSA_path = os.path.join(cwd, 'w2v.txt')
    
    data_path = 'pres_nos_{}.txt'.format(subj)
    rec_path = 'rec_nos_{}.txt'.format(subj)
    rec_times_path = 'rec_times_{}.txt'.format(subj)
    data_pres = np.loadtxt(data_path, delimiter=',')
    data_rec = np.loadtxt(rec_path, delimiter=',')
    data_rec_times = np.loadtxt(rec_times_path, delimiter=',')

    # read in LSA matrix
    LSA_mat = np.loadtxt(LSA_path)

    # run CMR2
    print 'running'
    rec_nos, times = CMR2.run_CMR2(LSA_path=LSA_path, LSA_mat=LSA_mat,
                              data_path=data_path,
                              params=params_to_test, sep_files=False)

    # save the output somewhere convenient
    np.savetxt(
        '.resp_pso_test_LTP093.txt', np.asmatrix(rec_nos),
        delimiter=',', fmt='%.0d')
    np.savetxt(
        '.times_pso_test_LTP093.txt', np.asmatrix(times),
        delimiter=',', fmt='%.0d')



    # raise ValueError("stop and check")
    ####################
    #
    #   Graph results from output (lag-CRP, SPC, PFC)
    #
    ####################

    # set list length here
    ll = 24

    # decide whether to save figs out or not
    save_figs = False

    ####
    #
    #   Get data output
    #
    ####

    # get data file, presented items, & recalled items
    #data_file = scipy.io.loadmat(
    #    data_path, squeeze_me=True, struct_as_record=False)

    #data_pres = data_file['data'].pres_itemnos      # presented
    #data_rec = data_file['data'].rec_itemnos        # recalled
 
    
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

    target_left_crp = target_crp[center_val-5:center_val]
    target_left_crp_sem = target_crp_sem[center_val-5:center_val]

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
    this_left_crp = this_crp[center_val-5:center_val]

    # get right crp values
    this_right_crp = this_crp[center_val+1:center_val+6]
    
    #Intrusion stuff
    #TODO Again using ll here is an error to happen the coincidentally works for ltpFR
    this_ppli, this_sem_ppli, this_tic, this_sem_tic = pso2_cmr2.handle_intrusions(rec_nos, data_pres,  \
                                           ll, times)
    target_ppli, target_sem_ppli, target_tic, target_sem_tic =  \
    pso2_cmr2.handle_intrusions(data_rec, data_pres, ll,  data_rec_times)
    
    #Do error stuff:
    print('RMSE for lag_CRP is {}'.format(rmse(target_crp, this_crp)))
    print('RMSE for SPC is {}'.format(rmse(target_spc, this_spc)))
    print('RMSE for PFR is {}'.format(rmse(target_pfc, this_pfc)))
    print('RMSE for PPLI is {}'.format(rmse(target_ppli, this_ppli)))
    print('RMSE for TIC is {}'.format(rmse(target_tic, this_tic)))
    
    """
    print("Data vals: ")
    print("target spc".format(target_spc))
    print("target pfc".format(target_pfc))
    print("Target left crp: {}".format(target_left_crp))
    print("Target right crp: {}".format(target_right_crp))
    print("Target ppli: {}".format(target_ppli))
    print("Target tic: {}".format(target_tic))
    print("Model vals: ")
    print("this spc".format(this_spc))
    print("this pfc".format(this_pfc))
    print("this left crp: {}".format(this_left_crp))
    print("this right crp: {}".format(this_right_crp))
    print("this ppli: {}".format(this_ppli))
    print("this tic: {}".format(this_tic))
    """
    # raise ValueError("stop and check")
    ####
    #
    #   Plot graphs
    #
    ####

    # line width
    lw = 2
    #plot intrusion stuff
    #TODO if you are using a different number of divisions, put that number of divisions + 1 in here instead of 7
    fig_ppli = plt.figure()
    xvals = range(1,7, 1)
    plt.plot(xvals, this_ppli, lw=lw, c='k', linestyle='--', label='CMR2')
    plt.plot(xvals, target_ppli, lw=lw, c='k', label='Data')
    plt.ylabel("Probability of intrusion given an intrusion occurs")
    plt.xlabel('Number of lists back', size='large')
    plt.xticks(xvals, size='large')
    plt.yticks(np.arange(0.0, 1.2, 0.2), size='large')
    plt.title('Probability of prior list intrusion (contingent on intrusion)', size='large')
    plt.legend(loc='upper left')
    
    
    fig_tic = plt.figure()
    xvals = range(1,7, 1)
    plt.plot(xvals, this_tic, lw=lw, c='k', linestyle='--', label='CMR2')
    plt.plot(xvals, target_tic, lw=lw, c='k', label='Data')
    plt.ylabel("Probability of intrusion")
    plt.xlabel('Segment of recall period', size='large')
    plt.xticks(xvals, size='large')
    plt.yticks(np.arange(0.0, 1.2, 0.2), size='large')
    plt.title('Temporal intrusion curve', size='large')
    plt.legend(loc='upper left')
    
    
    
    #_______________________ plot spc
    fig1 = plt.figure()
    xvals = range(1, ll+1, 1)     # ticks for x-axis

    plt.plot(xvals, this_spc, lw=lw, c='k', linestyle='--', label='CMR2')
    plt.plot(xvals, target_spc, lw=lw, c='k', label='Data')

    plt.ylabel('Probability of Recall', size='large')
    plt.xlabel('Serial Position', size='large')
    plt.xticks(xvals, size='large')
    plt.yticks(np.arange(0.0, 1.2, 0.2), size='large')
    plt.axis([0.5, ll+.5, 0, 1], size='large')
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
    plt.plot(xvals_left, target_left_crp, lw=lw, c='k', label='Data')

    # right
    plt.plot(xvals_right, this_right_crp, lw=lw, c='k', linestyle='--')
    plt.plot(xvals_right, target_right_crp, lw=lw, c='k')

    xticks_crp = range(-5, 6, 1)
    plt.xticks(xticks_crp, size='large')
    plt.yticks(np.arange(0.0, 1.0, 0.1), size='large')
    plt.axis([-6, 6, 0, 1.0], size='large')
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
    plt.plot(xvals, target_pfc, lw=lw, c='k', label='Data')

    plt.title('Probability of First Recall', size='large')
    plt.xlabel('Serial Position', size='large')
    plt.ylabel('Probability of First Recall', size='large')
    plt.xticks(xvals, size='large')
    plt.yticks(np.arange(0.0, 1.2, 0.2), size='large')
    plt.axis([0.5, ll+.5, 0, 1], size='large')
    plt.legend(loc='upper left')

    # save fig nicely
    if save_figs:
        plt.savefig('./Figs/pfc_fig.eps', format='eps', dpi=1000)

    plt.show()


if __name__ == "__main__": main()




