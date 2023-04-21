#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:28:01 2017

@author: shai.goldman
"""

import glob, os
from matplotlib import pyplot as plt
import numpy as np
#import scipy.io
import CMR2_pack_cyth_LTP228 as CMR2
import pso_par_cmr2 as pso2_cmr2
import lagCRP2
import plis
import opR_crl

files = glob.glob('kappa=*')

kappas = np.asarray([float(f[f.index('=')+1:f.index('_')]) for f in files])
lambdas = np.loadtxt(files[0])[0]

rmses = np.asarray([np.loadtxt(f)[1] for f in files])

plt.pcolor(kappas, lambdas, rmses, cmap='Reds')
plt.xticks(np.round(kappas, 2))
plt.yticks(np.round(lambdas, 2))
plt.ylabel('Lambda')
plt.xlabel('Kappa')
plt.colorbar()

best = np.where(rmses==rmses.min())
best_k = kappas[best[1]]
best_l = lambdas[best[0]]

params_093_spc_crp_pli_31 = {
        'beta_enc': 2.785291934596597074e-01,
        'beta_rec': 6.383527530661591287e-01,
        'gamma_fc': 3.968139462336542911e-01,
        'gamma_cf': 9.494140002163982128e-01,
        'scale_fc': 1 - 3.968139462336542911e-01,
        'scale_cf': 1 - 9.494140002163982128e-01,
    
        'phi_s': 2.157968652715054780e+00,
        'phi_d': 1.312590804776602615e+00,
        'kappa': 1.954898402828377513e-01,
    
        'eta': 2.115299892696877460e-01,
        's_cf': 1.439828498396066081e+00,
        's_fc': 0.0,
        'beta_rec_post': 5.539411142449401915e-01,
        'omega': 1.118186487766434389e+01,
        'alpha': 8.608259186179094691e-01,
        'c_thresh': 4.086471677982843054e-01,
        'dt': 10.0,
    
        'lamb': 7.677613206786167155e-02,
        'rec_time_limit': 75000,
    
        'dt_tau': 0.01,
        'sq_dt_tau': 0.10,
    
        'nlists_for_accumulator': 4
}

myparams = params_093_spc_crp_pli_31

myparams['kappa'] = best_k
myparams['lamb'] = best_l

params_to_test = myparams

if True:
    # Set LSA and data paths -- K02 data
    cwd = '/home1/shai.goldman/pyCMR2/Test_On_LTP093/'
    LSA_path = os.path.join(cwd,'w2v.txt')
    data_path = cwd + 'pres_nos_LTP093.txt'

    # read in LSA matrix
    LSA_mat = np.loadtxt(LSA_path)

    # run CMR2
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

    data_pres = np.loadtxt('pres_nos_LTP093.txt', delimiter=',')
    data_rec = np.loadtxt('rec_nos_LTP093.txt', delimiter=',')
    data_times = np.loadtxt('times_LTP093.txt', delimiter=',')

    # recode data lists for spc, pfc, and lag-CRP analyses
    recoded_lists = pso2_cmr2.recode_for_spc(data_rec, data_pres)

    # save out the recoded lists in case you want to read this in later
    np.savetxt('recoded_lists_pso_test_K02.txt', recoded_lists,
               delimiter=',', fmt='%.0d')

    # get spc & pfc
    target_spc, target_spc_sem, target_pfc, target_pfc_sem = \
        pso2_cmr2.get_spc_pfc(recoded_lists, ll, False)

    target_crp, target_crp_sem = lagCRP2.get_crp(recoded_lists, ll)

    # get Lag-CRP sections of interest
    center_val = ll - 1

    target_left_crp = target_crp[center_val-5:center_val]
    target_left_crp_sem = target_crp_sem[center_val-5:center_val]

    target_right_crp = target_crp[center_val+1:center_val+6]
    target_right_crp_sem = target_crp_sem[center_val+1:center_val+6]
    
    target_crl, target_crl_sem, szs = opR_crl.opR_crl(data_times, recoded_lists)
    
    target_plis, target_plis_sem = plis.plicount(recoded_lists)

    ####
    #
    #   Get CMR2 output
    #
    ####

    cmr_recoded_output = pso2_cmr2.recode_for_spc(rec_nos, data_pres)

    # get the model's spc and pfc predictions:
    (this_spc, this_spc_sem, this_pfc,
    this_pfc_sem) = pso2_cmr2.get_spc_pfc(cmr_recoded_output, ll, False)

    # get the model's crp predictions:
    this_crp, this_crp_sem = lagCRP2.get_crp(cmr_recoded_output, ll)

    # get left crp values
    this_left_crp = this_crp[center_val-5:center_val]

    # get right crp values
    this_right_crp = this_crp[center_val+1:center_val+6]
    
    this_crl, this_crl_sem, mszs = opR_crl.opR_crl(times, cmr_recoded_output, Szs=szs)
    
    this_plis, this_plis_sem = plis.plicount(cmr_recoded_output)
    
    opR_crl.opR_crl(data_times, recoded_lists, True, True, szs, times, cmr_recoded_output)

    print("Data vals: ")
    print(target_spc)
    print(target_pfc)
    print(target_left_crp)
    print(target_right_crp)

    print("Model vals: ")
    print(this_spc)
    print(this_pfc)
    print(this_left_crp)
    print(this_right_crp)


    # raise ValueError("stop and check")
    ####
    #
    #   Plot graphs
    #
    ####

    # line width
    lw = 2

    #_______________________ plot spc
    fig1 = plt.figure()
    xvals = range(1, 25, 1)     # ticks for x-axis

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

    #_____________________ plot pfc
    fig3 = plt.figure()
    xvals = np.arange(1, 25, 1)
    plt.plot(xvals, this_pfc, lw=lw, c='k', linestyle='--', label='CMR2')
    plt.plot(xvals, target_pfc, lw=lw, c='k', label='Data')

    plt.title('Probability of First Recall', size='large')
    plt.xlabel('Serial Position', size='large')
    plt.ylabel('Probability of First Recall', size='large')
    plt.xticks(xvals, size='large')
    plt.yticks(np.arange(0.0, 1.2, 0.2), size='large')
    plt.axis([0.5, ll+.5, 0, 1], size='large')
    plt.legend(loc='upper left')
    
    #_______________________ plot crl
    figr = plt.figure()
    plt.plot(this_crl, lw=lw, c='k', linestyle='--', label='CMR2')
    plt.plot(target_crl, lw=lw, c='k', label='Data')

    plt.title('CRL', size='large')
    plt.xlabel('Output-position', size='large')
    plt.ylabel('Inter-Response Time', size='large')
    plt.legend(loc='upper left')
    
    plt.show()
    
    fig4 = plt.figure()
    xvals = range(1, 25, 1)
    plt.plot(xvals, this_pfc, lw=lw, c='k', linestyle='--', label='CMR2')
    plt.plot(xvals, target_pfc, lw=lw, c='k', label='Data')

    plt.title('Probability of First Recall', size='large')
    plt.xlabel('Serial Position', size='large')
    plt.ylabel('Probability of First Recall', size='large')
    plt.xticks(xvals, size='large')
    plt.yticks(np.arange(0.0, 1.2, 0.2), size='large')
    plt.axis([0.5, ll+.5, 0, 1], size='large')
    plt.legend(loc='upper left')
    
    #_______________________ plot plis
    fig5 = plt.figure()
    xvals = np.arange(-5, -1, 1)
    plt.plot(xvals, this_plis, lw=lw, c='k', linestyle='--', label='CMR2')
    plt.plot(xvals, target_plis, lw=lw, c='k', label='Data')

    plt.xticks(xvals)

    plt.title('PLIs', size='large')
    plt.xlabel('Lists back', size='large')
    plt.ylabel('Count', size='large')
    plt.legend(loc='upper left')

    # save fig nicely
    if save_figs:
        plt.savefig('./Figs/pli_fig.eps', format='eps', dpi=1000)

    plt.show()



















