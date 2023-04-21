#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:22:14 2017

@author: shai.goldman
"""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import CMR2_pack_cyth_LTP228 as CMR2
import pso_par_cmr2 as pso2_cmr2
import lagCRP2
import plis
import opR_crl

# .3881706364026762945 - beta_enc
# .1248745531446266155 - beta_rec
# .4652669686436619045 - gamma_fc
# .5412247636058757916 - gamma_cf
# 2.219225796668485629 - phi_s
# .2978733416740619866 - phi_d
# .01127888408717198782 - kappa
# .3347122214798961548 - eta
# 2.605913213170756304 - s_cf
# 1.000000000000000000 - beta_rec_post
# 13.35721818704929120 - omega
# .9057420023551858712 - alpha
# .6580403715962824807 - c_thresh
# .03190466611937717301 - lambda

# SS = 100, n iterations = 24
# used regular rmse values
# rec_time_limit was accidentally left at 30000, so it is set to 30000 here
params_093_iter = {
    'beta_enc': .3881706364026762945,
    'beta_rec': .1248745531446266155,
    'gamma_fc': .4652669686436619045,
    'gamma_cf': .5412247636058757916,
    'scale_fc': 1 - .4652669686436619045,
    'scale_cf': 1 - .5412247636058757916,

    'phi_s': 2.219225796668485629,
    'phi_d': .2978733416740619866,
    'kappa': .01127888408717198782,

    'eta': .3347122214798961548,
    's_cf': 2.605913213170756304,
    's_fc': 0.0,
    'beta_rec_post': 1.00000,
    'omega': 13.35721818704929120,
    'alpha': .9057420023551858712,
    'c_thresh': .6580403715962824807,
    'dt': 10.0,

    'lamb': .03190466611937717301,
    'rec_time_limit': 30000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
}

# .4478377543321004906  - beta_enc
# .6122825554798102532  - beta_rec
# .4638347104174828650  - gamma_fc
# .4872202388997172728  - gamma_cf
# .9538032869280285153  - phi_s
# .2014997523916394473  - phi_d
# .06754820837488234586 - kappa
# .3173510594921117312  - eta
# 1.563671681986635642  - s_cf
# .5043051765556250121 - beta_rec_post
# 7.941082735536347137 - omega
# .8257459296378557578 -alpha
# .02218333005680216824 - c_thresh
# .03647599127074976910 - lambda

# used chi^2 fit metric
params_093_chi = {
    'beta_enc': .4478377543321004906,
    'beta_rec': .6122825554798102532,
    'gamma_fc': .4638347104174828650,
    'gamma_cf': .4872202388997172728,
    'scale_fc': 1 - .4638347104174828650,
    'scale_cf': 1 - .4872202388997172728,

    'phi_s': .9538032869280285153,
    'phi_d': .2014997523916394473,
    'kappa': .06754820837488234586,

    'eta': .3173510594921117312,
    's_cf': 1.563671681986635642,
    's_fc': 0.0,
    'beta_rec_post': .5043051765556250121,
    'omega': 7.941082735536347137,
    'alpha': .8257459296378557578,
    'c_thresh': .02218333005680216824,
    'dt': 10.0,

    'lamb': .03647599127074976910,
    'rec_time_limit': 75000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
}


# .4410051015463134494
# 1.000000000000000000
# .3983093118037470126
# .1986423720067171383
# .6848445866479200284
# 1.233469643303236829
# .1830820682032394675
# .4465486776144486636
# 2.096440686732088388
# .1000000000000000056
# 12.38323360198773493
# .7235405154277710915
# .6114585143630549835
# .3661075805212513079

# amplified the lag crp values; multipled each vector's nansum by 24/5
params_093_crp = {
    'beta_enc': .4410051015463134494,
    'beta_rec': 1.000000,
    'gamma_fc': .3983093118037470126,
    'gamma_cf': .1986423720067171383,
    'scale_fc': 1 - .3983093118037470126,
    'scale_cf': 1 - .1986423720067171383,

    'phi_s': .6848445866479200284,
    'phi_d': 1.233469643303236829,
    'kappa': .1830820682032394675,

    'eta': .4465486776144486636,
    's_cf': 2.096440686732088388,
    's_fc': 0.0,
    'beta_rec_post': .100000,
    'omega': 12.38323360198773493,
    'alpha': .7235405154277710915,
    'c_thresh': .6114585143630549835,
    'dt': 10.0,

    'lamb': .3661075805212513079,
    'rec_time_limit': 75000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
}


# only doing last 18-23 indices (start = 0) of the pfr; full spc, full lag-CRP
# .2653422373060996264 -- beta_enc
# .1533939172573373366 -- beta_rec
# .3553530862031602511 -- gamma_fc
# .7169531393575768741 -- gamma_cf
# .9859760789441397444 -- phi_s
# 1.226874054708524620 -- phi_d
# .2973361499620149617 -- kappa
# .4926053453508170699 -- eta
# 1.289456490473720907 -- s_cf
# .9176366756724834151 -- beta_rec_post
# 14.99495515578635008 -- omega
# .5455429499980215535 -- alpha
# .06862259447453303296 -- c_thresh
# .1546044175499650952 -- lambda

params_093_part_pfr = {
    'beta_enc': .2653422373060996264,
    'beta_rec': .1533939172573373366,
    'gamma_fc': .3553530862031602511,
    'gamma_cf': .7169531393575768741,
    'scale_fc': 1 - .3553530862031602511,
    'scale_cf': 1 - .7169531393575768741,

    'phi_s': .9859760789441397444,
    'phi_d': 1.226874054708524620,
    'kappa': .2973361499620149617,

    'eta': .4926053453508170699,
    's_cf': 1.289456490473720907,
    's_fc': 0.0,
    'beta_rec_post': .9176366756724834151,
    'omega': 14.99495515578635008,
    'alpha': .5455429499980215535,
    'c_thresh': .06862259447453303296,
    'dt': 10.0,

    'lamb': .1546044175499650952,
    'rec_time_limit': 75000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
}

# line 77 after 10 iterations
# Fitting only the lag-CRP

# .4837304721587586398  - beta_enc
# .6298044114567187268  - beta_rec
# .3324101683260898832  - gamma_fc
# .8044093324912163778  - gamma_cf
# .8975550418915092532  - phi_s
# .7298599491715541676  - phi_d
# .1743914347892096539  - kappa
# .3306021264787895042  - eta
# 1.366625787944510728  - s_cf
# .6536834559913272669  - beta_rec_post
# 13.69051463068548991  - omega
# .8178043220754483977  - alpha
# .4747254871773082807  - c_thresh
# .1714679850952394724  - lambda

params_093_crp = {
    'beta_enc': .4837304721587586398,
    'beta_rec': .6298044114567187268,
    'gamma_fc': .3324101683260898832,
    'gamma_cf': .8044093324912163778,
    'scale_fc': 1 - .3324101683260898832,
    'scale_cf': 1 - .8044093324912163778,

    'phi_s': .8975550418915092532,
    'phi_d': .7298599491715541676,
    'kappa': .1743914347892096539,

    'eta': .3306021264787895042,
    's_cf': 1.366625787944510728,
    's_fc': 0.0,
    'beta_rec_post': .6536834559913272669,
    'omega': 13.69051463068548991,
    'alpha': .8178043220754483977,
    'c_thresh': .4747254871773082807,
    'dt': 10.0,

    'lamb': .1714679850952394724,
    'rec_time_limit': 75000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
}

# line 52, after 27 iterations
# fitting only the lag-CRP

# .2391946680112451151 - beta_enc
# .7528692098548239731 - beta_rec
# .4110180704921635453 - gamma_fc
# .7651198115058943650 - gamma_cf
# 1.895431860967812465 - phi_s
# 1.216163939234208735 - phi_d
# .03211576202625544135 - kappa
# .2397953130371312247 - eta
# 1.912681329041025391 - s_cf
# .7857135665114277634 - beta_rec_post
# 13.18300848439370121 - omega
# .7617797134494085354 - alpha
# .6002237048381811046 - c_thresh
# .1983749600870549634 - lambda

# rmse was 3.0-something
params_093_crp_27 = {
    'beta_enc': .2391946680112451151,
    'beta_rec': .7528692098548239731,
    'gamma_fc': .4110180704921635453,
    'gamma_cf': .7651198115058943650,
    'scale_fc': 1 - .4110180704921635453,
    'scale_cf': 1 - .7651198115058943650,

    'phi_s': 1.895431860967812465,
    'phi_d': 1.216163939234208735,
    'kappa': .03211576202625544135,

    'eta': .2397953130371312247,
    's_cf': 1.912681329041025391,
    's_fc': 0.0,
    'beta_rec_post': .7857135665114277634,
    'omega': 13.18300848439370121,
    'alpha': .7617797134494085354,
    'c_thresh': .6002237048381811046,
    'dt': 10.0,

    'lamb': .1983749600870549634,
    'rec_time_limit': 75000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
}

# fitting for just spc and lag-crp
# iteration 14

# .8469057443075604930 - beta_enc
# .7238803172467324076 - beta_rec
# .3004624859479314014 - gamma_fc
# 1.000000000000000000 - gamma_cf
# 1.070224483568049445 - phi_s
# .5931258050218595201 - phi_d
# .04126642819020793357 - kappa
# .4903567133705323822 - eta
# 2.133548759228674374 - s_cf
# .4264101536927418223 - beta_rec_post
# 14.49883250886011687 - omega
# .7077580484148883189 - alpha
# .2291586834307451914 - c_thresh
# .08532452226308903653 - lamb

params_093_spc_crp_14 = {
    'beta_enc': .8469057443075604930,
    'beta_rec': .7238803172467324076,
    'gamma_fc': .3004624859479314014,
    'gamma_cf': 1.000000,
    'scale_fc': 1 - .3004624859479314014,
    'scale_cf': 1 - 1.000000,

    'phi_s': 1.070224483568049445,
    'phi_d': .5931258050218595201,
    'kappa': .04126642819020793357,

    'eta': .4903567133705323822,
    's_cf': 2.133548759228674374,
    's_fc': 0.0,
    'beta_rec_post': .4264101536927418223,
    'omega': 14.49883250886011687,
    'alpha': .7077580484148883189,
    'c_thresh': .2291586834307451914,
    'dt': 10.0,

    'lamb': .08532452226308903653,
    'rec_time_limit': 75000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
}

# .8456605188324133326 - beta_enc
# .7466192842401355723 - beta_rec
# .3004295123602649942 - gamma_fc
# 1.000000000000000000 - gamma_cf
# 1.085136410545422203 - phi_s
# .6046768608433781278 - phi_d
# .04089212701000504574 - kappa
# .4896128554675797662 - eta
# 2.123625257897016283 - s_cf
# .4277019952932875624 - beta_rec_post
# 14.34023549409545950 - omega
# .7090430591615091149 - alpha
# .2282694913886026133 - c_thresh
# .08684434330102780430 - lamb

# 6.96
params_093_spc_crp_22 = {
    'beta_enc': .8456605188324133326,
    'beta_rec': .7466192842401355723,
    'gamma_fc': .3004295123602649942,
    'gamma_cf': 1.000000,
    'scale_fc': 1 - .3004295123602649942,
    'scale_cf': 1 - 1.000000,

    'phi_s': 1.085136410545422203,
    'phi_d': .6046768608433781278,
    'kappa': .04089212701000504574,

    'eta': .4896128554675797662,
    's_cf': 2.123625257897016283,
    's_fc': 0.0,
    'beta_rec_post': .4277019952932875624,
    'omega': 14.34023549409545950,
    'alpha': .7090430591615091149,
    'c_thresh': .2282694913886026133,
    'dt': 10.0,

    'lamb': .08684434330102780430,
    'rec_time_limit': 75000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
}

# spc, lag-crp, and pli & eli
# 2.785291934596597074e-01 - beta_enc
# 6.383527530661591287e-01 - beta_rec
# 3.968139462336542911e-01 - gamma_fc
# 9.494140002163982128e-01 - gamma_cf
# 2.157968652715054780e+00 - phi_s
# 1.312590804776602615e+00 - phi_d
# 1.954898402828377513e-01 - kappa
# 2.115299892696877460e-01 - eta
# 1.439828498396066081e+00 - s_cf
# 5.539411142449401915e-01 - beta_rec_post
# 1.118186487766434389e+01 - omega
# 8.608259186179094691e-01 - alpha
# 4.086471677982843054e-01 - c_thresh
# 7.677613206786167155e-02 - lambda

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

    
def get_my_curr_params():
    f = open('/home1/shai.goldman/pyCMR2/Test_On_LTP093/Outs-001/xopt_LTP325.txt')
    txt = f.read()
    f.close()
    txt = np.asarray(txt.split('\n'))[:-1].astype(float)
    dta = .01
    return {
        'beta_enc': txt[0],
        'beta_rec': txt[1],
        'gamma_fc': txt[2],
        'gamma_cf': txt[3],
        'scale_fc': 1 - txt[2],
        'scale_cf': 1 - txt[3],
    
        'phi_s': txt[4],
        'phi_d': txt[5],
        'kappa': txt[6],
    
        'eta': txt[7],
        's_cf': txt[8],
        's_fc': 0.0,
        'beta_rec_post': txt[9],
        'omega': txt[10],
        'alpha': txt[11],
        'c_thresh': txt[12],
        'dt': 10.0,
    
        'lamb': txt[13],
        'rec_time_limit': 75000,
    
        'dt_tau': dta,
        'sq_dt_tau': np.sqrt(dta),
    
        'nlists_for_accumulator': 4
        }



if True:
    import os
    ###################
    #
    #   Get data & CMR2-predicted data
    #
    ###################

    # select params to test from among the options defined above
    params_to_test = get_my_curr_params()
    #params_to_test = params_093_spc_crp_pli_31

    # Set LSA and data paths -- K02 data
    cwd = '/home1/shai.goldman/pyCMR2/Test_On_LTP093/'
    LSA_path = os.path.join(cwd,'w2v.txt')
    fnum = np.loadtxt('sbj.txt').tolist()
    fnum = 325
    data_path = cwd + 'pres_nos_LTP%d.txt' % fnum

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

    data_pres = np.loadtxt(data_path, delimiter=',')
    data_rec = np.loadtxt('rec_nos_LTP%d.txt' % fnum, delimiter=',')
    data_times = np.loadtxt('times_LTP%d.txt' % fnum, delimiter=',')

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
    
    target_irt_dist, target_irt_dist_sem = opR_crl.irt_dist(data_times, recoded_lists)

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
    
    this_irt_dist, this_irt_dist_sem = opR_crl.irt_dist(times, cmr_recoded_output)
    
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
    plt.xticks(xvals[::3], size='large')
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
    
    #_______________________ plot plis
    fig5 = plt.figure()
    xvals = np.arange(1, 7, 1)
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
    
    
    fig5 = plt.figure()
    xvals = np.arange(1, 7, 1)
    plt.plot(xvals, this_irt_dist, lw=lw, c='k', linestyle='--', label='CMR2')
    plt.plot(xvals, target_irt_dist, lw=lw, c='k', label='Data')

    plt.xticks(xvals)

    plt.ylabel('Freq', size='large')
    plt.xlabel('IRT', size='large')
    plt.legend(loc='upper right')
    # save fig nicely
    if save_figs:
        plt.savefig('./Figs/irt_dist_fig.eps', format='eps', dpi=1000)

    plt.show()

