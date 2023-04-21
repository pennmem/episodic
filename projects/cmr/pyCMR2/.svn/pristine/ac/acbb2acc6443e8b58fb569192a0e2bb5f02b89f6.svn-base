import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import CMR2_pack_cyth as CMR2
import pso_par_cmr2_eval as pso2_cmr2
import lagCRP2
import warnings
import pandas

# iter = 12, SS = 100
# 2.253780040066615253e-01 - beta enc
# 4.260348999729270947e-01 - beta rec
# 4.494940527039775757e-01 - gamma fc
# 6.694723191673451757e-01 - gamma cf
# 1.085769733622018007e+00 - phi s
# 3.666568790786536858e-01 - phi d
# 2.716580908964218999e-01 - kappa
# 2.427902950429073337e-01 - eta
# 2.169717142825280831e+00 - s_cf
# 2.347880732868359299e-01
# 1.032131665719187730e+01 - omega
# 9.251302062818872463e-01 - alpha
# 3.584008911233489414e-01
# 1.401249207934799623e-01

params_228_spc_crp_pli_12 = {
    'beta_enc': 2.253780040066615253e-01,
    'beta_rec': 4.260348999729270947e-01,
    'gamma_fc': 4.494940527039775757e-01,
    'gamma_cf': 6.694723191673451757e-01,
    'scale_fc': 1 - 4.494940527039775757e-01,
    'scale_cf': 1 - 6.694723191673451757e-01,

    'phi_s': 1.085769733622018007e+00,
    'phi_d': 3.666568790786536858e-01,
    'kappa': 2.716580908964218999e-01,

    'eta': 2.427902950429073337e-01,
    's_cf': 2.169717142825280831e+00,
    's_fc': 0.0,
    'beta_rec_post': 2.347880732868359299e-01,
    'omega': 1.032131665719187730e+01,
    'alpha': 9.251302062818872463e-01,
    'c_thresh': 3.584008911233489414e-01,
    'dt': 10.0,

    'lamb': 1.401249207934799623e-01,
    'rec_time_limit': 75000,

    'dt_tau': 0.01,
    'sq_dt_tau': 0.10,

    'nlists_for_accumulator': 4
}

# for comparison:

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

    'nlists_for_accumulator': 4,

    'beta_source': 0.5,
    'L_CF_tw': 1.0,             # NW quadrant
    'L_CF_ts': 0.0,             # NE quadrant
    'L_CF_sw': 0.516,           # SW quadrant
    'L_CF_ss': 0.0              # SE quadrant
}

eval_params = [2.087409871274440720e-01,
4.417548505574213080e-01,
5.607375129618237253e-01,
5.865810999586569263e-01,
1.025813969562999306e+00,
7.727624397241358301e-01,
7.933540269243204157e-02,
2.677142294440553183e-01,
1.975860952699988182e+00,
5.450174885608769504e-01,
1.288027383856686114e+01,
8.442225013716252446e-01,
4.519158416077235785e-01,
1.417321239136376254e-01]

# rmse = 4.74308314278
eval_params_8_lim = [2.618232805817717335e-01, 5.543089679951223037e-01,
                     4.258711967343156712e-01, 6.180311261973433501e-01,
                     2.110927789977050217e+00, 5.274036131606105737e-01,
                     2.524920717901178446e-01, 3.141004891831752355e-01,
                     1.647448485137032126e+00, 5.064662303976787960e-01,
                     9.029929604194762760e+00, 7.202601276247641016e-01,
                     7.758609098522326608e-01, 2.825198598871148659e-01]

eval_params_12_lim = [2.660034306715760577e-01, 6.029587824377693472e-01,
                      4.478310861763507011e-01, 6.385274771993195708e-01,
                      2.102080690878928682e+00, 5.639051305181405072e-01,
                      2.778836893965428434e-01, 3.332162386070143811e-01,
                      1.233502122042202442e+00, 4.796845842890386513e-01,
                      7.565943847798275890e+00, 7.163197930724808371e-01,
                      6.587796341679699186e-01, 2.524037844284643950e-01]

eval_params_16_lim = [2.457145011427358250e-01, 5.939983043205657731e-01,
                      4.416853418276440180e-01, 6.677600813998119111e-01,
                      1.835312817806019181e+00, 5.194668997963091117e-01,
                      2.736198815421247432e-01, 3.338746162092399716e-01,
                      1.350687105006719291e+00, 4.755741359612904451e-01,
                      8.296476292989565238e+00, 7.260923282618370056e-01,
                      5.900591826029830678e-01, 2.629919802366982862e-01]

eval_params_18_lim = [2.384286533797432273e-01, 6.146085568669237276e-01,
                      4.411262590510121040e-01, 6.791699420562914424e-01,
                      1.739863745656733407e+00, 5.092590571035078284e-01,
                      2.729462878064732312e-01, 3.349668834023873387e-01,
                      1.362869944709371506e+00, 4.806567844654729971e-01,
                      8.134513846337645404e+00, 7.251719760588070107e-01,
                      5.650158026144929124e-01, 2.585854974352264635e-01]

# rmse = 3.58379
eval_params_24_lim = [2.379410019354659400e-01, 6.069446552704620412e-01,
                      4.458909203617581474e-01, 6.782021986377470002e-01,
                      1.764003840755353547e+00, 5.180403495658876256e-01,
                      2.779153409685543585e-01, 3.370976656220047163e-01,
                      1.353511895590898462e+00, 4.851633956457476748e-01,
                      8.120252716294730888e+00, 7.195984765026066654e-01,
                      5.362993963728615032e-01, 2.585447986666354336e-01]

#------------------------ params, partial-spc fit -----------------------

eval_params_partial_31 = [0.2420457, 0.62119201, 0.44498159, 0.67771403,
                          1.78601999, 0.51594315, 0.27726596, 0.338817,
                          1.34697484, 0.48764351, 8.1965843, 0.71865383,
                          0.54565971, 0.25807545]



def get_dict(param_vec):

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
        'rec_time_limit': 30000,

        'dt_tau': 0.01,
        'sq_dt_tau': 0.10,

        'nlists_for_accumulator': 4
    }

    return param_dict

# eval_ltp228_iter6
# eval_ltp228_iter6 = [1.716809329416443819e-01, 3.384802789328152373e-01,
#                      5.278854148179108474e-01, 8.526512588392437531e-01,
#                      6.995507883861772358e-01, 1.500000000000000000e+00,
#                      1.590025974049480428e-01, 2.195736874567962571e-01,
#                      8.131401955773863710e-01, 8.469151984074398953e-01,
#                      1.035579982757124995e+01, 9.167894549370505519e-01,
#                      3.642663994944037587e-01, 3.598538416670629680e-01,
#                      8.630962287790123755e-01]
#
# eval_ltp228_iter12 = [1.318106346980016108e-01, 3.125415647585478851e-01,
#                       5.726118851402727250e-01, 8.112087325165593388e-01,
#                       7.169355061419873110e-01, 6.970455576646857887e-01,
#                       2.317143196747519784e-01, 2.523034490253219242e-01,
#                       7.664344870232021600e-01, 7.812154630108903985e-01,
#                       1.258037990186466537e+01, 9.137970895958434925e-01,
#                       4.673921631725889703e-01, 3.098854419968781748e-01,
#                       9.197003740656600757e-01]
#
# # eval ltp 228 iter 6, limited spc
# eval_ltp228_iter6_lim = [1.792660260505461456e-01, 8.106792311970560938e-01,
#                          6.999999999999999556e-01, 2.708005992436705034e-01,
#                          4.235520677709700421e-01, 1.301499799125044099e+00,
#                          4.290859951720893406e-01, 1.244994828998331471e-01,
#                          1.917075897085287028e+00, 8.312198342534764528e-01,
#                          1.804461497929068514e+01, 5.000000000000000000e-01,
#                          7.023422614570099531e-01, 2.249255539588189134e-01,
#                          1.000000000000000056e-01]
#
# eval_ltp228_iter8_lim = [2.022720031201254154e-01, 7.962355255063156001e-01,
#                          6.702653140162302403e-01, 3.323870869601093792e-01,
#                          3.048285700504894113e-01, 1.315912035440348404e+00,
#                          3.958794930337313223e-01, 1.154837752587090927e-01,
#                          1.963593795061992608e+00, 8.620888536143426206e-01,
#                          1.789197293960100765e+01, 5.170671145504418531e-01,
#                          7.304755959145361466e-01, 2.179096792498745883e-01,
#                          1.000000000000000056e-01]


def get_eval_param_dict(param_vec):
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

        'rec_time_limit': 75000,

        'dt_tau': 0.01,
        'sq_dt_tau': 0.10,

        'nlists_for_accumulator': 2,
    }

    return param_dict


def getEmotCounts(presented_list):
    """Define helper function return the emotional valence counts
    for presented items"""

    # init no.s valenced items in this list
    sum_neg  = 0
    sum_pos  = 0
    sum_neut = 0

    # for each item in that list,
    for j in range(len(presented_list)):

        # if valid item ID,
        if presented_list[j] >= 0:

            # get index of this item in the valence key
            valence_key_index = int(presented_list[j]) - 1

            # get valence for this item
            val_j = word_valence_key.iloc[valence_key_index][1]

            # if item has valence, increment sum of valenced items
            if val_j < 4.0:
                sum_neg += 1
            elif val_j > 6.0:
                sum_pos += 1
            else:
                sum_neut += 1

    return [sum_neg, sum_pos, sum_neut]


def codeList(orig_list):
    """Define helper function to recode a given list into emotional pool IDs"""

    recoded_list = []
    for j in range(len(orig_list)):
        # if valid item ID,
        if orig_list[j] >= 0:

            # get index of this item in the valence key
            valence_key_index = int(orig_list[j]) - 1

            # get valence for this item
            val_j = word_valence_key.iloc[valence_key_index][1]

            # if item has valence, increment sum of valenced items
            if val_j < 4.0:
                recoded_list.append('Ng')
            elif val_j > 6.0:
                recoded_list.append('P')
            else:
                recoded_list.append('N')
        else:
            recoded_list.append('-1')

    return np.asarray(recoded_list)


def recode_rep_intrs(rec_list_0, pres_list_0):
    """Define helper function to recode repeats & intrusions as -1"""
    cleaned_list = np.zeros(len(rec_list_0))

    # for each item in list
    for i in range(len(rec_list_0)):

        item = rec_list_0[i]
        cleaned_list[i] = item

        # recode intrusions as -3
        if item not in pres_list_0:
            cleaned_list[i] = -3

        # recode repeats as -2
        if i > 0:
            if (item in rec_list_0[0:i - 1]):
                cleaned_list[i] = -2

    return cleaned_list


def emot_val(pres_lists, rec_lists):
    """Main method to calculate emotional valences, etc."""

    # Initialize lists to hold transition probability scores
    # for this subject, across lists
    list_probs = []

    # if no. presented lists != no. recalled lists, skip this session
    # (assumed some kind of recording error)
    if pres_lists.shape[0] != rec_lists.shape[0]:
        all_list_means = []
    else:
        for row in range(pres_lists.shape[0]):  # for each row:

            # get item nos. in row w/o 0's
            rec_row = rec_lists[row][rec_lists[row] != 0]
            pres_row = pres_lists[row]

            # if no responses, skip this list
            if len(rec_row) == 0:
                continue
            else:
                # get number of emotion words in presented list
                # Ng, P, N <-- order of counts in list
                emot_counts = getEmotCounts(pres_row)

                # if pres list doesn't have >= 1 of at least one emot. word,
                # then skip it
                if (emot_counts[0] < 1) \
                        or (emot_counts[1] < 1) or (emot_counts[2] < 1):
                    continue
                else:
                    # recode any repeats or intrusions as -1
                    # squeeze out all -1 values
                    cleaned_list = recode_rep_intrs(rec_row, pres_row)

                    # recode rec_row w/ emotional valence pool IDs
                    rec_list_emot = codeList(cleaned_list)

                    # remove all -1 values
                    rec_list_analyze = rec_list_emot[rec_list_emot != '-1']

                    # if N valid words recalled is < 2, skip list
                    if len(rec_list_analyze) < 2:
                        continue
                    else:
                        count_ng = 0
                        count_p  = 0
                        count_n  = 0

                        # iterate through the list and sum
                        # no. of val-val transitions that occur
                        for i in range(len(rec_list_analyze)):

                            # keep running count of # items of
                            # particular valence
                            this_item = rec_list_analyze[i]

                            # keep running count of # items of
                            # particular valence remaining
                            list_tot_ng = emot_counts[0]
                            list_tot_p  = emot_counts[1]
                            list_tot_n  = emot_counts[2]

                            ng_remaining = list_tot_ng - count_ng
                            p_remaining  = list_tot_p - count_p
                            n_remaining  = list_tot_n - count_n

                            # initialize transition scores
                            score_ng = 0
                            score_p  = 0
                            score_n  = 0

                            if i == 0:
                                if this_item == 'Ng':
                                    count_ng += 1
                                elif this_item == 'P':
                                    count_p += 1
                                elif this_item == 'N':
                                    count_n += 1
                            if i > 0:
                                # get previous item (base of the transition)
                                prev_item = rec_list_analyze[i - 1]

                                # get transition scores
                                if this_item == 'Ng':
                                    score_ng += 1
                                elif this_item == 'P':
                                    score_p  += 1
                                elif this_item == 'N':
                                    score_n  += 1

                                # get observed probabilities for this step
                                if ng_remaining != 0:
                                    obs_prob_ng  = score_ng / ng_remaining
                                else:
                                    obs_prob_ng  = np.nan

                                if p_remaining  != 0:
                                    obs_prob_p   = score_p / p_remaining
                                else:
                                    obs_prob_p   = np.nan

                                if n_remaining  != 0:
                                    obs_prob_n   = score_n / n_remaining
                                else:
                                    obs_prob_n   = np.nan

                                # append to appropriate base-pair list
                                if prev_item   == 'Ng':
                                    list_probs.append(
                                        [obs_prob_ng, obs_prob_p, obs_prob_n,
                                         np.nan, np.nan, np.nan,
                                         np.nan, np.nan, np.nan])
                                elif prev_item == 'P':
                                    list_probs.append(
                                        [np.nan, np.nan, np.nan,
                                         obs_prob_ng, obs_prob_p, obs_prob_n,
                                         np.nan, np.nan, np.nan])
                                elif prev_item == 'N':
                                    list_probs.append(
                                        [np.nan, np.nan, np.nan,
                                         np.nan, np.nan, np.nan,
                                         obs_prob_ng, obs_prob_p, obs_prob_n])

                                # update running item counts
                                if this_item == 'Ng':
                                    count_ng += 1
                                elif this_item == 'P':
                                    count_p += 1
                                elif this_item == 'N':
                                    count_n += 1

        # Ignore np.nanmean warning for averaging across NaN-only slices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # average down columns to get ave transition probs across lists
            all_list_means = np.nanmean(list_probs, axis=0)

            # get SEM down columns
            all_list_std = np.nanstd(list_probs, axis=0)
            all_list_sem = all_list_std/(np.count_nonzero(
                ~np.isnan(list_probs), axis=0)**0.5)

    return all_list_means, all_list_sem

def main():

    ###################
    #
    #   Get data & CMR2-predicted data
    #
    ###################

    # select params to test from among the options defined above
    params_to_test = get_eval_param_dict(eval_params_partial_31) # get_dict(eval_params)
    # params_to_test = get_eval_param_dict(np.loadtxt('best_params_28'))

    # Set LSA and data paths -- K02 data
    LSA_path = 'w2v.txt'
    data_path = 'pres_nos_LTP138.txt'

    # read in LSA matrix
    LSA_mat = np.loadtxt(LSA_path)

    # run CMR2
    rec_nos, times = CMR2.run_CMR2(LSA_path=LSA_path, LSA_mat=LSA_mat,
                                   data_path=data_path,
                                   params=params_to_test, sep_files=False)

    # save the output somewhere convenient
    np.savetxt(
        'resp_pso_test_LTP138.txt', np.asmatrix(rec_nos),
        delimiter=',', fmt='%.0d')
    np.savetxt(
        'times_pso_test_LTP138.txt', np.asmatrix(times),
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
    save_figs = True

    global word_valence_key
    valence_key_path = '/home1/rivkat.cohen/wordproperties_CSV.csv'
    word_valence_key = pandas.read_csv(valence_key_path)

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

    data_pres = np.loadtxt('pres_nos_LTP138.txt', delimiter=',')
    data_rec = np.loadtxt('rec_nos_LTP138.txt', delimiter=',')

    # recode data lists for spc, pfc, and lag-CRP analyses
    recoded_lists = pso2_cmr2.recode_for_spc(data_rec, data_pres)

    # save out the recoded lists in case you want to read this in later
    np.savetxt('recoded_lists_LTP138.txt', recoded_lists,
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

    target_eval_mean, target_eval_sem = emot_val(data_pres, data_rec)

    ####
    #
    #   Get CMR2 output
    #
    ####

    cmr_recoded_output = pso2_cmr2.recode_for_spc(rec_nos, data_pres)

    # get the model's emot clustering predictions
    cmr_eval_mean, cmr_eval_sem = emot_val(data_pres, rec_nos)

    # get the model's spc and pfc predictions:
    (this_spc, this_spc_sem, this_pfc,
    this_pfc_sem) = pso2_cmr2.get_spc_pfc(cmr_recoded_output, ll)

    # get the model's crp predictions:
    this_crp, this_crp_sem = lagCRP2.get_crp(cmr_recoded_output, ll)

    # get left crp values
    this_left_crp = this_crp[center_val-5:center_val]

    # get right crp values
    this_right_crp = this_crp[center_val+1:center_val+6]

    # get model's eval predictions

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

    print("Data Eval:")
    print(target_eval_mean)
    print("Model Eval:")
    print(cmr_eval_mean)

    # raise ValueError("stop and check")
    ####
    #
    #   Plot graphs
    #
    ####

    # line width
    lw = 2

    # gray color for CMR predictions
    gray = '0.50'

    #_______________________ plot spc
    fig1 = plt.figure()
    xvals = range(1, ll+1, 1)     # ticks for x-axis

    plt.plot(xvals, this_spc, lw=lw, c=gray, linestyle='--', label='CMR2')
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
    plt.plot(xvals_left, this_left_crp, lw=lw, c=gray,
             linestyle='--', label='CMR2')
    plt.plot(xvals_left, target_left_crp, lw=lw, c='k', label='Data')

    # right
    plt.plot(xvals_right, this_right_crp, lw=lw, c=gray, linestyle='--')
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
    plt.plot(xvals, this_pfc, lw=lw, c=gray, linestyle='--', label='CMR2')
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

    fig4 = plt.figure()

    # plot base on x-axis (Ng-, P-) and output on y-axis (Ng-, P-)

    # Ng-Ng, Ng-P
    plt.plot([0, 1], [cmr_eval_mean[0], cmr_eval_mean[1]], lw=lw,
             c='0.60', linestyle='solid', label='Neg, CMR')
    plt.plot([0, 1], [target_eval_mean[0], target_eval_mean[1]],
             c='k', linestyle='solid', label='Neg, Data')
    # P-Ng, P-P
    plt.plot([0, 1], [cmr_eval_mean[3], cmr_eval_mean[4]], lw=lw,
             c='0.60', linestyle='dashed', label='Pos, CMR')
    plt.plot([0, 1], [target_eval_mean[3], target_eval_mean[4]], lw=lw,
             c='k', linestyle='dashed', label='Pos, Data')

    plt.title('Emotional Clustering Effect', size='large')
    plt.xlabel('Output Type', size='large')
    plt.ylabel('Probability Score', size='large')
    plt.xticks((0,1),('Negative', 'Positive'))
    plt.axis([-.25, 1.25, 0, .14], size='large')
    plt.legend(loc='upper right')

    # save fig nicely
    if save_figs:
        plt.savefig('./Figs/eval_fig.eps', format='eps', dpi=1000)

    plt.show()


if __name__ == "__main__": main()


