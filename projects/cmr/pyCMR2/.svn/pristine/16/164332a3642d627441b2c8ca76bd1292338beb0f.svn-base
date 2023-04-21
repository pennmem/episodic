#!/home1/rivkat.cohen/anaconda3/bin/python
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from pybeh.mask_maker import make_clean_recalls_mask2d


def lag1_all(lag_1=None):
    return True
def lag1_range_1(lag_1=None):
    return lag_1 == 1
def lag1_range_2(lag_1=None):
    return lag_1 == -1
def lag1_range_3(lag_1=None):
    return lag_1 > 3 or lag_1 < -3

from pybeh.crp import crp
from pybeh.sem_crp import sem_crp

def crp2(recalls=None, subjects=None, listLength=None, lag_num=None, lag_range=None, skip_first_n=0):
    # Convert recalls and subjects to numpy arrays
    recalls = np.array(recalls)
    subjects = np.array(subjects)
    # Get a list of unique subjects -- we will calculate a CRP for each
    usub = np.unique(subjects)
    # Number of possible lags = (listLength - 1) * 2 + 1; e.g. a length-24 list can have lags -23 through +23
    num_lags = 2 * listLength - 1
    # Initialize array to store the CRP for each subject (or other unique identifier)
    result = np.zeros((usub.size, num_lags))
    # Initialize arrays to store transition counts
    actual = np.empty(num_lags)
    poss = np.empty(num_lags)
    

    # For each subject/unique identifier
    for i, subj in enumerate(usub):
        # Reset counts for each participant
        actual.fill(0)
        poss.fill(0)
        # Create trials x items matrix where item j, k indicates whether the kth recall on trial j was a correct recall
        clean_recalls_mask = np.array(make_clean_recalls_mask2d(recalls[subjects == subj]))
        # For each trial that matches that identifier
        for j, trial_recs in enumerate(recalls[subjects == subj]):
            seen = set()
            for k, rec in enumerate(trial_recs[:-1]):
                seen.add(rec)
                lag1_valid = False
                if(k!=0):
                    prev_rec = trial_recs[k - 1]
                    lag1_valid = lag_range(rec - prev_rec)
                # Only increment transition counts if the current and next recall are BOTH correct recalls
                if clean_recalls_mask[j, k] and clean_recalls_mask[j, k + 1] and k >= skip_first_n and lag1_valid:
                    next_rec = trial_recs[k + 1]
                    pt = np.array([trans for trans in range(1 - rec, listLength + 1 - rec) if rec + trans not in seen], dtype=int)
                    poss[pt + listLength - 1] += 1
                    trans = next_rec - rec
                    # Record the actual transition that was made
                    actual[trans + listLength - 1] += 1

        result[i, :] = actual / poss
        result[i, poss == 0] = np.nan

    result[:, listLength - 1] = np.nan

    return result[:, listLength - lag_num - 1:listLength + lag_num]


def get_crp(recall_matrix, ll, func_num = 0):
    """Get crp across a matrix of lists"""

    desired_length = 2
    func = lag1_all
    if func_num == 1:
        func = lag1_range_1
    elif func_num == 2:
        func = lag1_range_2
    elif func_num == 3:
        func = lag1_range_3
        
    subjects_input = ['LTP093' for i in range(len(recall_matrix)-12)]+['LTP106' for i in range(12)]
    
    crps = crp2(recalls=recall_matrix, subjects=subjects_input, listLength=24, lag_num=ll-1, lag_range=func, skip_first_n=0)

    # if crps is empty / there were no eligible lists, return
    # a vector of 0's.
    if not crps.any():
        crps = np.asarray(np.zeros((1, (ll-1)*2 + 1)))
    else:
        crps = np.asarray(crps)
        
    # now compute sem!

    # return mean crp across lag bins
    return crps


def main():

    data_path = '/home1/rivkat.cohen/PycharmProjects/CMR2/K02_files/K02_data.mat'
    data_file = scipy.io.loadmat(
        data_path, squeeze_me=True, struct_as_record=False)  # get data
    data_mat = data_file['data'].recalls  # get presented items

    # set list length
    listlength = 10

    output_mean, output_std = get_crp(data_mat, listlength)

    output_mean = np.squeeze(np.asarray(output_mean))
    output_std  = np.squeeze(np.asarray(output_std))

    # lag-CRP output
    print("\nLag-CRP values: ")
    print(output_mean)
    print(output_std)

    ############
    #
    #   Graph the lag-CRP output out to 5 lags in either direction
    #
    ############

    left_crp = output_mean[4:int((len(output_mean) - 1)/2)]
    right_crp = output_mean[int((len(output_mean) - 1)/2) + 1:15]

    left_crp_std = output_std[4:int((len(output_mean) - 1)/2)]
    right_crp_std = output_std[int((len(output_mean) - 1)/2) + 1:15]

    xleft = range(-5, 0, 1)
    xright = range(1, 6, 1)

    plt.plot(xleft, left_crp, color='k', lw=2.5)
    plt.plot(xright, right_crp, color='k', lw=2.5)

    plt.ylim([0.0, 0.6])

    plt.show()


if __name__ == "__main__": main()