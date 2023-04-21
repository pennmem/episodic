import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time

def get_possible_transitions(rec_list, list_length):

    """Return all possible transitions, for each item in the list.

    Need to update this to ignore all transitions with a lag greater than or
    less than a pre-determined maximum or minimum threshold value.  This can be
    accomplished after the fact by simply truncating the results vector,
    but for speed's sake, better to a priori not calculate CRP for those lags.

    """

    collected_possible_transitions = []

    # for each item in the list, other than the last item:
    for idx, item in enumerate(rec_list[:len(rec_list) - 1]):

        # base of all possible transitions is,
        # that item minus all the steps between that item and 0,
        # and that item plus all the steps between that item and list_length,
        # inclusive
        pos_lag = 1
        valid_pos_lags = []

        # get all lags that don't put us past list-length
        while item + pos_lag <= list_length:
            valid_pos_lags.append(pos_lag)
            pos_lag += 1

        neg_lag = -1
        valid_neg_lags = []
        # get all lags that don't put us below serial pos. 1
        while item + neg_lag >= 1:
            valid_neg_lags.append(neg_lag)
            neg_lag -= 1

        # combine these possible lags
        # (these don't yet take into account previously recalled items)
        base_possible_transitions = np.sort(valid_neg_lags + valid_pos_lags)

        # calculate the lags between the current item and each item
        # that has come prior to it on the recalled-items list
        prior_items = rec_list[0:idx]

        invalid_lags = (np.repeat(item, len(prior_items)) - prior_items) * -1

        # get booleans where invalid lags occur in the
        # collected_possible_transitions list
        invalid_indices = np.searchsorted(
            base_possible_transitions, invalid_lags)

        # replace the items at these indices with 0's.
        # later, we will be ignoring these.
        base_possible_transitions[invalid_indices] = 0

        # append the final, screened list of possible lags
        collected_possible_transitions.append(base_possible_transitions)

    return collected_possible_transitions


def get_actual_transitions(rec_list):
    """for a given list of recall no's, output what types of
    transitions were made"""

    # calculate lags between each pair of items, except for the
    # last item recalled
    all_transitions_made = np.subtract(
        rec_list[1:len(rec_list)], rec_list[0:(len(rec_list) - 1)])

    return all_transitions_made


def list_crp(sample_list, ll):

    """Calculate lag-crp for an individual list"""

    # remove any duplicates while maintaining original list order
    unique_indexes = np.unique(np.asarray(sample_list), return_index=True)[1]

    if len(unique_indexes) < 2:

        return np.zeros((1, (ll - 1)*2 + 1))

    sample_list = np.squeeze(np.array(sample_list))
    sample_list = np.array([sample_list[index] for index
                            in sorted(unique_indexes)])

    # remove zeros
    sample_list = sample_list[sample_list > 0]

    # get actual & possible transitions in the sample list
    all_poss_trans = get_possible_transitions(sample_list, ll)
    all_made_trans = get_actual_transitions(sample_list)

    # collect possible transitions so we can get counts per lag type
    flatten_poss_trans = [item for sublist in all_poss_trans
                          for item in sublist]

    # Each gives a tuple of two arrays:
    # Array 0 contains array of unique values in the list.
    # Array 1 contains their frequencies.
    actual_bins = np.unique(all_made_trans, return_counts=True)
    possible_bins = np.unique(flatten_poss_trans, return_counts=True)

    # init. vector into which to place the possible bin values
    # max lag is ll-1; then we want to examine in both + and - directions,
    # so we take (ll-1)*2. Add 1 for the 0th value.
    spaced_poss_vec = np.zeros((1, (ll-1)*2 + 1))

    # get indices in the zero-padded vec (one slot per lag under consideration)
    pos_bin_indices = possible_bins[0] + (ll-1)

    # place counts of each lag type into the appropriate bins,
    # but place no counts into bin of 0.
    spaced_poss_vec[0, pos_bin_indices] = possible_bins[1]

    # get indices where the actual_bin identities are located in the
    # possible_bin identities the 0th index in actual_bins and possible_bins
    # holds the lag identities
    spaced_indices = np.asarray(range(-(ll-1), ll, 1))
    matching_lag_indices = np.searchsorted(spaced_indices, actual_bins[0])

    # zero-pad actual_bins values so that we can appropriately
    # divide them into the possible_bins values at matching locations.
    actual_bins_zp = np.zeros((1, spaced_poss_vec[0].shape[0]))
    actual_bins_zp[0, matching_lag_indices] = actual_bins[1]

    # divide counts of actual transitions that took place by the values
    # of possible transitions that took place.

    list_crps = np.zeros(actual_bins_zp.shape[1])

    # divide all actual transitions by possible transitions
    for idx, poss_val in enumerate(spaced_poss_vec[0]):
        # don't divide by zero!
        if poss_val != 0:
            list_crps[idx] = actual_bins_zp[0, idx] / poss_val
        else:
            list_crps[idx] = np.nan

    return list_crps


def get_crp(list_matrix, ll):
    """Get crp across a matrix of lists"""

    desired_length = 2

    crps = []
    for this_list in list_matrix:

        this_list = np.asarray(this_list)
        unique_list = np.unique(this_list[this_list > 0])
        # skip if list length is less than the desired length
        if len(unique_list) < desired_length:
            continue
        else:
            this_crp = list_crp(this_list, ll)
            crps.append(np.squeeze(this_crp))

    # if crps is empty / there were no eligible lists, return
    # a vector of 0's.
    if not crps:
        crps = np.asarray(np.zeros((1, (ll-1)*2 + 1)))
    else:
        crps = np.asarray(crps)

    # return mean crp across lag bins
    return np.nanmean(crps, axis=0), (np.nanstd(crps, axis=0)
                                      / (crps.shape[0]**0.5))


def main():

    data_path = '/Users/KahaNinja/PycharmProjects/CMR2/K02_files/K02_data.mat'
    data_file = scipy.io.loadmat(
        data_path, squeeze_me=True, struct_as_record=False)  # get data
    data_mat = data_file['data'].recalls  # get presented items

    # set list length
    listlength = 10

    output_mean, output_std = get_crp(data_mat, listlength)

    output_mean = np.squeeze(np.asarray(output_mean))
    output_std  = np.squeeze(np.asarray(output_std))

    # Print lag-CRP output
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