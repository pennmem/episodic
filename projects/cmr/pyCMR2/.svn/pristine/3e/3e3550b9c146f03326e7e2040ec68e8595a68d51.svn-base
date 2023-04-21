import scipy.io
import numpy as np

path_to_load = '/data5/eeg/scalp/ltp/ltpFR2/behavioral/data/stat_data_LTP138.mat'
pres_nos = scipy.io.loadmat(
    path_to_load, squeeze_me=True, struct_as_record=False)['data'].pres_itemnos

rec_nos = scipy.io.loadmat(
    path_to_load, squeeze_me=True, struct_as_record=False)['data'].rec_itemnos

recalls = scipy.io.loadmat(
    path_to_load, squeeze_me=True, struct_as_record=False)['data'].recalls

print(pres_nos[:5])
print(rec_nos[:5])
print(recalls[:5])

path_to_save = 'rec_nos_LTP138.txt'
np.savetxt(path_to_save, rec_nos, delimiter=',', fmt='%i')

pres_path_to_save = 'pres_nos_LTP138.txt'
np.savetxt(pres_path_to_save, pres_nos, delimiter=',', fmt='%i')

np.savetxt('recalls_LTP063.txt', recalls, delimiter=',', fmt='%i')