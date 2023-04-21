#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:16:21 2017

@author: shai.goldman
"""

import numpy as np

from numpy import matlib

def create_txts_for_subj(fnum, tilenum=4):
    times = np.loadtxt('times_LTP228.txt', delimiter=',')
    rec_itemnos = np.loadtxt('rec_nos_LTP228.txt', delimiter=',')
    pres_itemnos = np.loadtxt('pres_nos_LTP228.txt', delimiter=',')

    times = matlib.repmat(times, 4, 1)
    rec_nos = matlib.repmat(rec_itemnos, 4, 1)
    pres_nos = matlib.repmat(pres_itemnos, 4, 1)

    np.savetxt ('times_LTP%d_x4.txt'% fnum, times, delimiter=',', fmt='%i')
    np.savetxt ('pres_nos_LTP%d_x4.txt'% fnum, pres_nos, delimiter=',', fmt='%i')
    np.savetxt ('rec_nos_LTP%d_x4.txt'% fnum, rec_nos, delimiter=',', fmt='%i')

if __name__ == '__main__':
    create_txts_for_subj(228, 4)