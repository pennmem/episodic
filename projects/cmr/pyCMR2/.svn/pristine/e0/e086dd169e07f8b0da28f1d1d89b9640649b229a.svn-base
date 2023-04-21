#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:37:50 2017

@author: shai.goldman
"""

import numpy as np

def sterr(data):
    # just magnified standard deviation up to 95% confidence.
    data = np.asarray(data)
    return(np.nanstd(data, axis=0)/ (data.shape[0]**0.5))

def plicount(recalls, startlist=7):
    plis = [[] for i in range(startlist-1)]
    errs = [[] for i in range(startlist-1)]
    for trial in range(startlist, recalls.shape[0]):
        for rec in recalls[trial]:
            for i in np.arange(-(startlist-1), 0):
                plis[np.absolute(i)-1].append((recalls[trial][recalls[trial]==i]).size)
    
    for i in range(len(plis)):
        errs[i] = sterr(plis[i])
        plis[i] = np.nanmean(plis[i])
        
    return np.asarray(plis), np.asarray(errs)