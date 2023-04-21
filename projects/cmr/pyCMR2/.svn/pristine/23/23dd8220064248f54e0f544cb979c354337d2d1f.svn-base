#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:51:23 2020

@author: lumdusislife
"""

import numpy as np
import scipy.stats as ss
import warnings

def get_irts(rectimes, recalls, listLength=24):  
    # Makes an array of calculated interresponse times
    # corresponding to the recalls matrix, where irts[0] 
    # would be up until the first item, and irts[1]
    # would be rectimes[1]-rectimes[0]. a
    
    irts = np.zeros([recalls.shape[0], recalls.shape[1]])
    for trial in range(0, recalls.shape[0]):
        try:
            irts[trial, 0] = rectimes[trial, 0]
        except IndexError:
            continue
        for pos in range(0, recalls.shape[1] - 1):
            irts[trial, pos+1] = rectimes[trial, pos + 1] - rectimes[trial, pos]
            if irts[trial, pos+1] <= 0:
                if irts[trial, pos+1] < 0 and rectimes[trial, pos+1]>0:
                    print ('negative IRT of %d detected at %d,%d' %(irts[trial, pos+1], trial, pos))
                irts[trial, pos+1] = np.nan
            
    for i in range(0, irts.shape[0]):
        if True in [np.isnan(j) for j in irts[i]]:
            irts[i][[np.isnan(j) for j 
                in irts[i]].index(True):] = np.nan
                
    return irts/1000

def R_crl(irts, return_sem=False, maxRToTest=15):
    
    irts = np.asarray(irts)
    
    Rs = np.asarray([len(trial[~np.isnan(trial)]) for trial in irts])
    
    irts_by_R = []
    errs = []
    
    rRange = np.arange(4, maxRToTest)
    
    for rVal in rRange:
        irts_by_R.append(np.nanmean(irts[Rs==rVal]))
        errs.append(ss.sem(np.nanmean(irts[Rs==rVal], axis=1)))
    
    irts_by_R = np.asarray(irts_by_R)
    irts_by_R[np.isnan(irts_by_R)] = 0 # not sure if this is correct
    
    if return_sem:
        return irts_by_R, errs
    
    return irts_by_R

def op_crl(irts, return_sem=False, maxOPToTest=15):
    
    irts = np.asarray(irts)
    
    z = np.zeros(maxOPToTest-1)
    av = np.nanmean(irts, axis=0)[1:maxOPToTest]
    z[:av.size] = av
    av = z
    
    z = np.zeros(maxOPToTest-1)
    errs = ss.sem(irts, axis=0, nan_policy='omit')[1:maxOPToTest]
    z[:errs.size] = errs
    errs = z
    
    if return_sem:
        return av, errs
    
    return av    
            
            
def firstRecallDist(responseTimes):
    
    firstRt = responseTimes[:, 0]
    
    quantiles = np.asarray([np.quantile(firstRt,q=i) for i in [.2,.4,.6,.8]])
    
    return quantiles
            
            
def nthIrtDist(irts, n=1):
    
    try:
        nthIrts = irts[:, n]
        nthIrts = nthIrts[~np.isnan(nthIrts)]
    except IndexError:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    
    quantiles = np.asarray([np.quantile(nthIrts,q=i) for i in [.2,.4,.6,.8]])
    
    return quantiles


def avgTotalRecalls(recalls):
    
    numRecalled = [len(np.unique(lst[lst>0])) for lst in recalls]
    return np.mean(numRecalled)


def opR_crl(irts, subjs, minR=8, max_output=16, ll=24,
                    graphit=False, colors=True, title=None):
    
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    irts_by_R_all_subj = []
    lens = [i for i in range(minR, max_output+1, 2)]
    for subj in np.unique(subjs):
        this_irts = irts[np.where(subjs==subj)]
        
        Rs = np.asarray([len(trial[~np.isnan(trial)]) for trial in this_irts])
        this_irts[this_irts == np.inf] = np.nan
          
        irts_by_R = []
        for i in lens:
            these_irts = this_irts[Rs==i].tolist()
            if len(these_irts) == 0:
                these_irts.append([])
            for i in these_irts:
                while len(i) < ll:
                    i.append(np.nan)
                while len(i) > ll:
                    i.remove(i[-1])
            irts_by_R.append(np.asarray(these_irts))
   
        irts_by_R = np.array([irts_by_R[i][:, 1:] for i in range(len(irts_by_R))])
        irts_by_R_all_subj.append(np.asarray(
                [np.nanmean(irt[~np.isnan(irt[:,0])], axis=0) for irt in irts_by_R]))
    
    irts_by_R_all_subj = np.asarray(irts_by_R_all_subj)
    toGraph = np.nanmean(irts_by_R_all_subj, axis=0)
    errs = ss.sem(irts_by_R_all_subj, axis=0)
    
    # -1 here bc RT1 discounted.
    errs = [e[:num_recalls-1] for num_recalls, e in zip(lens, errs)]
    toGraph = [d[:num_recalls-1] for num_recalls, d in zip(lens, toGraph)]
    
    linestyles = []
    markers = []
    for i in np.arange(0, len(toGraph)+1, 2):
        linestyles.extend(['-', '--', '-.', ':'])
        markers.extend(['o', 's'])
    if graphit:
        from matplotlib import pyplot as plt
        ax = plt.subplot(111)
        for i in range(0,len(toGraph)):
            graph = toGraph[i]
            err = errs[i]
            
            x = np.arange(1,len(graph)+1)
            
            y = graph#[~np.isnan(graph)]
            error = err#[~np.isnan(graph)]
            
            if colors:
                ax.errorbar(x, y, yerr=error, linewidth=2, 
                             color='k',
                             marker=markers[i], linestyle=linestyles[i], 
                             label=len(y)+1)
            else:
                ax.errorbar(x, y, yerr=error, linewidth=2, 
                             marker=markers[i], linestyle=linestyles[i], 
                             label=len(y)+1)
                
        plt.xlabel('Output Position')
        plt.ylabel('Inter-Response Time (s)')
        plt.xticks(range(1, max_output+1, 2), 
                   [f'{i}-{i+1}' for i in range(1, max_output+1, 2)])
        plt.tight_layout()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, 
                         box.width * 0.7, box.height])
        handles, labels = ax.get_legend_handles_labels() # remove the errorbars 
        handles = [h[0] for h in handles] 
        ax.legend(handles, labels, loc='center left', 
                  bbox_to_anchor=(1.0, 0.5), title='Total Recalls')
        if title is not None:
            plt.title(title)
        plt.show()
    
    warnings.resetwarnings()
    concat = np.concatenate(toGraph)
    concat[np.isnan(concat)] = 0
    return concat





            
            