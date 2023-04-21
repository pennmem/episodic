#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:31:45 2017

@author: shai.goldman
"""
import numpy as np
from matplotlib import pyplot as plt

def sterr(data, axis=0):
    # just magnified standard deviation up to 95% confidence.
    data = np.asarray(data)
    return(np.nanstd(data, axis=axis)/ (data.shape[0]**0.5))
    

def split_sessions(irts):
    Ss_data = []

    
    subj_id_map = np.repeat(np.arange(1, (irts.shape[0]/irts.shape[1])+1), irts.shape[1])

    # Get locations where each Subj's data starts & stops.
    new_subj_locs = np.unique(
        np.searchsorted(subj_id_map, subj_id_map))

    # Separate data into sets of lists presented to each subject
    for i in range(new_subj_locs.shape[0]):
        # for all but the last list, get the lists that were presented
        # between the first time that subj ID occurs and when the next ID occurs
        if i < new_subj_locs.shape[0] - 1:
            start_lists = new_subj_locs[i]
            end_lists = new_subj_locs[i + 1]

        # once you have reached the last subj, get all the lists from where
        # that ID first occurs until the final list in the dataset
        else:
            start_lists = new_subj_locs[i]
            end_lists = irts.shape[0]

        # append subject's sheet
        Ss_data.append(irts[start_lists:end_lists, :])

    return np.asarray(Ss_data)

def get_irts(rectimes, recalls):  # Makes an array of calculated interresponse times
    # corresponding to the recalls matrix, where irts[0] would be up until the first item, and irts[1]
    # would be rectimes[1]-rectimes[0]. It ignores false recalls as if they 
    # never occured
    
    ll = rectimes.shape[1]
    
    irts = np.zeros([recalls.shape[0], recalls.shape[1]])
    for trial in range(0, recalls.shape[0]):
        seen = [] #to exclude repeats
        irts[trial, 0] = rectimes[trial, 0]
        for pos in range(0, ll - 1):
            if not recalls[trial, pos] in seen and recalls[trial, pos+1] > 0: 
                irts[trial, pos+1] = rectimes[trial, pos + 1] - rectimes[trial, pos]
                seen.append(recalls[trial, pos])
            else:
                irts[trial, pos+1] = np.nan
            if irts[trial, pos+1] < 0:
                irts[trial, pos+1] = np.nan
            
    for i in range(0, irts.shape[0]):
        if True in [np.isnan(j) for j in irts[i]]:
            irts[i][[np.isnan(j) for j 
                in irts[i]].index(True):] = np.nan
    irts = irts[:, :24]
    return irts


def op_crl(rectimes, recalls, graphit=False):
    irts = get_irts(rectimes, recalls)[:, 1:15]
    
    ops = np.zeros([irts.shape[0], irts.shape[1]])
    ops[:] = np.nan
    for trial in range(irts.shape[0]):
        for rec in range(irts.shape[1]):
            ops[trial, rec] = irts[trial][~np.isnan(irts[trial])].size - rec
    
    
    x = np.arange(1, irts.shape[1]+1)
    avgs = np.nanmean(irts, axis=0)
    errs = sterr(irts)
    if graphit:
        plt.errorbar(x, avgs, yerr=errs, marker='o', linewidth=3)
    return avgs, errs

def R_crl(rectimes, recalls, graphit=False):
    irts =  get_irts(rectimes, recalls)
    Rs = np.asarray([len(trial[~np.isnan(trial)]) for trial in irts])
    irts_by_R = []
    for i in np.arange(4, Rs.max()+1):
        irts_by_R.append(np.nanmean(irts[Rs==i]))
        try:
            if not irts[Rs==i]:
                irts_by_R.append(np.array(0))
        except ValueError:
            pass
    plt.plot(irts_by_R)

def irt_dist(rectimes, recalls, num_bins=6, graphit=False):
    irts = get_irts(rectimes, recalls)
    hirts = irts.flatten()
    hirts = hirts[~np.isnan(hirts)]
    hist, bins = np.histogram(hirts[hirts<8000], bins = num_bins, normed=1)
    irts = split_sessions(irts)
    hists = []
    for session in irts:
        thist, tbins = np.histogram(session[session<8000], bins=bins, normed=1)
        hists.append(thist)
    
    if graphit:
        plt.plot(bins[:-1], hist)
    
    return np.nanmean(hists, axis=0), sterr(hists)


def opR_crl(rectimes, recalls, graphit=False, colors=True, 
            Szs=None, rectimes2=None, recalls2=None, numstrands=3, Rmax=24):
    print 'in shaisvers'
    import numpy as np
    #return np.full(24*6, 0), np.full(24*6, 0)
    
    print 'Szs:',Szs
    if type(Szs) != type(None):
        try:
            Szs = np.sort(Szs)
        except ValueError:
            print Szs, type(Szs), 'haderr'
    
    irts = get_irts(rectimes, recalls)
    irts2 = None
    if type(rectimes2) != type(None):
        irts2 = get_irts(rectimes2, recalls2)
    
    if type(irts2) != type(None):
        irtslst = [irts, irts2]
    else:
        irtslst = [irts]
    
    if len(irtslst) > 1:
        sepcolors = (i for i in ['blue', 'm'])
    else:
        sepcolors = (i for i in ['k'])
    
    for irts in irtslst:
        irts = split_sessions(irts)
        errs = []
        full_crls = [[] for i in range(24)]
        for session in irts:
        
            Rs = np.asarray([len(trial[~np.isnan(trial)]) for trial in session])
            #nonedszs = []
            irts_by_R = []
            if type(Szs) == type(None):
                itr = np.arange(3, Rmax+1)
            else:
                itr = Szs
                
            for i in itr:
                try:
                    if not session[Rs==i]:
                        irts_by_R.append(np.array([np.append(
                                                np.full(i, np.nan), 
                                                np.full(session.shape[1]-i, np.nan))]))
                    else:
                        irts_by_R.append(session[Rs==i])
                except ValueError:
                    irts_by_R.append(session[Rs==i])
        
            irts_by_R = np.asarray(irts_by_R)
            #irts_by_R = irts_by_R[0:len(irts_by_R):2]
            
            irts_by_R = np.array([irts_by_R[i][:, 1:] for i in range(len(irts_by_R))])
            
            for i in range(len(full_crls)):
                try:
                    full_crls[i].extend(irts_by_R[i])
                except IndexError:
                    pass
            errs.append(np.asarray([np.nanmean(irt[~np.isnan(irt[:,0])], axis=0) for irt in irts_by_R]))
        
        full_crls = np.asarray(full_crls)
        errs = sterr(np.asarray(errs), axis=0)
        sizes = np.asarray([len(i) for i in full_crls])
        full_crls = full_crls[np.argsort(sizes)[-numstrands:]]
        errs = errs[np.argsort(sizes)[-numstrands:]]
        if type(Szs) == type(None):
            forlegend = np.arange(1, Rs.max()+1)[np.argsort(sizes)[-numstrands:]] + 1
            full_crls = full_crls[np.argsort(forlegend)]
            errs = errs[np.argsort(forlegend)]
            forlegend = np.sort(forlegend) + 1
            Szs = forlegend
        else:
            forlegend = Szs
        
        
        toGraph = np.array([np.nanmean(i, axis=0) for i in full_crls])
        
        for i in range(len(Szs)):
            if len(toGraph[i][~np.isnan(toGraph[i])]) == 0:
                toGraph[i] = (np.array([np.append(
                                                np.full(Szs[i]-1, 100000), 
                                                np.full(session.shape[1]-(Szs[i]), np.nan))]))
                errs[i] = (np.array([np.append(
                                                np.full(Szs[i]-1, 1), 
                                                np.full(session.shape[1]-(Szs[i]), np.nan))]))
        
        linestyles = []
        markers = []
        for i in np.arange(0, len(toGraph)+1, 4):
            linestyles.extend(['-', '--', '-', '--'])
            markers.extend(['o', 's', 's', 'o'])
        if graphit:
            ax = plt.subplot(111)
            mycolor = sepcolors.next()
            for i in range(0,len(toGraph)):
                graph = toGraph[i]
                err = errs[i]
                    
                x = np.arange(1,len(graph[~np.isnan(graph)])+1)
                    
                y = graph[~np.isnan(graph)]
                error = err[~np.isnan(graph)]
                    
                if colors:
                    plt.errorbar(x, y, yerr=error, linewidth=2, 
                                    color=mycolor,
                                    marker=markers[i], linestyle=linestyles[i])
                else:
                    plt.errorbar(x, y, yerr=error, linewidth=2, 
                                    marker=markers[i], linestyle=linestyles[i])
                        
            plt.xlabel('Output Position')
            plt.ylabel('Inter-Response Time (s)')
            plt.tight_layout()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, 
                                 box.width * 0.7, box.height])
            plt.legend(forlegend,
                    bbox_to_anchor=(1.05, 1), loc=2, 
                    borderaxespad=0., title='Total Recalls')
                #plt.text(max_output+1.3, toGraph[-1][~np.isnan(toGraph[-1])][-3]-.3, 'Total Recalls')
    if graphit:
        plt.show()
            
            
            
    supergraph = []
    for i in toGraph:
        supergraph=np.append(supergraph, i[~np.isnan(i)])
    supererr = []
    for i in errs:
        supererr=np.append(supererr, i[~np.isnan(i)])
    
    try:
        if not supergraph:
            supergraph = np.full(np.sum(Szs-1), 0)
            print 'reverted to old method, not sure why'
            supererr = np.full(np.sum(Szs=1), 0)
    except ValueError:
        pass
    
    #print forlegend

    return np.asarray(supergraph), np.asarray(supererr), forlegend