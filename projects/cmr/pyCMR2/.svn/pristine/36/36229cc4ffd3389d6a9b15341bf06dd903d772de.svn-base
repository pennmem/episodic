#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:31:26 2020

@author: lumdusislife
"""

import numpy as np
import CMR2_pack_cyth as CMR2
from pybeh.spc import spc
from pybeh.crp import crp
from pybeh.pfr import pfr
from pybeh.make_recalls_matrix import make_recalls_matrix
from pybeh.create_intrusions import intrusions
from optimization_utils import param_vec_to_dict, pad_into_array
from optimization_utils import get_data
from glob import glob


def obj_func(param_vec, data_pres, sessions, w2v, source_mat=None):

    # Reformat parameter vector to the dictionary format expected by CMR2
    
    param_dict = param_vec_to_dict(param_vec)
    print(param_dict)
    print('pres_nos')
    print(data_pres[0][0:])
    
    # Run model with the parameters given in param_vec
    rec_nos, rts = CMR2.run_cmr2_multi_sess(param_dict, data_pres, sessions, w2v, source_mat=source_mat, mode='DFR')
    print('Doing DFR')
    
    print('rec_nos')
    print(rec_nos[0][0:])
    
    # Create recalls and intrusion matrices
    cmr_rec_nos = pad_into_array(rec_nos, min_length=1)
    cmr_recalls = make_recalls_matrix(pres_itemnos=data_pres, rec_itemnos=rec_nos)
    #cmr_intrusions = intrusions(pres_itemnos=data_pres, rec_itemnos=rec_nos, subjects=np.zeros(rec_nos.shape[0]),
    #                            sessions=sessions)
    print('recalls')
    print(cmr_recalls[0][0:])
    return cmr_rec_nos, cmr_recalls, rts


if __name__ == "__main__":
    
    import os
    
    subj = 229
    vers = 15
    
    onRhino = True
    try:
        os.chdir('/data')
    except:
        onRhino = False
    
    if onRhino:
        
        basepath = ""
    
        outdir = basepath+'/home1/shai.goldman/pyCMR2/IRT_Optimizations/model_params/'
        
        print('one vers %d' % vers)
        
        f = open (outdir + 'xopt_ltpFR2_%d.txt' % vers)
        param_vec = []
        for param in f:
            param = param[:len(param)-1]
            param_vec.append(float(param))
        f.close()
        
        print(param_vec)
                        
        #bestfitfile=outdir+'141data251.pkl'
        #import pickle
        #with open(bestfitfile, 'rb') as f:
        #    pkldata = pickle.load(f)
        #param_vec = pkldata['params']
        
        # Load lists from participants who were not excluded in the behavioral analyses
        file_list = glob(basepath+'/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP*.json')
        for i in file_list:
            if "incomplete" in i:
                file_list.remove(i)
        
        #file_list = [basepath+'/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP%s.json' % subj]
        
        # Set file paths for data, wordpool, and semantic similarity matrix
        wordpool_file = basepath+'/home1/shai.goldman/pyCMR2/CMR2_Optimized/wordpools/PEERS_wordpool.txt'
        w2v_file = basepath+'/home1/shai.goldman/pyCMR2/CMR2_Optimized/wordpools/PEERS_w2v.txt'
    
        # Load data
        print('Loading data...')
        data_pres, sessions, sources = get_data(file_list, wordpool_file, number_sessions=900)
        sources = None  # Leave source features out of the model for the between-subjects experiment
        #run 5X on the data:
        #data_pres = np.concatenate(np.repeat(np.array([data_pres]), 5, axis=0))  
        #sessions = np.concatenate(np.repeat(np.array([sessions]), 5, axis=0))   
        
        # Load semantic similarity matrix (word2vec)
        w2v = np.loadtxt(w2v_file)
        
        
        rec_nos, recalls, rts = obj_func(param_vec, data_pres, sessions, w2v, source_mat=None)
        
        print(recalls)
        print(rts)
        
        print('Done! Saving files now.')
        
        np.save(outdir+'data%s_pres.npy' % subj, data_pres)
        np.save(outdir+'model%s_recnos.npy' % subj, rec_nos)
        np.save(outdir+'model%s_recalls.npy' % subj, recalls)
        np.save(outdir+'model%s_rts.npy' % subj, rts)
        np.save(outdir+'w2v.npy', w2v)
        
    else:
        
        runfile('/Users/lumdusislife/Desktop/IRT/pyCMR2/IRT_Optimizations/fitting/optimization_utils.py', wdir='/Users/lumdusislife/Desktop/IRT/pyCMR2/IRT_Optimizations/fitting')
        
        basepath = '/Users/lumdusislife/Desktop/IRT/pyCMR2/IRT_Optimizations/'
        
        pres = np.load(basepath+'model_params/data%s_pres.npy' % subj)[:,:24]
        recalls = np.load(basepath+'model_params/model%s_recalls.npy' % subj)[:,:24]
        rec_nos = np.load(basepath+'model_params/model%s_recnos.npy' % subj)[:,:24]
        rts = np.load(basepath+'model_params/model%s_rts.npy' % subj)[:,:24]
        #w2v = np.load(basepath+'w2v.npy' % subj)
        
        sessions = 200
        lists_per_session =24
        ll = 24
        
        #rts = rts[:,:ll]
        #r = np.zeros((sessions*lists_per_session, ll))
        #r[:, :recalls.shape[1]] = recalls 
        #recalls = r
        
        #r = np.zeros((sessions*lists_per_session, ll))
        #r[:, :rts.shape[1]] = rts
        #rts = r
        
        os.chdir(basepath+'/CrlScripts')
        
        
        runfile('/Users/lumdusislife/Desktop/IRT/Scripts/misc_funcs.py', wdir='/Users/lumdusislife/Desktop/IRT/Scripts')
        runfile('/Users/lumdusislife/Desktop/IRT/Scripts/crls.py', wdir='/Users/lumdusislife/Desktop/IRT/Scripts')
        runfile('/Users/lumdusislife/Desktop/IRT/Scripts/data_collection.py', wdir='/Users/lumdusislife/Desktop/IRT/Scripts')
        runfile('/Users/lumdusislife/Desktop/IRT/pyCMR2/IRT_Optimizations/fitting/crls.py', wdir='/Users/lumdusislife/Desktop/IRT/pyCMR2/IRT_Optimizations/fitting')
        #from crls import opR_crl, lag_crl, sem_crl, SimbyLag_crl, prevIrt_crl, intrasess_crl
        #from misc_funcs import irt_dist, plt_std_graph, semantic_crp, lag_crp, spc
        from matplotlib import pyplot as plt
        #os.chdir(basepath+'/CrlScripts')
        #from data_collection import get_irts
    
        
        irts = get_irts(rts, recalls)
        
        data = {'recalls': recalls.astype(int),
                'rec_itemnos': rec_nos.astype(int),
                'pres_itemnos': pres.astype(int),
                'rectimes': rts/1000,
                'irts': irts,
                'subject': np.zeros(recalls.shape[0]),
                'model': True
                }
        
        #from data_collection import get_alldata
        
        adata = get_alldata()
        adata['irts'] = get_irts(adata['rectimes']*1000, adata['recalls'], 24)

        
        
        #semantic_crp(data)
                
        """     
        bestfitfile=basepath+'model_params/141data251.pkl'
        import pickle
        with open(bestfitfile, 'rb') as f:
            pkldata = pickle.load(f)
        
        
        plt.plot(np.arange(.2,1,.2), pkldata['fRT'], marker = 'o', label='model')
        plt.plot(np.arange(.2,1,.2), firstRecallDist(adata['rectimes']), marker = 'o', label='actual')
        plt.xticks(np.arange(.2,1,.2))
        plt.legend()
        plt.xlabel('Quantile')
        plt.ylabel('First Response Time')
        plt.show()
        
        plt.plot(range(1,25), pkldata['pfr'], marker = 'o', label='model')
        plt.plot(range(1,25), np.nanmean(pfr(adata['recalls'], adata['subject'], 24), axis=0), marker = 'o', label='actual')
        plt.xticks(range(1,25,2))
        plt.legend()
        plt.xlabel('Serial Position')
        plt.ylabel('PFR')
        plt.show()
        """
        ###############3
        
        model_color = 'r'
        actual_color = 'black'
        
        opR_crl(data['irts'], data['subject'], minR=8, max_output=16, ll=24,
                    graphit=True, colors=True, title='Model')
        opR_crl(adata['irts'], adata['subject'], minR=8, max_output=16, ll=24,
                    graphit =True, colors=True, title='Actual')
        
        
        plt.plot(range(1,25), pfr(data['recalls'], data['subject'], 24)[0], 
                 marker = 'o', label='model', color = model_color)
        plt.plot(range(1,25), 
                 np.nanmean(pfr(adata['recalls'], adata['subject'], 24), axis=0), 
                 marker = 'o', label='actual', color = actual_color)
        plt.xticks(range(1,25,2))
        plt.legend()
        plt.xlabel('Serial Position')
        plt.ylabel('PFR')
        plt.show()
        
        plt.plot(np.arange(.2,1,.2), 
                 firstRecallDist(data['rectimes']), 
                 marker = 'o', label='model', color = model_color)
        plt.plot(np.arange(.2,1,.2), 
                 firstRecallDist(adata['rectimes']), 
                 marker = 'o', label='actual', color = actual_color)
        plt.xticks(np.arange(.2,1,.2))
        plt.legend()
        plt.xlabel('Quantile')
        plt.ylabel('First Response Time')
        plt.show()
        
                
        plt.plot(np.arange(.2,1,.2), 
                 nthIrtDist(data['irts'], 1), 
                 marker = 'o', label='model', color = model_color)
        plt.plot(np.arange(.2,1,.2), 
                 nthIrtDist(adata['irts'], 1), 
                 marker = 'o', label='actual', color = actual_color)
        plt.xticks(np.arange(.2,1,.2))
        plt.legend()
        plt.xlabel('Quantile')
        plt.ylabel('First IRT')
        plt.show()
        
        plt.plot(range(-8,9), 
                 crp(data['recalls'][:, 0:2], data['subject'], 24, 8)[0], 
                 marker = 'o', label='model', color = model_color)
        plt.plot(range(-8,9), 
                 np.nanmean(crp(adata['recalls'][:, 0:2], adata['subject'], 24, 8), axis=0), 
                 marker = 'o', label='actual', color = actual_color)
        plt.xticks(range(-8,9))
        plt.legend()
        plt.xlabel('Lag')
        plt.ylabel('First Transition CRP')
        plt.show()
        
        plt.bar([0,1], [avgTotalRecalls(data['recalls']),
                        avgTotalRecalls(adata['recalls'])],
                edgecolor='k', linewidth=2, color='lightblue'
        )
        plt.xticks([0,1], ['model', 'actual'])
        plt.ylabel('Average total recalls')
        plt.show()
        
        
        """
        plt.plot(range(1,25), spc(data['recalls']), marker = 'o', label='model')
        plt.plot(range(1,25), spc(adata['recalls']), marker = 'o', label='actual')
        plt.xticks(range(1,25))
        plt.legend()
        plt.xlabel('Lag')
        plt.ylabel('SPC')
        plt.show()
        
        plt.plot(range(-8,9), crp(data['recalls'], data['subject'], 24, 8)[0], marker = 'o', label='model')
        plt.plot(range(-8,9), np.nanmean(crp(adata['recalls'], adata['subject'], 24, 8), axis=0), marker = 'o', label='actual')
        plt.xticks(range(-8,9))
        plt.legend()
        plt.xlabel('Lag')
        plt.ylabel('CRP')
        plt.show()
        
        
        plt.plot(range(2,16), op_crl(data['irts']), marker = 'o', label='model')
        plt.plot(range(2,16), op_crl(adata['irts']), marker = 'o', label='actual')
        plt.xticks(range(2,16))
        plt.legend()
        plt.xlabel('Output Position')
        plt.ylabel('Mean IRT (s)')
        plt.show()
        
        plt.plot(range(4,15), R_crl(data['irts']), marker = 'o', label='model')
        plt.plot(range(4,15), R_crl(adata['irts']), marker = 'o', label='actual')
        plt.xticks(range(4,15))
        plt.legend()
        plt.xlabel('Total Number of Recalls')
        plt.ylabel('Mean IRT (s)')
        plt.show()
        """
        """lag_crp(data['recalls'])
        plt.plot(spc(data['recalls']))
        
        
        irt_dist(data)
        opR_crl(data)
        lag_crl(data)
        prevIrt_crl(data)
        intrasess_crl(data)
        sem_crl(data)"""
    
    
    
    
    
    
    
    
    
    
    
    
    
    