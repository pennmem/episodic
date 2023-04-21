#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:52:58 2020

@author: lumdusislife
"""

#from data_collection import get_alldata
import numpy as np
from pybeh.spc import spc
from pybeh.crp import crp
from pybeh.pfr import pfr
import json
from crls import firstRecallDist,nthIrtDist
from crls import avgTotalRecalls
from crls import opR_crl
#from pybeh.pli import pli

adata = get_alldata()



listLength=24

stats = {}

stats['spc'] = spc(adata['recalls'], subjects=adata['subject'], listLength=listLength)
stats['pfr'] = np.array(pfr(adata['recalls'], subjects=adata['subject'], listLength=listLength))
stats['fRT'] = firstRecallDist(adata['rectimes'])

stats['crp'] = crp(adata['recalls'], adata['subject'], 24, 8)
stats['crp1'] = crp(adata['recalls'][:, 0:2], adata['subject'], 24, 8)

#pli = np.array(pli(intrusions, subjects=sessions, per_list=True))

from crls import get_irts, R_crl, op_crl

adata['irts'] = get_irts(adata['rectimes']*1000, adata['recalls'], 24)

stats['irt0']=nthIrtDist(adata['irts'], n=1)

stats['opcrl'] = op_crl(adata['irts'])
stats['rcrl'] = R_crl(adata['irts'])
stats['opR_crl'] = opR_crl(adata['irts'], adata['subject'])

stats['avgTotalRecalls'] = avgTotalRecalls(adata['recalls'])

for key in stats:
    stats[key] = stats[key].tolist()

with open('/Users/lumdusislife/Desktop/IRT/pyCMR2/IRT_Optimizations/target_stats.json', 'w') as outfile:
    json.dump(stats, outfile)
