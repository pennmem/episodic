#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:01:06 2020

@author: lumdusislife
"""

OUTDIR = '/home1/shai.goldman/pyCMR2/IRT_Optimizations/outfiles/'

scores = []

generations = 150
populations = 1000

for gen in range(generations):
    for j in range(populations):
        scores.append(np.loadtxt(OUTDIR + '%stempfile%s.txt' % (gen, j)))
        
from matplotlib import pyplot

plt.plot(scores)
#plt.xticks(np.arange(.2,1,.2))
plt.xlabel('Mean Squared Error')
plt.ylabel('Iteration and Populations')
plt.show()