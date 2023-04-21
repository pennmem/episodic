#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:02:55 2020

@author: lumdusislife
"""

import os

base = '/home1/shai.goldman/pyCMR2/IRT_Optimizations/'


for direc in ['outfiles/', 'noise_files/']:
    for file in os.listdir(base + direc):
        try:
            os.remove(base+direc+file)
        except OSError:
            print('OSError at %s' % file)
            pass
    print ('removed %s!' % direc)
