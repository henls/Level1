#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:14:31 2020

@author: wangxinhua
"""

import sys
import numpy as np
from astropy.io import fits
import scipy.fftpack as fft
import cupy as cp
from xyy_lib import xyy_lib as xyy
import os
from skimage import filters
import json
import time


def online_flat():
    f = open(r"/home/wangxinhua/level1/Level1/json.txt",'r')
    para = json.load(f)
    f.close()
    path = para['path']#"/home/wangxinhua/20190518/HA"
    redrive = para['redrive']#"/home/wangxinhua/nvst"
    dark_flag = int(para['dark_flag'])
    flat_flag = int(para['flat_flag'])
    darked_path = para['darked_path']
    datapath, flatpath, darkpath = xyy.path_paser(path)
    #mean flat
    darkdata = xyy.readfits(os.path.join(redrive,path[1:],'Dark','dark.fits'))[0]
    for i in flatpath:
        if os.path.exists(os.path.join(redrive,i[1:],'flat.fits')):
            print('flat have been calculated')
        else:
            xyy.mkdir(os.path.join(redrive,i[1:]))
            flatdata = xyy.online_mean(i)
            xyy.writefits(os.path.join(redrive,i[1:],'flat.fits'),flatdata)
            print(os.path.join(redrive,i[1:],'flat.fits'))
            #(data-dark)/(flat-dark)*max(flat-dark)
            for j in datapath:
                bandoff = i.split('/')[-1]
                if bandoff in j and bandoff in i:
                    datafitspath = os.listdir(j)
                    for k in datafitspath:
                        xyy.mkdir(os.path.join(redrive,j[1:]))
                        print(os.path.join(redrive,j[1:],k))
                        data = xyy.readfits(os.path.join(j,k))[0]
                        xyy.writefits(os.path.join(redrive,j[1:],k),(data-darkdata)/(flatdata-darkdata)*np.max(flatdata-darkdata))
                '''elif 'CENT' in j and 'CENT' in i:
                    datafitspath = os.listdir(j)
                    for k in datafitspath:
                        xyy.mkdir(os.path.join(redrive,j[1:]))
                        print(os.path.join(redrive,j[1:],k))
                        data = xyy.readfits(os.path.join(j,k))[0]
                        xyy.writefits(os.path.join(redrive,j[1:],k),(data-darkdata)/(flatdata-darkdata)*np.max(flatdata-darkdata))
                elif 'R050' in j and 'R050' in i:
                    datafitspath = os.listdir(j)
                    for k in datafitspath:
                        xyy.mkdir(os.path.join(redrive,j[1:]))
                        print(os.path.join(redrive,j[1:],k))
                        data = xyy.readfits(os.path.join(j,k))[0]
                        xyy.writefits(os.path.join(redrive,j[1:],k),(data-darkdata)/(flatdata-darkdata)*np.max(flatdata-darkdata))'''
    print('flat is over')
if __name__ == '__main__':
    start = time.time()
    online_flat()
    print('elapse:',time.time()-start)