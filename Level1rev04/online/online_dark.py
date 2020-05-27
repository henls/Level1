# -*- coding: utf-8 -*-
import sys
import numpy as np
from astropy.io import fits
import scipy.fftpack as fft
import cupy as cp
from xyy_lib import xyy_lib as xyy
import os
from skimage import filters
import json


#process dark

def online_dark():
    f = open(r"/home/wangxinhua/level1/Level1/json.txt",'r')
    para = json.load(f)
    f.close()
    path = para['path']#"/home/wangxinhua/20190518/HA"
    redrive = para['redrive']#"/home/wangxinhua/nvst"
    dark_flag = int(para['dark_flag'])
    flat_flag = int(para['flat_flag'])
    darked_path = para['darked_path']
    datapath, flatpath, darkpath = xyy.path_paser(path)
    #path of a group of fits
    try:
        path = path.split(':')[1][1:]
    except Exception as e:
        path = path[1:]
    if os.path.exists(os.path.join(redrive,path,'Dark','dark.fits')):
        print('dark have been calculated,pass')
    else:
        for i in darkpath:
            darkeddata = xyy.online_mean(i)
            xyy.mkdir(os.path.join(redrive,path,'Dark'))
            #print(os.path.join(redrive,path,'Dark'))
            xyy.writefits(os.path.join(redrive,path,'Dark','dark.fits'),darkeddata)
    print('Dark is over')
if __name__ == "__main__":
    online_dark()
    