#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 20:31:42 2020

@author: wangxinhua
"""
from mpi4py import MPI 
import os
from astropy.io import fits
import numpy as np
import re
import imageio
import time
from numba import vectorize
import cupy as cp
import multiprocessing


def test(fileslist):
    for i in fileslist:
        index = int(re.findall('\d+',i)[0])
        data = fits.open(os.path.join(path,i))[0].data
        cube[index,:,:] = data
        
    return cube

if __name__ == '__main__':
    '''start = time.time()
    global path
    path = r'/home/wangxinhua/20190518/HA/12741/010704/B050/010704/'
    files = os.listdir(path)    
    base = files[0]
    first = int(re.findall('\d+',base)[0])
    numb = len(files)
    basedata = fits.open(os.path.join(path,base))[0].data
    files.remove(base)
    comm_size = multiprocessing.cpu_count()
    nums = len(files)/comm_size
    [rcxsize,rcysize] = basedata.shape
    global cube
    cube = np.zeros([numb,rcxsize,rcysize], dtype = np.float32)
    pool = multiprocessing.Pool(comm_size)
    a = []
    for j in range(comm_size):
        a.append(files[int(j*nums):int((j+1)*nums)])
    data = np.zeros([numb,rcxsize,rcysize], dtype = np.float32)
    for i in pool.imap_unordered(test, a):
        data += i
    data[first,:,:] = basedata
    print(time.time()-start)'''
    start = time.time()
    global path
    path = r'/home/wangxinhua/20190518/HA/12741/010704/B050/010704/'
    files = os.listdir(path)    
    base = files[0]
    first = int(re.findall('\d+',base)[0])
    numb = len(files)
    basedata = fits.open(os.path.join(path,base))[0].data
    [rcxsize,rcysize] = basedata.shape
    global cube
    cube = np.zeros([numb,rcxsize,rcysize], dtype = np.float32)
    files.remove(base)
    print(test(files))
    print(time.time()-start)