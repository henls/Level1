#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:58:04 2020

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
import pyinotify
import platform
from shutil import copyfile


def online_dark(path):
    f = open(r"/home/wangxinhua/level1/Level1rev06/json.txt",'r')
    para = json.load(f)
    f.close()
    #path = para['path']#"/home/wangxinhua/20190518/HA"
     
    archivedarkdir = para['archivedarkdir']
    
    redrive = para['redrive']#"/home/wangxinhua/nvst"
    dark_flag = int(para['dark_flag'])
    
    darked_path = para['darked_path']
    datapath, flatpath, darkpath = xyy.path_paser(path)
    #path of a group of fits
    flag = 1
    try:
        os.makedirs(archivedarkdir)
    except Exception as e:
        print(e)
    today_time = time.strftime("%Y%m%d",time.localtime())
    workdir = path
    use_other_dark = 0
    t0 = 0   #count time
    
    while flag:
        if use_other_dark == 0 and t0 < 300:
            if darkpath != None:
                print('Using '+darkpath[0])
                flag = 0
                operating_sys = platform.system()
                if operating_sys == 'Linux':
                    try:
                        path = path[path.index('\\')+1:]
                    except ValueError:   
                        path = path[path.index('/')+1:]
                elif operating_sys == 'Windows':
                    path = path.split(':')[1]
                    try:
                        path = path[path.index('\\')+1:]
                    except ValueError:   
                        path = path[path.index('/')+1:]
                print(os.path.join(redrive,path,'Dark','dark.fits'))
                #decide which dark will be used    writting
                if os.path.exists(os.path.join(redrive,path,'Dark','dark.fits')):
                    print('dark have been calculated,pass')
                else:
                    for i in darkpath:
                        darkeddata = xyy.online_mean(i)
                        xyy.mkdir(os.path.join(redrive,path,'Dark'))
                        #print(os.path.join(redrive,path,'Dark'))
                        xyy.writefits(os.path.join(redrive,path,'Dark','dark.fits'),darkeddata)
                #copy dark file to a folder
                copyfile(os.path.join(redrive,path,'Dark','dark.fits'),archivedarkdir+'/'+today_time+'dark.fits')
                dark_log = open(r'/home/wangxinhua/Observation_log/'+today_time+'.log','a+')
                dark_log.writelines('\nused Dark:'+os.path.join(redrive,path,'Dark','dark.fits'))#which dark file is used     writting
                dark_log.close()
            else:
                datapath, flatpath, darkpath = xyy.path_paser(path)
                flag = 1
            time.sleep(1)
            t0 += 1
        else:
            #5min have no new dark file then use the latest dark file 
            #define a folder that record all dark fits
            latestdarkpath = os.listdir(archivedarkdir)[-1]
            latestdarkfile = archivedarkdir+'\\'+latestdarkpath
            dark_log = open(r'/home/wangxinhua/Observation_log/'+today_time+'.log','w')
            dark_log.writelines('used Dark:'+latestdarkfile)#which dark file is used     writting
            dark_log.close()
    print('Dark is over')
if __name__ == "__main__":
    start = time.time()
    while 1:
        #try:
        today_time = time.strftime("%Y%m%d",time.localtime())
        f = open(r'/home/wangxinhua/Observation_log/'+today_time+'.log','r')
        path = f.readlines()
        f.close()
        print('starting...............')
        print('processing '+path[0])
        if len(path[0]) != 0:
            online_dark(path[0]+'/HA')
            print(time.time()-start)
        #except Exception as e:
        #    print(e)
        #    pass
    
    