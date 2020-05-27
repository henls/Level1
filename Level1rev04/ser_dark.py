# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 16:14:13 2019

@author: a
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

def dark():
    f = open(r"/home/wangxinhua/level1/Level1rev04/json.txt",'r')
    para = json.load(f)
    f.close()
    path = para['path']
    redrive = para['redrive']
    dark_flag = int(para['dark_flag'])
    flat_flag = int(para['flat_flag'])
    darked_path = para['darked_path']
    subpaths = os.listdir(path)
    darkpath = []
    datapath = []
    for i in range(len(subpaths)):
        subpath = os.path.join(path,subpaths[i])
        if ('D' in subpaths[i]) or ('d' in subpaths[i]):
            darkpath.append(subpath)
        elif ('F' not in subpaths[i]) and ('f' not in subpaths[i]):
            datapath.append(subpath)
    if len(darkpath)==0:
        print('没有暗场数据！')
        darkpath = input('请输入暗场的路径（格式例如：E:\dark20180312）：')
    elif len(datapath)==0:
        print('没有观测数据，停止数据处理！')
    print('观测数据文件夹：',datapath)
    print('使用的暗场数据文件夹：',darkpath)
    print('开始计算暗场！')
    
    redarkpath=redrive+os.path.splitdrive(darkpath[0])[1]
    print(redarkpath)
    xyy.mkdir(redarkpath)
    darkfile=os.path.join(redarkpath,'dark.fits')
    if not os.path.exists(darkfile):
        dark= np.array(xyy.dirfitsaddmean(darkpath[0]),dtype=np.float32)
        xyy.writefits(darkfile,dark)
        print('暗场计算完毕')
    else:

        print('暗场已计算过！')
    if dark_flag == 1 and flat_flag == 0:
        dark = xyy.readfits(redarkpath+'\\'+'dark.fits')[0]
        print('开始计算只做暗场处理的请求')
        try:
            xyy.mkdir(darked_path)
        except Exception as e:
            print('folder has existed')
        #读取原数据
        dirs = xyy.nvst_dirsandfiles_path(path)
        roots = dirs[0]
        fitsfile = dirs[1]
        t = 0
        for i in roots:
            if 'f' not in i and 'd' not in i and 'F' not in i and 'D' not in i:
                data_root = i
                data_fits = dirs[1][t]
            t+=1
        t = 0
        for i in data_fits:
            files = os.listdir(i)
            for j in files:
                savepath =darked_path+os.path.splitdrive(i)[1]
                xyy.mkdir(savepath)
                print('正在计算第'+str(t)+'组')
                xyy.writefits(savepath+'\\'+j,np.array(xyy.readfits(os.path.join(i,j))[0] - dark+32768,dtype = np.float32))
                t+=1
        print('处理完成，文件保存在：'+darked_path)


if __name__ =="__main__":
    start = time.time()
    dark()
    print('elapse:',str(time.time()-start))