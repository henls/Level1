# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 19:45:22 2019

@author: a
"""
import numpy as np
from astropy.io import fits
import scipy.fftpack as fft
import cupy as cp
from xyy_lib import xyy_lib as xyy
import os
from skimage import filters
import json
import time
import matplotlib.pyplot as plt


def flat():
    f = open(r"/home/wangxinhua/level1/Level1/json.txt",'r')
    para = json.load(f)
    f.close()
    path = para['path']
    flated_path = para['flated_path']
    redrive = para['redrive']
    subpaths = os.listdir(path)
    flatpath=[]
    darkpath=[]
    datapath=[]
    
    for i in range(len(subpaths)):   
        subpath=os.path.join(path,subpaths[i])
    
        if ('F' in subpaths[i]) or ('f' in subpaths[i]) :        
            flatpath.append(subpath)
        elif ('D' in subpaths[i]) or ('d' in subpaths[i]): 
            darkpath.append(subpath)
        else:
            datapath.append(subpath)
            
    #----------------
         
    if  len(datapath)==0 :
        print('没有观测数据，停止数据处理！') 
        
    print('观测数据文件夹:', datapath)
    
    #----------------------  平场     
            
    if  len(flatpath)==0 :
        print('没有平场数据！请输入邻近观测日的平场数据路径！')
        flatpath=input('请输入路径（格式例如：H:\20190112\HA\FLAT00）：')
        
        
    print('平场数据文件夹:', flatpath)
    
    #-------------------    暗场 
    
    if  len(darkpath)==0 :
        print('没有暗场数据！')
        darkpath=input('请输入暗场的路径（格式例如：E:\dark20180312）：')
    
        
    print('暗场数据文件夹:', darkpath)    
    print()    
    
    
    #========================================================================
    redarkpath=os.path.join(redrive,os.path.splitdrive(darkpath[0])[1])
    xyy.mkdir(redarkpath)
        
    darkfile=os.path.join(redarkpath,'dark.fits')
    #----------------------
    dark = xyy.readfits(darkfile)[0]
    print('开始计算平场！') 
    #xyy.nvst_dirsandfiles_path(path)
    dirs = xyy.nvst_dirsandfiles_path(path)
    roots = dirs[0]
    fitsfile = dirs[1]
    t = 0
    for i in roots:
        if 'f' in i or 'F' in i :
            flat_root = i
            flat_fits = dirs[1][t]
        t+=1
    t = 0
    for i in roots:
        if 'f' not in i and 'd' not in i and 'F' not in i and 'D' not in i:
            data_root = i
            data_fits = dirs[1][t]
        t+=1
    for i in flat_fits:
        xyy.mkdir(os.path.join(redrive,os.path.splitdrive(i)[1]))
        flatfile = os.path.join(redrive,os.path.splitdrive(i)[1])
        if os.path.exists(flatfile+'\\'+'flat.fits') != True:
            addmean = np.array(xyy.dirfitsaddmean(i),dtype=np.float32)#平均平场
            xyy.writefits(flatfile+'\\'+'flat.fits',addmean)
        else:
            print('平场已经计算过')
            addmean = xyy.readfits(flatfile+'\\'+'flat.fits')[0]
        t = 0
        for j in data_fits:
            #print('正在处理第'+str(t)+'组')
            if 'B050' in j and 'B050' in i:
                datafits = os.listdir(j)
                for k in datafits:
                    if os.path.exists(os.path.join(redrive,os.path.splitdrive(j)[1])+'\\'+k) != True:
                        xyy.mkdir(os.path.join(redrive,os.path.splitdrive(j)[1]))
                        datatmp = np.array(xyy.readfits(os.path.join(j,k))[0],dtype = np.float32)
                        xyy.writefits(os.path.join(redrive,os.path.splitdrive(j)[1])+'\\'+k,np.array((datatmp-dark)/(addmean-dark)*np.median(addmean-dark),dtype=np.float32))
                        print('在处理的数据：'+os.path.join(j,k))
                        print('使用的flat：'+flatfile+'\\'+'flat.fits')
                        print(os.path.join(redrive,os.path.splitdrive(j)[1])+'\\'+k)
            elif 'CENT' in j and 'CENT' in i:
                datafits = os.listdir(j)
                for k in datafits:
                    if os.path.exists(os.path.join(redrive,os.path.splitdrive(j)[1])+'\\'+k) != True:
                        xyy.mkdir(os.path.join(redrive,os.path.splitdrive(j)[1]))
                        datatmp = np.array(xyy.readfits(os.path.join(j,k))[0],dtype = np.float32)
                        xyy.writefits(os.path.join(redrive,os.path.splitdrive(j)[1])+'\\'+k,np.array((datatmp-dark)/(addmean-dark)*np.median(addmean),dtype=np.float32))#归一化后做完平场处理的数据
                        print('在处理的数据：'+os.path.join(j,k))
                        print('使用的flat：'+flatfile+'\\'+'flat.fits')
                        print(os.path.join(redrive,os.path.splitdrive(j)[1])+'\\'+k)
            elif 'R050' in j and 'R050' in i:
                datafits = os.listdir(j)
                for k in datafits:
                    if os.path.exists(os.path.join(redrive,os.path.splitdrive(j)[1])+'\\'+k) != True:
                        xyy.mkdir(os.path.join(redrive,os.path.splitdrive(j)[1]))
                        datatmp = np.array(xyy.readfits(os.path.join(j,k))[0],dtype = np.float32)
                        xyy.writefits(os.path.join(redrive,os.path.splitdrive(j)[1])+'\\'+k,np.array((datatmp-dark)/(addmean-dark)*np.median(addmean),dtype=np.float32))
                        print('在处理的数据：'+os.path.join(j,k))
                        print('使用的flat：'+flatfile+'\\'+'flat.fits')
                        print(os.path.join(redrive,os.path.splitdrive(j)[1])+'\\'+k)
            t+=1
            
            
if __name__=='__main__':
    start = time.time()
    flat()
    print('elapse time:'+str(time.time()-start))