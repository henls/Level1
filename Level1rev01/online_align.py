#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:26:23 2020
rev01
@author: wangxinhua
"""

import numpy as np
from astropy.io import fits
import scipy.fftpack as fft
import cupy as cp
from xyy_lib import xyy_lib as xyy
import os
from skimage import filters
import json
import re
import matplotlib.pyplot as plt
import time


def align():
    f = open(r"/home/wangxinhua/level1/Level1/json.txt",'r')
    para = json.load(f)
    f.close()
    rcxsize = int(para['rcxsize'])
    rcysize = int(para['rcysize'])
    corstart = re.findall('\d+',para['corstart'])
    corstart = [int(i) for i in corstart]
    corsize = re.findall('\d+',para['corsize'])
    corsize = [int(i) for i in corsize]
    flated_path = para['flated_path']
    sobel = int(para['sobel'])
    path = para['path']
    only_align_no_luckyimage = int(para['only_align_no_luckyimage'])
    redrive = para['redrive']
    only_align_no_luckyimage_path = para['only_align_no_luckyimage_path']
    pfstart = re.findall('\d+',para['pfstart'])
    pfstart = [int(i) for i in pfstart]
    pfsize = re.findall('\d+',para['pfsize'])
    pfsize = [int(i) for i in pfsize]
    lucky_align_path = para['lucky_align_path']
    win=xyy.win(int(pfsize[0]),int(pfsize[1]),0.5,winsty='hann')     #----窗函数
    diameter = float(para['diameter'])
    wavelen = float(para['wavelen'])
    pixsca = float(para['pixsca'])
    fsp = float(para['fsp'])
    srstx = int(para['srstx'])
    srsty = int(para['srsty'])
    srxsize = int(para['srxsize'])
    srysize = int(para['srysize'])
    postprocess_flag = int(para['postprocess_flag'])
    srsize = int(para['srsize'])
    winsr=xyy.win(srsize,srsize, 0.5, winsty='hann')
    diaratio = float(para['diaratio'])
    start_r0 = float(para['start_r0'])
    step_r0 = float(para['step_r0'])
    maxfre=wavelen*10.0**(-10.0)/(2.0*diameter*pixsca)*(180.0*3600.0/np.pi)
    filename = para['filename']
    sitfdata=fits.getdata(filename)
    gussf=xyy.gaussf2d(rcxsize,rcysize,1.5)
    infrq=(pfsize[0]//2)*0.05/maxfre
    otfrq=(pfsize[0]//2)*0.10/maxfre
    #做对齐
    #读预处理后的数据做对齐
    proceed_path = os.path.join(redrive,path[1:])
    datapath, flatpath, darkpath = xyy.path_paser(proceed_path)
    for i in datapath:
        data_path_fits = os.listdir(i)
        numb = len(data_path_fits)
        try:
            assert numb == 100
        except Exception as e:
            print('You are working on the last set of data')
        cube = np.empty([numb,rcxsize,rcysize], dtype = np.float32)
        try:
            data_dir_fitstmp = os.path.join(i,data_path_fits[0])
        except Exception as e:
            print('warning:目录'+i+'下没有fits文件')
            continue
        ini = xyy.readfits(data_dir_fitstmp)[0]
        initmp = ini[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
        #initmp_gpu = cp.asarray(initmp) 
        print('basefile:'+ data_dir_fitstmp)
        if sobel == 1:
            initmp = filters.sobel(filters.gaussian(initmp,5.0))
        t = 0
        #align 
        for j in data_path_fits:
            head=fits.getheader(os.path.join(i,j))
            if t !=0:
                data = xyy.readfits(i+"/"+j)[0]
                datatmp = data[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
                if sobel == 1:
                    datatmp = filters.sobel(filters.gaussian(datatmp,5.0))
                #datatmp_gpu = cp.asarray(datatmp)
                cc,corr = xyy.corrmaxloc_gpu(initmp,datatmp)
                #cc,corr = xyy.corrmaxloc(initmp,datatmp)
                tmp = xyy.imgshift(data,[-cc[0],-cc[1]])#对齐后的图
                if only_align_no_luckyimage == 1:
                    #不选帧，直接叠加
                    print('不选帧对齐模式')
                    ini += tmp
                else:
                    #print('选帧后对齐模式')
                    cube[t,:,:] = tmp[0:rcxsize,0:rcysize]
                    cubepf=cube[:,pfstart[0]:pfstart[0]+pfsize[0],pfstart[1]:pfstart[1]+pfsize[1]]
                    cubemean=np.mean(cubepf, axis=0)
                    psdcube = np.empty([numb,pfsize[0],pfsize[1]], dtype=np.float32) 
                    
                    for nn in range(numb):
                        tmp=cubepf[nn,:,:].copy()
                        meantmp=np.mean(tmp)
                        tmp=(tmp-meantmp)*win+meantmp
                        psd=np.abs(fft.fftshift(fft.fft2(tmp)))**2
                        psd=(psd/psd[pfsize[0]//2,pfsize[1]//2]).astype(np.float32)
                        psdcube[nn,:,:]=psd   
                    psdmean=np.mean(psdcube, axis=0)
                    psdcube=psdcube/psdmean
                    [Y,X]=np.meshgrid(np.arange(pfsize[1]),np.arange(pfsize[0])) 
                    dist=((X-pfsize[0]//2)**2.0+(Y-pfsize[1]//2)**2.0)**0.5
                    ring=np.where((dist>=infrq)&(dist<=otfrq), 1.0, 0.0).astype(np.float32)
                    psdcube=psdcube*ring
                    ringcube=np.mean(np.mean(psdcube, axis=1),axis=1)
                    index0=np.argsort(ringcube)[::-1]
                    #---------------------------------------------------------------------------------------
                    #--------------------------------  取排序前**帧, 再次相关对齐，叠加   
                    cubesort0=cube.copy()[index0][0:int(fsp*numb),:,:]
                    ini=np.mean(cubesort0, axis=0).astype(np.float32)
                    initmp=ini[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
                    if sobel==1:
                        initmp=filters.sobel(filters.gaussian(initmp,5.0))      
                      
                    
                    # ----------------------   对齐   
                    for nn in range(cubesort0.shape[0]):                        
                        data=cubesort0[nn,:,:].copy()
                        datatmp=data[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
                        if sobel==1:
                            datatmp=filters.sobel(filters.gaussian(datatmp,5.0))
                                  
                        #datatmp_gpu=cp.asarray(datatmp)
                        cc,corr=xyy.corrmaxloc_gpu(initmp, datatmp)
                        #cc,corr = xyy.corrmaxloc(initmp,datatmp)
                        ####cc,corr=xyy.corrmaxloc(initmp, datatmp)
                        
                        tmp=xyy.imgshift(data,[-cc[0],-cc[1]])
                        cubesort0[nn,:,:]=tmp
                        
                    averg=np.mean(cubesort0, axis=0).astype(np.float32)#叠加
                    
            t +=1
        if only_align_no_luckyimage == 1:
            averg = ini/t
        #----------------------------    选帧（1计算功率谱，2环带积分，3排序）
        
        #.................................................
        aligned_path = i+'/aligned'
        try:
            print('location of aligned:'+path+os.path.splitdrive(aligned_path)[1])
        except Exception as e:
            print('location of aligned:'+aligned_path)
        if only_align_no_luckyimage == 1:
            try:
                os.mkdir(path+os.path.splitdrive(aligned_path)[1])
            except Exception as e:
                print('warning:'+aligned_path+'existed')
            # closed 
            xyy.writefits(path+os.path.splitdrive(aligned_path)[1]+'/'+'aligned.fits',initmp/len(data_path_fits))
            #xyy.writefits(os.path.join(redrive,i,'aligned','aligned.fits'),initmp/len(data_path_fits))
        else:
            try:
                os.mkdir(path+os.path.splitdrive(aligned_path)[1])
            except Exception as e:
                #print(path+aligned_path+'existed')
                xyy.mkdir(aligned_path)
            #closed
            xyy.writefits(aligned_path+'/'+'aligned.fits',averg)
            #xyy.writefits(os.path.join(redrive,i,'aligned','aligned.fits'),averg)
        #退卷积
        if postprocess_flag == 1:
            cubesr=cube[:,srstx:srstx+srxsize,srsty:srsty+srysize]
            r0,index=xyy.cubesrdevr0(cubesr,srsize,winsr,sitfdata,diameter,diaratio,maxfre,0.00,0.06,start_r0,step_r0)
            sitf=xyy.GetSitf(sitfdata,maxfre,rcxsize,index)
            img=xyy.ImgPSDdeconv(averg,sitf)
                
            head['CODE2'] = r0
                
            result=xyy.ImgFilted(img,gussf)
                
            result=result/np.median(result)*np.median(averg)
            try:
                fitsname = redrive+os.path.splitdrive(aligned_path)[1]+'/'+'post_aligned.fits'
                xyy.mkdir(redrive+os.path.splitdrive(aligned_path)[1])
            except Exception as e:
                xyy.mkdir(os.path.join(redrive,i,'aligned'))
                fitsname = os.path.join(redrive,i,'aligned','post_aligned.fits')
            xyy.writefits(fitsname,result.astype(np.float32),head)
            #plt.imshow(result)
    print('align is over')
if __name__ == "__main__":
    start = time.time()
    align()
    print('elapse:',time.time()-start)
