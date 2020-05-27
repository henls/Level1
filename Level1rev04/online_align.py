#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:26:23 2020
rev01
@author: wangxinhua
2020/1/9    data cache in GPU in one circle
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
import sys
import numba as nb
from numba import cuda
from mpi4py import MPI


def align():
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    with cp.cuda.Device(comm_rank%4):
        f = open(r"/home/wangxinhua/level1/Level1rev04/json.txt",'r')
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
        win=xyy.win_gpu(int(pfsize[0]),int(pfsize[1]),0.5,winsty='hann')     #----窗函数
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
        winsr=xyy.win_gpu(srsize,srsize, 0.5, winsty='hann')
        diaratio = float(para['diaratio'])
        start_r0 = float(para['start_r0'])
        step_r0 = float(para['step_r0'])
        maxfre=wavelen*10.0**(-10.0)/(2.0*diameter*pixsca)*(180.0*3600.0/np.pi)
        filename = para['filename']
        sitfdata=cp.array(fits.getdata(filename),'<f4')
        gussf=xyy.gaussf2d_gpu(rcxsize,rcysize,1.5)
        infrq=(pfsize[0]//2)*0.05/maxfre
        otfrq=(pfsize[0]//2)*0.10/maxfre
        #做对齐
        #读预处理后的数据做对齐
        try:
            proceed_path = os.path.join(redrive,path[2:])
            proceed_path.split(':')[1]
        except Exception as e:
            proceed_path = os.path.join(redrive,path[1:])
        datapath, flatpath, darkpath = xyy.path_paser(proceed_path)
        #############################################################################
        
        #if comm_size >= len(datapath):
        #    nums = len(datapath)-1
        #else:
        nums = len(datapath)/comm_size
        if comm_rank == 0:
            a = []
            for j in range(comm_size):
                a.append(datapath[int(j*nums):int((j+1)*nums)])
        else:
            a = None
        local_data = comm.scatter(a, root=0)
        #############################################################################
        
        for i in local_data:
            
            data_path_fits = os.listdir(i)
            numb = len(data_path_fits)
            try:
                assert numb == 100
            except Exception as e:
                print('You are working on the last set of data')
            #cube = cuda.shared.array([numb,rcxsize,rcysize], dtype = cp.float32)
            cubedata = cp.empty([numb,rcxsize,rcysize], dtype = cp.float32)
            try:
                data_dir_fitstmp = os.path.join(i,data_path_fits[0])
            except Exception as e:
                print('warning:目录'+i+'下没有fits文件')
                continue
            cubedata = cp.array(xyy.cubedata(i, data_path_fits,rcxsize,rcysize),dtype='<f4')
            
            
            ini = cubedata[0,:,:]
            initmp = ini[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
            #initmp_gpu = cp.asarray(initmp) 
            print('basefile:'+ data_dir_fitstmp)
            if sobel == 1:
                initmp = filters.sobel(filters.gaussian(initmp,5.0))
            
            t = 0
            #align 
            head=fits.getheader(os.path.join(i,data_path_fits[0]))
            for j in range(1,numb):
                data = cubedata[j,:,:]
                datatmp = data[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
                    
                if sobel == 1:
                    datatmp = filters.sobel(filters.gaussian(datatmp,5.0))
                #datatmp_gpu = cp.asarray(datatmp)
                cc,corr = xyy.corrmaxloc_gpu(initmp,datatmp)
                
                tmp = xyy.imgshift_gpu(data,[-cc[0],-cc[1]])#对齐后的图
                
                if only_align_no_luckyimage == 1:
                    #不选帧，直接叠加
                    print('不选帧对齐模式')
                    ini += tmp
                else:
                    #print('选帧后对齐模式')
                    #100,1024,1028
                    
                    cubedata[j,:,:] = tmp[0:rcxsize,0:rcysize]
            
            
            cubepf=cubedata[:,pfstart[0]:pfstart[0]+pfsize[0],pfstart[1]:pfstart[1]+pfsize[1]]
            cubemean=cp.mean(cubepf, axis=0)
            psdcube = cp.empty([numb,pfsize[0],pfsize[1]], dtype=cp.float32) 
                        
            for nn in range(numb):
                tmp=cubepf[nn,:,:].copy()
                meantmp=cp.mean(tmp)
                tmp=(tmp-meantmp)*win+meantmp
                psd=cp.abs(cp.fft.fftshift(cp.fft.fft2(tmp)))**2
                psd=(psd/psd[pfsize[0]//2,pfsize[1]//2]).astype(cp.float32)
                psdcube[nn,:,:]=psd   
            psdmean=cp.mean(psdcube, axis=0)
            psdcube=psdcube/psdmean
            [Y,X]=cp.meshgrid(cp.arange(pfsize[1]),cp.arange(pfsize[0])) 
            dist=((X-pfsize[0]//2)**2.0+(Y-pfsize[1]//2)**2.0)**0.5
            ring=cp.where((dist>=infrq)&(dist<=otfrq), 1.0, 0.0).astype(cp.float32)
            psdcube=psdcube*ring
            ringcube=cp.mean(cp.mean(psdcube, axis=1),axis=1)
            index0=cp.argsort(ringcube)[::-1]
                #---------------------------------------------------------------------------------------
                #--------------------------------  取排序前**帧, 再次相关对齐，叠加 
                #################
                        
            #cube = cp.asnumpy(cube)
            #index0 = cp.asnumpy(index0)
                #################
                        
                #cubesort0=cube.copy()[index0][0:int(fsp*numb),:,:]
            cubesort0=cubedata.copy()[index0][0:int(fsp*numb),:,:]
                ########################
            #cubesort0 = cp.array(cubesort0)
            #cube = cp.array(cube,dtype='<f4')
                ########################
                        
            ini=cp.mean(cubesort0, axis=0).astype(cp.float32)
            initmp=ini[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
            if sobel==1:
                initmp=filters.sobel(filters.gaussian(cp.asnumpy(initmp),5.0))      
                  
                        
                        # ----------------------   对齐   
            for nn in range(cubesort0.shape[0]):                        
                data=cubesort0[nn,:,:].copy()
                datatmp=data[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
                if sobel==1:
                    datatmp=filters.sobel(filters.gaussian(cp.asnumpy(datatmp),5.0))
                                      
                            #datatmp_gpu=cp.asarray(datatmp)
                cc,corr=xyy.corrmaxloc_gpu(initmp, datatmp)
                            #cc,corr = xyy.corrmaxloc(initmp,datatmp)
                            ####cc,corr=xyy.corrmaxloc(initmp, datatmp)
                            
                tmp=xyy.imgshift_gpu(data,[-cc[0],-cc[1]])
                cubesort0[nn,:,:]=tmp
                        #print(tmp)
            
            averg=cp.mean(cubesort0, axis=0).astype(cp.float32)#叠加
                      
                
            if only_align_no_luckyimage == 1:
                averg = ini/t
            #----------------------------    选帧（1计算功率谱，2环带积分，3排序）
            
            #.................................................
            aligned_path = '/home/wangxinhua/Desktop/align'+'/'.join(path.split('/')[path.split('/').index('Desktop')+1:])+'/aligned'
            try:
                print('location of aligned:'+path+os.path.splitdrive(aligned_path)[1])
            except Exception as e:
                print('location of aligned:'+aligned_path)
            if only_align_no_luckyimage == 1:
                try:
                    os.mkdir(path+os.path.splitdrive(aligned_path)[1])
                except Exception as e:
                    print('warning:'+aligned_path+'existed')
                 
                xyy.writefits(path+os.path.splitdrive(aligned_path)[1]+'/'+'aligned.fits',cp.asnumpy(initmp/len(data_path_fits)))
                
            else:
                try:
                    os.mkdir(path+os.path.splitdrive(aligned_path)[1])
                except Exception as e:
                    #print(path+aligned_path+'existed')
                    xyy.mkdir(aligned_path)
                
                xyy.writefits(aligned_path+'/'+'aligned.fits',cp.asnumpy(averg))
                
            #退卷积
            if postprocess_flag == 1:
                cubesr=cubedata[:,srstx:srstx+srxsize,srsty:srsty+srysize]
                
                try:
                    r0,index=xyy.cubesrdevr0_gpu(cubesr,srsize,winsr,sitfdata,diameter,diaratio,maxfre,0.00,0.06,start_r0,step_r0)
                except Exception as e:
                    #print(cube)
                    print(cubesr)
                    sys.exit()
                sitf=xyy.GetSitf_gpu(sitfdata,maxfre,rcxsize,index)
        
                img=xyy.ImgPSDdeconv_gpu(averg,sitf)
                    
                head['CODE2'] = r0
                    
                result=xyy.ImgFilted_gpu(img,gussf)
                    
                result=result/np.median(cp.asnumpy(result))*np.median(cp.asnumpy(averg))
                try:
                    fitsname = redrive+os.path.splitdrive(aligned_path)[1]+'/'+'post_aligned.fits'
                    xyy.mkdir(redrive+os.path.splitdrive(aligned_path)[1])
                except Exception as e:
                    xyy.mkdir(os.path.join(redrive,i,'aligned'))
                    fitsname = os.path.join(redrive,i,'aligned','post_aligned.fits')
                xyy.writefits(fitsname,cp.asnumpy(result).astype(np.float32),head)
                #plt.imshow(result)'''
   # print('align is over')
if __name__ == "__main__":
    start = time.time()
    align()
    print('elapse:'+str(time.time()-start))