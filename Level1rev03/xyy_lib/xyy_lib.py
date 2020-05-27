# -*- coding: utf-8 -*-
"""


作者：向永源

说明:  cupy 函数输入变量是device变量，输出也是device变量！！！  
       切记！DEVICE中为防止FFT失败，输入前用 astype(np.xxxx) 进行类型转换！！！


更新：

2019-10-15

2019-11-22

2019-11-26

2019-12-26

2020-01-06 计算全部在GPU中计算

"""

import math
import numpy as np 
import astropy.io.fits as fits
import os
import numpy.fft as fft
import imageio
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

import scipy.ndimage as ndm

from collections import Counter

import numpy.random as rand
from numpy import sinc

import cupy as cp


####=================================================   文件操作

''' 读一帧简单格式的fits文件
参数： 文件名
返回： [数据，头文件] 
'''
def readfits(filename):
    data=fits.getdata(filename)
    header=fits.getheader(filename)
    return [data,header]

def readfits_gpu(filename):
    '''
    输入：文件名
    输出：[data， header]  GPU
    '''
    cp.cuda.Device(0).use()
    data = cp.array(fits.getdata(filename),dtype='<f4')
    header = cp.array(fits.getheader(filename),dtype='<f4')
    return [data, header]

''' 保存一帧简单格式的fits文件
参数： 文件名，数据，头文件
备注：hdr=fits.Header() 是默认变量，当调用函数时候无hdr,则默然创建一个头文件
调用： xyy.writefits(filename, data, hdr),  xyy.writefits(filename, data)
返回： FITS文件 
'''
def writefits(filename, data, hdr=fits.Header()):
    if os.path.exists(filename):
        os.remove(filename)
    fits.writeto(filename, data, hdr)
    
def writefits_gpu(filename, data, hdr=fits.Header()):
    '''
    输入：文件名、数据（GPU）
    '''
    cp.cuda.Device(0).use()
    if os.path.exists(filename):
        os.remove(filename)
    fits.writeto(filename,cp.asnumpy(data),hdr)
  
''' 寻找目录下所有的文件
参数：主路径，文件名包含的字符
返回：所有含指定字符串的文件列表 
'''
def file_search(dirname,filt):
    result=[]
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath=os.path.join(maindir, filename)
            if filt in apath:
                result.append(apath)
    return result


def file_search_gpu(dirname,filt):
    result=[]
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath=os.path.join(maindir, filename)
            if filt in apath:
                result.append(apath)
    return result
''' 寻找目录下所有的子目录
参数：主路径
返回：所有子目录的列表 
'''
def subdir_search(dirname):
    result=[]
    for maindir, subdir, file_name_list in os.walk(dirname):
        for subname in subdir:
            apath=os.path.join(maindir, subname)
            result.append(apath)
    return result

def subdir_search_gpu(dirname):
    result=[]
    for maindir, subdir, file_name_list in os.walk(dirname):
        for subname in subdir:
            apath=os.path.join(maindir, subname)
            result.append(apath)
    return result
''' 创建文件夹
参数：tmppath
'''
def mkdir(tmppath):
    if os.path.exists(tmppath):
       pass
    else:
       os.makedirs(tmppath)

def mkdir_gpu(tmppath):
    if os.path.exists(tmppath):
       pass
    else:
       os.makedirs(tmppath)
''' 目录里FITS文件打包
参数：dirname
返回：cube
'''
def  dirfitstocube(dirname):
    
    files = file_search(dirname,'.fits') 
    zsize = len(files)
    data = fits.getdata(files[0])
    xsize = data.shape[0]
    ysize = data.shape[1]  
    cube = np.empty([zsize,xsize,ysize], dtype = data.dtype)  
    for i in range(zsize):
        data = fits.getdata(files[i]) 
        cube[i,:,:] = data 
        print(i)
        
    return cube

def  dirfitstocube_gpu(dirname):
    '''
    输入：文件夹名
    输出：cube  gpu
    '''
    files = file_search(dirname,'.fits') 
    zsize = len(files)
    data = fits.getdata(files[0])
    xsize = data.shape[0]
    ysize = data.shape[1] 
    cp.cuda.Device(0).use()
    data = cp.array(data,dtype='<f4')
    cube = cp.empty([zsize,xsize,ysize], dtype = data.dtype)  
    for i in range(zsize):
        data = cp.array(fits.getdata(files[i]),dtype='<f4')
        cube[i,:,:] = data 
        print(i)
        
    return cube
'''  子目录所有FITS求平均，并保存为FITS
参数：path
返回：addmean
'''
def  dirfitsaddmean(path):
    
    files = file_search(path,'.fits')
    zsize = len(files)
    
    if zsize == 0:
        print('文件夹下没有FITS文件！！！')   
        addmean=[]
        
    if zsize >= 1:

        nn=zsize
        addmean=0
        for i in range(zsize):
           filename=files[i] 
           
           if not os.path.getsize(filename):
               os.remove(filename)
               nn=nn-1
               continue
                
           data = np.array(fits.getdata(filename),dtype = np.float32)
           addmean = addmean + data

        addmean = addmean/nn

    return addmean


def  dirfitsaddmean_gpu(path):
    cp.cuda.Device(0).use()
    files = file_search(path,'.fits')
    zsize = len(files)
    
    if zsize == 0:
        print('文件夹下没有FITS文件！！！')   
        addmean=[]
        
    if zsize >= 1:

        nn=zsize
        addmean=cp.array(0)
        for i in range(zsize):
           filename=files[i] 
           
           if not os.path.getsize(filename):
               os.remove(filename)
               nn=nn-1
               continue
                
           data = cp.array(fits.getdata(filename),dtype = '<f4')
           addmean = addmean + data

        addmean = addmean/nn

    return addmean
 
######+==========================================================   图像显示和电影     
       
''' 
显示一张图像
'''
def showimg(data):

    plt.close()
    mi=max([data.min(),data.mean()-3*data.std()])
    mx=min([data.max(),data.mean()+3*data.std()])

    plt.imshow(data,vmin=mi,vmax=mx,cmap='gray')
    
    return
            
def showimg_gpu(data):

    plt.close()
    mi=max([data.min(),data.mean()-3*data.std()])
    mx=min([data.max(),data.mean()+3*data.std()])

    plt.imshow(cp.asnumpy(data),vmin=mi,vmax=mx,cmap='gray')
    
    return
''' 三维CUBE做成GIF
参数：cube，gif_name, nx,ny(视频的大小), gap(取图像的间隔)
返回：GIF
'''
def  cubetogif(cube, gif_name, nx, ny, gap):
 
    size =  cube.shape 
    zsize = size[0]
    xsize = size[1]
    ysize = size[2]
    xv = min(xsize,nx)
    yv = min(ysize,ny)
    zv = zsize//gap
    frames = np.empty([zv,xv,yv], dtype = np.uint8) 
    for i in range(zv):  
        data = cube[i*gap,:,:].astype(np.float32)
        tmp = cv2.resize(data, (xv,yv), interpolation = cv2.INTER_CUBIC)
        tmp0 = ((tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))*255.0).astype(np.uint8)  
        frames[i,:,:] = tmp0
        imageio.mimsave(gif_name, frames, 'GIF', duration = 0.05)  
    return

def  cubetogif_gpu(cube, gif_name, nx, ny, gap):
    size =  cube.shape 
    zsize = size[0]
    xsize = size[1]
    ysize = size[2]
    xv = min(xsize,nx)
    yv = min(ysize,ny)
    zv = zsize//gap
    cp.cuda.Device(0).use()
    frames = cp.empty([zv,xv,yv], dtype = cp.uint8) 
    for i in range(zv):  
        data = cube[i*gap,:,:].astype(cp.float32)
        tmp = cv2.resize(cp.asnumpy(data), (xv,yv), interpolation = cv2.INTER_CUBIC)
        tmp = cp.array(tmp)
        tmp0 = ((tmp-cp.min(tmp))/(cp.max(tmp)-cp.min(tmp))*255.0).astype(cp.uint8)  
        frames[i,:,:] = tmp0
        imageio.mimsave(gif_name, cp.asnumpy(frames), 'GIF', duration = 0.05)  
    return
''' 三维CUBE做成GIF,添加水印
参数：cube，gif_name, gsize(画布大小)，gap(取图像的间隔), nfps
返回：GIF
备注：优于cubetogif 
'''
def  cubetogif2(cube, gif_name, gsize, gap, nfps):
    
    fig = plt.figure(figsize = (gsize, gsize))
    zsize = cube.shape[0]
    xsize = cube.shape[1]
    ysize = cube.shape[2]
    zv = zsize//gap
    frames = []
    for i in range(zv):
        fn = i*gap
        data = cube[fn,:,:]
        im = plt.imshow(cp.asnumpy(data), animated=True, cmap='gray')
        text = plt.text(xsize*0.15, ysize*0.85, '{:0>4d}'.format(fn), fontsize=18, style='italic', ha='left',va='bottom',wrap=True)   
        frames.append([im,text])
        
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True,repeat_delay=1000)
    writer = PillowWriter(fps=nfps, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(gif_name,writer=writer)
    return ani


''' 目录里FITS文件做成GIF,添加水印
参数：dirname，gif_name, gap(取图像的间隔), nfps
返回：GIF 
'''
def  dirfitstogif2(dirname, gif_name, gsize, gap, nfps):
    
    fig = plt.figure(figsize = (gsize, gsize))
    
    files = file_search(dirname,['.fits']) 
    zsize = len(files)
    head = fits.getheader(files[0])
    xsize = head['NAXIS1']
    ysize = head['NAXIS2']  
    zv = zsize//gap
    frames = []
    for i in range(zv):
        fn = i*gap
        data = fits.getdata(files[i]) 
        im = plt.imshow(data, animated=True, cmap='gray')
        text = plt.text(xsize*0.15, ysize*0.85, '{:0>4d}'.format(fn), fontsize=18, style='italic', ha='left',va='bottom',wrap=True)   
        frames.append([im,text])
        
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True,repeat_delay=1000)
    writer = PillowWriter(fps=nfps, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(gif_name,writer=writer)
    return ani


######=========================================================   图像相关和移动



''' 计算相关最大值位置
参数：参考图ini，目标图obj  ;   如果关键字win无，则默认win=1.0
返回：最大值坐标向量（目标相对于参考的位置向量）

'''
def corrmaxloc(ini, obj, win=1.0):
     
    xsize = ini.shape[0]
    ysize = ini.shape[1]
       
    initmp = (ini - np.mean(ini))*win
    inifft = fft.fft2(initmp)
    
    objtmp = (obj - np.mean(obj))*win
    objfft = fft.fft2(objtmp)
    
    corr = np.real(fft.fftshift(fft.ifft2(np.conj(objfft)*inifft)))
    maxid = np.where(corr == np.max(corr))
    shiftxy = [xsize//2-maxid[0][0], ysize//2-maxid[1][0]]
    
    return shiftxy, corr



''' 计算相关最大值位置 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  GPU    
参数：参考图ini_gpu，目标图obj_gpu  (device 中的变量);   如果关键字win无，则默认win=1.0
返回：最大值坐标向量（目标相对于参考的位置向量）

'''
def corrmaxloc_gpu(ini_gpu, obj_gpu, win=1.0):
    cp.cuda.Device(0).use()
    #ini_gpu = cp.array(ini_gpu,dtype='<f4')
    #obj_gpu = cp.array(obj_gpu,dtype='<f4')
    #print(ini_gpu)
    #print(obj_gpu)
    xsize = ini_gpu.shape[0]
    ysize = ini_gpu.shape[1]
    initmp = (ini_gpu - cp.mean(ini_gpu))*win
    inifft = cp.fft.fft2(initmp)
    objtmp = (obj_gpu - cp.mean(obj_gpu))*win
    objfft = cp.fft.fft2(objtmp)
    corr_gpu = cp.real(cp.fft.fftshift(cp.fft.ifft2(cp.conj(objfft)*inifft)))
    
    maxid = cp.where(corr_gpu == cp.max(corr_gpu))#最大值的索引
    shiftxy = [xsize//2-maxid[0][0], ysize//2-maxid[1][0]]
    
    return shiftxy, corr_gpu



''' 计算相关最大值位置(亚像元)
参数：参考图ini，目标图obj
返回：最大值坐标向量（目标相对于参考的位置向量）
'''
def corrmaxsubloc(ini, obj, win=1.0):
    
    xsize = ini.shape[0]
    ysize = ini.shape[1]
     
    initmp = (ini - np.mean(ini))*win
    inifft = fft.fft2(initmp)
#
    objtmp = (obj - np.mean(obj))*win
    objfft = fft.fft2(objtmp)
    
    corr = np.real(fft.fftshift(fft.ifft2(np.conj(objfft)*inifft)))
    maxid = np.where(corr == np.max(corr))
    dxy0 = [-xsize//2+maxid[0][0], -ysize//2+maxid[1][0]]
    
    dxy0=np.minimum(np.maximum(dxy0,[-xsize//5*2,-ysize//5*2]),[xsize//5*2,ysize//5*2])

    tmp = imgshift(obj,dxy0)

    objtmp = (tmp - np.mean(tmp))*win
    objfft = fft.fft2(objtmp)  
    corr = np.real(fft.fftshift(fft.ifft2(np.conj(objfft)*inifft)))
    maxid = np.where(corr == np.max(corr))

    nn=3
    index=np.maximum([maxid[0][0],maxid[1][0]],[nn//2,nn//2]) 
    tmp=corr[index[0]-nn//2:index[0]-nn//2+nn,index[1]-nn//2:index[1]-nn//2+nn]
    
    tmp=tmp-np.min(tmp)
    cent=centroid(tmp)

    ddxy=[-dxy0[0]+xsize//2-index[0]+nn//2-cent[0], -dxy0[1]+ysize//2-index[1]+nn//2-cent[1]]
    
    return ddxy, corr

def corrmaxsubloc_gpu(ini, obj, win=1.0):
    
    xsize = ini.shape[0]
    ysize = ini.shape[1]
     
    initmp = (ini - cp.mean(ini))*win
    inifft = cp.fft.fft2(initmp)
#
    objtmp = (obj - cp.mean(obj))*win
    objfft = cp.fft.fft2(objtmp)
    
    corr = cp.real(cp.fft.fftshift(cp.fft.ifft2(cp.conj(objfft)*inifft)))
    maxid = cp.where(corr == cp.max(corr))
    dxy0 = [-xsize//2+maxid[0][0], -ysize//2+maxid[1][0]]
    
    dxy0=cp.minimum(cp.maximum(dxy0,[-xsize//5*2,-ysize//5*2]),[xsize//5*2,ysize//5*2])

    tmp = imgshift_gpu(obj,dxy0)

    objtmp = (tmp - cp.mean(tmp))*win
    objfft = cp.fft.fft2(objtmp)  
    corr = cp.real(cp.fft.fftshift(cp.fft.ifft2(cp.conj(objfft)*inifft)))
    maxid = cp.where(corr == cp.max(corr))

    nn=3
    index=cp.maximum([maxid[0][0],maxid[1][0]],[nn//2,nn//2]) 
    tmp=corr[index[0]-nn//2:index[0]-nn//2+nn,index[1]-nn//2:index[1]-nn//2+nn]
    
    tmp=tmp-cp.min(tmp)
    cent=centroid(tmp)

    ddxy=[-dxy0[0]+xsize//2-index[0]+nn//2-cent[0], -dxy0[1]+ysize//2-index[1]+nn//2-cent[1]]
    
    return ddxy, corr
''' 相移定理,可用于亚像素平移
参数：img，shift  ( [dx,dy] )
返回：
'''
def  phaseshift(img,shift):
    
    tmp=img.copy() 
    fftimg=fft.fftshift(fft.fft2(tmp))
    
    xsize=img.shape[0]
    ysize=img.shape[1]
    [Y,X]=np.meshgrid(np.arange(ysize)-ysize//2,np.arange(xsize)-xsize//2) 
    
    tmp0=fftimg*np.exp(2.0*np.pi*(X*shift[0]/xsize+Y*shift[1]/ysize)*(-1j))
    result=fft.ifft2(fft.ifftshift(tmp0)).real

    return result

def  phaseshift_gpu(img,shift):
    
    tmp=img.copy() 
    fftimg=cp.fft.fftshift(cp.fft.fft2(tmp))
    
    xsize=img.shape[0]
    ysize=img.shape[1]
    [Y,X]=cp.meshgrid(cp.arange(ysize)-ysize//2,cp.arange(xsize)-xsize//2) 
    
    tmp0=fftimg*cp.exp(2.0*cp.pi*(X*shift[0]/xsize+Y*shift[1]/ysize)*(-1j))
    result=cp.fft.ifft2(cp.fft.ifftshift(tmp0)).real

    return result
''' 相移定理,可用于亚像素平移   ~~~~~~~~~~~  GPU          

参数：img_cupy ，shift  ( [dx,dy] )
返回：

结论：速度   phaseshift_cupy > imgsubshift > imgsubshift_cupy > phaseshift
'''
def  phaseshift_cupy(img_cupy,shift):
    
    fftimg_cupy=cp.fft.fftshift(cp.fft.fft2(img_cupy))
    
    xsize=img_cupy.shape[0]
    ysize=img_cupy.shape[1]
    [Y,X]=cp.meshgrid(cp.arange(ysize)-ysize//2,cp.arange(xsize)-xsize//2) 
    
    phas=cp.zeros([xsize,ysize],dtype=cp.complex64)
    phas.imag=-2.0*cp.pi*cp.add(X*shift[0]/xsize,Y*shift[1]/ysize)
    
    tmp0=cp.multiply(fftimg_cupy,cp.exp(phas))
    result_cupy=cp.fft.ifft2(cp.fft.ifftshift(tmp0)).real

    return result_cupy


'''  亚像素平移
参数：img，shift
返回：
备注： 结果和相移定理一毛一样!
'''
def  imgsubshift(img,shift):
    
    tmp = img.copy()
    fftimg = fft.fft2(tmp)
    tmp0 = ndm.fourier_shift(fftimg, shift)
    result = fft.ifft2(tmp0).real

    return result

def  imgsubshift_gpu(img,shift):
    
    tmp = img.copy()
    fftimg = cp.fft.fft2(tmp)
    tmp0 = ndm.fourier_shift(cp.asnumpy(fftimg), shift)
    result = cp.fft.ifft2(cp.array(tmp0)).real

    return result
'''  亚像素平移
参数：img_cupy，shift
返回：

'''
def  imgsubshift_cupy(img_cupy,shift):

    fftimg_cupy = cp.fft.fft2(img_cupy)    
    fftimg=cp.asnumpy(fftimg_cupy) 
    tmp0=ndm.fourier_shift(fftimg, shift)  
    tmp0_cupy=cp.asarray(tmp0)
    result_cupy=cp.fft.ifft2(tmp0_cupy).real
    
    return result_cupy


''' 二维数组的平移
参数：数组，[dx, dy]
返回：平移后的数组
备注：此处不宜加上@jit,运行速度不升反降
'''
def imgshift(img,dxy):
    
    imgout=np.copy(img)
    imgout=np.roll(imgout,int(dxy[0]),axis=0)
    imgout=np.roll(imgout,int(dxy[1]),axis=1)
    
    return imgout

def imgshift_gpu(img,dxy):
    
    imgout=cp.copy(img)
    imgout=cp.roll(imgout,int(dxy[0]),axis=0)
    imgout=cp.roll(imgout,int(dxy[1]),axis=1)
    
    return imgout
''' 二维数组的平移  ~~~~~~~~~~~~~  GPU
参数：img_cupy，[dx, dy]    (输入变量必须是  device 中的变量)  
  
返回：平移后的数组(device 变量) 

'''
def imgshift_cupy(img_cupy,dxy):
    
    imgout_cupy=cp.copy(img_cupy)
    imgout_cupy=cp.roll(imgout_cupy,int(dxy[0]),axis=0)
    imgout_cupy=cp.roll(imgout_cupy,int(dxy[1]),axis=1)
    
    return imgout_cupy


''' 三维数组对齐
参数：subcube，lxp, corsize, win相关的窗函数
返回：nsubcube
'''
def  cube_align(subcube,lxp,corsize,win=1.0):
    
    zsize = subcube.shape[0]
    xsize = subcube.shape[1]
    ysize = subcube.shape[2]

    corstart=[(xsize-corsize[0])//2,(ysize-corsize[1])//2]
    ini = lxp[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
    nsubcube = np.zeros([zsize,xsize,ysize], dtype = subcube.dtype) 
    
    for i in range(zsize):
        data = subcube[i,:,:].copy()
        obj = data[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]  
        cc,corr = corrmaxloc(ini, obj, win)
        nsubcube[i,:,:] = imgshift(data,[-cc[0],-cc[1]])
  
    return nsubcube

def  cube_align_gpu(subcube,lxp,corsize,win=1.0):
    
    zsize = subcube.shape[0]
    xsize = subcube.shape[1]
    ysize = subcube.shape[2]

    corstart=[(xsize-corsize[0])//2,(ysize-corsize[1])//2]
    ini = lxp[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
    nsubcube = cp.zeros([zsize,xsize,ysize], dtype = subcube.dtype) 
    
    for i in range(zsize):
        data = subcube[i,:,:].copy()
        obj = data[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]  
        cc,corr = corrmaxloc_gpu(ini, obj, win)
        nsubcube[i,:,:] = imgshift_gpu(data,[-cc[0],-cc[1]])
  
    return nsubcube
''' 三维数组对齐（两组）
参数：cube2（信噪比高），cube1（信噪比低），lxp2（参考图）, corsize, win相关的窗函数
返回：cube2， cube1
'''
def  twocube_align(subcube2,subcube1,lxp2,corsize,win=1.0):
    
    zsize = subcube2.shape[0]
    xsize = subcube2.shape[1]
    ysize = subcube2.shape[2]

    corstart=[(xsize-corsize[0])//2,(ysize-corsize[1])//2]
    
    ini = lxp2[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
    
    nsubcube2 = np.empty([zsize,xsize,ysize], dtype = subcube2.dtype) 
    nsubcube1 = np.empty([zsize,xsize,ysize], dtype = subcube2.dtype) 
    
    for i in range(zsize):
        
        data = subcube2[i,:,:].copy()
        obj = data[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
        
        cc,corr = corrmaxloc(ini, obj, win)
        
        nsubcube2[i,:,:] = imgshift(data,[-cc[0],-cc[1]])
        
        data = subcube1[i,:,:].copy()
        nsubcube1[i,:,:] = imgshift(data,[-cc[0],-cc[1]])
        
    return nsubcube2,nsubcube1


def  twocube_align_gpu(subcube2,subcube1,lxp2,corsize,win=1.0):
    
    zsize = subcube2.shape[0]
    xsize = subcube2.shape[1]
    ysize = subcube2.shape[2]

    corstart=[(xsize-corsize[0])//2,(ysize-corsize[1])//2]
    
    ini = lxp2[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
    
    nsubcube2 = cp.empty([zsize,xsize,ysize], dtype = subcube2.dtype) 
    nsubcube1 = cp.empty([zsize,xsize,ysize], dtype = subcube2.dtype) 
    
    for i in range(zsize):
        
        data = subcube2[i,:,:].copy()
        obj = data[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
        
        cc,corr = corrmaxloc_gpu(ini, obj, win)
        
        nsubcube2[i,:,:] = imgshift_gpu(data,[-cc[0],-cc[1]])
        
        data = subcube1[i,:,:].copy()
        nsubcube1[i,:,:] = imgshift_gpu(data,[-cc[0],-cc[1]])
        
    return nsubcube2,nsubcube1

''' 三维数组对齐（亚像元）
参数：cube2（信噪比高），cube1（信噪比低），lxp2（参考图）, corsize, win相关的窗函数
返回：cube2， cube1
'''
def  twocube_align_sub(subcube2,subcube1,lxp2,corsize,win=1.0):
    
    zsize = subcube2.shape[0]
    xsize = subcube2.shape[1]
    ysize = subcube2.shape[2]

    corstart=[(xsize-corsize[0])//2,(ysize-corsize[1])//2]
    ini = lxp2[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
    nsubcube2 = np.empty([zsize,xsize,ysize], dtype = subcube2.dtype) 
    nsubcube1 = np.empty([zsize,xsize,ysize], dtype = subcube2.dtype) 
    
    for i in range(zsize):  
        data = subcube2[i,:,:].copy()
        obj = data[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
        cc,corr = corrmaxsubloc(ini, obj, win)
        
        nsubcube2[i,:,:] = imgsubshift(data,[-cc[0],-cc[1]])
        data = subcube1[i,:,:].copy()
        nsubcube1[i,:,:] = imgsubshift(data,[-cc[0],-cc[1]])
        
    return nsubcube2,nsubcube1

def  twocube_align_sub_gpu(subcube2,subcube1,lxp2,corsize,win=1.0):
    
    zsize = subcube2.shape[0]
    xsize = subcube2.shape[1]
    ysize = subcube2.shape[2]

    corstart=[(xsize-corsize[0])//2,(ysize-corsize[1])//2]
    ini = lxp2[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
    nsubcube2 = cp.empty([zsize,xsize,ysize], dtype = subcube2.dtype) 
    nsubcube1 = cp.empty([zsize,xsize,ysize], dtype = subcube2.dtype) 
    
    for i in range(zsize):  
        data = subcube2[i,:,:].copy()
        obj = data[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
        cc,corr = corrmaxsubloc_gpu(ini, obj, win)
        
        nsubcube2[i,:,:] = imgsubshift_gpu(data,[-cc[0],-cc[1]])
        data = subcube1[i,:,:].copy()
        nsubcube1[i,:,:] = imgsubshift_gpu(data,[-cc[0],-cc[1]])
        
    return nsubcube2,nsubcube1


    
######========================================   窗函数和滤波器



'''
#---说明： 返回图像振幅和相位  
#---参数： img   
#---返回： modul，phase
'''
def imgmodpha(img):
    
    [xsize,ysize]=img.shape
    sp=fft.fftshift(fft.fft2(img))/xsize/ysize
    modul=np.abs(sp)
    phase=np.angle(sp)
    
    return modul,phase  


def imgmodpha_gpu(img):
 
    img = img
    [xsize,ysize]=img.shape
    sp = cp.divide(cp.divide(cp.fft.fftshift(cp.fft.fft2(img)),xsize),ysize)
    modul = cp.abs(sp)
    phase = cp.angle(sp)
    
    return modul, phase
'''
#---说明： 振幅和相位返回图像  
#---参数： modul，phase 
#---返回： img 
'''
def modphaimg(modul,phase):
    
    [xsize,ysize]=modul.shape
    sp=modul*np.exp(0+1j*phase)*xsize*ysize 
    img=fft.ifft2(fft.ifftshift(sp)).real
    
    return img  

def modphaimg_gpu(modul,phase):
    
    [xsize,ysize]=modul.shape
    sp = modul*cp.exp(0+1j*phase)*xsize*ysize
    img = cp.fft.ifft2(cp.fft.ifftshift(sp)).real
    return img
'''  窗函数
参数：nx,ny 窗函数大小，
      apod切趾的比例,
      a0=0.5 for hanning,  a0=25.0/46.0 for hamming, 当无参数winsty时，默认a0=0.5
      
调用：xyy.win(xsize,ysize,0.2,'hamm')

返回：win
'''
def  win(nx,ny,apod,winsty=0.0):
    
    if winsty == 'hann':
        a0=0.5
    if winsty == 'hamm':
        a0=25.0/46.0
    if winsty == 0.0:
        a0=0.5

    nn = np.int16((apod*nx)//2*2+1)
    wx = a0-(1.0-a0)*np.cos(2.0*np.pi*np.arange(nn).reshape(nn,1)/(nn-1))
    maxp = np.max(wx)
    maxid = np.where(wx == maxp)
    c = maxid[0][0]
    w1 = np.empty([nx,1], dtype = np.float32) 
    w1[0:c]=wx[0:c]
    w1[c:nx-(nn-c)]=maxp
    w1[nx-(nn-c):nx]=wx[c:nn]
    
    nn = np.int16((apod*ny)//2*2+1)
    wy = a0-(1.0-a0)*np.cos(2.0*np.pi*np.arange(nn).reshape(nn,1)/(nn-1))
    maxp = np.max(wy)
    maxid = np.where(wy == maxp)
    c = maxid[0][0]
    w2 = np.empty([ny,1], dtype = np.float32) 
    w2[0:c]=wy[0:c]
    w2[c:ny-(nn-c)]=maxp
    w2[ny-(nn-c):ny]=wy[c:nn]
    
    win = np.dot(w1, w2.T)
    win=win/np.max(win)
   
    return win

def  win_gpu(nx,ny,apod,winsty=0.0):
    
    if winsty == 'hann':
        a0=0.5
    if winsty == 'hamm':
        a0=25.0/46.0
    if winsty == 0.0:
        a0=0.5

    nn = cp.int16((apod*nx)//2*2+1)
    wx = a0-(1.0-a0)*cp.cos(2.0*cp.pi*cp.arange(nn).reshape(nn,1)/(nn-1))
    maxp = cp.max(wx)
    maxid = cp.where(wx == maxp)
    c = maxid[0][0]
    w1 = cp.empty([nx,1], dtype = cp.float32) 
    w1[0:c]=wx[0:c]
    w1[c:nx-(nn-c)]=maxp
    w1[nx-(nn-c):nx]=wx[c:nn]
    
    nn = cp.int16((apod*ny)//2*2+1)
    wy = a0-(1.0-a0)*cp.cos(2.0*cp.pi*cp.arange(nn).reshape(nn,1)/(nn-1))
    maxp = cp.max(wy)
    maxid = cp.where(wy == maxp)
    c = maxid[0][0]
    w2 = cp.empty([ny,1], dtype = cp.float32) 
    w2[0:c]=wy[0:c]
    w2[c:ny-(nn-c)]=maxp
    w2[ny-(nn-c):ny]=wy[c:nn]
    
    win = cp.dot(w1, w2.T)
    win=win/cp.max(win)
   
    return win

'''
#----说明： 频域图像滤波(卷积)
#----参数:  data, filt
#----返回： img
'''

def  ImgFilted(data,filt):
    
    fftobj=fft.fft2(data)
    fftpsf=fft.fft2(filt)
    
    img=fft.fftshift((fft.ifft2(fftobj*fftpsf))).real.astype(np.float32)
    
    return img

def  ImgFilted_gpu(data,filt):
    
    fftobj=cp.fft.fft2(data)
    fftpsf=cp.fft.fft2(filt)
    
    img=cp.fft.fftshift((cp.fft.ifft2(fftobj*fftpsf))).real.astype(cp.float32)
    
    return img


'''
#----说明： 图像功率谱推卷积
#----参数:  data, sitf
#----返回： img
'''

def  ImgPSDdeconv(data,sitf):
    
    modul,phase=imgmodpha(data)
    
    mod=np.sqrt(modul**2/(sitf+0.0001))
    
    img=modphaimg(mod,phase)
    
    return img


def ImgPSDdeconv_gpu(data,sitf):

    modul,phase = imgmodpha_gpu(data)

    mod = cp.sqrt(modul**2/(sitf+0.0001))
    img = modphaimg_gpu(mod,phase)
    return img
  

''' 二维高斯函数
参数：xsize，ysize，delta
返回：
'''
def  gaussf2d(xsize,ysize,delta):
    
     xline=np.exp(-(np.arange(xsize)-xsize//2)**2/(delta)**2.0)
     yline=np.exp(-(np.arange(ysize)-ysize//2)**2/(delta)**2.0)                        
     [Y,X]=np.meshgrid(yline,xline) 
     
     return Y*X 
 
def  gaussf2d_gpu(xsize,ysize,delta):
    
     xline=cp.exp(-(cp.arange(xsize)-xsize//2)**2/(delta)**2.0)
     yline=cp.exp(-(cp.arange(ysize)-ysize//2)**2/(delta)**2.0)                        
     [Y,X]=cp.meshgrid(yline,xline) 
     
     return Y*X 
    
######==================================================================================== 其他

'''     圆孔
参数： xsize,ysize,radus
说明：  圆孔中心（xsize//2,ysize//2）

'''

def circlepupil(xsize,ysize,radus):
    
    pupil=np.zeros([xsize,ysize],dtype=np.float32)
    [Y,X]=np.meshgrid(np.arange(ysize)-ysize//2,np.arange(xsize)-xsize//2) 
    R=np.sqrt(Y*Y+X*X)
    pupil=np.where(R<=radus,1.0,0.0)

    return pupil

def circlepupil_gpu(xsize,ysize,radus):
    
    pupil=cp.zeros([xsize,ysize],dtype=cp.float32)
    [Y,X]=cp.meshgrid(cp.arange(ysize)-ysize//2,cp.arange(xsize)-xsize//2) 
    R=cp.sqrt(Y*Y+X*X)
    pupil=cp.where(R<=radus,1.0,0.0)

    return pupil
'''     图像径向求平均
参数：  data
返回：  datanew
'''

def imgradusmean(data):
    
    xsize,ysize=data.shape
    [Y,X]=np.meshgrid(np.arange(ysize)-ysize//2,np.arange(xsize)-xsize//2) 
    R=np.sqrt(X*X+Y*Y)
    
    datanew=np.zeros([xsize,ysize],dtype=np.float32)
    datanew[xsize//2,ysize//2]=data[xsize//2,ysize//2]
    
    for i in np.arange(0,np.min([xsize,ysize])//2):
    
        mask=np.where(R<=i+1,1.0,0.0)-np.where(R<=i,1.0,0.0)
        val=np.sum(mask*data)/np.sum(mask)
        datanew=datanew+val*mask
        
        
    return datanew
        
def imgradusmean_gpu(data):
    
    xsize,ysize=data.shape
    [Y,X]=cp.meshgrid(cp.arange(ysize)-ysize//2,cp.arange(xsize)-xsize//2) 
    R=cp.sqrt(X*X+Y*Y)
    
    datanew=cp.zeros([xsize,ysize],dtype=cp.float32)
    datanew[xsize//2,ysize//2]=data[xsize//2,ysize//2]
    
    for i in cp.arange(0,cp.min([xsize,ysize])//2):
    
        mask=cp.where(R<=i+1,1.0,0.0)-cp.where(R<=i,1.0,0.0)
        val=cp.sum(mask*data)/cp.sum(mask)
        datanew=datanew+val*mask
        
        
    return datanew

''' 返回质心坐标
参数：img
返回：[dx,dy]
'''
def  centroid(img):
     
    xsize=img.shape[0]
    ysize=img.shape[1]
    
    [Y,X]=np.meshgrid(np.arange(ysize),np.arange(xsize)) 
    
    dx = np.sum(X*img)/np.sum(img)
    dy = np.sum(Y*img)/np.sum(img)
    
    return [dx,dy]

    
def  centroid_gpu(img):
     
    xsize=img.shape[0]
    ysize=img.shape[1]
    
    [Y,X]=cp.meshgrid(cp.arange(ysize),cp.arange(xsize)) 
    
    dx = cp.sum(X*img)/cp.sum(img)
    dy = cp.sum(Y*img)/cp.sum(img)
    
    return [dx,dy]


''' 二维数组BINNING
参数：参考图img，bins
返回：图像new
'''
def imgbin(img, bins):
     
    xsize = img.shape[0]
    ysize = img.shape[1]
    newxsize = xsize//bins
    newysize = ysize//bins
    cpy = img.copy()
    
    tmp = np.empty([xsize,newysize], dtype = np.int16)
    for i in range(xsize):
        tmp[i,:] = cpy[i,:newysize*bins].reshape(-1,bins).mean(axis=1)
        
    new = np.empty([newxsize,newysize], dtype = np.int16)
    for i in range(newysize):
        new[:,i] = tmp[:newxsize*bins,i].reshape(-1,bins).mean(axis=1)
    
    return new


def imgbin_gpu(img, bins):
     
    xsize = img.shape[0]
    ysize = img.shape[1]
    newxsize = xsize//bins
    newysize = ysize//bins
    cpy = img.copy()
    
    tmp = cp.empty([xsize,newysize], dtype = cp.int16)
    for i in range(xsize):
        tmp[i,:] = cpy[i,:newysize*bins].reshape(-1,bins).mean(axis=1)
        
    new = cp.empty([newxsize,newysize], dtype = cp.int16)
    for i in range(newysize):
        new[:,i] = tmp[:newxsize*bins,i].reshape(-1,bins).mean(axis=1)
    
    return new
'''  计算两幅图像（二维矩阵）的皮尔森积矩相关系数
参数：imga，imgb,         
返回： 
'''
def  prs_cor_coef(ima,imb):
    
    imga=ima.copy()/np.mean(ima)
    imgb=imb.copy()/np.mean(imb)
    
    cov=np.sum((imga-np.mean(imga))*(imgb-np.mean(imgb)))
    
    tha=(np.sum((imga-np.mean(imga))**2.0))**0.5
    thb=(np.sum((imgb-np.mean(imgb))**2.0))**0.5
    
    cc=cov/(tha*thb)
    
    return cc

   
def  prs_cor_coef_gpu(ima,imb):
    
    imga=ima.copy()/cp.mean(ima)
    imgb=imb.copy()/cp.mean(imb)
    
    cov=cp.sum((imga-cp.mean(imga))*(imgb-cp.mean(imgb)))
    
    tha=(cp.sum((imga-cp.mean(imga))**2.0))**0.5
    thb=(cp.sum((imgb-cp.mean(imgb))**2.0))**0.5
    
    cc=cov/(tha*thb)
    
    return cc
'''
#---说明： 计算各点到原点（二维图像中心)）的欧式距离
#---参数： row（行数），col（列数）
#---返回： array
'''
def dist(row,col):
      
    [X,Y]=np.meshgrid(np.arange(col)-col//2,np.arange(row)-row//2) 
    dist=(X**2.0+Y**2.0)**0.5
    
    return dist

def dist_gpu(row,col):
      
    [X,Y]=cp.meshgrid(cp.arange(col)-col//2,cp.arange(row)-row//2) 
    dist=(X**2.0+Y**2.0)**0.5
    
    return dist
######################=====================================================    传递函数

'''
#  函数： 计算特定频率下环形光瞳的传递函数, 几何算法, 归一化的自相关面积
#  参数： a：遮挡比，a=0 代表清澈圆孔径； rho：空间频率
#  返回：该频率下的归一化OTF值
'''

def ringotfcal(a, rho):
    
    if (rho < 0) or (rho > 1): return 0.0
    if (a < 0) or (a > 1): return 0.0
 
    #----------------------
    r=0.5  
    if (rho < 2.0*r):   
        c=2.0*math.acos(rho/(r*2.0))*r**2-rho*math.sqrt(r**2-rho**2/4.0)
    else: 
        c=0.0

    if (rho < 2.0*r*a): 
        e=2.0*math.acos(rho/(a*r*2.0))*(r*a)**2-rho*math.sqrt((r*a)**2-rho**2/4.0)
    else: 
        e=0.0

    if (rho <= r+a*r) and (rho > r-a*r):
        s1=0.5*math.acos(((r*a)**2+rho**2-r**2)/(2.0*r*a*rho))*(r*a)**2
        s2=0.5*math.acos((rho**2+r**2-(r*a)**2)/(2.0*r*rho))*r**2
        s3=0.5*math.sin(math.acos(((r*a)**2+rho**2-r**2)/(2.0*a*r*rho)))*a*r*rho
        d=2.0*(s1+s2-s3)
    else:
        if rho <= r-a*r :
            d=math.pi*(a*r)**2
        else :
            d=0.0
    #--------------------------
    h=(c+e-2*d)/(math.pi*r**2)
    
    if rho == 0 :
        h=1-a**2

    return h


def ringotfcal_gpu(a, rho):
    
    if (rho < 0) or (rho > 1): return 0.0
    if (a < 0) or (a > 1): return 0.0
 
    #----------------------
    r=0.5  
    if (rho < 2.0*r):   
        c=2.0*cp.arccos(rho/(r*2.0))*r**2-rho*cp.sqrt(r**2-rho**2/4.0)
    else: 
        c=0.0

    if (rho < 2.0*r*a): 
        e=2.0*cp.arccos(rho/(a*r*2.0))*(r*a)**2-rho*cp.sqrt((r*a)**2-rho**2/4.0)
    else: 
        e=0.0

    if (rho <= r+a*r) and (rho > r-a*r):
        s1=0.5*cp.arccos(((r*a)**2+rho**2-r**2)/(2.0*r*a*rho))*(r*a)**2
        s2=0.5*cp.arccos((rho**2+r**2-(r*a)**2)/(2.0*r*rho))*r**2
        s3=0.5*cp.sin(cp.arccos(((r*a)**2+rho**2-r**2)/(2.0*a*r*rho)))*a*r*rho
        d=2.0*(s1+s2-s3)
    else:
        if rho <= r-a*r :
            d=cp.pi*(a*r)**2
        else :
            d=0.0
    #--------------------------
    h=(c+e-2*d)/(cp.pi*r**2)
    
    if rho == 0 :
        h=1-a**2

    return h
'''
#   函数：计算指定环型光瞳，指定采样比例尺的望远镜OTF, a=0代表清澈圆孔径
#   返回：二维数组，OTF
'''
def telotf(a, maxfre, width):
    
    half=width//2
    cent=width//2
    otf=np.zeros([width,width],dtype=np.float32)
    for i in range(width):
        for j in range(width):
            fre=np.sqrt((i-cent)**2+(j-cent)**2)/half*maxfre
            freq=np.minimum(fre,1.0)
            otf[i,j]= ringotfcal(a, freq)
            
    otf=otf/otf[cent,cent]
        
    return otf


def telotf_gpu(a, maxfre, width):
    
    half=width//2
    cent=width//2
    otf=cp.zeros([width,width],dtype=cp.float32)
    for i in range(width):
        for j in range(width):
            fre=cp.sqrt((i-cent)**2+(j-cent)**2)/half*maxfre
            freq=cp.minimum(fre,1.0)
            otf[i,j]= ringotfcal(a, freq)
            
    otf=otf/otf[cent,cent]
        
    return otf
'''
#   函数： 大气短曝光传递函数计算
#   返回： 特定空间频率下的 OTF 值
'''
def atsotfcal(diameter, r0, fre):
    #otf=np.exp(-3.44*(diameter/r0*fre)**(5/3)*(1-fre**(1/3)))      
    otf=np.exp(-3.44*(diameter/r0*fre)**(5/3)*(1-np.exp(-fre**3)*fre**(1/3)))   #  此举抑制高频上翘
    return otf

def atsotfcal_gpu(diameter, r0, fre):
    #otf=np.exp(-3.44*(diameter/r0*fre)**(5/3)*(1-fre**(1/3)))      
    otf=cp.exp(-3.44*(diameter/r0*fre)**(5/3)*(1-cp.exp(-fre**3)*fre**(1/3)))   #  此举抑制高频上翘
    return otf
'''
#   函数： 大气短曝光传递函数计算
#   返回： 二维数组
'''
def atsotf(diameter, r0, width, maxfre):
      
    half=width//2
    cent=width//2
    
    [Y,X]=np.meshgrid(np.arange(width)-half,np.arange(width)-half) 
    fre=np.sqrt(X*X+Y*Y)/half*maxfre
    
    freq=np.minimum(fre,1.0)
        
    sotf=np.exp(-3.44*(diameter/r0*freq)**(5/3)*(1-np.exp(-freq**3)*freq**(1/3)))   #  此举抑制高频上翘
    
    sotf=sotf/sotf[cent,cent]
    
    return sotf

def atsotf_gpu(diameter, r0, width, maxfre):
      
    half=width//2
    cent=width//2
    
    [Y,X]=cp.meshgrid(cp.arange(width)-half,cp.arange(width)-half) 
    fre=cp.sqrt(X*X+Y*Y)/half*maxfre
    
    freq=cp.minimum(fre,1.0)
        
    sotf=cp.exp(-3.44*(diameter/r0*freq)**(5/3)*(1-cp.exp(-freq**3)*freq**(1/3)))   #  此举抑制高频上翘
    
    sotf=sotf/sotf[cent,cent]
    
    return sotf
'''
#   函数：计算综合系统短曝光传递函数
#   返回：二维数组，OTF
'''
def sotf(diameter, a, r0, maxfre, width):
    half=width//2
    cent=width//2
    sotf=np.zeros([width,width],dtype=np.float32)
    for i in range(width):
        for j in range(width):
            fre=np.sqrt((i-cent)**2+(j-cent)**2)/half*maxfre
            freq=np.minimum(fre,1.0)
            sotf[i,j]=ringotfcal(a, freq)*atsotfcal(diameter, r0, freq) 
            
    sotf=sotf/sotf[cent,cent]
    
    return sotf

def sotf_gpu(diameter, a, r0, maxfre, width):
    half=width//2
    cent=width//2
    sotf=cp.zeros([width,width],dtype=cp.float32)
    for i in range(width):
        for j in range(width):
            fre=cp.sqrt((i-cent)**2+(j-cent)**2)/half*maxfre
            freq=cp.minimum(fre,1.0)
            sotf[i,j]=ringotfcal(a, freq)*atsotfcal(diameter, r0, freq) 
            
    sotf=sotf/sotf[cent,cent]
    
    return sotf
'''
#   函数： 大气长曝光传递函数计算
#   返回： 特定空间频率下的 OTF 值
'''
def atlotfcal(diameter, r0, fre):
    otf=np.exp(-3.44*(diameter/r0*fre)**(5/3))
    return otf

def atlotfcal_gpu(diameter, r0, fre):
    otf=cp.exp(-3.44*(diameter/r0*fre)**(5/3))
    return otf
'''
#   函数： 大气长曝光传递函数计算
#   返回： 二维数组
'''
def atlotf(diameter, r0, width, maxfre):
     
    half=width//2
    cent=width//2
    
    [Y,X]=np.meshgrid(np.arange(width)-half,np.arange(width)-half) 
    fre=np.sqrt(X*X+Y*Y)/half*maxfre
    
    freq=np.minimum(fre,1.0)
        
    lotf=np.exp(-3.44*(diameter/r0*freq)**(5/3))
    lotf=lotf/lotf[cent,cent]
    
    return lotf

def atlotf_gpu(diameter, r0, width, maxfre):
     
    half=width//2
    cent=width//2
    
    [Y,X]=cp.meshgrid(cp.arange(width)-half,cp.arange(width)-half) 
    fre=cp.sqrt(X*X+Y*Y)/half*maxfre
    
    freq=cp.minimum(fre,1.0)
        
    lotf=cp.exp(-3.44*(diameter/r0*freq)**(5/3))
    lotf=lotf/lotf[cent,cent]
    
    return lotf
'''
#   函数：计算综合系统长曝光传递函数
#   返回：二维数组，OTF
'''
def lotf(diameter, a, r0, maxfre, width):
    cent=width//2
    half=width//2
    lotf=np.zeros([width,width],dtype=np.float32)
    for i in range(width):
        for j in range(width):
            fre=np.sqrt((i-cent)**2+(j-cent)**2)/half*maxfre
            freq=np.minimum(fre,1.0)
            lotf[i,j]=ringotfcal(a, freq)*atlotfcal(diameter, r0, freq)
            
    lotf=lotf/lotf[cent,cent]
    
    return lotf

def lotf_gpu(diameter, a, r0, maxfre, width):
    cent=width//2
    half=width//2
    lotf=cp.zeros([width,width],dtype=cp.float32)
    for i in range(width):
        for j in range(width):
            fre=cp.sqrt((i-cent)**2+(j-cent)**2)/half*maxfre
            freq=cp.minimum(fre,1.0)
            lotf[i,j]=ringotfcal_gpu(a, freq)*atlotfcal_gpu(diameter, r0, freq)
            
    lotf=lotf/lotf[cent,cent]
    
    return lotf
'''
#   说明： 计算标准谱比（短曝光）
#   输入：  sitfdata, diameter, diaratio, maxfre, subsize, start_r0, step_r0
#   返回：  三维数组

'''

def  sotfsrstand(sitfdata,diameter,diaratio,maxfre,subsize,start_r0,step_r0):
    
    r0num=sitfdata.shape[0]

    TelOtf=telotf(diaratio, maxfre, subsize)

    sotfsrstand=np.zeros([r0num,subsize, subsize], dtype=np.float32) 

    for i in range(r0num):

        subsitf=GetSitf(sitfdata,maxfre,subsize,i)
        
        r0=start_r0+step_r0*i
        AtSotf=atsotf(diameter, r0, subsize, maxfre)
        sotf=TelOtf*AtSotf
  
        sotfsr=sotf**2/(subsitf)
        sotfsr=sotfsr/sotfsr[subsize//2,subsize//2]
    
        sotfsrstand[i,:,:]=sotfsr
        
    return sotfsrstand


def  sotfsrstand_gpu(sitfdata,diameter,diaratio,maxfre,subsize,start_r0,step_r0):
    
    r0num=sitfdata.shape[0]

    TelOtf=telotf_gpu(diaratio, maxfre, subsize)

    sotfsrstand=cp.zeros([r0num,subsize, subsize], dtype=cp.float32) 

    for i in range(r0num):

        subsitf=GetSitf_gpu(sitfdata,maxfre,subsize,i)
        
        r0=start_r0+step_r0*i
        AtSotf=atsotf_gpu(diameter, r0, subsize, maxfre)
        sotf=TelOtf*AtSotf
  
        sotfsr=sotf**2/(subsitf)
        sotfsr=sotfsr/sotfsr[subsize//2,subsize//2]
    
        sotfsrstand[i,:,:]=sotfsr
        
    return sotfsrstand
'''
#   说明： 计算标准谱比（长曝光）
#   输入：  sitfdata, diameter, diaratio, maxfre, subsize, start_r0, step_r0
#   返回：  三维数组

'''

def  lotfsrstand(sitfdata,diameter,diaratio,maxfre,subsize,start_r0,step_r0):
    
    r0num=sitfdata.shape[0]

    TelOtf=telotf(diaratio, maxfre, subsize)

    lotfsrstand=np.zeros([r0num,subsize, subsize], dtype=np.float32) 

    for i in range(r0num):

        subsitf=GetSitf(sitfdata,maxfre,subsize,i)
        
        r0=start_r0+step_r0*i
        AtLotf=atlotf(diameter, r0, subsize, maxfre)
        lotf=TelOtf*AtLotf
  
        lotfsr=lotf**2/(subsitf)
        lotfsr=lotfsr/lotfsr[subsize//2,subsize//2]
    
        lotfsrstand[i,:,:]=lotfsr
        
    return lotfsrstand

def  lotfsrstand_gpu(sitfdata,diameter,diaratio,maxfre,subsize,start_r0,step_r0):
    
    r0num=sitfdata.shape[0]

    TelOtf=telotf_gpu(diaratio, maxfre, subsize)

    lotfsrstand=cp.zeros([r0num,subsize, subsize], dtype=cp.float32) 

    for i in range(r0num):

        subsitf=GetSitf_gpu(sitfdata,maxfre,subsize,i)
        
        r0=start_r0+step_r0*i
        AtLotf=atlotf_gpu(diameter, r0, subsize, maxfre)
        lotf=TelOtf*AtLotf
  
        lotfsr=lotf**2/(subsitf)
        lotfsr=lotfsr/lotfsr[subsize//2,subsize//2]
    
        lotfsrstand[i,:,:]=lotfsr
        
    return lotfsrstand
'''
#   说明：  计算一组三维数组的谱比
#   输入：  cubesub,winsr
#   返回：  sr (二维数组)
'''

def  cubesrcal(cubesub,winsr):
    
    srsize=cubesub.shape[1]
    corsize=[int(srsize*0.8),int(srsize*0.8)]
    
    #-----   计算平均帧     
    meanf=np.mean(cubesub,axis=0)
    #-----   平均帧的平均值
    mfval=np.mean(meanf)
    #-----   以平均帧对齐  
    cubesubalign=cube_align(cubesub,meanf,corsize)
    
    #-------------  平均叠加帧加窗
    lxp=(meanf-mfval)*winsr+mfval

    #-----   计算每一帧图像的均值         
    meanv=np.mean(cubesubalign,axis=(1,2))
    mvcast=meanv[:,None,None]
     
    #----（每一帧图像-其均值）* 窗函数 + 其均值  
    cubesubalignwin=(cubesubalign-mvcast)*winsr+mvcast
      
    #-------傅里叶变换（得到每一帧频谱）
    cubesp=fft.fftshift(fft.fft2(cubesubalignwin,axes=(1,2)),axes=(1,2)) 

    psd=Psdcubecal(cubesp)
    psd=psd/psd[srsize//2,srsize//2]
         
    psdnd,noise=Psdnd(psd)
    psdnd=psdnd/psdnd[srsize//2,srsize//2]
     
    #--------------计算平均短曝光传递函数  
    sotf2=np.abs(np.fft.fftshift(np.fft.fft2(lxp)))**2 
    sotf2=sotf2/sotf2[srsize//2,srsize//2]
    
    sotf2nd,noise2=Psdnd(sotf2)
    sotf2nd=sotf2nd/sotf2nd[srsize//2,srsize//2]
   
    #--------------计算谱比
    sr=sotf2/(psdnd+0.0000001)
    sr=sr/sr[srsize//2,srsize//2]
        
    return sr,sotf2,psd,noise

def  cubesrcal_gpu(cubesub,winsr):
    
    srsize=cubesub.shape[1]
    corsize=[int(srsize*0.8),int(srsize*0.8)]
    
    #-----   计算平均帧     
    meanf=cp.mean(cubesub,axis=0)
    #-----   平均帧的平均值
    mfval=cp.mean(meanf)
    #-----   以平均帧对齐  
    cubesubalign=cube_align_gpu(cubesub,meanf,corsize)
    
    #-------------  平均叠加帧加窗
    lxp=(meanf-mfval)*winsr+mfval

    #-----   计算每一帧图像的均值         
    meanv=cp.mean(cubesubalign,axis=(1,2))
    mvcast=meanv[:,None,None]
     
    #----（每一帧图像-其均值）* 窗函数 + 其均值  
    cubesubalignwin=(cubesubalign-mvcast)*winsr+mvcast
      
    #-------傅里叶变换（得到每一帧频谱）
    cubesp=cp.fft.fftshift(cp.fft.fft2(cubesubalignwin,axes=(1,2)),axes=(1,2)) 

    psd=Psdcubecal_gpu(cubesp)
    psd=psd/psd[srsize//2,srsize//2]
         
    psdnd,noise=Psdnd_gpu(psd)
    psdnd=psdnd/psdnd[srsize//2,srsize//2]
     
    #--------------计算平均短曝光传递函数  
    sotf2=cp.abs(cp.fft.fftshift(cp.fft.fft2(lxp)))**2 
    sotf2=sotf2/sotf2[srsize//2,srsize//2]
    
    sotf2nd,noise2=Psdnd_gpu(sotf2)
    sotf2nd=sotf2nd/sotf2nd[srsize//2,srsize//2]
   
    #--------------计算谱比
    sr=sotf2/(psdnd+0.0000001)
    sr=sr/sr[srsize//2,srsize//2]
        
    return sr,sotf2,psd,noise
'''
#   说明：  谱比导出r0
#   输入：  sr, srarry（标准谱比）, maxfre（用于确定截止频率位置）, low（环带积分内环位置）, hig（环带积分外环位置）
#   返回：  r0
'''

def  srdevr0(sr,srarry,maxfre,low,hig,start_r0,step_r0):
    
    srsize=sr.shape[1]
    [Y,X]=np.meshgrid(np.arange(srsize)-srsize//2,np.arange(srsize)-srsize//2) 
    mask1=np.where(np.sqrt(X**2+Y**2)<=(srsize//2)/(1.0*maxfre)*hig, 1.0, 0.0)  
    mask2=np.where(np.sqrt(X**2+Y**2)>=(srsize//2)/(1.0*maxfre)*low, 1.0, 0.0)  
    masksr=mask1*mask2
 
    diff1=(srarry-sr)*masksr
    diff2=diff1**2.0
    valarr=np.sum(diff2,axis=(1,2))
       
    idex=np.where(valarr==np.min(valarr))[0][0]
    r0=start_r0+step_r0*idex
      
    return r0,masksr

def  srdevr0_gpu(sr,srarry,maxfre,low,hig,start_r0,step_r0):
    
    srsize=sr.shape[1]
    [Y,X]=cp.meshgrid(cp.arange(srsize)-srsize//2,cp.arange(srsize)-srsize//2) 
    mask1=cp.where(cp.sqrt(X**2+Y**2)<=(srsize//2)/(1.0*maxfre)*hig, 1.0, 0.0)  
    mask2=cp.where(cp.sqrt(X**2+Y**2)>=(srsize//2)/(1.0*maxfre)*low, 1.0, 0.0)  
    masksr=mask1*mask2
 
    diff1=(srarry-sr)*masksr
    diff2=diff1**2.0
    valarr=cp.sum(diff2,axis=(1,2))
       
    idex=cp.where(valarr==cp.min(valarr))[0][0]
    r0=start_r0+step_r0*idex
      
    return r0,masksr
'''
#   说明：   cube分块谱比导出r0
#   输入：   cubesr,srsize,winsr,sitfdata,diameter,diaratio,maxfre,low,hig,start_r0,step_r0
#   返回：   r0,index
'''

def  cubesrdevr0(cubesr,srsize,winsr,sitfdata,diameter,diaratio,maxfre,low,hig,start_r0,step_r0):
    
    srarry=sotfsrstand(sitfdata,diameter,diaratio,maxfre,srsize,start_r0,step_r0)
    xnum=cubesr.shape[2]//srsize 
    ynum=cubesr.shape[1]//srsize 
    r0arr=[]
    ###corsize=[int(srsize*0.9),int(srsize*0.9)]
    
    for i in range(xnum):
        for j in range(ynum):
            
            cubesub=cubesr[:,i*srsize:i*srsize+srsize,i*srsize:i*srsize+srsize]

            ###meanf=np.mean(cubesub,axis=0)
            ###cubesubalign=cube_align(cubesub,meanf,corsize,win=1.0)
            
            sr,sotf2,psd,noise=cubesrcal(cubesub,winsr)
            
            sr_filter=ndm.gaussian_filter(sr, sigma=0.8)
            sr_filter=sr_filter/sr_filter[srsize//2,srsize//2]
            
            r0,masksr=srdevr0(sr_filter,srarry,maxfre,low,hig,start_r0,step_r0)
            r0arr.append(r0)
 
    r0=(Counter(r0arr).most_common(1)[0][0]).astype(np.float32)
    index=np.rint((r0-start_r0)/step_r0).astype(np.int)
    
    print('r0 =',r0)

    return r0,index    


def  cubesrdevr0_gpu(cubesr,srsize,winsr,sitfdata,diameter,diaratio,maxfre,low,hig,start_r0,step_r0):
    
    srarry=sotfsrstand_gpu(sitfdata,diameter,diaratio,maxfre,srsize,start_r0,step_r0)
    xnum=cubesr.shape[2]//srsize 
    ynum=cubesr.shape[1]//srsize 
    r0arr=[]
    ###corsize=[int(srsize*0.9),int(srsize*0.9)]
    
    for i in range(xnum):
        for j in range(ynum):
            
            cubesub=cubesr[:,i*srsize:i*srsize+srsize,i*srsize:i*srsize+srsize]

            ###meanf=np.mean(cubesub,axis=0)
            ###cubesubalign=cube_align(cubesub,meanf,corsize,win=1.0)
            
            sr,sotf2,psd,noise=cubesrcal_gpu(cubesub,winsr)
            
            sr_filter=ndm.gaussian_filter(cp.asnumpy(sr), sigma=0.8)
            sr_filter=cp.array(sr_filter/sr_filter[srsize//2,srsize//2])
            
            r0,masksr=srdevr0_gpu(sr_filter,srarry,maxfre,low,hig,start_r0,step_r0)
            r0arr.append(np.float32(cp.asnumpy(r0)))
 
    r0=(Counter(r0arr).most_common(1)[0][0])
    index=cp.rint(cp.array((r0-start_r0)/step_r0)).astype(cp.int)
    
    print('r0 =',r0)

    return r0,index    
'''
#----说明： 计算斑点干涉术传递函数  
#----参数： stfdata, Maxfre, IMsize, idx
#----返回： sitf
'''
def  GetSitf(stfdata,maxfre,imsize,idx):
    
    sitfsize=stfdata.shape[1]
    mpr=(sitfsize)*(maxfre*2.0)/(imsize)
    
    [Y,X]=np.meshgrid(np.arange(imsize)-imsize//2,np.arange(imsize)-imsize//2) 
    mprtx=np.int64(np.sqrt(X*X+Y*Y)*mpr)
    
    sitf=stfdata[idx,np.minimum(mprtx,sitfsize-1)]    
    sitf=sitf/sitf[imsize//2,imsize//2]
    
    return sitf

def  GetSitf_gpu(stfdata,maxfre,imsize,idx):
    
    sitfsize=stfdata.shape[1]
    mpr=(sitfsize)*(maxfre*2.0)/(imsize)
    [Y,X]=cp.meshgrid(cp.arange(imsize)-imsize//2,cp.arange(imsize)-imsize//2) 
    mpr = cp.array(mpr)
    mprtx=(cp.sqrt(X*X+Y*Y)*mpr).astype(cp.int64)
    idx = cp.array(idx)
    imsize = cp.array(imsize)
    sitf=stfdata[idx,cp.minimum(mprtx,sitfsize-1)]    
    sitf=sitf/sitf[imsize//2,imsize//2]
    
    return sitf
'''
#----说明： 功率谱退卷积
#----参数:  fdata, subsitf
#----返回： img
'''
def  PsdDeconv(data,subsitf):
    
    [xsize,ysize]=data.shape
    datasp=fft.fftshift(fft.fft2(data))/xsize/ysize
    
    pha=np.angle(datasp)
    psd=np.abs(datasp)**2
    mod=np.sqrt(psd/(subsitf+0.0005))

    sp=mod*np.exp(0+1j*pha)*xsize*ysize 
    img=fft.ifft2(fft.ifftshift(sp)).real
    
    return img

def  PsdDeconv_gpu(data,subsitf):
    
    [xsize,ysize]=data.shape
    datasp=cp.fft.fftshift(cp.fft.fft2(data))/xsize/ysize
    
    pha=cp.angle(datasp)
    psd=cp.abs(datasp)**2
    mod=cp.sqrt(psd/(subsitf+0.0005))

    sp=mod*cp.exp(0+1j*pha)*xsize*ysize 
    img=cp.fft.ifft2(cp.fft.ifftshift(sp)).real
    
    return img


def nvst_dirsandfiles_path(path):
    '''return [[database,darkbase,flatbase],[datapath,darkbase,flatpath]]'''
    direction = path
    import os
    target = []#g:\20190518\ha\     活动区、dark、flat
    dirstmp = ''
    a = 0
    paths = [[],[],[]]
    for roots, path, files in os.walk(direction):
        if a == 1:
            target.append(roots)
        a+=1
        if a> 1 and target[-1] not in roots:
            target.append(roots)
    for roots, path, files in os.walk(direction):
        for i in range(len(target)):
            if target[i] in roots:
                paths[i].append(roots)
    path0_len = len(paths[0][-1])
    path1_len = len(paths[1][-1])
    path2_len = len(paths[2][-1])
    path1 = []
    path2 = []
    path3 = []
    for i in range(len(paths[0])):
        if path0_len == len(paths[0][i]):
            path1.append(paths[0][i])
    for i in range(len(paths[1])):
        if path1_len == len(paths[1][i]):
            path2.append(paths[1][i])
    for i in range(len(paths[2])):
        if path2_len == len(paths[2][i]):
            path3.append(paths[2][i])
    return [[target[0],target[1],target[2]],[path1,path2,path3]]


'''
#---说明：计算功率谱  
#---参数： cubesp 频谱(三维)  
#---返回： psdmean
'''
def Psdcubecal(cubesp):
    cubespconj = np.conj(cubesp)
    cubepsd = cubesp * cubespconj
    psdmean = np.mean(cubepsd, axis=0)
    return psdmean.real


def Psdcubecal_gpu(cubesp):
    cubespconj = cp.conj(cubesp)
    cubepsd = cubesp * cubespconj
    psdmean = cp.mean(cubepsd, axis=0)
    return psdmean.real
'''
#----说明： 功率谱噪声剔除
#----参数:  psd 
#----返回： psdnd
'''


def Psdnd(psd):
    xsize, ysize = psd.shape
    noise = (np.sum(psd[0:4, 0:4]) + np.sum(psd[0:4, ysize - 4:ysize]) + np.sum(psd[xsize - 4:xsize, 0:4]) + np.sum(
        psd[xsize - 4:xsize, ysize - 4:ysize])) / 16 / 4.0

    psdnd = np.maximum(psd - 1.0 * noise, 0)

    return psdnd, noise

def Psdnd_gpu(psd):
    xsize, ysize = psd.shape
    noise = (cp.sum(psd[0:4, 0:4]) + cp.sum(psd[0:4, ysize - 4:ysize]) + cp.sum(psd[xsize - 4:xsize, 0:4]) + cp.sum(
        psd[xsize - 4:xsize, ysize - 4:ysize])) / 16 / 4.0

    psdnd = cp.maximum(psd - 1.0 * noise, 0)

    return psdnd, noise


def path_paser(path):
    #return direction
    path = path
    dirs = nvst_dirsandfiles_path(path)
    roots = dirs[0]
    fitsfile = dirs[1] 
    t = 0
    for i in roots:
        '''try:
            i = i.split(':')[1]
            print('You are running in windows')
        except Exception as e:
            i = i
            print('You are running in linux')
        try:
            i = i.split('/')
            i.remove('data')
            i = '/'.join(i)
        except Exception as e:
            i = i
        if 'f' not in i and 'F' not in i and 'D' not in i and 'd' not in i:
            datapath = fitsfile[t]
        elif 'f' in i or 'F' in i:
            flatpath = fitsfile[t]
        elif 'd' in i or 'D' in i:
            darkpath = fitsfile[t]
        t += 1'''
        if 'FLAT' in i.lower() or 'flat' in i.lower():
            flatpath = fitsfile[t]
        elif 'Dark' in i.lower() or 'dark' in i.lower():
            darkpath = fitsfile[t]
        else:
            datapath = fitsfile[t]
        t += 1
    return datapath,flatpath,darkpath


def online_mean(path):
    #contribute for hadata dark flat
    cp.cuda.Device(0).use()
    path = path
    data = cp.array(0,dtype=cp.float32)
    cal_path = []
    while True:
        #fitsfilepath = os.listdir(path)
        if 'TIO' in path:
            raise Exception('Warning: you are proceeing TIO data, its forbidden')
        if ('dark' in path.lower() or 'Dark' in path.lower()):
            fitsfilepath = os.listdir(path)
            for f in cal_path:
                fitsfilepath.remove(f)
            for i in fitsfilepath:
                data = cp.add(data,cp.array(readfits(path+'/'+i)[0],dtype=cp.float32))
                cal_path.append(i)
            if len(cal_path) == 1000:
                return cp.asnumpy(data/1000)
        elif ('FLAT' in path.lower() or 'flat' in path.lower()):
            # 3000
            fitsfilepath = os.listdir(path)
            for f in cal_path:
                fitsfilepath.remove(f)
            for i in fitsfilepath:
                data = cp.add(data,cp.array(readfits(path+'/'+i)[0],dtype=cp.float32))
                cal_path.append(i)
            if len(cal_path) == 1000:
                return cp.asnumpy(data/1000)
        #notice:amount of data <=100            
        '''else:
            fitsfilepath = os.listdir(path)
            for f in cal_path:
                fitsfilepath.remove(f)
            for i in fitsfilepath:
                data = cp.add(data,cp.array(readfits(path+'/'+i)[0],dtype=cp.float32))
                cal_path.append(i)
            if len(cal_path) == 100:
                return cp.asnumpy(data/100)'''
                
            
def cubedata(pathname,fitsname,rcxsize,rcysize):
    cube = np.empty([len(fitsname),rcxsize,rcysize],dtype=np.float32)
    t = 0
    for i in fitsname:
        cube[t,:,:] = readfits(os.path.join(pathname,i))[0][0:rcxsize,0:rcysize]
        t += 1
    return cube