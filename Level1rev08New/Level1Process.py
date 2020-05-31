# encoding:utf-8
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
#import numba as nb
#from numba import cuda
#from mpi4py import MPI
import datetime
import threading
import multiprocessing
#2020/4/29
#添加对无偏带的支持
#2020/5/7
#将同步执行改成异步执行
#2020/5/8
#增加数据完整性检测
#2020/5/18
#改日志格式为json
class myThread(threading.Thread):


    def __init__(self, DirFlat,DirDark,DirLog):
        threading.Thread.__init__(self)
        self.DirFlat = DirFlat
        self.DirDark = DirDark
        self.DirLog = DirLog

    def run(self):
        count = 0
        while 1:
            try:
                #print(self.DirLog)
                newfile = open(self.DirLog,'r')
                new = newfile.readlines()
                newfile.close()
                while len(new) != count+1:
                    count += 1
                    DirFits = new[count].replace('\n','')
                    #print(DirFits)
                    #print(self.DirFlat)
                    #print(self.DirDark)
                    #print(self.DirLog)
                    Align(DirFits,self.DirFlat,self.DirDark)
            except Exception as e:
                pass

class myProcessing(object):
    #def __init__(self, DirFlat,DirDark,DirLog):
    #    self.DirFlat = DirFlat
    #    self.DirDark = DirDark
    #    self.DirLog = DirLog

    def run(self,DirFlat,DirDark,DirLog,DeviceNumber):
        self.DirFlat = DirFlat
        self.DirDark = DirDark
        self.DirLog = DirLog
        self.DeviceNumber = DeviceNumber
        FileBandoff = DirLog.split('.')[0]
        f = open(FileBandoff+'.run','a')
        f.close()
        count = 0
        while 1:
            try:
                #print(12)
                time.sleep(0.1)
                newfile = open(self.DirLog,'r')
                new = newfile.readlines()
                newfile.close()
                #print(self.DirLog)
                #print(new)
                #print('new:'+str(len(new)))
                #print('count:'+str(count))
                while len(new) > count+1:
                    #print(45)
                    DirFits = new[count].replace('\n','')
                    #count += 1
                    Align(DirFits,self.DirFlat,self.DirDark,DeviceNumber)
                    count += 1
            except Exception as e:
                pass


def KeepNewestFlat(flatpath):
    flatpath = flatpath
    flats = os.listdir(flatpath)
    archive = {}
    bandoff = set([i[14:18] for i in flats])
    for i in flats:
        l = []
        for j in bandoff:
            if j in i:
                try:
                    List = archive[j]
                    List.append(i)
                    archive.update({j:List})
                except Exception as e:
                    archive.update({j:[i]})
    a = archive.copy()
    for i in bandoff:
        historyflat = archive[i]
        IntHistoryFlat = [int(i[:8]) for i in historyflat]
        newest = historyflat[IntHistoryFlat.index(max(IntHistoryFlat))]
        archive[i].remove(newest)
        for j in archive[i]:
            os.remove(os.path.join(flatpath,j))


def LtstDtlFts(filepath,r0,fitsname):
    #print(1)
    f = open(filepath,'a')
    f.write(fitsname+'\t'+str(r0)+'\n')
    f.close()
    #print(fitsname+'\t'+r0+'\n')

#@profile
def Align(DirFits,DirFlat,DirDark,DeviceNumber):
    #传入参数：待处理的fits文件的路径。字符串类型一次处理一组
    #处理后的Flat路径 处理后的Dark路径
    DirFits = DirFits
    DirFlat = DirFlat
    DirDark = DirDark
    DeviceNumber = DeviceNumber
    #print(DirFlat)
    #print(DirDark)
    #print(DeviceNumber)
    print(DirFits)
    #print(DirFlat)
    #print(DirDark)
    #print(DeviceNumber)
    cp.cuda.Device(DeviceNumber).use()
    win = cp.array(win_host)
    winsr = cp.array(winsr_host)
    sitfdata = cp.array(sitfdata_host,'<f4')
    gussf = cp.array(gussf_host)
    #读取待处理数据
    data_path_fits = os.listdir(DirFits)
    Year = datetime.datetime.now().strftime('%Y')
    aligned_path = os.path.join(DirFits[DirFits.index(Year):-7],DirFits[-6:]+'.fits')#2020/20200215/...
    #增加处理过的文件不再处理的判断
    SaveFits = redrive+os.path.splitdrive(aligned_path[0:-11])[1]
    xyy.mkdir(SaveFits)
    SaveFitsName = os.path.join(SaveFits,aligned_path[-11:])
    if os.path.exists(SaveFitsName) == 0:
        Filter = data_path_fits.copy()
        for i in data_path_fits:
            if os.path.getsize(os.path.join(DirFits,i)) != 2111040:
                Filter.remove(i)
        numb = len(Filter)
        #print('Align')
        #numb = len(data_path_fits)
        #在cubedata函数补上预处理过程,添加完毕
        #print(DirFits)
        #print(numb)
        #datas,numb = xyy.cubedata(DirFlat,DirDark,DirFits, data_path_fits,rcxsize,rcysize)
        #cubedata = cp.array(datas,dtype='<f4')
        #datas = 0#释放内存
        #cp.cuda.Device(np.random.randint(0,4)).use()
        cubedata = cp.array(xyy.cubedata(DirFlat,DirDark,DirFits, Filter,rcxsize,rcysize),dtype='<f4')#将文件夹所有fits转成三维数组
        #if numb > 1:
        try:
            start = time.time()
            ini = cubedata[0,:,:]
            initmp = ini[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
            if sobel == 1:
                initmp = filters.sobel(filters.gaussian(initmp,5.0))
            #开始对齐
            t = 1
            head=fits.getheader(os.path.join(DirFits,Filter[0]))
            for j in range(1,numb):#从第二个开始处理并与第一张图对齐
                data = cubedata[j,:,:]
                datatmp = data[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
                if sobel == 1:
                    datatmp = filters.sobel(filters.gaussian(datatmp,5.0))
                cc,corr = xyy.corrmaxloc_gpu(initmp,datatmp)
                tmp = xyy.imgshift_gpu(data,[-cc[0],-cc[1]])#对齐后的图
                if only_align_no_luckyimage == 1:
                    #不选帧，直接叠加
                    print('不选帧对齐模式')
                    ini += tmp
                    t += 1
                else:
                    #开始对位移后的图选帧
                    cubedata[j,:,:] = tmp[0:rcxsize,0:rcysize]
            if only_align_no_luckyimage == 1:
                averg = ini/t
            else:
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
                index0=cp.argsort(ringcube)[::-1]#排序
                cubesort0=cubedata.copy()[index0][0:int(fsp*numb),:,:]#取排序前*的帧，再次相关对齐，叠加
                ini=cp.mean(cubesort0, axis=0).astype(cp.float32)
                initmp=ini[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
                if sobel==1:
                    initmp=filters.sobel(filters.gaussian(cp.asnumpy(initmp),5.0))
                for nn in range(cubesort0.shape[0]):
                    data=cubesort0[nn,:,:].copy()
                    datatmp=data[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
                    if sobel==1:
                        datatmp=filters.sobel(filters.gaussian(cp.asnumpy(datatmp),5.0))
                    cc,corr=xyy.corrmaxloc_gpu(initmp, datatmp)
                    tmp=xyy.imgshift_gpu(data,[-cc[0],-cc[1]])#相关对齐
                    cubesort0[nn,:,:]=tmp
                averg=cp.mean(cubesort0, axis=0).astype(cp.float32)#叠加
                if postprocess_flag == 1:
                    print('开始退卷积')
                    cubesr=cubedata[:,srstx:srstx+srxsize,srsty:srsty+srysize]
                    try:
                        r0,index=xyy.cubesrdevr0_gpu(cubesr,srsize,winsr,sitfdata,diameter,diaratio,maxfre,0.00,0.06,start_r0,step_r0)
                    except Exception as e:
                        sys.exit()
                    sitf=xyy.GetSitf_gpu(sitfdata,maxfre,rcxsize,index)#读取理论点扩散函数
                    img=xyy.ImgPSDdeconv_gpu(averg,sitf)
                    head['CODE2'] = r0#r0写到fits头文件中
                    result=xyy.ImgFilted_gpu(img,gussf)
                    result=result/np.median(cp.asnumpy(result))*np.median(cp.asnumpy(averg))
                    try:#redrive:/*
                        SaveFits = redrive+os.path.splitdrive(aligned_path[0:-11])[1]
                        #print(aligned_path)
                        #print(SaveFits)
                        xyy.mkdir(SaveFits)
                        SaveFitsName = os.path.join(SaveFits,aligned_path[-11:])
                    except Exception as e:
                        #print(e)
                        pass
                    xyy.writefits(SaveFitsName,cp.asnumpy(result).astype(np.float32),head)
                else:
                    try:
                        SaveFits = redrive+os.path.splitdrive(aligned_path[0:-11])[1]
                        xyy.mkdir(SaveFits)
                        SaveFitsName = os.path.join(SaveFits,aligned_path[-11:])
                    except Exception as e:
                        #print(e)
                        pass
                    result = averg
                    xyy.writefits(SaveFitsName,cp.asnumpy(result).astype(np.float32),head)
            #print(SaveFitsName)
            print('elapse:'+str(time.time()-start)+'s')
            print('计算完毕，等待下一组数据')
            try:
                os.mkdir(LatestFitsR0)
            except Exception as e:
                pass
            LtstDtlFts(os.path.join(LatestFitsR0,SaveFitsName.split('/')[-2]+'.latest'),r0,SaveFitsName)
        #else:
        #    print('文件夹下无文件，跳过')
        except Exception as e:
            #print('文件夹下无文件，跳过')
            print(e)
            pass

        
def initial(jsonfile):
    f = open(jsonfile,'r')
    para = json.load(f)
    f.close()
    global rcxsize, rcysize, corstart, corsize, sobel, only_align_no_luckyimage, redrive, pfstart, pfsize
    global win_host, diameter, wavelen, pixsca, fsp, srstx, srsty, srxsize, postprocess_flag, srsize, srysize
    global winsr_host, diaratio, start_r0, step_r0, maxfre, filename, sitfdata_host, gussf_host, infrq, otfrq, OnlyBand, LatestFitsR0
    LatestFitsR0 = para['LatestFitsR0']
    rcxsize = int(para['rcxsize'])
    rcysize = int(para['rcysize'])
    corstart = re.findall('\d+',para['corstart'])
    corstart = [int(i) for i in corstart]
    corsize = re.findall('\d+',para['corsize'])
    corsize = [int(i) for i in corsize]
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
    #win=xyy.win_gpu(int(pfsize[0]),int(pfsize[1]),0.5,winsty='hann')     #----窗函数
    win_host=xyy.win(int(pfsize[0]),int(pfsize[1]),0.5,winsty='hann')
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
    #winsr=xyy.win_gpu(srsize,srsize, 0.5, winsty='hann')
    winsr_host=xyy.win(srsize,srsize, 0.5, winsty='hann')
    diaratio = float(para['diaratio'])
    start_r0 = float(para['start_r0'])
    step_r0 = float(para['step_r0'])
    maxfre=wavelen*10.0**(-10.0)/(2.0*diameter*pixsca)*(180.0*3600.0/np.pi)
    filename = para['filename']
    #sitfdata=cp.array(fits.getdata(filename),'<f4')
    sitfdata_host=fits.getdata(filename)
    #gussf=xyy.gaussf2d_gpu(rcxsize,rcysize,1.5)
    gussf_host=xyy.gaussf2d(rcxsize,rcysize,1.5)
    infrq=(pfsize[0]//2)*0.05/maxfre
    otfrq=(pfsize[0]//2)*0.10/maxfre
    log_dir = para['log_dir']
    Dir_NewFile = para['Dir_NewFile']
    today_time = datetime.datetime.now().strftime('%Y%m%d')
    worker = myProcessing()
    while 1:
        try:
            logs = open(os.path.join(log_dir,today_time+'.log'),'r')
            jsinf = json.load(logs)
            logs.close()
            break
        except Exception as e:
            pass
#添加flat的归档管理功能，保证只保留某波段最新的flat
    try:
        DirDarknew = jsinf['AveDark']
        DirFlatnew = jsinf['AveFlat']
        NewFilenew = jsinf['NewFile']
    except Exception as e:
        DirDarknew = []
        DirFlatnew = []
        NewFilenew  = []
    while 1:
        try:#判断程序所需文件是否完整且正确
            KeepNewestFlat(jsinf['ArchiveFlat'][0])
            logs = open(os.path.join(log_dir,today_time+'.log'),'r')
            jsinf = json.load(logs)
            logs.close()
            NewFile = jsinf['NewFile']
            size_newfile = [os.path.getsize(i) for i in NewFile]
            #print(size_newfile)
            if 'AveDark' not in jsinf.keys() and sum(size_newfile)>10 and 'AveFlat' not in jsinf.keys():
                #NewFile = jsinf['NewFile']
                DirFlat = jsinf['ArchiveFlat'][0]
                DirDark = jsinf['ArchiveDark'][0]
                AllDark = os.listdir(DirDark)
                #print(AllDark)
                DirDark = [os.path.join(DirDark,AllDark[0])]
                #print(DirDark)
                AllFlat = os.listdir(DirFlat)
                archflat = []
                for i in AllFlat:
                    archflat.append(os.path.join(DirFlat,i))
                DirFlat = archflat
            elif 'AveDark' not in jsinf.keys() and sum(size_newfile)>10:
                #NewFile = jsinf['NewFile']#"/home/wangxinhua/Desktop/NewFile/20200518Disk_Center365874B363.log"
                DirDark = jsinf['ArchiveDark'][0]
                DirFlat = jsinf['AveFlat']
                AllDark = os.listdir(DirDark)
                DirDark = [os.path.join(DirDark,AllDark[0])]
            elif sum(size_newfile)>10 and 'AveFlat' not in jsinf.keys():
                #NewFile = jsinf['NewFile']#"/home/wangxinhua/Desktop/NewFile/20200518Disk_Center365874B363.log"
                DirDark = jsinf['AveDark']
                DirFlat = jsinf['ArchiveFlat'][0]
                AllFlat = os.listdir(DirFlat)
                archflat = []
                for i in AllFlat:
                    archflat.append(os.path.join(DirFlat,i))
                DirFlat = archflat
            else:
                DirDark = jsinf['AveDark']
                DirFlat = jsinf['AveFlat']#"/home/wangxinhua/Desktop/Flat/20200518FLAT01B060.fits"
                #NewFile = jsinf['NewFile']#"/home/wangxinhua/Desktop/NewFile/20200518Disk_Center365874B363.log"
            #print(DirDark)
            #print(DirFlat)
            #匹配NewFile的偏带和DirFlat的偏带
            Flatbandoff = [i.split('.')[0][-4:] for i in DirFlat]
            #print(Flatbandoff)
            for i in NewFile:
                bandFile = i.split('.')[0][-4:]
                for j in DirFlat:
                    bandFlat= j.split('.')[0][-4:]
                    if bandFile == bandFlat and os.path.exists(i.split('.')[0]+'.run') == False:
                        #print(i.split('.')[0]+'.run')
                        #run(self,DirFlat,DirDark,DirNewFileLog,DeviceNumber):
                        processing = multiprocessing.Process(target=worker.run,args=(j,DirDark[0],i,np.random.randint(4)))
                        processing.start()
                        time.sleep(0.1)
                    #elif bandFile not in Flatbandoff and os.path.exists(i.split('.')[0]+'.run') == False:
                    #    processing = multiprocessing.Process(target=worker.run,args=(DirFlat[0],DirDark[0],i,np.random.randint(4)))#随便取个平场偏带算
                    #    processing.start()
                    #    time.sleep(0.1)
        except Exception as e:
            #print(e)
            pass
            
if __name__ == '__main__':
    jsonfile = r'/home/wangxinhua/level1/Level1/Level1rev08New/json.txt'
    initial(jsonfile)
