"""
"""

"""  ------------色球LEVEL1程序 , 适用如下路径格式   

#  数据路径格式：   H:\20190112\HA\Prominence\024502\CENT\024536
                   盘符:\日期\波段\目标\起始时间\线心偏带\组数（以时间命名）
                    
#  平场路径格式：   H:\20190112\HA\FLAT00\CENT
                   盘符:\日期\波段\FLAT00\线心偏带
                   
                   
 



说明：找到所有数据目录，根据头文件信息，完成LEV1并命名

                  
                   
                   
测试：       日期2019.12.25    数据20190516
                   

""" 





import numpy as np
import astropy.io.fits as fits
import scipy.fftpack as fft

import cupy as cp


from xyy_lib import xyy_lib as xyy

import os
from skimage import filters

##################################################################################


#----------输入 路径 ------------

#path = r'G:\20190516\HA'   #----  观测数据所在目录（到HA）
path = r'G:\20190518\HA'
redrive = r'F:'    # ---保存结果的盘符



#--------------  参数(请确认)


# 波长、口径、遮挡比、起始r0、步长r0、比例尺
(wavelen, diameter, diaratio, start_r0, step_r0, pixsca) = (6563.00, 0.9800, 0.03, 0.0098, 0.0049, 0.165)

maxfre=wavelen*10.0**(-10.0)/(2.0*diameter*pixsca)*(180.0*3600.0/np.pi)


#------------------------------第一次相关区域

corstart=[100,100]
corsize=[800,800]

#  ----- 如果是边缘日珥，sobel=1 ； 如果是日面内，sobel=0

sobel=0 


# 谱比区域大小和起始(从重建区域大小里挖取)
(srxsize, srysize, srstx, srsty) = (384, 384, 320, 320) 


rcxsize=1024
rcysize=1024


gussf=xyy.gaussf2d(rcxsize,rcysize,1.5)

srsize=32
winsr=xyy.win(srsize,srsize, 0.5, winsty='hann')  


#-----------------------------选帧计算所需区域

pfstart=[300,300]
pfsize=[128,128]

win=xyy.win(pfsize[0],pfsize[1],0.5,winsty='hann')     #----窗函数

#---------------------   选帧比例

fsp=0.3

#  --------功率谱积分的区域（取中频段 5cm~10cm 范围）

infrq=(pfsize[0]//2)*0.05/maxfre
otfrq=(pfsize[0]//2)*0.10/maxfre



filename=r'G:\20160501\sitf\korff_sitf_98.fits'
sitfdata=fits.getdata(filename)

#-=========================================================================  寻找平暗场路径

subpaths = os.listdir(path)

flatpath=[]#['G:\\20190518\\HA\\FLAT00']
darkpath=[]#['G:\\20190518\\HA\\Dark']
datapath=[]#['G:\\20190518\\HA\\12741']

for i in range(len(subpaths)):   
    subpath=os.path.join(path,subpaths[i])

    if ('F' in subpaths[i]) or ('f' in subpaths[i]) :        
        flatpath.append(subpath)#['G:\\20190518\\HA\\FLAT00']
    elif ('D' in subpaths[i]) or ('d' in subpaths[i]): 
        darkpath.append(subpath)#['G:\\20190518\\HA\\Dark']
    else:
        datapath.append(subpath)#[['G:\\20190518\\HA\\12741']]

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


#=========================================================================


print('开始计算暗场！') 
                            #('G:', '\\20190518\\HA\\Dark')
redarkpath=os.path.join(redrive,os.path.splitdrive(darkpath[0])[1])
xyy.mkdir(redarkpath)
darkfile=os.path.join(redarkpath,'dark.fits')

if not os.path.exists(darkfile):
    dark= xyy.dirfitsaddmean(darkpath[0]) 
    xyy.writefits(darkfile,dark)    
else:             
    print('暗场已计算过！')
    print() 

print() 


#----------------------

print('开始计算平场！') 

flatfile=[] 
for i in range(len(flatpath)):#['G:\\20190518\\HA\\FLAT00']
    
    fdir=flatpath[i]
    subfdir=os.listdir(fdir)#['B050', 'CENT', 'R050']
    for j in range(len(subfdir)):
        
        subdir=os.path.join(fdir,subfdir[j])
        #G:\20190518\HA\FLAT00\B050  G:\20190518\HA\FLAT00\CENT   G:\20190518\HA\FLAT00\R050 
        
        reflatpath=os.path.join(redrive,os.path.splitdrive(subdir)[1])
        xyy.mkdir(reflatpath)
        
        file=os.path.join(reflatpath,'flat.fits')
        
        if not os.path.exists(file):
            addmean= xyy.dirfitsaddmean(subdir)
            xyy.writefits(file,addmean)
            print('请等待。。。。。')
            print() 
        else:             
            print('平场已计算过！')
            print() 
                           
        flatfile.append(file)
                           
print()  

#========================================================================     开始 Lev1

dark=fits.getdata(darkfile)#dark就一类

      
for i in range(len(datapath)):
    
    tag=datapath[i]               #-----  观测目标 G:\20190518\HA\12741
    ser=os.listdir(tag)           #----- 获取时间段   ['010704']
    for j in range(len(ser)):
         
        subdir=os.path.join(tag,ser[j])
        wave=os.listdir(subdir)            #  -----   获取线心偏带 ['B050', 'CENT', 'R050']
        
        for k in range(len(wave)):
            
            wavesubdir=os.path.join(subdir,wave[k])
       
            waverepath=os.path.join(redrive,os.path.splitdrive(wavesubdir)[1])
            xyy.mkdir(waverepath)#G:\20190518\HA\12741\010704\B050
            print('开始处理',wavesubdir)
            for m in range(len(flatfile)):#['E:\\20190518\\HA\\FLAT00\\B050\\flat.fits', 'E:\\20190518\\HA\\FLAT00\\CENT\\flat.fits', 'E:\\20190518\\HA\\FLAT00\\R050\\flat.fits']
                if flatfile[m].find(wave[k])>=0:
                    flatname=flatfile[m]
                    print('所用平场：',flatname)
                    print() 
                    flat=fits.getdata(flatname) 
                    
                    flat=flat-dark 
                    flatmp=flat/np.median(flat)
            
            tim=os.listdir(wavesubdir)       # -----  获取组数  某偏带下的文件夹名。每个文件夹有100个fits文件
            
            for n in range(len(tim)):
                
                timdir=os.path.join(wavesubdir,tim[n])
                print('第',n+1,'组',tim[n])
                
                files=os.listdir(timdir)   #这一组下的fits文件
                numb=len(files)
                
                head=fits.getheader(os.path.join(timdir,files[0]))
                data_obs=head['DATE_OBS'][0:19].replace(':', '_') 
         
                ###ysize=head['NAXIS1']
                ###xsize=head['NAXIS2'] 
                 
                fitsname=os.path.join(waverepath,'Ha_'+wave[k]+'_'+data_obs)+'_lev1.fits'
                
                if os.path.exists(fitsname):
                    
                    print('已经处理过 ! 开始下一组')
                    print() 
                    continue
      
                print('请等待。。。') 
                
                cube=np.empty([numb,rcxsize,rcysize], dtype=np.float32) 
                filename=os.path.join(timdir,files[0])
                ini=(fits.getdata(filename)-dark)/flatmp
                initmp=ini[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
                if sobel==1:
                    initmp=filters.sobel(filters.gaussian(initmp,5.0))
                initmp_gpu=cp.asarray(initmp)    
                # ----------------------    平场并且第一次对齐   
                for nn in range(numb):
                    
                    filename=os.path.join(timdir,files[nn])
                    data=(fits.getdata(filename)-dark)/flatmp 
                    datatmp=data[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
                    if sobel==1:
                        datatmp=filters.sobel(filters.gaussian(datatmp,5.0)) 
                           
                    datatmp_gpu=cp.asarray(datatmp)
                    cc,corr=xyy.corrmaxloc_gpu(initmp_gpu, datatmp_gpu)                   
                    #cc,corr=xyy.corrmaxloc(initmp, datatmp)
                    
                    tmp=xyy.imgshift(data,[-cc[0],-cc[1]])
                    cube[nn,:,:] = tmp[0:rcxsize,0:rcysize]
                               
                ##---------------------------------------------
                
                ###------------------------------------- 视宁度和传递函数
                cubesr=cube[:,srstx:srstx+srxsize,srsty:srsty+srysize]
                r0,index=xyy.cubesrdevr0(cubesr,srsize,winsr,sitfdata,diameter,diaratio,maxfre,0.00,0.06,start_r0,step_r0)

                sitf=xyy.GetSitf(sitfdata,maxfre,rcxsize,index)

                print() 
                
                #----------------------------    选帧（1计算功率谱，2环带积分，3排序）

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
                    
                initmp_gpu=cp.asarray(initmp)    
                # ----------------------   对齐   
                for nn in range(cubesort0.shape[0]):
                    
                    data=cubesort0[nn,:,:].copy()
                    datatmp=data[corstart[0]:corstart[0]+corsize[0],corstart[1]:corstart[1]+corsize[1]]
                    if sobel==1:
                        datatmp=filters.sobel(filters.gaussian(datatmp,5.0))
                              
                    datatmp_gpu=cp.asarray(datatmp)
                    cc,corr=xyy.corrmaxloc_gpu(initmp_gpu, datatmp_gpu)
                    
                    ####cc,corr=xyy.corrmaxloc(initmp, datatmp)
                    
                    tmp=xyy.imgshift(data,[-cc[0],-cc[1]])
                    cubesort0[nn,:,:]=tmp
                    
                averg=np.mean(cubesort0, axis=0).astype(np.float32)#叠加
                           
                ##-------------   功率谱退卷积    
                img=xyy.ImgPSDdeconv(averg,sitf)
                
                head['CODE2'] = r0
                
                result=xyy.ImgFilted(img,gussf)
                
                result=result/np.median(result)*np.median(averg)

                xyy.writefits(fitsname,result.astype(np.float32),head)
                
                                                 
print('==================  Level1 处理结束！==================')