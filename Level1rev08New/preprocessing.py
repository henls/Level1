# encoding:utf-8
import numpy as np
from astropy.io import fits
import os
import datetime
import json
import re
from xyy_lib import xyy_lib as xyy
from multiprocessing import Process
import time
#预处理程序
#平均Dark，归一化FLAT
#将结果写到log日志中
#2020/4/29
#添加对无偏带的支持
#2020/5/9
#增加文件完整性检验
#2020/5/17
#更加清晰的检索方式
def AveDark(DirDark,DirSaveDark,DirLog):
    #1000张不分偏带
    #DirDark:Dark的路径
    #DirSaveDark:处理后Dark保存的路径
    #Dirlog：日志的路径，dark处理好后将dark的路径写入日志
    today_time = datetime.datetime.now().strftime('%Y%m%d')
    DirLog = DirLog
    DirDark = DirDark
    DirSaveDark = DirSaveDark
    while os.path.exists(os.path.join(DirSaveDark,today_time+'dark.fits')) == False:
        number = 10
        DirLog = DirLog
        DirDark = DirDark
        DarkFits = os.listdir(DirDark)
        if len(DarkFits) >= number:
            t = 0
            DirDarkFits = []
            for i in DarkFits:
                DirDarkFits.append(os.path.join(DirDark,DarkFits[t]))
                t += 1
            data = 0
            Filter = DirDarkFits.copy()
            for i in DirDarkFits:
                if os.path.getsize(i) != 2111040:
                    Filter.remove(i)
            for i in Filter:
                data += np.array(fits.open(i)[0].data,dtype = np.float32)
            data /= len(Filter)
            try:
                os.mkdir(DirSaveDark)
            except Exception as e:
                pass
            #today_time = datetime.datetime.now().strftime('%Y%m%d')
            #today_time = '20190525'
            try:
                fits.writeto(os.path.join(DirSaveDark,today_time+'dark.fits'),np.array(data))
            except Exception as e:
                pass
            print('Dark 计算完毕')
            #log = open(DirLog,'a+')
            #log.writelines('\n处理后的Dark路径：'+os.path.join(DirSaveDark,today_time+'dark.fits'))
            #log.close()
            xyy.JsonWrite(DirLog,'AveDark',[os.path.join(DirSaveDark,today_time+'dark.fits')])
            #print(DirLog)
            #print([os.path.join(DirSaveDark,today_time+'dark.fits')])

def AveFlat(DirFlat,DirSaveFlat,DirLog):
    #每个偏带3000张
    #DirFlat是flat路径
    #DirSaveFlat是处理过后对应偏带的flat路径
    today_time = datetime.datetime.now().strftime('%Y%m%d')
    DirLog = DirLog
    DirFlat = DirFlat
    DirSaveFlat = DirSaveFlat
    bandoff = DirFlat.split('/')[-1]
    Sig_Flat = DirFlat.split('/')[-2]
    SaveName = os.path.join(DirSaveFlat,today_time+Sig_Flat+bandoff+'.fits')
    while os.path.exists(SaveName) == False:
        #print('一次')
        try:
            os.mkdir(DirSaveFlat)
        except Exception as e:
            pass
        #today_time = datetime.datetime.now().strftime('%Y%m%d')
        #today_time = '20190525'
        #print(DirFlat)
        number = 10
        DirFlatFits = os.listdir(DirFlat)
        if len(DirFlatFits) >= number:
            #print('Flat Starting.....')
            data = 0
            Filter = DirFlatFits.copy()
            for i in DirFlatFits:
                if os.path.getsize(os.path.join(DirFlat,i)) != 2111040:
                    Filter.remove(i)
            for i in Filter:
                data += np.array(fits.open(os.path.join(DirFlat,i))[0].data,dtype = np.float32)
            data /= len(Filter)
            #today_time = datetime.datetime.now().strftime('%Y%m%d')
            #today_time = '20190518'
            try:
                fits.writeto(SaveName,np.array(data))
                #log = open(DirLog,'a+')
                #log.writelines('\n处理后的Flat路径：'+os.path.join(DirSaveFlat,today_time+'CFlat.fits'))
                #log.close()
                xyy.JsonWrite(DirLog,'AveFlat',[SaveName])
                print('Flat计算完毕')
            except Exception as e:
                #print(e)
                pass


def DarkProc(log_dir,DirDarknew,DirSaveDark):
    while 1:
        try:
            time.sleep(0.1)
            today_time = datetime.datetime.now().strftime('%Y%m%d')
            logs = open(os.path.join(log_dir,today_time+'.log'),'r')
            jsinf = json.load(logs)
            logs.close()
            DirDark = jsinf['Darkoff']
            if DirDark != DirDarknew and len(DirDarknew) != 0:
                newDark = DirDark.copy()
                [DirDark.remove(i) for i in DirDarknew]
                for i in range(len(DirDark)):
                    AveDark(DirDark[i],DirSaveDark,log_dir)
                DirDarknew = newDark
            elif DirDark != DirDarknew and len(DirDarknew) == 0:
                DirDarknew = DirDark
                for i in range(len(DirDark)):
                    AveDark(DirDark[i],DirSaveDark,log_dir)
            break
        except Exception as e:
            #print(e)
            pass
        
        

def FlatProc(log_dir,DirFlatnew,DirSaveFlat):
    while 1:
        try:
            today_time = datetime.datetime.now().strftime('%Y%m%d')
            time.sleep(0.1)
            logs = open(os.path.join(log_dir,today_time+'.log'),'r')
            jsinf = json.load(logs)
            logs.close()
            DirFlat = jsinf['Flatoff']
            #print(DirFlat)
            #print(DirFlatnew)
            if DirFlat != DirFlatnew and len(DirFlatnew) != 0:
                newFlat = DirFlat.copy()
                [DirFlat.remove(i) for i in DirFlatnew]
                for i in range(len(DirFlat)):
                    AveFlat(DirFlat[i],DirSaveFlat,log_dir)
                DirFlatnew = newFlat
            elif DirFlat != DirFlatnew and len(DirFlatnew) == 0:
                DirFlatnew = DirFlat
                for i in range(len(DirFlat)):
                    AveFlat(DirFlat[i],DirSaveFlat,log_dir)
        except Exception as e:
            #print(e)
            pass


def main(jsonfile):
    jsonfile = jsonfile
    f = open(jsonfile,'r')
    paras = json.load(f)
    f.close()
    log_dir = paras['log_dir']
    DirSaveDark = paras['DirSaveDark']
    DirSaveFlat = paras['DirSaveFlat']
    today_time = datetime.datetime.now().strftime('%Y%m%d')
    while 1:
        try:
            time.sleep(0.1)
            logs = open(os.path.join(log_dir,today_time+'.log'),'r')
            jsinf = json.load(logs)
            logs.close()
            break
        except Exception as e:
            pass
#分离Dark和Flat
    '''try:
        DirDarknew = jsinf['Darkoff']
    except Exception as e:
        DirDarknew = []
    if len(DirDarknew) != 0:
        for i in range(len(DirDarknew)):
            AveDark(DirDarknew[i],DirSaveDark,log_dir)
    try:
        DirFlatnew = jsinf['Flatoff']
    except Exception as e:
        DirFlatnew = []
    if len(DirFlatnew) != 0:
        for i in range(len(DirFlatnew)):
            AveFlat(DirFlatnew[i],DirSaveFlat,log_dir)'''
    DirDarknew = []
    DirFlatnew = []
    #print(1)
    P1 = Process(target=DarkProc,args=(log_dir,DirDarknew,DirSaveDark))
    P1.start()
    #print('启动flat进程')
    P2 = Process(target=FlatProc,args=(log_dir,DirFlatnew,DirSaveFlat))
    P2.start()
    #print('flat启动成功')
            



if __name__ == '__main__':
    jsonfile = r'/home/wangxinhua/level1/Level1rev08New/json.txt'
    main(jsonfile)
