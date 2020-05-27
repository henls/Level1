# encoding:utf-8
import numpy as np
from astropy.io import fits
import os
import datetime
import json
import re
#预处理程序
#平均Dark，归一化FLAT
#将结果写到log日志中
#2020/4/29
#添加对无偏带的支持
#2020/5/9
#增加文件完整性检验
def AveDark(DirDark,DirSaveDark,DirLog):
    #1000张不分偏带
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
            log = open(DirLog,'a+')
            log.writelines('\n处理后的Dark路径：'+os.path.join(DirSaveDark,today_time+'dark.fits'))
            log.close()

def AveFlat(DirFlat,DirSaveFlat,DirLog):
    #每个偏带3000张
    #DirFlat是各偏带下flat路径的列表
    #DirSaveFlat是处理过后对应偏带的flat路径
    today_time = datetime.datetime.now().strftime('%Y%m%d')
    DirLog = DirLog
    DirFlat = DirFlat
    DirSaveFlat = DirSaveFlat
    while os.path.exists(os.path.join(DirSaveFlat,today_time+'CFlat.fits')) == False:
        
        try:
            os.mkdir(DirSaveFlat)
        except Exception as e:
            pass
        #today_time = datetime.datetime.now().strftime('%Y%m%d')
        #today_time = '20190525'
        #print(DirFlat)
        number = 10
        if len(DirFlat) == 1:
            DirFlatFits = os.listdir(DirFlat[0])
            if len(DirFlatFits) >= number:
                data = 0
                Filter = DirFlatFits.copy()
                for i in DirFlatFits:
                    if os.path.getsize(os.path.join(DirFlat[0],i)) != 2111040:
                        Filter.remove(i)
                for i in Filter:
                    data += np.array(fits.open(os.path.join(DirFlat[0],i))[0].data,dtype = np.float32)
                data /= len(Filter)
                #today_time = datetime.datetime.now().strftime('%Y%m%d')
                #today_time = '20190518'
                try:
                   
                    fits.writeto(os.path.join(DirSaveFlat,today_time+'CFlat.fits'),np.array(data))
           
                    log = open(DirLog,'a+')
                    log.writelines('\n处理后的Flat路径：'+os.path.join(DirSaveFlat,today_time+'CFlat.fits'))
                    log.close()
                except Exception as e:
                    #print(e)
                    pass
        else:
            DirFlatFits0 = os.listdir(DirFlat[0])#C
            DirFlatFits1 = os.listdir(DirFlat[1])#R
            DirFlatFits2 = os.listdir(DirFlat[2])#B
            if len(DirFlatFits0) >= number and len(DirFlatFits1) >= number and len(DirFlatFits2) >= number:
                data = 0
                #print(DirFlat[0])
                Filter = DirFlatFits0.copy()
                for i in DirFlatFits0:
                    if os.path.getsize(os.path.join(DirFlat[0],i)) != 2111040:
                        Filter.remove(i)
                
                for i in Filter:
                    data += np.array(fits.open(os.path.join(DirFlat[0],i))[0].data,dtype = np.float32)
                data /= len(Filter)
                
                a = len(re.findall(r'[C][E][N][T]',DirFlat[0]))
                b = len(re.findall(r'[B]\d+',DirFlat[0]))
                c = len(re.findall(r'[R]\d+',DirFlat[0]))
                if a != 0:
                    #today_time = datetime.datetime.now().strftime('%Y%m%d')
                    #today_time = '20190518'
                    try:
                        fits.writeto(os.path.join(DirSaveFlat,today_time+'CFlat.fits'),np.array(data))
                        log = open(DirLog,'a+')
                        log.writelines('\n处理后的Flat路径：'+os.path.join(DirSaveFlat,today_time+'CFlat.fits'))
                        log.close()
                    except Exception as e:
                        #print(e)
                        pass
                elif b != 0:
                    #today_time = datetime.datetime.now().strftime('%Y%m%d')
                    #today_time = '20190518'
                    try:
                        fits.writeto(os.path.join(DirSaveFlat,today_time+'BFlat.fits'),np.array(data))
                        log = open(DirLog,'a+')
                        log.writelines('\n处理后的Flat路径：'+os.path.join(DirSaveFlat,today_time+'BFlat.fits'))
                        log.close()
                    except Exception as e:
                        #print(e)
                        pass
                elif c != 0:
                    #today_time = datetime.datetime.now().strftime('%Y%m%d')
                    #today_time = '20190518'
                    try:
                        fits.writeto(os.path.join(DirSaveFlat,today_time+'RFlat.fits'),np.array(data))
                        log = open(DirLog,'a+')
                        log.writelines('\n处理后的Flat路径：'+os.path.join(DirSaveFlat,today_time+'RFlat.fits'))
                        log.close()
                    except Exception as e:
                        #print(e)
                        pass
                Filter = DirFlatFits1.copy()
                for i in DirFlatFits1:
                    if os.path.getsize(os.path.join(DirFlat[1],i)) != 2111040:
                        Filter.remove(i)
                for i in Filter:
                    data += np.array(fits.open(os.path.join(DirFlat[1],i))[0].data,dtype = np.float32)
                data /= len(Filter)
        
                a = len(re.findall(r'[C][E][N][T]',DirFlat[1]))
                b = len(re.findall(r'[B]\d+',DirFlat[1]))
                c = len(re.findall(r'[R]\d+',DirFlat[1]))
                if a != 0:
                    #today_time = datetime.datetime.now().strftime('%Y%m%d')
                    #today_time = '20190518'
                    try:
                        fits.writeto(os.path.join(DirSaveFlat,today_time+'CFlat.fits'),np.array(data))
                        log = open(DirLog,'a+')
                        log.writelines('\n处理后的Flat路径：'+os.path.join(DirSaveFlat,today_time+'CFlat.fits'))
                        log.close()
                    except Exception as e:
                        #print(e)
                        pass
                elif b != 0:
                    #today_time = datetime.datetime.now().strftime('%Y%m%d')
                    #today_time = '20190518'
                    try:
                        fits.writeto(os.path.join(DirSaveFlat,today_time+'BFlat.fits'),np.array(data))
                        log = open(DirLog,'a+')
                        log.writelines('\n处理后的Flat路径：'+os.path.join(DirSaveFlat,today_time+'BFlat.fits'))
                        log.close()
                    except Exception as e:
                        print(e)
                elif c != 0:
                    #today_time = datetime.datetime.now().strftime('%Y%m%d')
                    try:
                        fits.writeto(os.path.join(DirSaveFlat,today_time+'RFlat.fits'),np.array(data))
                        log = open(DirLog,'a+')
                        log.writelines('\n处理后的Flat路径：'+os.path.join(DirSaveFlat,today_time+'RFlat.fits'))
                        log.close()
                    except Exception as e:
                        #print(e)
                        pass
                Filter = DirFlatFits2.copy()
                for i in DirFlatFits2:
                    if os.path.getsize(os.path.join(DirFlat[2],i)) != 2111040:
                        Filter.remove(i)
                for i in Filter:
                    data += np.array(fits.open(os.path.join(DirFlat[2],i))[0].data,dtype = np.float32)
                data /= len(Filter)
                a = len(re.findall(r'[C][E][N][T]',DirFlat[2]))
                b = len(re.findall(r'[B]\d+',DirFlat[2]))
                c = len(re.findall(r'[R]\d+',DirFlat[2]))
                if a != 0:
                    #today_time = datetime.datetime.now().strftime('%Y%m%d')
                    try:
                        fits.writeto(os.path.join(DirSaveFlat,today_time+'CFlat.fits'),np.array(data))
                        log = open(DirLog,'a+')
                        log.writelines('\n处理后的Flat路径：'+os.path.join(DirSaveFlat,today_time+'CFlat.fits'))
                        log.close()
                    except Exception as e:
                        #print(e)
                        pass
                elif b != 0:
                    #today_time = datetime.datetime.now().strftime('%Y%m%d')
                    try:
                        fits.writeto(os.path.join(DirSaveFlat,today_time+'BFlat.fits'),np.array(data))
                        log = open(DirLog,'a+')
                        log.writelines('\n处理后的Flat路径：'+os.path.join(DirSaveFlat,today_time+'BFlat.fits'))
                        log.close()
                    except Exception as e:
                        #print(e)
                        pass
                elif c != 0:
                    #today_time = datetime.datetime.now().strftime('%Y%m%d')
                    try:
                        fits.writeto(os.path.join(DirSaveFlat,today_time+'RFlat.fits'),np.array(data))
                        log = open(DirLog,'a+')
                        log.writelines('\n处理后的Flat路径：'+os.path.join(DirSaveFlat,today_time+'RFlat.fits'))
                        log.close()
                    except Exception as e:
                        #print(e)
                        pass
                print('Flat计算完毕')


def main(jsonfile):
    jsonfile = jsonfile
    f = open(jsonfile,'r')
    paras = json.load(f)
    f.close()
    log_dir = paras['log_dir']
    DirSaveDark = paras['DirSaveDark']
    DirSaveFlat = paras['DirSaveFlat']
    while 1:
        today_time = datetime.datetime.now().strftime('%Y%m%d')
        if os.path.exists(os.path.join(DirSaveFlat,today_time+'CFlat.fits')) or os.path.exists(os.path.join(DirSaveDark,today_time+'dark.fits')):
            pass
        else:
            try:
                #today_time = '20190525'
                DirLog = os.path.join(log_dir,today_time+'.log')
                logs = open(DirLog,'r')
                log = logs.readlines()
                logs.close()
                OnlyBand = int(log[4].split(':')[1])
                if OnlyBand == 0:
                    DarkFlatDirs = log[8:12]
                else:
                    DarkFlatDirs = log[6:8]
                DirDark = DarkFlatDirs[0].split(':')[1][1:-1]#字符串
                DirFlat = []#列表
            #print(DarkFlatDirs[1:])
                if OnlyBand == 0:
                    for i in DarkFlatDirs[1:]:
                        if '\n' in i:
                            DirFlat.append(i.split(':')[1][1:-1])
                        else:
                            DirFlat.append(i.split(':')[1][1:])
                else:
                    i = DarkFlatDirs[1]
                    if '\n' in i:
                        DirFlat.append(i.split(':')[1][1:-1])
                    else:
                        DirFlat.append(i.split(':')[1][1:])
                AveDark(DirDark,DirSaveDark,DirLog)
                AveFlat(DirFlat,DirSaveFlat,DirLog)
            except Exception as e:
                #print(e)
                pass

if __name__ == '__main__':
    jsonfile = r'/home/wangxinhua/level1/Level1rev07New/json.txt'
    main(jsonfile)
