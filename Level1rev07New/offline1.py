# encoding:utf-8
import pyinotify
import json
import re
import os
import _thread
import datetime
import threading
def main(jsonfile):
    #Now_time = datetime.datetime.now().strftime('%Y%m%d')
    f = open(jsonfile,'r')
    para = json.load(f)
    f.close()
    Dir_NewFile = para['Dir_NewFile']
    try:
        os.mkdir(Dir_NewFile)
    except Exception as e:
        pass
    while 1:
        log_dir = para['log_dir']
        Now_time = datetime.datetime.now().strftime('%Y%m%d')
        try:
            logs = open(os.path.join(log_dir,Now_time+'.log'),'r')
            log = logs.readlines()
            logs.close()
            OnlyBand = int(log[4].split(':')[1])
            if OnlyBand == 0:
                DirOffBand = log[5:8]
                OffBands = []
                for i in DirOffBand:
                    OffBands.append(i.split('/')[-1][:-1])#去掉换行符
            else:
                DirOffBand = log[5]
                OffBands = []
                OffBands.append(DirOffBand.split('/')[-1][:-1])
            a = os.listdir(DirOffBand[0].split(':')[-1][1:-1])
            b = os.listdir(DirOffBand[1].split(':')[-1][1:-1])
            c = os.listdir(DirOffBand[2].split(':')[-1][1:-1])
            afile = open(os.path.join(Dir_NewFile,Now_time+OffBands[0][0]+'.log'),'w')
            for i in a:
                afile.writelines(os.path.join(DirOffBand[0].split(':')[-1][1:-1],i)+'\n')
            afile.close()
            bfile = open(os.path.join(Dir_NewFile,Now_time+OffBands[1][0]+'.log'),'w')
            for i in b:
                bfile.writelines(os.path.join(DirOffBand[1].split(':')[-1][1:-1],i)+'\n')
            bfile.close()
            cfile = open(os.path.join(Dir_NewFile,Now_time+OffBands[2][0]+'.log'),'w')
            for i in c:
                cfile.writelines(os.path.join(DirOffBand[2].split(':')[-1][1:-1],i)+'\n')
            cfile.close()
        except Exception as e:
            jsonfile = r'/home/wangxinhua/level1/Level1rev07New/json.txt'
            main(jsonfile)
        break



if __name__ == '__main__':
    jsonfile = r'/home/wangxinhua/level1/Level1rev07New/json.txt'
    main(jsonfile)
