# encoding:utf-8
import pyinotify
import json
import re
import os
import _thread
import datetime
import threading
from xyy_lib import xyy_lib as xyy
import time
from multiprocessing import Process


#负责监控观测文件夹，在newfile文件夹下产生新文件的log文件
#2020/4/29
#添加对无偏带的支持

class EventHandler(pyinotify.ProcessEvent):


    def Dir(self,Dir_NewFile):
        self.Dir_NewFile = Dir_NewFile


    def process_IN_CREATE(self,event):
        path = event.pathname
        Dir_NewFile = self.Dir_NewFile
        file = open(Dir_NewFile,'a+')
        file.writelines(path+'\n')
        file.close


def inotify(NewFile,Dir_NewFile_offband):
    NewFile = NewFile
    Dir_NewFile_offband = Dir_NewFile_offband
    handler = EventHandler()
    handler.Dir(Dir_NewFile_offband)
    wm = pyinotify.WatchManager()
    mask = pyinotify.IN_CREATE
    notifier = pyinotify.Notifier(wm,handler)
    wm.add_watch(NewFile,mask)#CENT,B050,R050
    notifier.loop()


class myThread(threading.Thread):


    def __init__(self,log_dir,NewFile, Dir_NewFile_offband):
        threading.Thread.__init__(self)
        self.NewFile = NewFile
        self.Dir_NewFile_offband = Dir_NewFile_offband
        self.log_dir = log_dir
        #print(self.log_dir)

    def run(self):
        try:
            #print(self.log_dir)
            f = open(self.Dir_NewFile_offband,'a')
            f.close()
            xyy.JsonWrite(self.log_dir,'NewFile',[self.Dir_NewFile_offband])
        except Exception as e:
            #print(e)
            xyy.JsonWrite(self.log_dir,'NewFile',[self.Dir_NewFile_offband])
            pass
        
        inotify(self.NewFile,self.Dir_NewFile_offband)


class myProcessing(Process):


    def __init__(self,log_dir,NewFile, Dir_NewFile_offband):
        super().__init__()
        self.NewFile = NewFile
        self.Dir_NewFile_offband = Dir_NewFile_offband
        self.log_dir = log_dir
        #print(self.log_dir)

    def run(self):
        try:
            #print(self.log_dir)
            f = open(self.Dir_NewFile_offband,'a')
            f.close()
            xyy.JsonWrite(self.log_dir,'NewFile',[self.Dir_NewFile_offband])
        except Exception as e:
            #print(e)
            xyy.JsonWrite(self.log_dir,'NewFile',[self.Dir_NewFile_offband])
            pass

        inotify(self.NewFile,self.Dir_NewFile_offband)


def main(jsonfile):
    #Now_time = datetime.datetime.now().strftime('%Y%m%d')
    f = open(jsonfile,'r')
    para = json.load(f)
    f.close()
    Dir_NewFile = para['Dir_NewFile']
    log_dir = para['log_dir']
    try:
        os.mkdir(Dir_NewFile)
    except Exception as e:
        pass
    while 1:
        try:
            Now_time = datetime.datetime.now().strftime('%Y%m%d')
            f = open(os.path.join(log_dir,Now_time+'.log'),'r')
            jsinf = json.load(f)
            f.close()
            bandoff = jsinf['Bandoff']
            
            for i in bandoff:
                ActiveReg = i.split('/')[-3]
                StartPoint = i.split('/')[-2]
                band = i.split('/')[-1]
                if os.path.exists(os.path.join(Dir_NewFile,Now_time+ActiveReg+StartPoint+band+'.log')) == False:
                    #thread = myThread(log_dir,i,os.path.join(Dir_NewFile,Now_time+ActiveReg+StartPoint+band+'.log'))
                    #thread.start()
                    #time.sleep(0.5)
                    p = myProcessing(log_dir,i,os.path.join(Dir_NewFile,Now_time+ActiveReg+StartPoint+band+'.log'))
                    p.start()
                    time.sleep(0.1)
                    #print(os.path.join(Dir_NewFile,Now_time+ActiveReg+StartPoint+band+'.log'))
        except Exception as e:
            #print(e)
            pass

if __name__ == "__main__":
    jsonfile = r'/home/wangxinhua/level1/Level1/Level1rev08New/json.txt'
    main(jsonfile)
