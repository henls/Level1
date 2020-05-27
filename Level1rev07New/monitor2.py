# encoding:utf-8
import pyinotify
import json
import re
import os
import _thread
import datetime
import threading
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


    def __init__(self, NewFile, Dir_NewFile_offband):
        threading.Thread.__init__(self)
        self.NewFile = NewFile
        self.Dir_NewFile_offband = Dir_NewFile_offband


    def run(self):
        inotify(self.NewFile,self.Dir_NewFile_offband)


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
    
    #t = 0
    
            if OnlyBand == 0:
                thread1 = myThread(DirOffBand[0].split(':')[-1][1:-1],os.path.join(Dir_NewFile,Now_time+OffBands[0][0]+'.log'))
                thread2 = myThread(DirOffBand[1].split(':')[-1][1:-1],os.path.join(Dir_NewFile,Now_time+OffBands[1][0]+'.log'))
                thread3 = myThread(DirOffBand[2].split(':')[-1][1:-1],os.path.join(Dir_NewFile,Now_time+OffBands[2][0]+'.log'))
                thread1.start()
                thread2.start()
                thread3.start()
                thread1.join()
                thread2.join()
                thread3.join()
                print('线程启动成功')
            else:
                thread1 = myThread(DirOffBand.split(':')[-1][1:-1],os.path.join(Dir_NewFile,Now_time+OffBands[0][0]+'.log'))
                thread1.start()
                thread1.join()
                print('线程启动成功')
        except Exception as e:
            pass
        
if __name__ == "__main__":
    jsonfile = r'/home/wangxinhua/level1/Level1rev07New/json.txt'
    main(jsonfile)
