# encoding:utf-8
import pyinotify
import json
import re
import os
import datetime

#负责监控data/2020并产生log日志
#2020/4/28
#添加对无偏带情况的支持
#2020/4/29
#添加无偏带标志位OnlyBand
#2020/5/4
#修改文件夹匹配方式，使其更容易匹配到想要的结果
#2020/5/8
#修复当日文件已经产生导致无法正常处理的bug
class EventHandler(pyinotify.ProcessEvent):
    def passlogdir(self,logdir):
        self.log_dir = logdir
    def process_IN_CREATE(self,event):
        log_dir = self.log_dir
        path = event.pathname
        today_time = re.findall(r'\d{8}', path)[0]
        try:
            os.mkdir(log_dir)
        except Exception as e:
            print('Warning '+log_dir+' existed')
        Now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        while True:
            try:
		#5/4改
                dir0 = os.listdir(path)[0]#HA
                
                #dir1 = re.findall(r'\d+[^FLAT00]\d+',str(os.listdir(os.path.join(path,dir0))))[0]#1274
                Imgs = os.listdir(os.path.join(path,dir0))
                if 'dark' in Imgs:
                    dark1 = 'dark'
                    flat1 = 'FLAT00'
                    Imgs.remove('dark')
                    Imgs.remove('FLAT00')
                    dir1 = Imgs[0]
                else:
                    dark1 = 'Dark'
                    flat1 = 'FLAT00'
                    Imgs.remove('Dark')
                    Imgs.remove('FLAT00')
                    dir1 = Imgs[0]
                #print(dark1)
                #print(flat1)
                #print(dir1)
              
                dir2 = os.listdir(os.path.join(path,dir0,dir1))[0]#010704
                try:
                    offband0 = re.findall(r'[B]\d+',str(os.listdir(os.path.join(path,dir0,dir1,dir2))))[0]#B050
                except Exception as e:
                    offband0=[]
                offband1 = re.findall(r'[C][E][N][T]',str(os.listdir(os.path.join(path,dir0,dir1,dir2))))[0]#CENT
                try:
                    offband2 = re.findall(r'[R]\d+',str(os.listdir(os.path.join(path,dir0,dir1,dir2))))[0]#R050
                except Exception as e:
                    offband2=[]
                if len(offband0) != 0 and len(offband2) != 0:
                    dirband0 = os.path.join(path,dir0,dir1,dir2,offband0)
                    dirband1 = os.path.join(path,dir0,dir1,dir2,offband1)
                    dirband2 = os.path.join(path,dir0,dir1,dir2,offband2)
                    
                    #dark1 = re.findall(r'[Dark]+',str(os.listdir(os.path.join(path,dir0))))[0]#Dark
                    dark2 = os.listdir(os.path.join(path,dir0,dark1))[0]#002804
                    dark3 = os.listdir(os.path.join(path,dir0,dark1,dark2))[0]#CENT
                    dark4 = os.listdir(os.path.join(path,dir0,dark1,dark2,dark3))[0]#002804
                    darkdir = os.path.join(path,dir0,dark1,dark2,dark3,dark4)
                    #flat1 = re.findall(r'[FLAT00]+',str(os.listdir(os.path.join(path,dir0))))[0]
                    flat2 = os.path.join(path,dir0,flat1,offband0)
                    flat3 = os.path.join(path,dir0,flat1,offband1)
                    flat4 = os.path.join(path,dir0,flat1,offband2)
                    OnlyBand = 0
                else:
                    #只有CENT band
                    OnlyBand = 1
                    dirband1 = os.path.join(path,dir0,dir1,dir2,offband1)
                    #dark1 = re.findall(r'[Dark]+',str(os.listdir(os.path.join(path,dir0))))[0]#Dark
                    dark2 = os.listdir(os.path.join(path,dir0,dark1))[0]#002804
                    dark3 = os.listdir(os.path.join(path,dir0,dark1,dark2))[0]#CENT
                    dark4 = os.listdir(os.path.join(path,dir0,dark1,dark2,dark3))[0]#002804
                    darkdir = os.path.join(path,dir0,dark1,dark2,dark3,dark4)
                    #flat1 = re.findall(r'[FLAT00]+',str(os.listdir(os.path.join(path,dir0))))[0]
                    flat3 = os.path.join(path,dir0,flat1,offband1)
                break
            except Exception as e:
                #print(e)
                #print('Note:observation have not begun yet')
                pass
        print('发现新观测数据，开始处理....................')
        file = open(os.path.join(log_dir,today_time)+'.log','w+')
        file.writelines('BeginTime: '+Now_time)
        file.writelines('\nRoot: '+event.pathname)
        file.writelines('\nActiveRegion: '+dir1)
        file.writelines('\nRecoreTime: '+dir2)
        #file.writelines('\nBandOff0: '+dirband0)
        #file.writelines('\nBandOff1: '+dirband1)
        #file.writelines('\nBandOff2: '+dirband2)
        #file.writelines('\nDark: '+darkdir)
        if OnlyBand == 0:
            file.writelines('\nOnlyBand: '+'0')
            file.writelines('\nBandOff0: '+dirband0)
            file.writelines('\nBandOff1: '+dirband1)
            file.writelines('\nBandOff2: '+dirband2)
            file.writelines('\nDark: '+darkdir)
            file.writelines('\nFlatOff0: '+flat2)
            file.writelines('\nFlatOff1: '+flat3)
            file.writelines('\nFlatOff2: '+flat4)
        else:
            file.writelines('\nOnlyBand: '+'1')
            file.writelines('\nBandOff1: '+dirband1)
            file.writelines('\nDark: '+darkdir)
            file.writelines('\nFlatOff1: '+flat3)
        file.close()


def DelayProcess(log_dir,path):
    log_dir = log_dir
    path = path
    today_time = re.findall(r'\d{8}', path)[0]
    try:
        os.mkdir(log_dir)
    except Exception as e:
        print('Warning '+log_dir+' existed')
    Now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    while True:
        try:
		#5/4改
            dir0 = os.listdir(path)[0]#HA
            
            #dir1 = re.findall(r'\d+[^FLAT00]\d+',str(os.listdir(os.path.join(path,dir0))))[0]#1274
            Imgs = os.listdir(os.path.join(path,dir0))
            if 'dark' in Imgs:
                dark1 = 'dark'
                flat1 = 'FLAT00'
                Imgs.remove('dark')
                Imgs.remove('FLAT00')
                dir1 = Imgs[0]
            else:
                dark1 = 'Dark'
                flat1 = 'FLAT00'
                Imgs.remove('Dark')
                Imgs.remove('FLAT00')
                dir1 = Imgs[0]
            #print(dark1)
            #print(flat1)
            #print(dir1)
          
            dir2 = os.listdir(os.path.join(path,dir0,dir1))[0]#010704
            try:
                offband0 = re.findall(r'[B]\d+',str(os.listdir(os.path.join(path,dir0,dir1,dir2))))[0]#B050
            except Exception as e:
                offband0=[]
            offband1 = re.findall(r'[C][E][N][T]',str(os.listdir(os.path.join(path,dir0,dir1,dir2))))[0]#CENT
            try:
                offband2 = re.findall(r'[R]\d+',str(os.listdir(os.path.join(path,dir0,dir1,dir2))))[0]#R050
            except Exception as e:
                offband2=[]
            if len(offband0) != 0 and len(offband2) != 0:
                dirband0 = os.path.join(path,dir0,dir1,dir2,offband0)
                dirband1 = os.path.join(path,dir0,dir1,dir2,offband1)
                dirband2 = os.path.join(path,dir0,dir1,dir2,offband2)
                
                #dark1 = re.findall(r'[Dark]+',str(os.listdir(os.path.join(path,dir0))))[0]#Dark
                dark2 = os.listdir(os.path.join(path,dir0,dark1))[0]#002804
                dark3 = os.listdir(os.path.join(path,dir0,dark1,dark2))[0]#CENT
                dark4 = os.listdir(os.path.join(path,dir0,dark1,dark2,dark3))[0]#002804
                darkdir = os.path.join(path,dir0,dark1,dark2,dark3,dark4)
                #flat1 = re.findall(r'[FLAT00]+',str(os.listdir(os.path.join(path,dir0))))[0]
                flat2 = os.path.join(path,dir0,flat1,offband0)
                flat3 = os.path.join(path,dir0,flat1,offband1)
                flat4 = os.path.join(path,dir0,flat1,offband2)
                OnlyBand = 0
            else:
                #只有CENT band
                OnlyBand = 1
                dirband1 = os.path.join(path,dir0,dir1,dir2,offband1)
                #dark1 = re.findall(r'[Dark]+',str(os.listdir(os.path.join(path,dir0))))[0]#Dark
                dark2 = os.listdir(os.path.join(path,dir0,dark1))[0]#002804
                dark3 = os.listdir(os.path.join(path,dir0,dark1,dark2))[0]#CENT
                dark4 = os.listdir(os.path.join(path,dir0,dark1,dark2,dark3))[0]#002804
                darkdir = os.path.join(path,dir0,dark1,dark2,dark3,dark4)
                #flat1 = re.findall(r'[FLAT00]+',str(os.listdir(os.path.join(path,dir0))))[0]
                flat3 = os.path.join(path,dir0,flat1,offband1)
            break
        except Exception as e:
            #print(e)
            #print('Note:observation have not begun yet')
            pass
    print('发现新观测数据，开始处理....................')
    file = open(os.path.join(log_dir,today_time)+'.log','w+')
    file.writelines('BeginTime: '+Now_time)
    file.writelines('\nRoot: '+path)
    file.writelines('\nActiveRegion: '+dir1)
    file.writelines('\nRecoreTime: '+dir2)
    #file.writelines('\nBandOff0: '+dirband0)
    #file.writelines('\nBandOff1: '+dirband1)
    #file.writelines('\nBandOff2: '+dirband2)
    #file.writelines('\nDark: '+darkdir)
    if OnlyBand == 0:
        file.writelines('\nOnlyBand: '+'0')
        file.writelines('\nBandOff0: '+dirband0)
        file.writelines('\nBandOff1: '+dirband1)
        file.writelines('\nBandOff2: '+dirband2)
        file.writelines('\nDark: '+darkdir)
        file.writelines('\nFlatOff0: '+flat2)
        file.writelines('\nFlatOff1: '+flat3)
        file.writelines('\nFlatOff2: '+flat4)
    else:
        file.writelines('\nOnlyBand: '+'1')
        file.writelines('\nBandOff1: '+dirband1)
        file.writelines('\nDark: '+darkdir)
        file.writelines('\nFlatOff1: '+flat3)
    file.close()    


def inotify(jsonfile):
    jsonfile = jsonfile
    f = open(jsonfile,'r')
    paras = json.load(f)
    f.close()
    workdir = paras['workdir']#"/home/user/data/2020"
    log_dir = paras['log_dir']
    handler = EventHandler()
    handler.passlogdir(log_dir)
    wm = pyinotify.WatchManager()
    mask = pyinotify.IN_CREATE
    notifier = pyinotify.Notifier(wm,handler)
    while 1:
        today = datetime.datetime.now().strftime('%Y%m%d')
        today_dir = os.path.join(workdir,today)
        if os.path.exists(workdir) and os.path.exists(today_dir) == False:
            #print(258)
            wm.add_watch(workdir,mask)
            notifier.loop()
            break
        elif os.path.exists(today_dir):
            DelayProcess(log_dir,today_dir)
            break
        else:
            pass

if __name__ == "__main__":
    jsonfile = r'/home/wangxinhua/level1/Level1rev07New/json.txt'
    inotify(jsonfile)
