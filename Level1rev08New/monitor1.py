# encoding:utf-8
import pyinotify
import json
import re
import os
import datetime
from xyy_lib import xyy_lib as xyy
#负责监控data/2020并产生log日志
#2020/4/28
#添加对无偏带情况的支持
#2020/4/29
#添加无偏带标志位OnlyBand
#2020/5/4
#修改文件夹匹配方式，使其更容易匹配到想要的结果
#2020/5/8
#修复当日文件已经产生导致无法正常处理的bug
#2020/5/16
#改用新的文件树检索结构和新的日志格式
class EventHandler(pyinotify.ProcessEvent):
    def passlogdir(self,logdir):
        self.log_dir = logdir
    def process_IN_DELETE(self,event):
        while 1:
            if os.path.exists(workdir) and os.path.exists(today_dir) == False:
                print(258)
                print(workdir)
                print(mask)
                wm.add_watch(workdir,mask)
                notifier.loop()
                break
            #elif os.path.exists(today_dir):
            #    DelayProcess(log_dir,today_dir)
            #    break
            else:
                pass
    def process_IN_CREATE(self,event):
        #print(1)
        log_dir = self.log_dir
        path = event.pathname#workdir/20201315/
        #today_time = re.findall(r'\d{8}', path)[0]
        today_time = datetime.datetime.now().strftime('%Y%m%d')
        try:
            os.mkdir(log_dir)
        except Exception as e:
            print('Warning '+log_dir+' existed')
        Now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        xyy.JsonWrite(log_dir,'BeginTime',[Now_time])
        xyy.JsonWrite(log_dir,'Root',[path])
        DarkoffNew = []
        FlatoffNew = []
        ActiveRegNew = []
        BandoffNew=[]
        while True:
            try:
		#5/4改  
                dir0 = os.listdir(path)[0]
                #dir1 = re.findall(r'\d+[^FLAT00]\d+',str(os.listdir(os.path.join(path,dir0))))[0]#1274
                Imgs = os.listdir(os.path.join(path,dir0))#Imgs:Dark、ActiveRegion、Flat
                #解析dark
                Dark = re.findall(r'[D][a][r][k]',str(Imgs),re.IGNORECASE)#Dark
                #解析flat
                Flat = re.findall(r'[F][L][A][T]\d+',str(Imgs),re.IGNORECASE)#FLAT00,FLAT01
                [Imgs.remove(i) for i in Dark]
                [Imgs.remove(i) for i in Flat]
                
                ActiveReg = Imgs#Disk_centrt,01254
                #print(Dark)
                #print(Flat)
                #print(ActiveReg)
                Darkoff = []
                for i in range(len(Dark)):
                    dark2 = os.listdir(os.path.join(path,dir0,Dark[i]))#001812,001854
                    #print(dark2)
                    for j in range(len(dark2)):
                        dark3 = os.listdir(os.path.join(path,dir0,Dark[i],dark2[j]))#CENT
                        dark4 = os.listdir(os.path.join(path,dir0,Dark[i],dark2[j],dark3[0]))#001812
                        Darkoff.append(os.path.join(path,dir0,Dark[i],dark2[j],dark3[0],dark4[0]))
                if Darkoff != DarkoffNew:
                    xyy.JsonWrite(log_dir,'Darkoff',Darkoff)
                    DarkoffNew = Darkoff
                Flatoff = []
                for i in range(len(Flat)):
                    flat2 = os.listdir(os.path.join(path,dir0,Flat[i]))#B080,CENT,R080
                    for j in range(len(flat2)):
                        Flatoff.append(os.path.join(path,dir0,Flat[i],flat2[j]))
                if Flatoff != FlatoffNew:
                    xyy.JsonWrite(log_dir,'Flatoff',Flatoff)
                    FlatoffNew = Flatoff
                Bandoff = []
                #ActiveRegNew = ActiveReg
                if ActiveReg!= ActiveRegNew:
                    xyy.JsonWrite(log_dir,'ActiveRegion',ActiveReg)
                    ActiveRegNew = ActiveReg
                for i in range(len(ActiveReg)):
                    dir2 = os.listdir(os.path.join(path,dir0,ActiveReg[i]))#010704,025426
                   # xyy.JsonWrite(log_dir,'RecordTime',dir2)
                    for j in range(len(dir2)):
                        dir3 = os.listdir(os.path.join(path,dir0,ActiveReg[i],dir2[j]))#CENT BO50 R050
                        for k in range(len(dir3)):
                            Bandoff.append(os.path.join(path,dir0,ActiveReg[i],dir2[j],dir3[k]))
                if Bandoff != BandoffNew:
                    BandoffNew = Bandoff
                    xyy.JsonWrite(log_dir,'Bandoff',Bandoff)
                    for i in range(len(ActiveReg)):
                        dir2 = os.listdir(os.path.join(path,dir0,ActiveReg[i]))#010704,025426
                        xyy.JsonWrite(log_dir,'RecordTime',dir2)
            except Exception as e:
                #print(e)
                pass

def DelayProcess(log_dir,path):
    #print('开始延时处理')
    log_dir = log_dir
    path = path#workdir/20201315/
    #today_time = re.findall(r'\d{8}', path)[0]
    today_time = datetime.datetime.now().strftime('%Y%m%d')
    try:
        os.mkdir(log_dir)
    except Exception as e:
        print('Warning '+log_dir+' existed')
    Now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    xyy.JsonWrite(log_dir,'BeginTime',[Now_time])
    xyy.JsonWrite(log_dir,'Root',[path])
    DarkoffNew = []
    FlatoffNew = []
    ActiveRegNew = []
    BandoffNew=[]
    while True:
        try:
		#5/4改  
            dir0 = os.listdir(path)[0]
            #dir1 = re.findall(r'\d+[^FLAT00]\d+',str(os.listdir(os.path.join(path,dir0))))[0]#1274
            Imgs = os.listdir(os.path.join(path,dir0))#Imgs:Dark、ActiveRegion、Flat
            #解析dark
            Dark = re.findall(r'[D][a][r][k]',str(Imgs),re.IGNORECASE)#Dark
            #解析flat
            Flat = re.findall(r'[F][L][A][T]\d+',str(Imgs),re.IGNORECASE)#FLAT00,FLAT01
            [Imgs.remove(i) for i in Dark]
            [Imgs.remove(i) for i in Flat]
            
            ActiveReg = Imgs#Disk_centrt,01254
            #print(Dark)
            #print(Flat)
            #print(ActiveReg)
            Darkoff = []
            for i in range(len(Dark)):
                dark2 = os.listdir(os.path.join(path,dir0,Dark[i]))#001812,001854
                #print(dark2)
                for j in range(len(dark2)):
                    dark3 = os.listdir(os.path.join(path,dir0,Dark[i],dark2[j]))#CENT
                    dark4 = os.listdir(os.path.join(path,dir0,Dark[i],dark2[j],dark3[0]))#001812
                    Darkoff.append(os.path.join(path,dir0,Dark[i],dark2[j],dark3[0],dark4[0]))
            if Darkoff != DarkoffNew:
                xyy.JsonWrite(log_dir,'Darkoff',Darkoff)
                DarkoffNew = Darkoff
            Flatoff = []
            for i in range(len(Flat)):
                flat2 = os.listdir(os.path.join(path,dir0,Flat[i]))#B080,CENT,R080
                for j in range(len(flat2)):
                    Flatoff.append(os.path.join(path,dir0,Flat[i],flat2[j]))
            if Flatoff != FlatoffNew:
                xyy.JsonWrite(log_dir,'Flatoff',Flatoff)
                FlatoffNew = Flatoff
            Bandoff = []
            #ActiveRegNew = ActiveReg
            if ActiveReg!= ActiveRegNew:
                xyy.JsonWrite(log_dir,'ActiveRegion',ActiveReg)
                ActiveRegNew = ActiveReg
            for i in range(len(ActiveReg)):
                dir2 = os.listdir(os.path.join(path,dir0,ActiveReg[i]))#010704,025426
               # xyy.JsonWrite(log_dir,'RecordTime',dir2)
                for j in range(len(dir2)):
                    dir3 = os.listdir(os.path.join(path,dir0,ActiveReg[i],dir2[j]))#CENT BO50 R050
                    for k in range(len(dir3)):
                        Bandoff.append(os.path.join(path,dir0,ActiveReg[i],dir2[j],dir3[k]))
            if Bandoff != BandoffNew:
                BandoffNew = Bandoff
                xyy.JsonWrite(log_dir,'Bandoff',Bandoff)
                for i in range(len(ActiveReg)):
                    dir2 = os.listdir(os.path.join(path,dir0,ActiveReg[i]))#010704,025426
                    xyy.JsonWrite(log_dir,'RecordTime',dir2)
        except Exception as e:
            #print(e)
            pass


def inotify(jsonfile):
    global today_dir, mask,workdir,wm,notifier
    jsonfile = jsonfile
    f = open(jsonfile,'r')
    paras = json.load(f)
    f.close()
    workdir = paras['workdir']#"/home/user/data/2020"
    log_dir = paras['log_dir']
    DirSaveDark = paras['DirSaveDark']
    DirSaveFlat = paras['DirSaveFlat']
    rootdir = os.path.dirname(workdir)
    xyy.JsonWrite(log_dir,'ArchiveDark',[DirSaveDark])
    xyy.JsonWrite(log_dir,'ArchiveFlat',[DirSaveFlat])
    #handler = EventHandler()
    #handler.passlogdir(log_dir)
    #wm = pyinotify.WatchManager()
    #mask = pyinotify.IN_CREATE
    #notifier = pyinotify.Notifier(wm,handler)
    #mask2 = pyinotify.IN_DELETE
    while 1:
        try:
            today = datetime.datetime.now().strftime('%Y%m%d')
            today_dir = os.path.join(workdir,today)
            #if os.path.exists(today_dir) == False:
            #    wm.add_watch(rootdir,mask2)
            #    notifier.loop()
               #else:
            if os.path.exists(today_dir) == True:
                DelayProcess(log_dir,today_dir)
                break
            '''if os.path.exists(workdir) and os.path.exists(today_dir) == False:
                print(258)
                wm.add_watch(workdir,mask)
                notifier.loop()
                break
            elif os.path.exists(today_dir):
                DelayProcess(log_dir,today_dir)
                break
            else:
                pass'''
        except Exception as e:
            pass

if __name__ == "__main__":
    jsonfile = r'/home/wangxinhua/level1/Level1/Level1rev08New/json.txt'
    inotify(jsonfile)
