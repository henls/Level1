#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:15:20 2020

@author: wangxinhua
2020/1/15 release:
    monitor workdir and return today folder to /home/wangxinhua/Observation log/*.log
    program of realtime_dark and realtime_flat reads the log file to START function
"""

import pyinotify
import json
import re
import os

#reload create function in peocessEvent
class EventHandler(pyinotify.ProcessEvent):
    def process_IN_CREATE(self,event):
        path = event.pathname
        today_time = re.findall(r'\d{8}',path)[0]
        log_dir = r'/home/wangxinhua/Observation_log/'
        try:
            os.mkdir(log_dir)
        except Exception as e:
            pass
        f = open(log_dir+today_time+'.log','w+')
        f.writelines(event.pathname)
        f.close()
        
        
#monitor the change of workdirectory      
def inotify(jsonfile):
    #f = open(r"/home/wangxinhua/level1/Level1rev06/json.txt",'r')
    f = open(jsonfile,'r')
    para = json.load(f)
    f.close()
    #"workdir": "/home/user/data/2020"
    workdir = para['workdir']
    handler = EventHandler()
    wm = pyinotify.WatchManager()
    mask = pyinotify.IN_CREATE
    notifier = pyinotify.Notifier(wm,handler)
    wm.add_watch(workdir,mask)
    notifier.loop()
    

if __name__ == "__main__":
    jsonfile = r"/home/wangxinhua/level1/Level1rev06/json.txt"
    inotify(jsonfile)
    
    