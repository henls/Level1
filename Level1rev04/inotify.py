#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 10:23:31 2020

@author: wangxinhua
"""

import os
import pyinotify
'''wm = pyinotify.WatchManager()
mask = pyinotify.IN_DELETE | pyinotify.IN_CREATE
wm.add_watch('/home/wangxinhua',mask)
notifiter = pyinotify.Notifier(wm)
notifiter.loop()'''
wm = pyinotify.WatchManager()
mask = pyinotify.IN_DELETE | pyinotify.IN_CREATE
class EventHandler(pyinotify.ProcessEvent):
    
    
    def process_IN_CREATE(self, event):
        print("Creating:", event.pathname)
 
    def process_IN_DELETE(self, event):
        print("Removing:", event.pathname)
        
handler = EventHandler()
notifier = pyinotify.Notifier(wm, handler)
wdd = wm.add_watch('/home/user/data/2020', mask)
notifier.loop()