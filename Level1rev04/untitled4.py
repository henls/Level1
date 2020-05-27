# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 18:50:40 2019

@author: a
"""
import numpy as np
import astropy.io.fits as fits
import os
import sys
import getopt
import cupy as cp
import time


class sr(object):
    
    
    def main(self,argv):
        try:
            opts ,args = getopt.getopt(argv[1:],"",["datapath=","flatdatapath=" , "darkdatapath=", "resultdir=", "sitfname=", "start=1", "final=1", "flag_flat=1", "sob=0"])
        except getopt.GetoptError:
            sys.exit(2)
        if(list.__len__(sys.argv) <= 1):
            print('python sr.py --datapath <data inputpath> --flatdatapath <flat inputpath> --darkdatapath <dark inputpath> --resultdir <result diretion> --sitfname <sit direction> --start <start> --final <final> --flag_flat <flag_flat> --sob <sob>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == "--datapath":
                self.datapath = arg
            elif opt == "--flatdatapath":
                self.flatdatapath = arg
            elif opt == "--darkdatapath":
                self.darkdatapath = arg
            elif opt == "--resultdir":
                self.resultdir = arg
            elif opt == "--sitfname":
                self.sitfname = arg
            elif opt == "--start":
                self.start = arg
            elif opt == "--final":
                self.final = arg
            elif opt == "--flag_flat":
                self.flag_flat = arg
            elif opt == "--sob":
                self.sob = arg
            else:
                print("usage: sr.py --obj_inputpath <object inputpath> --diameter <telescope aperture> --srsize <sr image size> --sitfpath <sitfpath> --diaratio <diaratio> --start_r0 <start_r0> --step_r0 <step_r0>")
                sys.exit()
        self.sr_process()
    
    
    def search_dir_scanning(self,datapath,datastep):
        '''共享变量：
                    bandpath = banddir
        ''''
        self.bandpath
    def ha_level1a_mkdir_temp2_scanning(self,bandpath,flatdatapath,resultdir)
        '''共享变量:
                    resultpath_tmp = resultpath
                    resultflatpath = resultflatpath
        '''
        self.resultpath_tmp
        self.resultflatpath
    
    def nvst_cirpf_gaus(self,xlth,wid,delta):
        
        
    def nvst_srstdcal_temp(self,sitfname,diameter,diaratio,maxfre,start_r0,step_r0,srsize):
        '''共享变量：
                    sr_standard
        '''
        self.sr_standard
        
    def sr_process(self):
        #计算谱比（spectrum ratio）   主函数
        self.level = '1A'
        self.datasizex = 512
        self.datasizey = 512
        self.datastartx = 0
        self.datastarty = 0
        self.rcsizex = 512
        self.rcsizey = 512
        self.rcstartx = 0
        self.rcstarty = 0
        self.srsize = 256
        self.datastep = 100.0
        self.diameter = 0.98
        self.diaratio = 0.3
        self.start_r0 = 0.0098
        self.step_r0 = 0.0049
        self.tel_r0 = 0.0600
        self.va = 0.15#选帧比例
        self.search_dir_scanning(self, datapath,datastep)
        n = len(self.bandpath)
        resultpath = []
        for i in range(n):
            band_tmp = self.bandpath[i]
            self.ha_level1a_mkdir_temp2_scanning(band_tmp,self.flatdatapath,self.resultdir)
            resultpath.append(self.resultpath_tmp)
        savepath = []
        for i in resultpath:
            savepath.append(i+'/'+'LuckImage')
            os.makedirs(savepath[i])
        f = []
        file = os.walk(self.obj_inputpath)
        for root, dirs, files in file:
            for filename in files:
                if filename.endswith('ts'):
                    f.append(os.path.join(root,filename))
        count = len(f)
        hdr = fits.open(f[0])[0].header
        wavelen = float(hdr['WAVELEN'])
        focallen = float(hdr['FOCALLEN'])
        xbining = float(hdr['XBINING'])
        xpixsz = float(hdr['XPIXSZ'])
        maxfre = (wavelen*(10.0**(-10))*focallen)/(2.0*self.diameter*(xbining*xpixsz*(10.0**(-6))))
        print('截止频率：'+str(maxfre))
        sr_pf = self.nvst_cirpf_gaus(self.srsize,int(self.srsize*6.5/10),10)
        self.nvst_srstdcal_temp(self.sitfname,self.diameter,self.diaratio,maxfre,self.start_r0,self.step_r0,self.srsize)
        #............................................Loop for different offband 
        for 