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
            opts ,args = getopt.getopt(argv[1:],"",["obj_inputpath=","diameter=" , "srsize=", "sitfpath=", "diaratio=", "start_r0=", "step_r0="])
        except getopt.GetoptError:
            sys.exit(2)
        if(list.__len__(sys.argv) <= 1):
            print('python sr.py --obj_inputpath <object inputpath> --diameter <telescope aperture> --srsize <sr image size> --sitfpath <sitfpath> --diaratio <diaratio> --start_r0 <start_r0> --step_r0 <step_r0>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == "--obj_inputpath":
                self.obj_inputpath = arg
            elif opt == "--diameter":
                self.diameter = arg
            elif opt == "--srsize":
                self.srsize = arg
            elif opt == "--sitfpath":
                self.sitfpath = arg
            elif opt == "--diaratio":
                self.diaratio = arg
            elif opt == "--start_r0":
                self.start_r0 = arg
            elif opt == "--step_r0":
                self.step_r0 = arg
            else:
                print("usage: sr.py --obj_inputpath <object inputpath> --diameter <telescope aperture> --srsize <sr image size> --sitfpath <sitfpath> --diaratio <diaratio> --start_r0 <start_r0> --step_r0 <step_r0>")
                sys.exit()
        self.sr_process()
    
    
    def test_conv_samesize(self):
        objsize = self.pf.shape
        consize = self.conv.shape
        oxsize = objsize[0]
        oysize = objsize[1]
        cxsize = consize[0]
        cysize = consize[1]
        result = self.conv[cxsize/2-oxsize/2:cxsize/2-oxsize/2+oxsize,cysize/2-oysize/2:cysize/2-oysize/2+oysize]
        return float(result)
    
    
    def nvst_ringotf_cal(self,fre):
        cp.cuda.Device(0).use
        rh0 = fre
        if (rh0 < 0) or (rh0 > 1):
            return 0.0
        if (self.diaratio < 0) or (self.diaratio > 1):
            return 0.0
        R = 0.5 
        A = self.diaratio
        if (rh0 < 2.0*R):
            c = 2.0*cp.arccos(rh0/(R*2.0))*R**2 - rh0*cp.sqrt(R**2-rh0**2/4.0)
        else:
            c = 0.0
        if (rh0 < 2.0*R*A):
            E = 2.0*cp.arccos(rh0/(A*R*2.0))*(R*A)**2-rh0*cp.sqrt((R*A)**2-rh0**2/4.0)
        else:
            E = 0.0
        if (rh0 <= R+R*A) and (rh0 > R-R*A):
            s1 = 0.5*cp.arccos(((R*A)**2+rh0**2-R**2)/(2.0*A*R*rh0))*(R*A)**2
            s2 = 0.5*cp.arccos((rh0**2+R**2-(R*A)**2)/(2.0*R*rh0))*R**2
            s3 = 0.5*cp.sin(cp.arccos(((R*A)**2+rh0**2-R**2)/(2.0*A*R*rh0)))*A*R*rh0
            D = s1 + s2 - s3
            D = D*2
        elif (rh0 <= R-A*R):
            D  = cp.pi*(A*R)**2
        else:
            D = 0.0
        H = (C+E-2*D)/(cp.pi*R**2)
        if rh0 == 0:
            H = 1-A**2
        return float(H)
        
    def nvst_soft_cal(self,fre):
        fre = fre
        cp.cuda.Device(0).use
        sle = cp.exp((-3.44*(self.diameter*fre)/self.R0)**(5/3.0)*(1-cp.exp(-fre**3)*fre**(1.0/3)))
        sle = sle*self.nvst_ringotf_cal(fre)
        return sle
        
        
    def nvst_tel_sotf(self):
        width = self.srsize
        center = width/2
        cp.cuda.Device(0).use
        setf_arr = cp.ndarray([width,width])
        dfre = self.maxfre/(width/2)
        for x in range(width):
            for y in range(width):
                fre = cp.sqrt((x-center)**2+(y-center)**2)*dfre
                if fre <=1:
                    setf_arr[x,y] = self.nvst_soft_cal()
                else:
                    setf_arr[x,y] = 0.0
        return setf_arr
        
    
    def nvst_srstdcal_temp(self):
        sitfline = fits.open(self.sitfpath)[0].data
        sitfsize = sitfline.shape
        R0_num = sitfsize[1]
        cp.cuda.Device(0).use
        setf = cp.ndarray([self.srsize,self.srsize])
        sitf = cp.ndarray([self.srsize,self.srsize])
        sitfcube = cp.ndarray([self.srsize,self.srsize,R0_num])
        self.sr_standard = cp.ndarray([self.srsize,self.srsize,R0_num])
        multipre = sitfsize[0]*self.maxfre*2.0/self.srsize
        index_matrix = cp.ndarray([self.srsize,self.srsize])
        x = cp.transpose(cp.mgrid[0:self.srsize,0:self.srsize][0])-self.srsize/2
        y = cp.mgrid[0:self.srsize,0:self.srsize][0]-self.srsize/2
        indexmatrix = cp.round_(cp.sqrt(x**2+y**2)*multipre)
        for index in range(R0_num):
            self.R0 = self.start_r0+self.step_r0*index
            sitfcube[:,:,index] =  sitfline[index,indexmatrix].reshape(self.srsize,self.srsize)
            sitf = sitfcube[:,:,index]
            sitf = sitf/sitf[self.srsize/2,self.srsize/2]
            setf = nvst_tel_sotf()
            sr = setf**2/(sitf+0.0000001*cp.max(sitf))
            sr = sr/sr[self.srsize/2,self.srsize/2]
            self.sr_standard[:,:,index] = sr
        print('标准谱比计算完成')
        
        
    
    def shift(self,data,axis0,axis1):
        axis0 = axis0
        axis1 = axis1
        cp.cuda.Device(0).use
        data = cp.array(data)
        return cp.roll(cp.roll(cp.roll(data,axis0,axis=0),axis1,axis=1))
    
    def test_extend_conv(self):
        objsize = self.pf.shape
        inisize = self.gaussff.shape
        imxsize = objsize[0] + inisize[0]
        imysize = objsize[1] + inisize[1]
        cp.cuda.Device(0).use
        objtmp = cp.ndarray([imxsize,imysize])
        initmp = cp.ndarray([imxsize,imysize])
        objtmp[0:objsize[0],0:objsize[1]] = self.pf
        initmp[0:inisize[0],0:inisize[1]] = self.gaussff
        objtmp = self.shift(objtmp,imxsize/2-objsize[0]/2,imysize/2-objsize[1]/2)
        initmp = self.shift(initmp,imxsize/2-inisize[0]/2,imysize/2-inisize[1]/2)
        objfft = self.shift(cp.fft.fft2(objtmp)*cp.sqrt(imxsize*imysize),imxsize/2,imysize/2)
        inifft= self.shift(cp.fft.fft2(initmp)*cp.sqrt(imxsize*imysize),imxsize/2,imysize/2)
        convfft = self.shift(cp.fft.ifft2(self.shift(objfft*inifft,imxsize/2,imysize/2)),imxsize/2,imysize/2)
        self.conv = cp.real(convfft)
        return float(self.conv)
    
    
    def nvst_cirpf_gaus(self):
        center = self.srsize/2
        cp.cuda.Device(0).use
        self.pf = cp.ndarray([self.srsize,self.srsize])
        wid = int(self.srsize*6.5/10.0)
        delta = 10
        for m in range(self.srsize):
            for n in range(self.srsize):
                R = cp.sqrt((m-center)**2+(n-center)**2)
                if R <= (wid/2.0):
                    pf[m,n] = 1.0
        self.gaussff = cp.ndarray([self.srsize,self.srsize])
        for m in range(self.srsize):
            for n in range(self.srsize):
                R2 = (m-center)**2+(n-center)**2
                gaussff = cp.exp(-R2/(1.0*delta**2))
        conv = self.test_extend_conv()
        pf = self.test_conv_samesize()
        return float(pf/max(pf))
    
    
    def search_dir_scanning(self,datapath,datastep):
        step = datastep
        datapath = datapath
        file = os.listdir(datapath)
        count = len(file)
        
    def sr_process(self):
        self.datasizex = 512
        self.datasizey = 512
        self.datastartx = 0
        self.datastarty = 0
        self.rcsizex = 512
        self.rcsizey = 512
        self.rcstartx = 0
        self.rcstarty = 0
        self.datastep = 100.0
        self.tel_r0 = 0.0600
        self.va = 0.15
        self.search_dir_scanning(self, datapath,datastep,bandpath)
        
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
        self.maxfre = (wavelen*(10.0**(-10))*focallen)/(2.0*self.diameter*(xbining*xpixsz*(10.0**(-6))))
        print('截止频率：'+str(maxfre))
        sr_pf = self.nvst_cirpf_gaus()
        self.nvst_srstdcal_temp()
        #............................................Loop for different offband 
        
    
        
    
