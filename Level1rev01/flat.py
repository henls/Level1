# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 17:02:46 2019

@author: wangxinhua
"""
import numpy as np
import astropy.io.fits as fits
import os
import sys
import getopt
import cupy as cp
import time


class flat(object):
    
    
    def main(self,argv):
        try:
            opts ,args = getopt.getopt(argv[1:],"",["obj_inputpath=","flat_inputpath=","dark_inputpath=" ,"outputpath="])
        except getopt.GetoptError:
            sys.exit(2)
        if(list.__len__(sys.argv) <= 1):
            print('python flat.py --obj_inputpath <object inputpath> --flat_inputpath <flat inputpath> --dark_inputpath <dark inputpath> --outputpath <output path>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == "--obj_inputpath":
                self.obj_inputpath = arg
            elif opt == "--flat_inputpath":
                self.flat_inputpath = arg
            elif opt == "--outputpath":
                self.outputpath = arg
            elif opt == "--dark_inputpath":
                self.dark_inputpath = arg
            else:
                print("usage: python flat.py --obj_inputpath <object inputpath> --dark_inputpath <dark inputpath> --flat_inputpath <flat inputpath> --outputpath <output path>")
                sys.exit()
        self.flat_processing()
    
    
    def flat_overlay(self):
        flat_item = os.listdir(self.flat_inputpath)
        for i in flat_item:
            data = fits.open(self.flat_inputpath+'/'+str(i))[0].data
            self.M, self.N = data.shape[0], data.shape[1]
            break
        try:
            os.mkdir(self.outputpath+'/flat')
        except Exception as e:
            print("warning:folder has existed")
        flat_data = cp.ndarray((self.M,self.N),dtype=np.float32)
        flat_data = 0
        cp.cuda.Device(0).use
        with cp.cuda.Device(0):
            for i in flat_item:
                flat_data += cp.asarray(fits.open(self.flat_inputpath+'/'+i)[0].data,dtype='float32')
            if len(flat_item)==0:
                raise 'flat file dosen\'t exist'
            flat_data = flat_data/len(flat_item)
            dark_item = os.listdir(self.dark_inputpath)
            dark_data = cp.ndarray((self.M,self.N),dtype=np.float32)
            dark_data = 0
            for i in dark_item:
                #dark_data +=fits.open(self.dark_inputpath+"//"+i)[0].data
                dark_data += cp.array(fits.open(self.dark_inputpath+'/'+i)[0].data,dtype='float32')
            dark_data = dark_data / len(dark_item)
            flat_data = flat_data - dark_data
            flat_data = cp.asnumpy(flat_data)
            flat_data = flat_data/flat_data.max()
            fits.writeto(self.outputpath+'/flat/flat.fits',flat_data,overwrite=True)
        
        
    def flat_processing(self):
        obj_item = os.listdir(self.obj_inputpath)
        try:
            os.mkdir(self.outputpath)
        except Exception as e:
            print("warning:folder have existed")
        self.flat_overlay()
        cp.cuda.Device(0).use
        with cp.cuda.Device(0):
            flat_data = cp.asarray(fits.open(self.outputpath+'/flat/flat.fits')[0].data,dtype='float32')
            for i in obj_item:
                flat_proceed_data = cp.asarray(fits.open(self.obj_inputpath+'/'+str(i))[0].data,dtype='float32')/(flat_data)
                fits.writeto(self.outputpath+'/flat_'+str(i),cp.asnumpy(flat_proceed_data),overwrite=True)
            
            
if __name__=='__main__':
    #obj_dir = r'G:\20160501\12536\011905\B070'
    #flat_dir = r'G:\20160501\FLAT00\B070'
    start = time.time()
    flat_pro = flat()
    flat_pro.main(sys.argv)
    print('flat_elapse:'+str(time.time()-start))