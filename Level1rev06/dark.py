# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:56:10 2019

@author: wangxinhua
"""
# -*- coding: utf-8 -*-
"""
2019/12/24 
create by wangxinhua

"""
import getopt
import time
import sys
import cupy as cp
import numpy as np
import os
import astropy.io.fits as fits


class dark(object):
    
    
    def main(self,argv):
        try:
            opts, args = getopt.getopt(argv[1:],"oi:di:o",["obj_inputpath=","dark_inputpath=","outputpath="])
        except getopt.GetoptError:
            sys.exit(2)
        if(list.__len__(sys.argv) <=1):
            print('python dark.py --obj_inputpath <object inputpath> --dark_inputpath <dark inputpath> --outputpath <output path>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == "--obj_inputpath":
                self.obj_inputpath = arg
            elif opt == "--dark_inputpath":
                self.dark_inputpath = arg
            elif opt == "--outputpath":
                self.outputpath = arg
            else:
                print("usage: python dark.py --obj_inputpath <object inputpath> --dark_inputpath <dark inputpath> --outputpath <output path>")
                sys.exit()
        self.processing()
        
        
    def processing(self):
        try:
            os.mkdir(self.outputpath)
        except Exception as e:
            print("warning:folder have existed")
        obj_item = os.listdir(self.obj_inputpath)
        dark_item = os.listdir(self.dark_inputpath)
        for i in obj_item:
            data = fits.open(self.obj_inputpath+'/'+str(i))[0].data
            M, N = data.shape[0], data.shape[1]
            break
        dark_data = cp.ndarray((M,N),dtype=np.float32)
        dark_data = 0
        cp.cuda.Device(0).use
        with cp.cuda.Device(0):
            for i in dark_item:
                #dark_data +=fits.open(self.dark_inputpath+"//"+i)[0].data
                dark_data += cp.array(fits.open(self.dark_inputpath+"/"+i)[0].data,dtype='float32')
                
            dark_data = dark_data/len(dark_item)
            fits.writeto(self.outputpath+'/'+'dark.fits',cp.asnum)
            for i in obj_item:
                data = cp.array(cp.array(fits.open(self.obj_inputpath+'/'+str(i))[0].data) - dark_data,dtype='float32')
                #data = np.array(fits.open(self.obj_inputpath+'\\'+str(i))[0].data - dark_data,dtype='float32')
                fits.writeto(self.outputpath+"/dark_"+str(i),cp.asnumpy(data),overwrite=True)
         
            
if __name__=='__main__':
    #obj_dir = r'G:\20160501\12536\011905\B070'
    #dark_dir = r'G:\20160501\dark'
    start = time.time()
    calibration = dark()
    calibration.main(sys.argv)
    print('dark_elapse:'+str(time.time()-start))
        
