# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:20:43 2020

@author: WangXh
"""
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import json
import time


def filters(src,data):
    #im = np.array(Image.open(src).convert('L'))
    im = data
    imhist,bins = np.histogram(im.flatten(),256)
    cdf = imhist.cumsum()
    cdf = 255*cdf/cdf[-1]
    im5 = np.interp(im.flatten(),bins[:-1],cdf).reshape(im.shape)
    plt.imsave(src,im5,cmap='gray')
def fits2jpg(src,dst):
    '''src:源文件路径
       dst:目标文件路径
    '''
    while 1:
        time.sleep(2)
        latx = os.listdir(src)
        try:
            for i in latx:
                f = open(os.path.join(src,i))
                detail = f.readlines()
                latest = detail[0].split('\t')[0]
                #last = detail[-2].split('\t')[0]
                save = os.path.join(dst,i[0]+'L.jpg')
                #save1 = os.path.join(dst,i[0]+'.jpg')
                data = fits.open(latest)[0].data
                #plt.imsave(save,data,cmap='gray')
                filters(save,data)
                last = detail[1].split('\t')[0]
                save1 = os.path.join(dst,i[0]+'.jpg')
                data1 = fits.open(last)[0].data
                #plt.imsave(save1,data1,cmap='gray')
                filters(save1,data1)
        except Exception as e:
            pass
    
if __name__ == "__main__":
    jsons = r'/home/wangxinhua/level1/Level1rev08New/json.txt'
    f = open(jsons,'r')
    src = json.load(f)["LatestFitsR0"]
    f.close()
    dst = r'/home/wangxinhua/QuickLookwebserver/www/LatestNvstImage'
    fits2jpg(src,dst)
