#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:48:31 2020

@author: wangxinhua
"""
from mpi4py import MPI 
import os
from astropy.io import fits
import numpy as np
import re
import imageio
import time
from numba import vectorize
import cupy as cp
import multiprocessing


def get_html(n):
    time.sleep(n)
    print('process is complete')
    return n


if __name__ == '__main__':
    process1 = multiprocessing.Process(target=get_html, args=(2, ),name='name1')

    process1.start()
    print(process1.name)  # »ñÈ¡¶à½ø³ÌµÄpid   ÔÚstartÖ®ºó²Å»áÓÐ½ø³Ìpid
    process1.join()
    print('')

    # Ê¹ÓÃ½ø³Ì³Ø
    pool = multiprocessing.Pool(multiprocessing.cpu_count())   # multiprocessing.cpu_count»ñÈ¡cpuÊýÁ¿
    result = pool.apply_async(get_html, args=(3, ))  # Òì²½Ö´ÐÐÒ»¸öÈÎÎñ

    # µÈ´ýËùÓÐÈÎÎñÍê³É
    pool.close()  # joinÖ®Ç°±ØÐëÏÈclose  ·ñÔò»á±¨´íµÄ
    pool.join()
    print(result.get())  # »ñÈ¡Ö´ÐÐº¯Êý·µ»ØµÄ½á¹û

    pool = multiprocessing.Pool(multiprocessing.cpu_count())  # multiprocessing.cpu_count»ñÈ¡cpuÊýÁ¿

    # imap·½·¨
    for result in pool.imap(get_html, [3, 43, 45]):   # »á¸ù¾Ý¿Éµü´ú¶ÔÏóµÄ²ÎÊý ´òÓ¡Ö´ÐÐ½á¹û
        print(result)

    for result in pool.imap_unordered(get_html, [6, 5, 3]):  # »á¸ù¾ÝË­ÏÈÖ´ÐÐÍê³É ´òÓ¡Ö´ÐÐ½á¹û
        print(result)