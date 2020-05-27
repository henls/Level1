#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:12:16 2020

@author: wangxinhua
"""
def hhh():
    print(1)
    f = open(r'/home/master/aaa.log','w')
    f.writelines('123')
    f.close()

if __name__ == "__main__":
    hhh()
