# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 13:44:06 2019

@author: a
"""
def nvst_dirsandfiles_path(path):
    '''return [[database,darkbase,flatbase],[datapath,darkbase,flatpath]]'''
    import os
    target = []#g:\20190518\ha\     活动区、dark、flat
    dirstmp = ''
    a = 0
    paths = [[],[],[]]
    for roots, path, files in os.walk(r'G:\20190518\HA'):
        if a == 1:
            target.append(roots)
        a+=1
        if a> 1 and target[-1] not in roots:
            target.append(roots)
    for roots, path, files in os.walk(r'G:\20190518\HA'):
        for i in range(len(target)):
            if target[i] in roots:
                paths[i].append(roots)
    path0_len = len(paths[0][-1])
    path1_len = len(paths[1][-1])
    path2_len = len(paths[2][-1])
    path1 = []
    path2 = []
    path3 = []
    for i in range(len(paths[0])):
        if path0_len == len(paths[0][i]):
            path1.append(paths[0][i])
    for i in range(len(paths[1])):
        if path1_len == len(paths[1][i]):
            path2.append(paths[1][i])
    for i in range(len(paths[2])):
        if path2_len == len(paths[2][i]):
            path3.append(paths[2][i])
    return [[target[0],target[1],target[2]],[path1,path2,path3]]



if __name__ =='__main__':
    path = r'G:\20190518\HA'
    dirs = nvst_dirsandfiles_path(path)
    data_base = dirs[0][0]
    flat_base = dirs[0][2]
    dark_base = dirs[0][1]
    data = dirs[1][0]
    dark = dirs[1][1]
    flat = dirs[1][2]
    print(data,'\n',flat,'\n',dark)