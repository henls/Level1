import os
import json
import numpy as np
import time


def three2one(src):
    while 1:
        if os.path.exists(os.path.join(src,'latest.log')) ==False:
            file = open(os.path.join(src,'latest.log'),'w')
            file.close()
        time.sleep(2)
        src = src
        try:
            Three = os.listdir(src)
            One = []
            for i in Three:
                file = open(os.path.join(src,i),'r')
                cont = file.readlines()
                file.close()
                [One.append(j) for j in cont]
            point = list(map(lambda x:int(x.split('/')[-1].split('\t')[0].split('.')[0]),One))
            maxlist = point.index(sorted(set(point))[-1])
            max = One[maxlist]
            file = open(os.path.join(src,'latest.log'),'r')
            cont = file.readlines()
            file.close()
            if max not in cont:
                file = open(os.path.join(src,'latest.log'),'a')
                file.writelines(max)
                file.close()
        except Exception as e:
            pass
if __name__ == '__main__':
    jsonfile = open(r'/home/wangxinhua/level1/Level1/Level1rev08New/json.txt','r')
    jsons = json.load(jsonfile)
    jsonfile.close()
    src = jsons['LatestFitsR0']
    three2one(src)
