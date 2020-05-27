#!/bin/sh


kill $(ps -ef|grep "offline2.py"|awk '{print $2}')
kill $(ps -ef|grep "offline1.py"|awk '{print $2}')
kill $(ps -ef|grep "preprocessing.py"|awk '{print $2}')
kill $(ps -ef|grep "Level1Process.py"|awk '{print $2}')
/home/wangxinhua/anaconda3/bin/python /home/wangxinhua/level1/Level1rev07New/offline2.py&
/home/wangxinhua/anaconda3/bin/python /home/wangxinhua/level1/Level1rev07New/offline1.py&
/home/wangxinhua/anaconda3/bin/python /home/wangxinhua/level1/Level1rev07New/preprocessing.py&
/home/wangxinhua/anaconda3/bin/python /home/wangxinhua/level1/Level1rev07New/Level1Process.py&
