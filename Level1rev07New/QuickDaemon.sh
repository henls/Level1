#!/bin/sh


kill $(ps -ef|grep "monitor2.py"|awk '{print $2}')
kill $(ps -ef|grep "monitor1.py"|awk '{print $2}')
kill $(ps -ef|grep "preprocessing.py"|awk '{print $2}')
kill $(ps -ef|grep "Level1Process.py"|awk '{print $2}')
/home/wangxinhua/anaconda3/bin/python /home/wangxinhua/level1/Level1rev07New/monitor1.py&
/home/wangxinhua/anaconda3/bin/python /home/wangxinhua/level1/Level1rev07New/monitor2.py&
/home/wangxinhua/anaconda3/bin/python /home/wangxinhua/level1/Level1rev07New/preprocessing.py&
/home/wangxinhua/anaconda3/bin/python /home/wangxinhua/level1/Level1rev07New/Level1Process.py&
