#!/bin/sh


kill $(ps -ef|grep "monitor2.py"|awk '{print $2}')
kill $(ps -ef|grep "monitor1.py"|awk '{print $2}')
kill $(ps -ef|grep "preprocessing.py"|awk '{print $2}')
kill $(ps -ef|grep "Level1Process.py"|awk '{print $2}')
