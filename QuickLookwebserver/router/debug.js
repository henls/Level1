const express=require('express');
const Router=express.Router();
const fs=require('fs');
var date = new Date();
var dateNow= date.toLocaleDateString();
dateYear=dateNow.split('-')[0]
dateNow=dateNow.split('-')[0]+'-0'+dateNow.split('-')[1]+'-0'+dateNow.split('-')[2]

file_name = '/home/pi/Desktop/communication/'+dateYear+'/data'+dateNow+'.csv'
//file_name='/data.csv';


var new_data,tem,hum,press,spreed,dir_wind,dew,rad,rain,new_number;
list =[1];
//当前时间
    


    //定时程序
    fs.readFile(file_name,function (err, data) {
        if (err){
            var flag='未接收到气象数据'
        }else {
            
            data = data.toString();
            //data=data.slice(1,-1)
            
            list = data.split('\n')
            
            //console.log(list.length);
            new_number=list.length-2;
            
            new_data=list[new_number];
            //console.log(new_data)
            //new_data=new_data.slice(1,-1).split(',');
            new_data=new_data.split(',');
            //console.log(new_data)
            speed=new_data[2];
            rain=new_data[3];
            tem=new_data[4];
            press=new_data[5];
            rad=new_data[6];
            dir_wind=new_data[7];
            hum=new_data[8];
            dew=new_data[9];
            /*********** */
            var date;
            date=Date();
            
            res.render('web',{
                title:'抚仙湖气象站数据',
                head:'系统测试',
                data:{
                    tem:tem,
                    hum:hum,
                    press:press,
                    spreed:spreed,
                    dir_wind:dir_wind,
                    dew:dew,
                    rad:rad,
                    rain:rain,
                    date:date

                    
                }
            });
        }
    });
//});