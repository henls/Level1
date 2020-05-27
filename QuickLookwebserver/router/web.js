const express=require('express');
const Router=express.Router();
const fs=require('fs');
var https = require('https');
var iconv = require('iconv-lite');
var date = new Date();
var dateNow= date.toLocaleDateString();
//dateNow=dateNow.split('-')[0]+'-0'+dateNow.split('-')[1]+'-0'+dateNow.split('-')[2]
dateYear=dateNow.split('-')[0]
//var month_=dateNow.split('-')[1]
//var day_ = dateNow.split('-')[2]
//switch(month_.length)
//{
//    case 2:switch (day_.length){case 2:dateNow=dateNow.split('-')[0]+'-'+dateNow.split('-')[1]+'-'+dateNow.split('-')[2];break;
//                                case 1:dateNow=dateNow.split('-')[0]+'-'+dateNow.split('-')[1]+'-0'+dateNow.split('-')[2];break; };break;
//    case 1:switch (day_.length){case 2:dateNow=dateNow.split('-')[0]+'-0'+dateNow.split('-')[1]+'-'+dateNow.split('-')[2];break;
//                                case 1:dateNow=dateNow.split('-')[0]+'-0'+dateNow.split('-')[1]+'-0'+dateNow.split('-')[2];break;};break;

//}
//if (month_.length<2 and day_.length<2){
//dateNow=dateNow.split('-')[0]+'-0'+dateNow.split('-')[1]+'-0'+dateNow.split('-')[2]
//}
//tiaoshi
//server=express();
Router.get('/LatestNvstImage',function (req,res,next) {
    let filenames = []
    jsons = fs.readFileSync('/home/wangxinhua/level1/Level1/Level1rev08New/json.txt').toString()
    
    latestpath = JSON.parse(jsons)['LatestFitsR0']
    //const files  = fs.readdirSync(latestpath)
    //files.forEach(function(item,index){
    //    filenames.push(item)
    //})
    let R0 = []
    let Bandoff = []
    let RecordTime = []
        value = fs.readFileSync(latestpath+'/'+'latest.log').toString().split('\n')
        //console.log(value)
        if (value.length>2){
        //parse = value.split('/')
        R0 = [value[value.length-2].split('\t')[1],value[value.length-3].split('\t')[1]]
        pathname1 = value[value.length-2].split('\t')[0].split('/')
        pathname2 = value[value.length-3].split('\t')[0].split('/')
        RecordTime = [pathname1[pathname1.length-1].split('.')[0],pathname2[pathname2.length-1].split('.')[0]]
        Bandoff = [pathname1[pathname1.length-2],pathname2[pathname2.length-2]]}
        else{
        if (value.length == 2){
        R0 = [value[value.length-2].split('\t')[1],'     ']
        
        pathname1 = value[value.length-2].split('\t')[0].split('/')
        RecordTime = [pathname1[pathname1.length-1].split('.')[0],'Nan']
        Bandoff = [pathname1[pathname1.length-2],'']
        }
        else{
            R0 = ['     ','     ']
            Bandoff = ['','']
            RecordTime = ['','']
            }
            
            
        }
        
        
            RR0 = R0[0]
            RLR0 = R0[1]
            RB = Bandoff[0]
            RLB = Bandoff[1]
            RT = RecordTime[0]
            RTT = RecordTime[1]
        res.render('web',{
            title:'NVST QuickLook',
            head:'系统测试',
            data:{
                RR0:RR0,
                RLR0:RLR0,
                RB:RB,
                RLB:RLB,
                RT:RT,
                RTT:RTT
            }
        });
        });

module.exports=Router;
