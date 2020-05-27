const express=require('express');
const fs=require('fs');
const path=require('path');
var server=express();
server.use(express.static(path.join(__dirname, 'www')));
//console.log(__dirname.constructor);
server.set('views',__dirname+'/views');
server.set('view engine','ejs');
server.use(require('./router/web.js'));
console.log(__dirname)
server.listen(8889,function () {
    console.log('服务器已启动');
});
