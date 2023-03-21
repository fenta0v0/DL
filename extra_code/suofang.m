% I = imread('D:\desktop\T\700.JPG');
% rect= [0,0,1944,1944];
% I1 = imcrop(I,rect);
% imwrite(I1,'D:\desktop\T\700_1.JPG');
% I2 = imresize(I1,0.05,'nearest');
% imwrite(I2,'D:\desktop\T\700_3.JPG');
% I3 = imresize(I2,[128,128],'nearest');
% imwrite(I2,'D:\desktop\T\700_4.JPG');

% ========拼接=========

% image3=imread('D:\desktop\z8\26-1-1\5.PNG');
% image4=imread("D:\desktop\z8\26-1-1\38.PNG");
% [rows3,cols3,m3]=size(image3);
% [rows4,cols4,m4]=size(image4);
% dou_image3=rgb2gray(image3);
% dou_image4=rgb2gray(image4);
% mb=dou_image3(:,cols3);
% mx=abs(dou_image4(:,:)-mb);
% sum_mx=sum(mx);
% min_mx=min(sum_mx);
% [rows,cols]=find(sum_mx==min_mx);
% mx_image4=image4(:,(cols:cols4),:);
% image=[image3,mx_image4];
% figure;
% subplot(2,2,1);
% imshow(image3);
% title('待拼接左');
% subplot(2,2,2);
% imshow(image4);
% title('待拼接右');
% subplot(2,2,[3,4]);
% imshow(image);
% title('左右拼接结果')



% I=rgb2gray(imread('D:\desktop\29-2\701.JPG'));
% figure;
% subplot(1,3,1),imshow(I);               %把绘图窗口分成一行三列，并将原图像I在第一行第一列显示           
% title('原图');
% [Height,Width]=size(I);                 %获取原图像的高度和宽度
% T1=affine2d([-1 0 0;0 1 0;Width 0 1]);  %构造空间变换结构T1，这里为水平镜像变换矩阵
% T2=affine2d([1 0 0;0 -1 0;0 Height 1]); %构造空间变换结构T2，这里为竖直镜像变换矩阵
% A1=imwarp(I,T1);                        %对原图像I进行水平镜像变换  
% A2=imwarp(I,T2);                        %对原图像I进行竖直镜像变换
% subplot(1,3,2),imshow(A1);              %把绘图窗口分成一行三列，并将A1在第一行第二列显示 
% title('水平镜像变换后的图片A1');              
% subplot(1,3,3),imshow(A2);              %把绘图窗口分成一行三列，并将A2在第一行第三列显示 
% title('竖直镜像变换后的图片A2');
% I = [A1,A2
%     I,A1];
% subplot(),imshow(I);


% % 采用LOG算子对含有噪声的图像进行边缘检测
% close all;
% clear all;
% clc;
% I=imread('D:\desktop\29-3\29-3\690.JPG');
% I1=imcomplement(I);
% I2 = imsubtract(I,I1);
% I3 = imcomplement(I2);
% imshow(I3);
% imwrite(I3,'D:\desktop\T\1_1.jpg')


% p=imread(['D:\desktop\z0\z4\train\131.png']);
% g=rgb2gray(p); % 转为灰阶图
% gg=double(g); % 转为数值矩阵
% gg=1-gg/255; % 将彩色值转为 0-1 的渐变值
% [x,y]=size(gg); % 取原图大小
% [X,Y]=meshgrid(1:y,1:x); % 以原图大小构建网格
% mesh(X,Y,gg); % 网格上画出图像
% colormap gray % 设为灰阶图像
% 
clear;
clc ;
file_path='D:\desktop\zaixian\zzz\crop\test\data\';%图片地址
img_path_list = dir(strcat(file_path,'*png'));
len = length(img_path_list);%数量
for i =1:len
    img_name = img_path_list(i).name;%图像名
    I1=imread(strcat(file_path,img_name));
%     I2=imresize(I1,1/2);
%     I2=imcrop(I1,[240,30,340,330]);
    I2=imresize(I1,[512,512]);
%     J1=flipdim(I2,1);%原图像的水平镜像
%     J2=flipdim(I2,2);%原图像的垂直镜像
    imwrite(mat2gray(abs(I2)),['D:\desktop\zaixian\zzz\crop\crop\test\data\',num2str(i,'%d'),'.png'] );
    
end

% clear;
% clc;
% I1= imread('D:\desktop\z8\26-1-1\16.png');
% % I = imresize(I1,1/4);
% I2=imcrop(I1,[240,30,340,330]);
% % I2=imresize(I,[512,512]);
% imwrite(mat2gray(abs(I2)),'D:\desktop\z8\26-1-2\2.png')


% close all;                  			%关闭当前所有图形窗口，清空工作空间变量，清除工作空间所有变量
% clear all;
% clc;
% I=imread('trees.tif'); 					%输入图像
% J1=transp(I);						%对原图像的转置
% I1=imread('lenna.bmp'); 				%输入图像
% J2=transp(I1);						%对原图像的转置
% set(0,'defaultFigurePosition',[100,100,1000,500]);%修改图形图像位置的默认设置
% set(0,'defaultFigureColor',[1 1 1])%修改图形背景颜色的设置
% figure,
% subplot(1,2,1),imshow(I);%绘制移动后图像
% subplot(1,2,2),imshow(J1);%绘制移动后图像
% figure,
% subplot(1,2,1),imshow(I1)
% subplot(1,2,2),imshow(J2)


% close all;                  %关闭当前所有图形窗口，清空工作空间变量，清除工作空间所有变量
% clear all;
% clc;
% I=imread('D:\desktop\HO\data1\1.png'); %输入图像
% J1=flipdim(I,1);%原图像的水平镜像
% J2=flipdim(I,2);%原图像的垂直镜像
% J3=flipdim(I,3);%原图像的水平垂直镜像
% % set(0,'defaultFigurePosition',[100,100,1000,500]);%修改图形图像位置的默认设置
% % set(0,'defaultFigureColor',[1 1 1])%修改图形背景颜色的设置
% figure,
% subplot(1,4,1),imshow(I) ,title('原图');%绘制原图像
% subplot(1,4,2),imshow(J1),title('水平镜像');%绘制水平镜像后图像
% 
% subplot(1,4,3),imshow(J2),title('垂直镜像');%绘制垂直镜像后图像
% subplot(1,4,4),imshow(J3),title('水平垂直镜像');%绘制水平垂直镜像后图像
% imwrite(mat2gray(abs(I)),'D:\desktop\HO\data2\4.png');
% imwrite(mat2gray(abs(J1)),'D:\desktop\HO\data2\1.png');
% imwrite(mat2gray(abs(J2)),'D:\desktop\HO\data2\2.png');
% imwrite(mat2gray(abs(J3)),'D:\desktop\HO\data2\3.png');

% clear;
% close all;
% I1 =imread('D:\desktop\data111\1\549.png');
% I2 = imread('D:\desktop\data111\1\550.png');
% I3 = imread('D:\desktop\data111\1\551.png');
% I4 =imread('D:\desktop\data111\1\552.png');
% 
% I = [I1,I2
%     I3,I4];
% imshow(I);

% clear;
% close all;
% x = [82,175,165,298,395,541];
% y = [195,479,172,290,360,416];
% z = [20,20,80,20,80,20]
% scatter3(x,y,z,'filled',"green");
% hold on
% x1 = [477,495];
% y1 = [310,229];
% z1 = [20,80]'
% scatter3(x1,y1,z1,'filled','red');

% I = imread('D:\desktop\zaixian\zzz\data\1.jpg');
% J = imread('D:\desktop\zaixian\zzz\label\1.png');
% I1 = imcrop(I,[360,360,1023,1023]);
% J1 = imcrop(J,[360,360,1023,1023]);
% k=2;
% filename_I = ['D:\desktop\zaixian\zzz\crop\test\data\',num2str(k),'.png'];
% filename_J = ['D:\desktop\zaixian\zzz\crop\test\label\',num2str(k),'.png']
% imwrite(I1,filename_I,'png');
% imwrite(J1,filename_J,'png')
