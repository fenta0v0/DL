% clear all
% I1=imread('D:\desktop\ZZ\11.png');
% figure(1);imshow(I1);
% f0=double(I1);
% [N1,N2]=size(f0);
% N=min(N1,N2);                        
% h=0.000532;             %波长(mm)
% pix=0.00465;            %像素宽(mm)
% z0=input('重建距离z0=');
% L=pix*N;                %CCD宽度(mm)                        
% f=zeros(N,N);
% Ih(1:N,1:N)=f0(1:N,1:N);
% 
% L0=L;              %赋值衍射面尺寸，单位:米                        %赋值观察屏到衍射面的距离,单位:米 
% fex=N/L;
% kethi=[-fex/2:fex/N:fex/2-fex/N];
% nenta=[-fex/2:fex/N:fex/2-fex/N];
% [kethi,nenta]=meshgrid(kethi,nenta);
% k=2*pi/h;	
% H=exp(j*k*z0*sqrt(1-(h*kethi).^2+(h*nenta).^2)); %传递函数H
% fa=fft2(Ih);                 %衍射面上光场的傅里叶变换
% Fuf=fa.*H;                            %光场的频谱与传递函数相乘
% U=ifft2(Fuf);                         %在观察屏上的光场分布
% I=U.*conj(U);                         %在观察屏上的光强分布
% % Gma=max(max(abs(U)));
% % Gmi=min(min(abs(U)));
% % p=10;
% % while p
% %     imshow(abs(U),[Gmi  Gma/p]);
% %     p=input('Gma/p,p=10?');
% % end
% figure,imshow(I,[0,max(max(I))]),colormap(gray)


% Uo=imread('D:\desktop\ZZ\11.png');             %调入作为物的图像
% Uo=double(Uo (:,:,1)); %取第一层，并转为双精度
% Uo = imresize(Uo,[256,256])
% [r,c]=size(Uo);
% Uo=ones(r,c)*0.98-Uo/255*0.5;       %将物转换为高透射率射系数体
% figure,imshow(Uo,[0,1]),title('物')
% lamda=6328*10^(-10);k=2*pi/lamda;   %赋值波长和波数
% Lo=5*10^(-3)                        %赋值衍射面(物)的尺寸
% xo=linspace(-Lo/2,Lo/2,r);yo=linspace(-Lo/2,Lo/2,c);
% [xo,yo]=meshgrid(xo,yo);            %生成衍射面(物)的坐标网格
% zo=0.20;                            %全息记录面到衍射面的距离,单位:米
% %下面用T-FFT算法完成物面到全息记录面的衍射计算
% F0=exp(j*k*zo)/(j*lamda*zo);
% F1=exp(j*k/2/zo.*(xo.^2+yo.^2));
% fF1=fft2(F1);
% fa1=fft2(Uo);
% Fuf1=fa1.*fF1; 
% Uh=F0.*fftshift(ifft2(Fuf1)); 
% Ih=Uh.*conj(Uh);
% figure,imshow(Ih,[0,max(max(Ih))/1]),title('全息图')
% %下面用T-FFT算法完成全息面到观察面的衍射计算(重构再现像)
% for t=1:2                         %分40幅图像再现聚、离焦过程
%     zi=0.01+t.*0.00001               %用不同的值赋值再现距离
%     F0i=exp(j*k*zi)/(j*lamda*zi);
%     F1i=exp(j*k/2/zi.*(xo.^2+yo.^2)); 
%     fF1i=fft2(F1i);
%     fIh=fft2(Ih); 
%     FufIh=fIh.*fF1i; 
%     Ui=F0i.*fftshift(ifft2(FufIh)); 
%     Ii=Ui.*conj(Ui);
%     imshow(Ii,[0,max(max(Ii))/1])
%     str=['成像距离:',num2str(zi),'米'];%设定显示内容
%     text(257,30,str,'HorizontalAlignment','center','VerticalAlignment','middle','background','white'); %设定在图中显示字符的位置及各式
%     m(t)=getframe;                 %获得并保存显示的图像
% end
% movie(m,2,5)                       %播放保存的图像

% clc;
% clear;
% rgb=imread('D:\desktop\ZZ\z\data\2.png');
% % rgb2 = imread('D:\desktop\ZZ\z\data\1.png');
% % rgb = imsubtract(rgb,rgb2)  %两张相减，消除噪声
% a = mat2gray(rgb);
% pix=0.000001 ;
% lam=0.000000532;
% k=2*pi/lam;
% d=0.0019;
% m =1024;
% n = 1024;
% [fx,fy]=meshgrid(linspace(-1/2/pix,1/2/pix,m),linspace(-1/2/pix,1/2/pix,n));
% g=exp(1i*k*d*sqrt(1-(lam*fx).^2-(lam*fy).^2));%H(fx,fy)
% af=fftshift(fft2(fftshift(a)));
% e=fftshift(ifft2(fftshift(af.*g)));
% figure,imshow(abs(e),[]);
%
% rgb=imread('D:\desktop\zaixian\z\187.png');
% g=mat2gray(rgb); % 转为灰阶图
% gg=double(g); % 转为数值矩阵
% gg=1-gg/255; % 将彩色值转为 0-1 的渐变值
% [x,y]=size(gg); % 取原图大小
% [X,Y]=meshgrid(1:y,1:x); % 以原图大小构建网格
% mesh(X,Y,gg); % 网格上画出图像
% colormap gray % 设为灰阶图像


clc;
clear;
% rgb=imread('D:\desktop\zaixian\yuan\1.jpg');
% % rgb2 = imread('D:\desktop\zaixian\2.jpg');
% % rgb = imsubtract(rgb,rgb2)  %两张相减，消除噪声
% a = mat2gray(rgb);
% pix=0.0000058 ;
% lam=0.000000532;
% k=2*pi/lam;
% d0=0.023;
% m =2592;
% n = 1944;
% zmax =255;
% z0 =1;
% [fx,fy]=meshgrid(linspace(-1/2/pix,1/2/pix,m),linspace(-1/2/pix,1/2/pix,n));
% while z0<zmax
%     d = d0+z0*0.0005
%     g=exp(1i*k*d*sqrt(1-(lam*fx).^2-(lam*fy).^2));%H(fx,fy)
%     af=fftshift(fft2(fftshift(a)));
%     e=fftshift(ifft2(fftshift(af.*g)));
%     I_new = abs(e);
%     filename=['D:\desktop\zaixian\1\',num2str(z0),'.png'];
%     imwrite(I_new,filename,'png');
%     z0=z0+1
% end

% [row col] =size(i11);
% for j=1:col
%     for i=1:row
%         darkestPixelValue = min(i11(i,j));    
%         [i,j]=find(i11==darkestPixelValue);
%         plot (i,j);
%     end
% end


% clc;
% clear;
% rgb=imread('D:\desktop\ZZ\2.png');
% rgb2 = imread('D:\desktop\ZZ\1.png');
% rgb = imsubtract(rgb,rgb2)  %两张相减，消除噪声
% I_new = mat2gray(rgb);
% z0 = 4;
% filename = ['D:\desktop\ZZ\',num2str(z0),'.png']
% imwrite(I_new,filename,'png');

% clc;
% clear;
% % 菲涅尔反向重现算法
% % 基本参数
% hologram1 = imread('D:\desktop\zaixian\66.png');
% % hologram1 = imcrop(hologram,[10,10,522,522])
% %z 表示物体离屏幕的距离，wavelength 表示激光波长，pixel_size 表示像素大小。
% z = 0.015
% wavelength = 0.000000532;
% pixel_size = 0.0000058;
% z0 = 6;
% % 计算二维 Fourier 变换
% 
% 
% f_hologram = fft2(hologram1);
% % 计算菲涅尔衍射公式
% [M, N] = size(hologram1);
% k = 2 * pi / wavelength;
% fx = fftshift(-N/2:N/2-1) / (N * pixel_size);
% fy = fftshift(-M/2:M/2-1) / (M * pixel_size);
% [u, v] = meshgrid(fx, fy);
% r_squared = u.^2 + v.^2;
% H = exp(1j * k * z) / (1j * wavelength * z) * exp(-1j * pi * wavelength * z * r_squared);
% f_propagated = f_hologram .* H;
% % 计算反 Fourier 变换
% hologram_reconstructed = ifft2(f_propagated);
% filename=['D:\desktop\zaixian\',num2str(z0),'.png'];
% %     imwrite(I_new,filename,'png');
% imwrite(hologram_reconstructed,filename,'png');


% clc;
% clear;
% % 菲涅尔反向重现算法
% % 基本参数
% hologram = imread('D:\desktop\zaixian\1.JPG');
% hologram1 = imcrop(hologram,[0,0,512,512])
% z0 = 66;
% filename=['D:\desktop\zaixian\',num2str(z0),'.png'];
% %     imwrite(I_new,filename,'png');
% imwrite(hologram1,filename,'png');

% clc;
% clear;
% 
% % 指定要读取的文件夹路径
% folder_path = 'D:\desktop\zaixian\yuan';
% 
% % 获取文件夹中所有文件的列表
% file_list = dir(fullfile(folder_path, '*.jpg'));
% 
% % 循环读取每个图片并进行处理
% for i = 1:length(file_list)
%     % 读取图片
%     img = imread(fullfile(folder_path, file_list(i).name));
%     img = mat2gray(img);
%     pix=0.0000058 ;
%     lam=0.000000532;
%     k=2*pi/lam;
%     d=0.145;
%     m =2592;
%     n = 1944;
%     zmax =255;
%     z0 =1;
%     [fx,fy]=meshgrid(linspace(-1/2/pix,1/2/pix,m),linspace(-1/2/pix,1/2/pix,n));
%     % 在此处添加对图片的处理代码
%     g=exp(1i*k*d*sqrt(1-(lam*fx).^2-(lam*fy).^2));%H(fx,fy)
%     af=fftshift(fft2(fftshift(img)));
%     e=fftshift(ifft2(fftshift(af.*g)));
%     I_new = abs(e);
%     filename=['D:\desktop\zaixian200\',num2str(i),'.png'];
%     imwrite(I_new,filename,'png');
%     % 显示图片
%     imshow(img);
% end




rgb=imread('D:\desktop\zaixian\yuan\1.jpg');
% rgb2 = imread('D:\desktop\zaixian\2.jpg');
% rgb = imsubtract(rgb,rgb2)  %两张相减，消除噪声
a = mat2gray(rgb);
pix=0.0000058 ;
lam=0.000000532;
k=2*pi/lam;
d0=0.120;
m =2592;
n = 1944;
zmax =255;
z0 =1;
[fx,fy]=meshgrid(linspace(-1/2/pix,1/2/pix,m),linspace(-1/2/pix,1/2/pix,n));
while z0<zmax
    d = d0+z0*0.00045
    g=exp(1i*k*d*sqrt(1-(lam*fx).^2-(lam*fy).^2));%H(fx,fy)
    af=fftshift(fft2(fftshift(a)));
    e=fftshift(ifft2(fftshift(af.*g)));
    I_new = abs(e);
    filename=['D:\desktop\zaixian\1\',num2str(z0),'.png'];
    imwrite(I_new,filename,'png');
    z0=z0+1
end

% 
% i = 100;
% file_name = ['D:\desktop\zaixian\yuan\',num2str(i),'.jpg']
% img = imread(file_name);
% img = mat2gray(img);
% pix=0.0000058 ;
% lam=0.000000532;
% k=2*pi/lam;
% d=0.135;
% m =2592;
% n = 1944;
% zmax =255;
% z0 =1;
% [fx,fy]=meshgrid(linspace(-1/2/pix,1/2/pix,m),linspace(-1/2/pix,1/2/pix,n));
% %在此处添加对图片的处理代码
% g=exp(1i*k*d*sqrt(1-(lam*fx).^2-(lam*fy).^2));%H(fx,fy)
% af=fftshift(fft2(fftshift(img)));
% e=fftshift(ifft2(fftshift(af.*g)));
% I_new = abs(e);
% filename=['D:\desktop\zaixian\zaixian100\',num2str(i),'.png'];
% imwrite(I_new,filename,'png');
% I1 = imread('D:\desktop\zaixian\zaixian100\1.png');
% I2 = imread('D:\desktop\zaixian\zaixian100\2.png');
% I = imsubtract(I1,I2);
% imshow(I)
