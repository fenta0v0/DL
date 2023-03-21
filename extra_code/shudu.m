clc
clear all


生成模拟全息图
data_number = 8000;
data_count =1450;
% 在每个粒子平面上生成粒子

初始位置平面
while data_count<data_number
    
    %初始化参数
        Nx=512;   %pixels along x   CCD像素大小为um,统一单位为um
        Ny=512;   %pixels along y
        mag=6;
        pixel_size=mag;%microns  像素大小(um)
        lambda=532*10^-3; %microns 波长
        zmax=256;         %pixels  最大深度
        working_dist=3000;   %microns
        depth_factor=8;%2^11/zmax; % scaling, now range is set as 2.048cm,2^11microns 8um
        
        particle_number = 50; % 模拟的粒子个数
%         particle_number = randi([50,150],1)
        sidex = Nx * pixel_size;  % CCD 尺寸
        sidey = Ny * pixel_size;
        pattern = zeros(Nx, Ny); % 初始图像为全0
        particle_size = randi([2,2], 1, particle_number); % 模拟粒子半径的大小，list[1,num]
        object_mask = ones(Nx, Ny, particle_number); % 生成多个粒子平面[512, 512, num]
        object_mask_out = ones(Nx, Ny);
        pattern_size = 2 * (particle_size); 
        count = 0;
        iota = sqrt(-1);  % i
        
        %%%生成一个全息图
        u = randi([1,5],1,particle_number);
        v = randi([1,5],1,particle_number);
        w = randi([1,5],1,particle_number);
        while count<particle_number   % 生成粒子
                X = randi([10,500],1);%生成随机整数(粒子横纵坐标)
                Y = randi([10,500],1);
                Xmax=min(X+pattern_size(count+1)-1,Nx-particle_size(count+1));
                Ymax=min(Y+pattern_size(count+1)-1,Ny-particle_size(count+1));
                if(pattern(X:Xmax,Y:Ymax)==zeros(Xmax-X+1,Ymax-Y+1))
                    pattern(X:Xmax,Y:Ymax)=ones(Xmax-X+1,Ymax-Y+1);
                    count=count+1;
                    depth(count)=working_dist+randi(zmax)*depth_factor;  %1000+z轴间距
                    %depth(count)=working_dist;
                    numpoints=pattern_size(count);
                    %生成粒子
                    for theta = 0:2*pi/numpoints:pi-2*pi/numpoints  % θ  0-π
                        xcirc=int16(particle_size(count)/2*cos(theta));
                        ycirc=int16(particle_size(count)/2*sin(theta));
                        Xmin=max(1,X-xcirc);
                        Xmax=min(X+xcirc,Nx-particle_size(count));
                        Ymin=max(1,Y-ycirc);
                        Ymax=min(Y+ycirc,Ny-particle_size(count));
                        
                        
                        object_mask(Xmin:Xmax,Ymin:Ymax,count) = 0;
                        object_mask_out = object_mask_out + object_mask(:,:,count);
                        figure(3)
                        imagesc(object_mask(:,:,count));
                        
                    end
                    r_out(count) = particle_size(count);  %参数R
                    x_out(count) = X;  %坐标x
                    y_out(count) = Y;  %坐标Y   >>>>lst(x,y,R)
                    %    object_mask(X:Xmax,Y:Ymax,count)=ones(Xmax-X+1,Ymax-Y+1); %square
                end
        end

生成全息图
        for y=1:1:Ny
            for x=1:1:Nx
                phasemap(x,y)=2*pi/lambda*(1-((y-Ny/2-1)*lambda/sidey)^2-((x-Nx/2-1)*lambda/sidex)^2)^0.5;
                    %           f1(x,y) = exp(iota*pi*(x + y));
            end
        end
        phasemap=ifftshift(phasemap);
        hologram=zeros(Nx,Ny);%空白图
        hologram_recontruct=ones(Nx,Ny);
        hologram2=zeros(Nx,Ny);
        for j=1:1:particle_number
            prop(:,:,j) = exp(-iota*depth(j)*phasemap);
            
            prop_fft = fft2(object_mask(:,:,j));
            U  = ifft2(prop_fft.*prop(:,:,j));
            hologram = hologram+abs(U).^2; %生成全息图
        end
        % noisy_hologram=hologram+max(max(hologram))/100*rand(Nx,Ny);  %加噪声
        I_new=mat2gray(hologram)
        % noise_levels = 15;
        % noisy_img = imnoise(I_new, 'gaussian', 0, (noise_levels/255)^2)
        % I_new = noisy_img;
        %     I_new = awgn(I_new,15);
        %     I_new=mat2gray(noisy_hologram);
        
        %imshow(I_new);
        filename=['E:\datasets\speed_test_20_150\train\data\1st\',num2str(data_count+1),'.png'];
        imwrite(I_new,filename,'png');

生成label
        NEW_rgb=zeros(Nx,Ny);
        z_max = max(depth);
        z_min = min(depth);
        z_change = 255/(z_max - z_min);  % 灰度值
        particle_number = length(r_out);
        filename_x=['E:\datasets\speed_test_20_150\train\text\1st/x_',num2str(data_count+1),'.txt'];
        filename_y=['E:\datasets\speed_test_20_150\train\text\1st/y_',num2str(data_count+1),'.txt'];
        filename_r=['E:\datasets\speed_test_20_150\train\text\1st/r_',num2str(data_count+1),'.txt'];
        filename_d=['E:\datasets\speed_test_20_150\train\text\1st/z_',num2str(data_count+1),'.txt'];
        dlmwrite(filename_x,x_out,'delimiter','\t','newline','pc');  % 将参数写入文本
        dlmwrite(filename_y,y_out,'delimiter','\t','newline','pc');
        dlmwrite(filename_r,r_out,'delimiter','\t','newline','pc');
        dlmwrite(filename_d,depth,'delimiter','\t','newline','pc');
        %  生成矩形(遍历所有像素点--512*512)
        for judge = 1:1:particle_number
            for particle_num = 1:1:particle_number
                
                for x = 1:1:Nx
                    for y = 1:1:Ny
                        if  y == y_out(particle_num) && x == x_out(particle_num)   %判断当前点是否为所需要的点
                            x_new = x - round(r_out(particle_num)/2);
                            y_new = y - round(r_out(particle_num)/2);
                            
                            if y_new < 1
                                y_new = 1;
                            end
                            if x_new < 1
                                x_new = 1; 
                            end
                            for x_2_new = x_new:1:x_new+r_out(particle_num)   %对点进行像素级更改
                                for y_2_new = y_new:1:y_new+r_out(particle_num)
                                    
                                    NEW_rgb(x_2_new,y_2_new) = (depth(particle_num)-working_dist)*z_change;%写入参数
                                end
                            end
                            
                        end
                    end
                    
                end
            end
            
        end
        %save label data
        I_new_1=mat2gray(NEW_rgb);
        figure(2)
        %imshow(I_new_1);
        filename_1=['E:\datasets\speed_test_20_150\train\label\1st\',num2str(data_count+1),'.png'];
        imwrite(I_new_1,filename_1,'png');
    
        
        

第二帧粒子
        
        X1= x_out;
        Y1 = y_out;
        Z1 = depth
        
        x11 = X1+u;
        y11 = Y1+v;
        z11 = Z1+w;
        
        sidex = Nx * pixel_size;  % CCD 尺寸
        sidey = Ny * pixel_size;
        pattern_1 = zeros(Nx, Ny); % 初始图像为全0
        particle_size_1 = particle_size; % 模拟粒子半径的大小，list[1,num]
        object_mask_1 = ones(Nx, Ny, particle_number); % 生成多个粒子平面[512, 512, num]
        object_mask_out_1 = ones(Nx, Ny);
        pattern_size_1 = 2 * (particle_size_1); 
        count = 0;
        iota = sqrt(-1);  % i
        while count<particle_number   % 生成粒子
                X = x11(count+1);%生成随机整数(粒子横纵坐标)
                Y = y11(count+1);
                Xmax=min(X+pattern_size_1(count+1)-1,Nx-particle_size_1(count+1));
                Ymax=min(Y+pattern_size_1(count+1)-1,Ny-particle_size_1(count+1));
    %             if(pattern(X:Xmax,Y:Ymax)==zeros(Xmax-X+1,Ymax-Y+1))
    %                 pattern(X:Xmax,Y:Ymax)=ones(Xmax-X+1,Ymax-Y+1);
                    count=count+1;
                    depth_1(count)=z11(count);  %1000+z轴间距
                    %depth(count)=working_dist;
                    numpoints=pattern_size_1(count);
                    %生成粒子
                    for theta = 0:2*pi/numpoints:pi-2*pi/numpoints  % θ  0-π
                        xcirc_1=int16(particle_size_1(count)/2*cos(theta));
                        ycirc_1=int16(particle_size_1(count)/2*sin(theta));
                        Xmin_1=max(1,X-xcirc_1);
                        Xmax_1=min(X+xcirc_1,Nx-particle_size(count));
                        Ymin_1=max(1,Y-ycirc_1);
                        Ymax_1=min(Y+ycirc_1,Ny-particle_size(count));
                        
                        
                        object_mask_1(Xmin_1:Xmax_1,Ymin_1:Ymax_1,count) = 0;
                        object_mask_out_1 = object_mask_out + object_mask_1(:,:,count);
                        figure(3)
                        imagesc(object_mask_1(:,:,count));
                    end
                    r_out_1(count) = particle_size(count);  %参数R
                    x_out_1(count) = X;  %坐标x
                    y_out_1(count) = Y;  %坐标Y   >>>>lst(x,y,R)
                
         end
        
        

第二帧image
        for y=1:1:Ny
                for x=1:1:Nx
                    phasemap_1(x,y)=2*pi/lambda*(1-((y-Ny/2-1)*lambda/sidey)^2-((x-Nx/2-1)*lambda/sidex)^2)^0.5;
                    %           f1(x,y) = exp(iota*pi*(x + y));
                end
        end
        phasemap_1=ifftshift(phasemap_1);
        hologram_1=zeros(Nx,Ny);%空白图
        hologram_recontruct_1=ones(Nx,Ny);
        hologram2=zeros(Nx,Ny);
        for j=1:1:particle_number
            prop(:,:,j) = exp(-iota*depth_1(j)*phasemap_1);
            
            prop_fft = fft2(object_mask_1(:,:,j));
            U  = ifft2(prop_fft.*prop(:,:,j));
            hologram_1 = hologram_1+abs(U).^2; %生成全息图
        end
    %     noisy_hologram=hologram+max(max(hologram))/100*rand(Nx,Ny);  %加噪声
        I_new=mat2gray(hologram)
        filename=['E:\datasets\speed_test_20_150\train\data\2nd\',num2str(data_count+1),'.png'];
        imwrite(I_new,filename,'png');

第二帧label
        NEW_rgb_2=zeros(Nx,Ny);
        z_max = max(depth_1);
        z_min = min(depth_1);
        z_change = 255/(z_max - z_min);  % 灰度值
        particle_number = length(r_out_1);
        filename_x=['E:\datasets\speed_test_20_150\train\text\2nd/x_',num2str(data_count+1),'.txt'];
        filename_y=['E:\datasets\speed_test_20_150\train\text\2nd/y_',num2str(data_count+1),'.txt'];
        filename_r=['E:\datasets\speed_test_20_150\train\text\2nd/r_',num2str(data_count+1),'.txt'];
        filename_d=['E:\datasets\speed_test_20_150\train\text\2nd/z_',num2str(data_count+1),'.txt'];
        dlmwrite(filename_x,x_out_1,'delimiter','\t','newline','pc');  % 将参数写入文本
        dlmwrite(filename_y,y_out_1,'delimiter','\t','newline','pc');
        dlmwrite(filename_r,r_out_1,'delimiter','\t','newline','pc');
        dlmwrite(filename_d,depth_1,'delimiter','\t','newline','pc');
        %  生成矩形(遍历所有像素点--512*512)
        for judge = 1:1:particle_number
            for particle_num = 1:1:particle_number
                
                for x1 = 1:1:Nx
                    for y1 = 1:1:Ny
                        if  y == y_out_1(particle_num) && x == x_out_1(particle_num)   %判断当前点是否为所需要的点
                            x_new_1 = x1 - round(r_out_1(particle_num)/2);
                            y_new_1 = y1 - round(r_out_1(particle_num)/2);
                            
                            if y_new_1 < 1
                                y_new_1 = 1;
                            end
                            if x_new_1 < 1
                                x_new_1 = 1; 
                            end
                            for x_2_new = x_new_1:1:x_new_1+r_out_1(particle_num)   %对点进行像素级更改
                                for y_2_new = y_new_1:1:y_new_1+r_out_1(particle_num)
                                    
                                    NEW_rgb_2(x_2_new,y_2_new) = (depth_1(particle_num)-working_dist)*z_change;%写入参数
                                end
                            end
                            
                        end
                    end
                    
                end
            end
            
        end
        %save label data
        I_new_2=mat2gray(NEW_rgb_2);
    %     figure(2)
        %imshow(I_new_1);
        filename_1=['E:\datasets\speed_test_20_150\train\label\2nd\',num2str(data_count+1),'.png'];
        imwrite(I_new_2,filename_1,'png');
        NEW_rgb_2=zeros(Nx,Ny);

U 、V、W
        NEW_rgb_3=zeros(Nx,Ny);
        x_out_2 = x_out;
        y_out_2 = y_out;
        depth_2 = depth;
        z_max = max(depth_2);
        z_min = min(depth_2);
        z_change = 255/(z_max - z_min);  % 灰度值
        particle_number = length(r_out);
    
        %  生成矩形(遍历所有像素点--512*512)
        for judge = 1:1:particle_number
            for particle_num = 1:1:particle_number
                
                for x = 1:1:Nx
                    for y = 1:1:Ny
                        if  y == y_out_2(particle_num) && x == x_out_2(particle_num)   %判断当前点是否为所需要的点
                            x_new_2 = x ;
                            y_new_2 = y ;
                            
                            if y_new_2 < 1
                                y_new_2 = 1;
                            end
                            if x_new_2 < 1
                                x_new_2 = 1; 
                            end
                            for x_2_new_2 = x_new_2:1:x_new_2+u(particle_num)   %对点进行像素级更改
                                for y_2_new_2 = y_new_2:1:y_new_2+v(particle_num)
                                    
                                    NEW_rgb_3(x_2_new_2,y_2_new_2) = (depth_2(particle_num)-working_dist)*z_change;%写入参数
                                end
                            end
                            
                        end
                    end
                    
                end
            end
            
        end
        %save label data
        I_new_3=mat2gray(NEW_rgb_3);
        filename_1=['E:\datasets\speed_test_20_150\train\label\uv\',num2str(data_count+1),'.png'];
        imwrite(I_new_3,filename_1,'png');
        NEW_rgb_3=zeros(Nx,Ny);
        
        NEW_rgb_4=zeros(Nx,Ny);
        for judge = 1:1:particle_number
            for particle_num = 1:1:particle_number
                
                for x = 1:1:Nx
                    for y = 1:1:Ny
                        if  y == y_out(particle_num) && x == x_out(particle_num)   %判断当前点是否为所需要的点
                            x_new = x ;
                            y_new = y ;
                            
                            if y_new < 1
                                y_new = 1;
                            end
                            if x_new < 1
                                x_new = 1; 
                            end
                            for x_2_new = x_new:1:x_new+u(particle_num)   %对点进行像素级更改
                                for y_2_new = y_new:1:y_new+w(particle_num)
                                    
                                    NEW_rgb_4(x_2_new,y_2_new) = (depth_2(particle_num)-working_dist)*z_change;%写入参数
                                end
                            end
                            
                        end
                    end
                    
                end
            end
            
        end
        %save label data
        I_new_4=mat2gray(NEW_rgb_4);
    %     figure(2)
        %imshow(I_new_1);
        
        filename_1=['E:\datasets\speed_test_20_150\train\label\uw\',num2str(data_count+1),'.png'];
        imwrite(I_new_4,filename_1,'png');
        
        filename_u = ['E:\datasets\speed_test_20_150\train\text\uvw/u_',num2str(data_count+1),'.txt'];
        filename_v = ['E:\datasets\speed_test_20_150\train\text\uvw/v_',num2str(data_count+1),'.txt'];
        filename_w = ['E:\datasets\speed_test_20_150\train\text\uvw/w_',num2str(data_count+1),'.txt'];
        dlmwrite(filename_u,u,'delimiter','\t','newline','pc');
        dlmwrite(filename_v,v,'delimiter','\t','newline','pc');
        dlmwrite(filename_w,w,'delimiter','\t','newline','pc');
        
        data_count = data_count +1;

end

