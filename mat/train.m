%{
Nx            -- Number of voxels along voxel voume x axis
Ny            -- Number of voxels along voxel volume y axis
zmax          -- Number of voxels along voxel volume z axis
pixel_size    -- size in microns
particle_size -- diameter of particle in microns
lambda        -- wavelength of light in microns from the laser source
working_dist  -- distance between voxel and camera 1 along z direction
depth_factor  -- voxel length in z-direction in microns
mag           -- magnification
%}

Initial value setting
clc
clear all
Nx=512;   %pixels along x   CCD像素大小为um,统一单位为um
Ny=512;   %pixels along y
mag=6;
pixel_size=mag;%microns  像素大小(um)
lambda=532*10^-3; %microns 波长
zmax=256;         %pixels  最大深度
working_dist=5000;   %microns
depth_factor=10;%2^11/zmax; % scaling, now range is set as 2.048cm,2^11microns 8um


generation hologram and corresponding ground true
data_cale = 2000;%datasets numbers  # 数据总数
data_count = 0;
while data_count < data_cale
    r_out = 0;% particle radius ground true  粒子半径
    x_out = 0;% x ground true     x轴
    y_out = 0;% y ground true     y轴
    depth = 0;% z ground true     z轴
    %Simulating holograms   
%     particle_number=randi([100,200],1);%particle number(200以内随机整数)
%     particle_number =randi([50,200],1);
    particle_number =100;
    sidex=Nx*pixel_size;  %CCD 尺寸
    sidey=Nx*pixel_size;
    pattern=zeros(Nx,Ny);
    particle_size= randi([2,5],1,particle_number); %particle radius  粒子半径list[1,num]
    %particle_size= ones(1,1,particle_number)
    object_mask=ones(Nx,Ny,particle_number);%Generate multiple particle planes  #粒子平面[512,512,num]
    object_mask_out = ones(Nx,Ny);
    pattern_size=2*(particle_size); 
    count=0;
    iota=sqrt(-1);  % i

Particles are generated in each plane
    while count<particle_number   % 生成粒子
        X = randi([particle_size(count+1),Nx-particle_size(count+1)],1);%生成随机整数(粒子横纵坐标)
        Y = randi([particle_size(count+1),Ny-particle_size(count+1)],1);
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

Generate hologram   生成全息图 
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
    noisy_hologram=hologram+max(max(hologram))/100*rand(Nx,Ny);  %加噪声
    I_new=mat2gray(hologram)
    noise_levels = 15;
    noisy_img = imnoise(I_new, 'gaussian', 0, (noise_levels/255)^2)
    I_new = noisy_img;
%     I_new = awgn(I_new,15);
%     I_new=mat2gray(noisy_hologram);
    figure(8)
    %imshow(I_new);
    filename=['E:\datasets\db\15db\data\',num2str(data_count+1),'.png'];
    imwrite(I_new,filename,'png');

Generate ground true
    NEW_rgb=zeros(Nx,Ny);
    z_max = max(depth);
    z_min = min(depth);
    z_change = 255/(z_max - z_min);  % 灰度值
    particle_number = length(r_out);
    filename_x=['E:\datasets\db\15db/text/x_',num2str(data_count+1),'.txt'];
    filename_y=['E:\datasets\db\15db\text/y_',num2str(data_count+1),'.txt'];
    filename_r=['E:\datasets\db\15db\text/r_',num2str(data_count+1),'.txt'];
    filename_d=['E:\datasets\db\15db\text/z_',num2str(data_count+1),'.txt'];
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
    filename_1=['E:\datasets\db\15db\label\',num2str(data_count+1),'.png'];
    imwrite(I_new_1,filename_1,'png');
    data_count = data_count +1;
end
