clear;
close all;
clc;
% data_count = 1;
% particle_number =10;
% 
% a = 210;
% b = 100;
% for j =1:1:31
%     y_out = [82,175,166,298,395,477,495,541,10,640];
%     x_out = [195,479,171,290,360,310,229,416,10,480];
%     depth = [a,a,b,a,b,a,b,a,500,10];
%     r_out = [20,20,20,20,20,20,20,20,20,20];
%     
%     z_change = 2;
%     filename_1=['D:\desktop\z0\z\label1\',num2str(j,'%d'),'.png'];
%     Nx = 486;
%     Ny = 648;
%     a = a+10;
%     b = b+10;
%     for particle_num = 1:1:particle_number
%     
%         for x = 1:1:Nx
%             for y = 1:1:Ny
%                 if  y == y_out(particle_num) && x == x_out(particle_num)   %判断当前点是否为所需要的点
%                     x_new = x - round(r_out(particle_num)/2);
%                     y_new = y - round(r_out(particle_num)/2);
%                     
%                     if y_new < 1
%                         y_new = 1;
%                     end
%                     if x_new < 1
%                         x_new = 1; 
%                     end
%                     for x_2_new = x_new:1:x_new+r_out(particle_num)   %对点进行像素级更改
%                         for y_2_new = y_new:1:y_new+r_out(particle_num)
%                             
%                             NEW_rgb(x_2_new,y_2_new) = (depth(particle_num))*z_change;%写入参数
%                             NEW_rgb(Nx,Ny)=1;
%                         end
%                     end
%                     
%                 end
%             end
%             
%         end
%            
%         %save label data
%         
%     end
%     I_new_1=mat2gray(NEW_rgb);
%     figure(2)
%     imshow(I_new_1);
%     % filename_1=['D:\desktop\train\label\3.png'];
%     imwrite(I_new_1,filename_1,'png');
%     % data_count = data_count +1;
%     a = a+10;
%     b = b+10;
%     j= j+1;
% end

% clear;
% clc ;
% y_out = [220,312,392,520,520,10,480];
% x_out = [408,240,329,338,176,10,640];
% r_out = [20,20,20,20,20,20,20];
% a =450;
% % b =410;
% j =36;
% depth = [a,a,a,a,a,500,10];
% particle_number=7;
% z_change = 2;
% 
% filename_1=['D:\desktop\z0\z4\label1\',num2str(j,'%d'),'.png'];
% Nx = 486;
% Ny = 648;
% % a = a+10;
% % b = b+10;
% for particle_num = 1:1:particle_number
% 
%     for x = 1:1:Nx
%         for y = 1:1:Ny
%             if  y == y_out(particle_num) && x == x_out(particle_num)   %判断当前点是否为所需要的点
%                 x_new = x - round(r_out(particle_num)/2);
%                 y_new = y - round(r_out(particle_num)/2);
%                 
%                 if y_new < 1
%                     y_new = 1;
%                 end
%                 if x_new < 1
%                     x_new = 1; 
%                 end
%                 for x_2_new = x_new:1:x_new+r_out(particle_num)   %对点进行像素级更改
%                     for y_2_new = y_new:1:y_new+r_out(particle_num)
%                         
%                         NEW_rgb(x_2_new,y_2_new) = (depth(particle_num))*z_change;%写入参数
%                         NEW_rgb(Nx,Ny)=1;
%                     end
%                 end
%                 
%             end
%         end
%         
%     end
%        
%     %save label data
%     
% end
% I_new_1=mat2gray(NEW_rgb);
% figure(2)
% imshow(I_new_1);
% % filename_1=['D:\desktop\train\label\3.png'];
% imwrite(I_new_1,filename_1,'png');


x_name = 'D:\desktop\zaixian\zzz\text\x_1.txt';
y_name = 'D:\desktop\zaixian\zzz\text\y_1.txt';
z_name = 'D:\desktop\zaixian\zzz\text\z_1.txt';

y_out = textread(x_name);
x_out = textread(y_name);
depth = textread(z_name);
r_out =20;

Nx = 1944;
Ny = 2592;
% a = a+10;
% b = b+10;
for particle_num = 1:1:length(x_out)

    for x = 1:1:Nx
        for y = 1:1:Ny
            if  y == y_out(particle_num) && x == x_out(particle_num)   %判断当前点是否为所需要的点
                x_new = x - round(r_out/2);
                y_new = y - round(r_out/2);
                
                if y_new < 1
                    y_new = 1;
                end
                if x_new < 1
                    x_new = 1; 
                end
                for x_2_new = x_new:1:x_new+r_out   %对点进行像素级更改
                    for y_2_new = y_new:1:y_new+r_out
                        
                        NEW_rgb(x_2_new,y_2_new) = (depth(particle_num));%写入参数
                        NEW_rgb(Nx,Ny)=1;
                    end
                end
                
            end
        end
        
    end
       
    %save label data
    
end
I_new_1=mat2gray(NEW_rgb);
figure(2)
imshow(I_new_1);
filename_1=['D:\desktop\zaixian\zzz\label\1.png'];
imwrite(I_new_1,filename_1,'png');

