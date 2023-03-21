clc
load wind

% partical_number = 100;
% x_data = randi([10,500],1,partical_number);
% y_data = randi([10,500],1,partical_number);
% z_data = randi([10,300],1,partical_number);
% 
% x1_data = x_data +randi([0,8],1,partical_number);
% y1_data = y_data +randi([-1,8],1,partical_number);
% z1_data = z_data +randi([0,2],1,partical_number);
% 
% 
% u = (x1_data -x_data)/0.6;
% v = (y1_data -y_data)/0.6;
% w = (z1_data-z_data)/0.6;
% xmax = max(x_data(:));
% xmin = min(x_data(:));
% ymax = max(y_data(:));
% ymin = min(y_data(:));
% zmax = max(z_data(:));
% zmin = min(z_data(:));
% figure(1)
% % streamslice(x,y,z,u,v,w,(xmax-xmin)/2,(ymax-ymin)/2,(zmax-zmin)/2)
% scatter3(x_data,y_data,z_data,10,'filled','r');
% hold on
% scatter3(x1_data,y1_data,z1_data,10,'filled','B');
% figure(2)
% quiver3(x_data,y_data,z_data,u,v,w,'b')
% subplot(1,2,1)
% 
% histogram((u.^2+v.^2+w.^2).^0.5,20)
% subplot(1,2,2)
% histogram((u.^2+v.^2).^0.5,20)
% % quiver(x_data,y_data,u,v,'r')
% % quiver(x_data,y_data,u,v,'r')
% % hold on
% % quiver(y1_data,z1_data,v,w,'g')
% axis equal
particle_number =50;
x_1_name = 'D:\desktop\spp\text\1st\x_3.txt';
y_1_name = 'D:\desktop\spp\text\1st\y_3.txt';
z_1_name = 'D:\desktop\spp\text\1st\z_3.txt';
u_1_name = 'D:\desktop\spp\text\uvw\u_3.txt';
v_1_name = 'D:\desktop\spp\text\uvw\v_3.txt'
w_1_name = 'D:\desktop\spp\text\uvw\w_3.txt'
x_label_name = 'D:\desktop\spp\label_result\text\x_2.txt';
y_label_name = 'D:\desktop\spp\label_result\text\y_2.txt';
z_label_name = 'D:\desktop\spp\label_result\text\d_2.txt';
u_label_name = 'D:\desktop\spp\label_result\text\w_2.txt';
v_label_name = 'D:\desktop\spp\label_result\text\h_2.txt';
w_label_name = 'D:\desktop\spp\text\uvw\w_1.txt'
x_1 = textread(x_1_name);
y_1 = textread(y_1_name);
z_1 = textread(z_1_name);
u_1 = textread(u_1_name);
v_1 = textread(v_1_name);
w_1 = textread(w_1_name);
z_max=max(z_1)
z_min = 1000
z_change = 255/(z_max-z_min)
depth_data = (z_1-1000)/z_change
depth_data = round(depth_data)
depth_data_mat2 = depth_data/max(depth_data)*255
depth_data = depth_data_mat2
depth_data =round(depth_data)
x_data = textread(x_label_name);
y_data = textread(y_label_name);
z_data = textread(z_label_name);
u_data = textread(u_label_name);
v_data = textread(v_label_name);

w =  randi([2,2],52,1);
scatter3(x_1,y_1,depth_data,'filled','b')  % 三维粒子场
hold on
quiver3(x_1,y_1,depth_data,u_1,v_1,w_1,'r')
% hold on
% scatter3(x_data,y_data,z_data,'filled','r')  % 预测
% hold on

% scatter(x_1,y_1,u_1,v_1,'r')
% hold on
% scatter(x_data,y_data,'b')
%  hold on
% quiver(x_1,y_1,v_1,u_1,'r')
% quiver(x_data,y_data,u_data,v_data,'b')
% quiver(x_data,y_data,z_data,u_data,v_data,'b')
