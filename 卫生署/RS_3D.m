clc
fig = 4
depth_file_name = 'D:\desktop\T2/z_1.txt';
x_file_name     = 'D:\desktop\T2/x_1.txt';
y_file_name     = 'D:\desktop\T2/y_1.txt';
r_file_name     = 'D:\desktop\T2/r_1.txt';

depth_label_file_name = 'D:\desktop\T1\text/d.txt';
x_label_file_name     = 'D:\desktop\T1\text/x.txt';
y_label_file_name     = 'D:\desktop\T1\text/y.txt';
r_label_file_name     = 'D:\desktop\T1\text/r.txt';

depth_data = textread(depth_file_name);
%depth_data = (depth_data*2.048/255)*1000+1000;
x_data = textread(x_file_name);
y_data = textread(y_file_name);
r_data = textread(r_file_name);
z_max=max(depth_data)
z_min = min(depth_data)
z_change = 255/(z_max-z_min)
depth_data = (depth_data-1000)/z_change
depth_data = round(depth_data)
depth_data_mat2 = depth_data/max(depth_data)*255
depth_data = depth_data_mat2
depth_data =round(depth_data)

figure(1);
scatter3(x_data,y_data,depth_data,r_data,'o','red')
hold on
%{
depth_data_label = textread(depth_label_file_name);
depth_data_label = (depth_data_label*2.048/255)*1000+1000;
x_data_label = textread(x_label_file_name);
y_data_label = textread(y_label_file_name);
r_data_label = textread(r_label_file_name);
scatter3(x_data_label,y_data_label,depth_data_label,r_data_label,'black')
%}
figure(2)

depth_data_label = textread(depth_label_file_name);
% depth_data_label = (depth_data_label*2.048/255)*1000+1000;
x_data_label = textread(x_label_file_name);
y_data_label = textread(y_label_file_name);
r_data_label = textread(r_label_file_name);
scatter3(x_data_label,y_data_label,depth_data_label,r_data_label*2,'+','b')  %(x,y,z,R)

figure(3);
scatter3(x_data,y_data,depth_data,r_data,'o','r')
hold on
depth_data_label = textread(depth_label_file_name);
%depth_data_label = (depth_data_label*2.048/255)*1000+1000;
x_data_label = textread(x_label_file_name);
y_data_label = textread(y_label_file_name);
r_data_label = textread(r_label_file_name);
scatter3(x_data_label,y_data_label,depth_data_label,r_data_label*2,'+','b')
legend({'Ground True','Predict'});

xlim('auto')
ylim('auto')
zlim('auto')
