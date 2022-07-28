"""
提取图片坐标信息
单个提取精度更高，循环提取精度低
"""
import cv2
import numpy as np
import os

path = 'D:/image/val/label'
img_name = os.listdir(os.path.join(path))
img_path = os.path.join(path, img_name[0])
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
ret, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # 阈值化处理
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 找到轮廓
result = cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
# cv2.imwrite('D:/desktop/T1/r3.png', result)
x_lst = []
y_lst = []
R_lst = []
d_lst = []

for c in range(0, len(contours)):
    x, y, w, h = cv2.boundingRect(contours[c])  # x,y:矩形左上点坐标  w,h:矩形的宽和高
    R = min(w, h)-1

    r_mid = int(R / 2)
    x_mid = x + r_mid + 1
    y_mid = y + r_mid + 1
    x_int = int(x)
    x_end = int(x + R)
    y_int = int(y)
    y_end = int(y + R)
    depth_lst = gray[y_int:y_end, x_int:x_end]
    depth_lst = np.ravel(depth_lst)
    depth_lst = np.sort(depth_lst)
    depth_lst = depth_lst[-(R + int(R / 2)):-int(R / 2)]
    d = round(np.mean(depth_lst))

    x_lst.append(x_mid)
    y_lst.append(y_mid)
    R_lst.append(r_mid)
    d_lst.append(d)
    cv2.rectangle(img, (x, y), (x + R, y + R), (0, 0, 255), 1)
cv2.imwrite('D:/desktop/T1/text/1.png', img)
np.savetxt('D:/desktop/T1/text/d.txt', d_lst, fmt='%f')
np.savetxt('D:/desktop/T1/text/y.txt', x_lst, fmt='%f')  # x,y坐标转换
np.savetxt('D:/desktop/T1/text/x.txt', y_lst, fmt='%f')
np.savetxt('D:/desktop/T1/text/r.txt', R_lst, fmt='%f')
print('over')
