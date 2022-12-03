# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:41:19 2021

@author: ion-m
"""
# -*- coding: utf-8 -*-
def get_mask_axis(count, mask, id):
    print(mask.shape)
    print(id)
    print(id.shape)
    x0 = 0
    y0 = 0
    height_nut= []
    if id[0] == 1:                  #mask的预测并不是按照类别来，他先检测到哪个，哪个就是第一个，之后是第二个
        for i in range(0,255):      #图像按行扫描，求得最大行x0
            if mask[i, 0:255 , 0].any():
                # print(i)
                x0 = i
            
        for j in range(0,255):     #在最大行x0的基础上，按列扫描，求得切点(x0, y0)
            if mask[x0, j, 0]:
                y0 = j

        for a in range(0, x0):      #在y0列，扫描得到a0
            a0 = a
            if mask[a, y0, 0]:
                break    
        
        for b in range(0, 255):    #在y0列，扫描得到b，数列长度即为nut高度
            if mask[b, y0, 1]:
                height_nut.append(b)
        
        with open('axis_nut.txt', 'a') as file:
            file.write(('%s '*4 + '\n' )%((count), str((x0-a0)), str(len(height_nut)),str( (x0-a0) / len(height_nut))))

        print('a0 : '+ str(a0))
        print('x0 : '+ str(x0))
        print('y0 : '+ str(y0))
        print('x0-a0 : ' + str(x0-a0))
        print(height_nut)
        print('b2-b1 : ' + str(len(height_nut)))
        try:
            print("x/b ： " + str( (x0-a0) / len(height_nut) ) )   #螺纹高度和螺母高度之比
        except ZeroDivisionError as e:
            print(e)
    
    elif id[0] == 2:
        for i in range(0,255):      #图像按行扫描，求得最大行x0
            if mask[i, 0:255 , 1].any():
                # print(i)
                x0 = i
            
        for j in range(0,255):     #在最大行x0的基础上，按列扫描，求得切点(x0, y0)
            if mask[x0, j, 1]:
                y0 = j

        for a in range(0, x0):      #在y0列，扫描得到a0
            a0 = a
            if mask[a, y0, 1]:
                break    
        
        for b in range(0, 255):    #在y0列，扫描得到b，数列长度即为nut高度
            if mask[b, y0, 0]:
                height_nut.append(b)
        
        with open('axis_nut.txt', 'a') as file:
            file.write(('%s '*4 + '\n' )%((count), str((x0-a0)), str(len(height_nut)),str( (x0-a0) / len(height_nut))))

        print('a0 : '+ str(a0))
        print('x0 : '+ str(x0))
        print('y0 : '+ str(y0))
        print('x0-a0 : ' + str(x0-a0))
        print(height_nut)
        print('b2-b1 : ' + str(len(height_nut)))
        try:
            print("x/b ： " + str( (x0-a0) / len(height_nut) ) )   #螺纹高度和螺母高度之比
        except ZeroDivisionError as e:
            print(e)        