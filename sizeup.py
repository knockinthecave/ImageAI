# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:47:17 2022

@author: ionman
"""

import cv2  
import numpy as np  
import matplotlib.pyplot as plt
 
#读取原始图像
img = cv2.imread('1.jpg')
 
#图像向上取样
r = cv2.pyrUp(img)
 
#显示图像
cv2.imshow('original', img)
cv2.imshow('PyrUp', r)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite('sizeup.jpg', r)
