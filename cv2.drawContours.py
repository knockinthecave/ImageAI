# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:13:32 2021

@author: ionman
"""

import cv2
import numpy as np

# Load image, grayscale, Otsu's threshold

img = cv2.imread('2.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
 
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
draw = cv2.drawContours(img,contours,-1,(0,0,255),3)  
 
cv2.imshow("img", img)  
cv2.waitKey(0)  

