# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob 
import pickle


# image size provided
img_h = 128
img_w = 128
img_c = 3


# I want take all rgb images as X and all masked one as y
file_loc = "Fish_Dataset/Fish_Dataset/*"
folder_main = sorted(glob.glob(file_loc))

folder_main

loc_of_rgb = []
loc_of_masked = []
for i in folder_main:
    each_fish = sorted(glob.glob(i+"/*"))
    rbg = each_fish[0]
    masked = each_fish[1]
    all_rgb_images = sorted(glob.glob(rbg + "/*"))
    all_mask_images =  sorted(glob.glob(masked + "/*"))
    loc_of_rgb.append(all_rgb_images)
    loc_of_masked.append(all_mask_images)
 
    
# Merge list items    
loc_of_rgb1 = []
loc_of_masked1 = []
for i in loc_of_rgb:
    for j in i:
        loc_of_rgb1.append(j)
        
for a in loc_of_masked:
    for b in a:
        loc_of_masked1.append(b)
        
len(loc_of_rgb1)
        
# Creating input and output shapes

X = np.zeros((len(loc_of_rgb1),img_h,img_w,img_c), dtype=np.uint8)

y = np.zeros((len(loc_of_masked1),img_h,img_w,1),dtype=np.bool)


for rgb_img in loc_of_rgb1:
    index = loc_of_rgb1.index(rgb_img)
    img_read = cv2.imread(rgb_img)
    img_size = cv2.resize(img_read,(img_h,img_w))
    img_array = np.array(img_size)
    X[index] = img_array
    percentage = (index * 100)/len(loc_of_rgb1)
    ## just chkening current status of loop 
    print("{}% of process done for RGB".format(round(percentage),1))
    
    
for mask in loc_of_masked1:
    index_mask = loc_of_masked1.index(mask)
    mask_img = cv2.imread(mask)
    mask_gray = cv2.cvtColor(mask_img,cv2.COLOR_BGR2GRAY)
    mask_gray_size = cv2.resize(mask_gray,(img_h,img_w))
    mask_array = np.array(mask_gray_size).reshape(img_h,img_w,1)
    y[index_mask] = mask_array
    percentage = (index_mask * 100)/len(loc_of_masked1)
    ## just chkening current status of loop 
    print("{}% of process done for mask".format(round(percentage), 1))


pickle_out = open('X.pickle','wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle','wb')
pickle.dump(y, pickle_out)
pickle_out.close()
