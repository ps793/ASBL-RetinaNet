#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 20:50:23 2018

@author: peng
"""

from __future__ import print_function, division
import os
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import random
import PIL
from PIL import Image, ImageOps
import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import cv2
import skimage.feature

import glob

#######training
'''
with open('/mnt/ssd_disk/naka247/peng/wider_hard_upload/ImageSets/Main/trainval.txt') as f:
    train_list = f.readlines()
with open('/mnt/ssd_disk/naka247/peng/wider_hard_upload/ImageSets/Main/test.txt') as f:
    val_list = f.readlines()
'''
label_root='/mnt/ssd_disk/naka247/peng/NWPU VHR-10 dataset/ground truth/'
image_root='/mnt/ssd_disk/naka247/peng/NWPU VHR-10 dataset/positive image set/'
#####val

import string
# =============================================================================
# prepare dataset for HBOX
# =============================================================================

from sklearn.model_selection import train_test_split          
def pre_retina(image_root,txt_root):
    #image_root='/home/peng/Desktop/DOTA/valsplit/images/'
    image_list=sorted(glob.glob(os.path.join(image_root,'*.jpg')))
    all_trainval=[]
    w_list=[]
    h_list=[]
    for item in image_list:
        name=item.split('/')[-1].split('.')[0]
        label=os.path.join(label_root,name+'.txt')
        annotations=[]
        with open(label, 'r') as f:
            for i, x in enumerate(f):
                annotations.append(x.split('\n')[0])
        image_name=item
        img=cv2.imread(image_name)
        w,h,c=img.shape
        w_list.append(w)
        h_list.append(h)

        output=[image_name]
        for anno in annotations:
            if anno=='\r':
                continue
            info=anno.split(',')
            info[0:4]=map(lambda x: x.translate(string.maketrans('', ''), '()'), info[0:4])
            x1,y1,x2,y2,label=map(int,info[:])
            class_name=label
            output.extend([x1,y1,x2,y2])
            output.extend([class_name])        
        all_trainval.append(output)
    
        
    trainval ,test = train_test_split(all_trainval,test_size=0.6,random_state=2018)  
    train ,val = train_test_split(trainval,test_size=0.5,random_state=2018)      
    total=[train,val,test]    
    o_name=['train','val','test']
    for i in range(len(total)):    
        output_name='%s_retina_vhr.txt' % o_name[i]
        outF = open(os.path.join('/mnt/ssd_disk/naka247/peng/NWPU VHR-10 dataset/',output_name), "w")
        for line in total[i]:
          outF.write(str(line)[1:len(str(line))-1])
          outF.write("\n")
        outF.close() 

pre_retina(image_root,label_root)

