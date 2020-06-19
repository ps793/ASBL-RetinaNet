import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
import os

# =============================================================================
# visualization and save into folder
# =============================================================================    
import glob 
import os
import numpy as np
nms_list=[0.3]
prob=0.4
for nms in nms_list:
    class_dic={'airplane':1,'baseball':2,'basketball':3,'bridge':4,'tenniscourt':5,
            'groundtrackfield':6,'harbor':7,'storagetank':8,'ship':9,'vehicle':10}
    ml_type='test'
    

    root='/home/peng/Desktop/NWPU VHR-10.v2 dataset/JPEGImages/'
    with open('/home/peng/Desktop/NWPU VHR-10.v2 dataset/train-test/test.txt') as f:
        test_list = f.readlines()
    
    image_list=test_list
    root1='/home/peng/Desktop/NWPU VHR-10.v2 dataset'
    
    ##load the model
    net = RetinaNet(num_classes=10)
    net=torch.load('./checkpoint/adam_e4_iou45_pre_fpn50_full_b2_vhr_50_nd_9ma_s20_105_5block_final1_flipflop_dy.pkl')
    net.eval()

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
        
    ##walk through all images and then make prediction   
    count=0  
    for i in range(len(image_list)):
        item=os.path.join(root,image_list[i].split(' \n')[0]+'.jpg')
        count+=1
        img = Image.open(item)
        rw,rh=img.size
        w = h = 600
        img = img.resize((w,h))
        
        #print('Predicting..')
        x = transform(img)
        x = x.unsqueeze(0)
        x = Variable(x, volatile=True)
        x = x.cuda()
        loc_preds, cls_preds = net(x)
        
        #print('Decoding..')
        encoder = DataEncoder()
    
        #CLS=0.3,NMS=0.15
        boxes, labels, score = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(),(w,h),prob,nms)
    
        all_blobs=[]
        if score.is_cuda:
            boxes=boxes.cpu().numpy()
            score=score.cpu().numpy()
            labels=labels.cpu().numpy()
        else:
            score=[]
        scale1=float(w)/rw
        scale2=float(h)/rh
            
        if score !=[]:
    
             # different order compared with mask rcnn and unet
            draw = ImageDraw.Draw(img)
            s_count=0
            for idx in range(boxes.shape[0]):
                cor = list(boxes[idx]) # (x1,y1, x2,y2)
                
                line = (cor[0],cor[1],cor[0],cor[3])
                draw.line(line, fill=(0,0,256/labels[idx]), width=5)
                line = (cor[0],cor[1],cor[2],cor[1])
                draw.line(line, fill=(0,0,256/labels[idx]), width=5)
                line = (cor[0],cor[3],cor[2],cor[3])
                draw.line(line, fill=(0,0,256/labels[idx]), width=5)
                line = (cor[2],cor[1],cor[2],cor[3])
                draw.line(line, fill=(0,0,256/labels[idx]), width=5)
                if score[idx] >=0.4:
                    s_count+=1
            ## save images into the output folder
            if s_count>0:   
                img.save(os.path.join(root1,'testvisual',item.split('/')[-1]))
# =============================================================================
# mask prediction 
# =============================================================================


import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

import glob
from skimage import io
import skimage.feature
import pandas as pd

def py_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]

        order = order[inds + 1]
    return keep


import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

import glob
from skimage import io
import skimage.feature
import pandas as pd
import xml
from shutil import copyfile

###load the model
print('Loading model..')
net = RetinaNet(num_classes=10)
net=torch.load('./checkpoint/adam_e4_s30_35_dy.pkl') ##change weight
#net.load_state_dict(state['net'])
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])


###load the path of images
root='/home/peng/Desktop/NWPU VHR-10.v2 dataset/JPEGImages/'
with open('/home/peng/Desktop/NWPU VHR-10.v2 dataset/train-test/test.txt') as f:
    test_list = f.readlines()

image_list=[]
for i in range(len(test_list)):
    input_path=os.path.join(root,test_list[i].split(' \n')[0]+'.jpg')
    image_list.append(input_path)

label_root='/home/peng/Desktop/NWPU VHR-10.v2 dataset/Annotations/'
class_name={'airplane':1,'baseball':2,'basketball':3,'bridge':4,'tenniscourt':5,
            'groundtrackfield':6,'harbor':7,'storagetank':8,'ship':9,'vehicle':10}





test_gt=[]
test_pred=[]
for idx in range(len(image_list)):
    print(idx)
    
    output=io.imread(image_list[idx])
    img_name=image_list[idx].split('/')[-1]
    

    all_gt=[]
    name=img_name.split('.')[0]
    label=os.path.join(label_root,name+'.xml')
    e=xml.etree.ElementTree.parse(label).getroot()
    for ob in e.findall('object'):
        xmin=int(ob[4][0].text)
        ymin=int(ob[4][1].text)
        xmax=int(ob[4][2].text)
        ymax=int(ob[4][3].text)
        category=ob[0].text
        all_gt.append([category,xmin,ymin,xmax,ymax])

    output_name=name+'.txt'
    outF = open(os.path.join('/home/peng/Desktop/NWPU VHR-10.v2 dataset/retina/ground-truth',output_name), "w")
    for line in all_gt:
      outF.write(str(line[0]) + " " + " ".join([str(a) for a in line[1:len(line)]]) + '\n')
      
    outF.close() 
    
    copyfile(image_list[idx],'/home/peng/Desktop/NWPU VHR-10.v2 dataset/retina/images/'+name+'.jpg')
    
    img = Image.open(image_list[idx])
    w = h = 600
    img = img.resize((w,h))
    
    #print('Predicting..')
    x = transform(img)
    x = x.unsqueeze(0)
    x = Variable(x, volatile=True)
    loc_preds, cls_preds = net(x)
    
    #print('Decoding..')
    encoder = DataEncoder()
    
    
    #CLS=0.3,NMS=0.15
    boxes, labels, score = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(),(w,h),0.001,0.15)

    all_blobs=[]

    boxes=boxes.cpu().numpy()
    score=score.cpu().numpy()
    labels=labels.cpu().numpy()

    scale=600.0/400
    if isinstance(boxes[0],np.ndarray):
        length=len(boxes)
        x1,y1,x2,y2=boxes[:,0]/scale,boxes[:,1]/scale,boxes[:,2]/scale,boxes[:,3]/scale # different order compared with mask rcnn and unet
        x1=x1.reshape(length,1)
        y1=y1.reshape(length,1)
        x2=x2.reshape(length,1)
        y2=y2.reshape(length,1)
        boxes=np.concatenate((x1,y1,x2,y2),axis=1)
        score=score.reshape(length,1)
        box=np.concatenate((boxes,score), axis=1)
        if len(score)>1:
           ind=py_nms(box,0.1)     ####most of time comsuming part!!!!!
           box=box[ind]   
           labels=labels[ind]   

        
        
        ### output prediction result and make them into right format for evaluation
        for ind_i in range(len(box)):
            
            xmin=int(box[ind_i][0])
            ymin=int(box[ind_i][1])
            xmax=int(box[ind_i][2])
            ymax=int(box[ind_i][3])
            score=box[ind_i][4]
            category=class_name.keys()[class_name.values().index(labels[ind_i]+1 )]
            all_blobs.append([category,score,xmin,ymin,xmax,ymax])    
        output_name=name+'.txt'
        outF = open(os.path.join('/home/peng/Desktop/NWPU VHR-10.v2 dataset/retina/predicted',output_name), "w")
        for line in all_blobs:
          outF.write(str(line[0]) + " " + " ".join([str(a) for a in line[1:len(line)]]) + '\n')
        outF.close() 




