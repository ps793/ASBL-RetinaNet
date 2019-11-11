import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
import os

print('Loading model..')
net = RetinaNet(num_classes=10)
net=torch.load('./checkpoint/adam_e4_iou45_pre_fpn50_full_b2_vhr_50_nd_9ma_s20_105_5block_final1_flipflop_dy.pkl')
#net.load_state_dict(state['net'])
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

root='/home/peng/Desktop/NWPU VHR-10.v2 dataset/JPEGImages/'
with open('/home/peng/Desktop/NWPU VHR-10.v2 dataset/train-test/test.txt') as f:
    test_list = f.readlines()

print('Loading image..')
img = Image.open(os.path.join(root,test_list[14].split(' \n')[0]+'.jpg'))
img = Image.open('/home/peng/Desktop/NWPU VHR-10.v2 dataset/JPEGImages/000872.jpg')
w = h = 600
img = img.resize((w,h))

print('Predicting..')
x = transform(img)
x = x.unsqueeze(0)
x = Variable(x, volatile=True)
#print('Decoding..')
encoder = DataEncoder()

import time
start = time.time()
loc_preds, cls_preds = net(x)



#loc_preds, cls_preds, input_size
#CLS=0.3,NMS=0.15
boxes, labels, score = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(),(w,h),0.001,0.15)
end = time.time()
print(end - start)


#loc_preds, cls_preds, input_size
#CLS=0.3,NMS=0.15
boxes, labels, score = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(),(w,h),0.001,0.15)

all_blobs=[]
if score.is_cuda:
    boxes=boxes.cpu().numpy()
    score=score.cpu().numpy()
    labels=labels.cpu().numpy()
else:
    score=[]

draw = ImageDraw.Draw(img)
for idx in range(boxes.shape[0]):
    ix,iy,x,y=boxes[idx]
    draw.rectangle(list(boxes[idx]), outline='red')
    #draw.text((ix-12, iy-12),"%.2f"%score[idx],(255,255,255))
    #draw.text((x, y),"%d"%labels[idx],(255,255,255))
img.show()
# =============================================================================
# ####make prediction for all images
# =============================================================================
boxes=boxes.numpy()
score=score.numpy()
scale=600.0/600
all_blobs=[]
for idx in range(boxes.shape[0]):
    x1,y1,x2,y2=boxes[idx]
    x=((x1+x2)/2)/scale
    y=((y1+y2)/2)/scale
    px=int(x)
    py=int(y)
    all_blobs.append([px,py,score[idx]])
    
    
    
# =============================================================================
# ####visualization 
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
    #os.mkdir(os.path.join(root1,'testvisual'))
    
    
    ####first we need to generate result on patches
    ####second merge patches into one image
    ####third generate format of submission
    net = RetinaNet(num_classes=10)
    net=torch.load('./checkpoint/adam_e4_iou45_pre_fpn50_full_b2_vhr_50_nd_9ma_s20_105_5block_final1_flipflop_dy.pkl')
    #net.load_state_dict(state['net'])
    net.eval()

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
        
        
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
            #img.show()
                #draw.rectangle(, outline='red')
                if score[idx] >=0.4:
                    s_count+=1
            if s_count>0:   
                img.save(os.path.join(root1,'testvisual',item.split('/')[-1]))
# =============================================================================
# ####mask prediction 
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
def IoU(box1, box2):

    [xmin1, ymin1, xmax1, ymax1] = box1
    [xmin2, ymin2, xmax2, ymax2] = box2
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    xmin_inter = max(xmin1, xmin2)
    xmax_inter = min(xmax1, xmax2)
    ymin_inter = max(ymin1, ymin2)
    ymax_inter = min(ymax1, ymax2)
    if xmin_inter > xmax_inter or ymin_inter > ymax_inter:
        return 0
    area_inter = (xmax_inter - xmin_inter) * (ymax_inter - ymin_inter)
    
    return float(area_inter) / (area1 + area2 - area_inter)




#blobs, pred_blobs=all_gt,all_blobs

def evaluation_blob(blobs, pred_blobs, iou_ratio):      
    if len(blobs)==0 and len(pred_blobs)!=0:
        red=0
        fp=len(pred_blobs)
        tp=0
        fn=0
    if len(blobs)!=0 and len(pred_blobs)==0:
        red=0
        fn=len(blobs)
        tp=0
        fp=0
    if len(blobs)==0 and len(pred_blobs)==0:
        red=0
        fp=0
        tp=0
        fn=0  
        
    if len(blobs)!=0 and len(pred_blobs)!=0: 
        fp=0
        tp=0
        fn=0
        red=0
        for truth in blobs:
            x1=int(truth[0])
            y1=int(truth[1])
            x2=int(truth[2])
            y2=int(truth[3])
            box1=[x1,y1,x2,y2]
            all_dis=[]
            all_score=[]
            pred_blobs=np.array(pred_blobs)           
            #logic1=(pred_blobs[:,0]>=nneg(ty-100)) & (pred_blobs[:,0]<=nneg(ty+100)) 
            #logic2=(pred_blobs[:,1]>=nneg(tx-100)) & (pred_blobs[:,1]<=nneg(tx+100)) 
            #small_pred=pred_blobs[logic1 & logic2]
            small_pred=pred_blobs
            for pred in small_pred:
                px1= int(pred[0])
                py1= int(pred[1])
                px2= int(pred[2])
                py2= int(pred[3])
                score=pred[4]
                
                box2=[px1,py1,px2,py2]
                #dis=np.sqrt(np.power((px-tx),2)+np.power((py-ty),2))
                #need to change to IoU
                iou=IoU(box1,box2)
                all_score.append(score)
                all_dis.append(iou)
            #print(all_dis)
            if sum(np.array(all_dis)>iou_ratio) !=0:
                #for two ture with one pred problem
                min_dis=max(all_dis) 
                min_ind=all_dis.index(min_dis)  
                
                small_pred[min_ind]=[-100,-100,-100,-100,-100]
                tp+=1
                #red+=sum(np.array(all_dis)>iou_ratio)-1
            
            if sum(np.array(all_dis)>iou_ratio) ==0:
                fn+=1
            #pred_blobs[logic1 & logic2]=small_pred
            pred_blobs=small_pred
        fp=len(pred_blobs)-tp
        #print(tp,fn,fp)
    
    return fp,tp,fn,red

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
print('Loading model..')
net = RetinaNet(num_classes=10)
net=torch.load('./checkpoint/adam_e4_s30_35_dy.pkl')
#net.load_state_dict(state['net'])
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])


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

import xml
from shutil import copyfile



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
    
    #loc_preds, cls_preds= loc_preds.cpu(), cls_preds.cpu()
    #print('Decoding..')
    encoder = DataEncoder()
    
    
    #loc_preds, cls_preds, input_size
    #CLS=0.3,NMS=0.15
    boxes, labels, score = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(),(w,h),0.001,0.15)
    #draw = ImageDraw.Draw(img)
    '''
    for box in boxes:
        draw.rectangle(list(box), outline='red')
    img.show()
    '''

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




#############################
    print('calculating f1 score for %d test' %len(image_list))
    thres_list=np.arange(0, 1.0, 0.05)
    
    
    # on easy one, 0.3 -> 0.8729 adam
    recall_list=[]
    precision_list=[]
    for thres in thres_list:
        print thres
        all_tp=[]
        all_fn=[]
        all_fp=[]
        iou_ratio=0.4
        all_pred=[]
        #thres=0.95
        all_recall=[]
        all_precision=[]
        all_true=[]
        for idx in range(len(image_list)):   
            #get gt cood
            all_gt=test_gt[idx]
        
            
            pred=test_pred[idx]
            all_blobs=[]           
            for i in range(len(pred)):
                for m in range(len(pred[i])):
                   x1,y1,x2,y2,score=pred[i][m]
                   if score>thres:
                       box=[x1,y1,x2,y2,score]
                       all_blobs.append(box)
                               
        
                        
            pred_len=len(all_blobs)  
            #print(pred_len)
            fp,tp,fn,red=evaluation_blob(all_gt,all_blobs,iou_ratio=iou_ratio)
            assert(tp+fn==len(all_gt))
            assert(tp+fp+red==pred_len)
            all_tp.append(tp)
            all_fn.append(fn)
            all_fp.append(fp)   
            all_pred.append(pred_len)
            if tp+fn==0:
                recall=0
            else:
                recall=float(tp)/(tp+fn)
                
            if tp+fp==0:
                precision=0
            else:
                precision=tp/(tp+fp)
            all_recall.append(recall)
            all_precision.append(precision)
            all_true.append(len(all_gt))
            
        #print(all_tp)
        #print(all_fn)
        #print(all_fp)
        
        recall_result=float(np.sum(all_tp))/(np.sum(all_tp)+np.sum(all_fn))
        precision_result=float(np.sum(all_tp))/(np.sum(all_tp)+np.sum(all_fp))
        recall_list.append(recall_result)
        precision_list.append(precision_result)
        f1=(2*precision_result*recall_result)/(recall_result+precision_result)
        
        
        print('full test f1: %.4f' % f1)    



    data=pd.DataFrame(list(zip(thres_list,precision_list, recall_list)),
                  columns=['thres','precision','recall'])
    data=data.dropna()
    new=data.sort_values('precision')
    pr_list[c]=new
    
    
from sklearn import metrics
new=pr_list['baseball']
pr=metrics.auc(new['recall'],new['precision'])
plt.plot(new['recall'],new['precision'])

from sklearn.metrics import average_precision_score


class_name={'airplane':1,'baseball':2,'basketball':3,'bridge':4,'tenniscourt':5,
            'groundtrackfield':6,'harbor':7,'storagetank':8,'ship':9,'vehicle':10}


img = img.resize((w,h))

draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
img.show()
