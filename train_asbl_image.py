from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss_asbl_image import FocalLoss
from retinanet import RetinaNet
from datagen_flipflop import ListDataset

from torch.autograd import Variable
import copy
from tqdm import tqdm

import runpy
from encoder import DataEncoder
import numpy as np

#==============================================================================
# parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
# parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# args = parser.parse_args()
#==============================================================================

lr=1e-4
resume=False
fix='full'
bs=2

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
best_map = float('-inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

trainset = ListDataset(list_file='/home/peng/Desktop/NWPU VHR-10.v2 dataset/train_retina_vhr2.txt' , train=True,  transform=transform, input_size=640)###may follow other papers
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

testset = ListDataset(list_file='/home/peng/Desktop/NWPU VHR-10.v2 dataset/val_retina_vhr2.txt' , train=False,  transform=transform, input_size=640)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=8, collate_fn=testset.collate_fn)

# Model
net = RetinaNet(num_classes=10)
net.load_state_dict(torch.load('./model/vhr_vgg16.pth'))
if resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()

if fix=='head':
    for param in net.module.fpn.conv1.parameters():
        param.requires_grad = False
    for param in net.module.fpn.bn1.parameters():
        param.requires_grad = False
    for param in net.module.fpn.layer1.parameters():
        param.requires_grad = False
    for param in net.module.fpn.layer2.parameters():
        param.requires_grad = False
    for param in net.module.fpn.layer3.parameters():
        param.requires_grad = False
    for param in net.module.fpn.layer4.parameters():
        param.requires_grad = False
else:
    for param in net.parameters():
        param.requires_grad = True
        
###optimizer
from itertools import ifilter
op_parameters = ifilter(lambda p: p.requires_grad, net.parameters())
 
criterion = FocalLoss(num_classes=10)
optimizer = optim.Adam(op_parameters, lr=lr, betas=(0.9, 0.99))
#later add scheduler into optimizer
#optimizer = optim.SGD(op_parameters, lr=lr, momentum=0.9, weight_decay=1e-4)
from torch.optim import lr_scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) ###need to update step size


###predefined functions
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

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    net.train()
    net.module.freeze_bn()
    train_loss = 0
    count=0
    t = tqdm(trainloader)
    for batch_idx, (inputs, loc_targets, cls_targets, flip) in enumerate(t):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()
        if num_pos==0:
            count+=1
            #print(batch_idx)
            continue
        
        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss,loc_loss,cls_loss,iw = criterion(inputs, flip, loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        #print('train_loss: %.3f | avg_loss: %.3f' % (loss.data[0], train_loss/(batch_idx-count+1)))
        t.set_description('loc_loss: %.3f|cls_loss: %.3f|iw: %.2f | train_loss: %.3f | avg_loss: %.3f, %d' % (loc_loss, cls_loss, iw, loss.data[0], train_loss/(batch_idx+1),flip[0]))
        
    return train_loss/ len(trainloader)
# Test
    
    
import xml
from shutil import copyfile
from skimage import io
from PIL import Image, ImageDraw
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    count=0
#t.set_description('loc_loss: %.3f|cls_loss: %.3f|iw: %.2f | train_loss: %.3f | avg_loss: %.3f, %d' % (loc_loss, cls_loss, iw, loss.data[0], train_loss/(batch_idx+1),flip[0]))
    t = tqdm(testloader)    
    for batch_idx, (inputs, loc_targets, cls_targets, flip) in enumerate(t):
        #need to check 10_G272_15Nov2016_0019.JPG, batch_idx=189
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()
        if num_pos==0:
            count+=1
            #print(batch_idx)
            continue
        
        loc_preds, cls_preds = net(inputs)
        loss,loc_loss,cls_loss,iw = criterion(inputs, flip, loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data[0]
        t.set_description('loc_loss: %.3f|cls_loss: %.3f|iw: %.2f | test_loss: %.3f | avg_loss: %.3f, %d' % (loc_loss, cls_loss, iw, loss.data[0], test_loss/(batch_idx+1),flip[0]))
        
    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        best_model_wts = copy.deepcopy(net.state_dict())
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        net.load_state_dict(best_model_wts)
        torch.save(net, './checkpoint/val.pkl' )
        best_loss = test_loss
    
    return test_loss



###select the best performance on validation dataset for mAP result.
def mAP(): 
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
    
    
    global best_map
    
    for idx in range(len(image_list)):
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
        
        #copyfile(image_list[idx],'/home/peng/Desktop/NWPU VHR-10.v2 dataset/retina/images/'+name+'.jpg')
        
        img = Image.open(image_list[idx])
        w = h = 640
        img = img.resize((w,h))
        
        #print('Predicting..')
        x = transform(img)
        x = x.unsqueeze(0)
        x = Variable(x, volatile=True)
        loc_preds, cls_preds = net(x)
        
        #loc_preds, cls_preds= loc_preds.cpu(), cls_preds.cpu()
        #print('Decoding..')
        encoder = DataEncoder()
        
        #CLS=0.3,NMS=0.15
        boxes, labels, score = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(),(w,h),0.1,0.15)# 0.0001
        #draw = ImageDraw.Draw(img)

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
    
    runpy.run_path('/home/peng/Desktop/NWPU VHR-10.v2 dataset/retina/main.py')
    result_txt= open('/home/peng/Desktop/NWPU VHR-10.v2 dataset/retina/results/results.txt', 'r')
    content = result_txt.read()
    content = content.split('\n')
    mAP=filter(lambda x: 'mAP' in x, content)[1]
    mAP=float(mAP.split(' ')[-1].split('%')[0])
    print(mAP)
    if mAP > best_map:
        print('Saving..')
        best_model_wts = copy.deepcopy(net.state_dict())
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        net.load_state_dict(best_model_wts)
        torch.save(net, './checkpoint/test_best_mAP.pkl' )
        best_mAP = mAP
    return best_mAP
        
    
    
### main function, for train/validation
train_loss_list=[]
test_loss_list=[]
mAP_list=[]
for epoch in range(start_epoch, start_epoch+100):
    loss=train(epoch)
    train_loss_list.append(loss)
    loss=test(epoch)
    test_loss_list.append(loss)
    if epoch % 5 ==0:
        map_value=mAP()
        mAP_list.append(map_value)
