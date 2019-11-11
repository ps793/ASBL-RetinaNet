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

from loss_sbl_dy import FocalLoss
from retinanet import RetinaNet
from datagen_flipflop import ListDataset

from torch.autograd import Variable
import copy
from tqdm import tqdm

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
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

trainset = ListDataset(list_file='/mnt/ssd_disk/naka247/peng/NWPU VHR-10 dataset/train_retina_vhr2.txt' , train=True,  transform=transform, input_size=600)###may follow other papers
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

testset = ListDataset(list_file='/mnt/ssd_disk/naka247/peng/NWPU VHR-10 dataset/val_retina_vhr2.txt' , train=False,  transform=transform, input_size=600)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=8, collate_fn=testset.collate_fn)

# Model
net = RetinaNet(num_classes=10)
net=torch.load('./checkpoint/adam_e4_iou45_pre_fpn50_full_b2_dota_50_nd_9ma_s20_105_5block_final1_flipflop.pkl')
net=net.module
net.fpn.eval()
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
        
        
from itertools import ifilter
op_parameters = ifilter(lambda p: p.requires_grad, net.parameters())
  
'''    
count=0
for param in net.parameters():
    count+=1
    print(param.requires_grad)
    print(count) 
'''
criterion = FocalLoss(num_classes=10)
optimizer = optim.Adam(op_parameters, lr=lr, betas=(0.9, 0.99))
#later add scheduler into optimizer
#optimizer = optim.SGD(op_parameters, lr=lr, momentum=0.9, weight_decay=1e-4)
from torch.optim import lr_scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

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

def train(epoch):
    print('\nEpoch: %d' % epoch)
    scheduler.step()
    net.train()
    net.module.freeze_bn()
    '''
    for param in fpn.parameters():
        param.requires_grad = False
    '''
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
        #loss,loc_loss,cls_loss,iw = criterion(inputs, flip, loc_preds, loc_targets, cls_preds, cls_targets)
        loss,loc_loss,cls_loss,iw= criterion(inputs, net.module.fpn,loc_preds, loc_targets, cls_preds, cls_targets)
        
        loss.backward()
        optimizer.step()
	#print(iw)
        train_loss += loss.data[0]
        #print('train_loss: %.3f | avg_loss: %.3f' % (loss.data[0], train_loss/(batch_idx-count+1)))
    	#t.set_description('loc_loss: %.3f|cls_loss: %.3f|iw: %.2f | train_loss: %.3f | avg_loss: %.3f, %d' % (loc_loss, cls_loss, iw, loss.data[0], train_loss/(batch_idx+1),flip[0]))
        t.set_description('loc_loss: %.3f|cls_loss: %.3f|train_loss: %.3f | avg_loss: %.3f, %d' % (loc_loss, cls_loss, loss.data[0], train_loss/(batch_idx+1),flip[0]))
        
    return train_loss/ len(trainloader)
# Test
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
        #loss,loc_loss,cls_loss,iw = criterion(inputs, flip, loc_preds, loc_targets, cls_preds, cls_targets)
        loss,loc_loss,cls_loss,iw= criterion(inputs, net.module.fpn,loc_preds, loc_targets, cls_preds, cls_targets)
        
        test_loss += loss.data[0]
        #print('test_loss: %.3f | avg_loss: %.3f' % (loss.data[0], test_loss/(batch_idx-count+1)))
        t.set_description('loc_loss: %.3f|cls_loss: %.3f| test_loss: %.3f | avg_loss: %.3f, %d' % (loc_loss, cls_loss, loss.data[0], test_loss/(batch_idx+1),flip[0]))
	#print('test_loss: %.3f | avg_loss: %.3f' % (loss.data[0], test_loss/(batch_idx-count+1)))
    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        best_model_wts = copy.deepcopy(net.state_dict())
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        net.load_state_dict(best_model_wts)
        torch.save(net, './checkpoint/adam_e4_iou45_pre_fpn50_full_b2_vhr_50_nd_9ma_s20_105_5block_flipflop_dy.pkl' )
        best_loss = test_loss
    
    return test_loss

    
import csv
train_loss_list=[]
test_loss_list=[]
for epoch in range(start_epoch, start_epoch+100):
    loss=train(epoch)
    train_loss_list.append(loss)
    loss=test(epoch)
    test_loss_list.append(loss)
    with open('/home/peng/Dropbox/retina/vhr/train_loss_e4_full_adam_nd_9ma_iou45_b2_fpn50_s20_105_5block1_flipflop_dy.csv', 'wb') as myfile:
    	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    	wr.writerow(train_loss_list)

    with open('/home/peng/Dropbox/retina/vhr/test_loss_e4_full_adam_nd_9ma_iou45_b2_fpn50_s20_105_5block1_flipflop_dy.csv', 'wb') as myfile:
    	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    	wr.writerow(test_loss_list)
    if epoch % 5 ==0:
       torch.save(net, './checkpoint/adam_e4_iou45_pre_fpn50_full_b2_vhr_50_nd_9ma_s20_105_5block_epoch1_%d_flipflop_dy.pkl' % epoch )

torch.save(net, './checkpoint/adam_e4_iou45_pre_fpn50_full_b2_vhr_50_nd_9ma_s20_105_5block_final1_flipflop_dy.pkl'  )   
    


'''
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    scheduler.step()
    net.module.freeze_bn()
    train_loss = 0
    count=0
    
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        batch_loss=Variable(torch.zeros(1).cuda())
        for b in range(bs):
            inputs[b] = Variable(inputs[b].cuda())
            loc_targets[b] = Variable(loc_targets[b].cuda())
            cls_targets[b] = Variable(cls_targets[b].cuda())
            inputs[b] = inputs[b].unsqueeze(0)
            loc_targets[b] = loc_targets[b].unsqueeze(0)
            cls_targets[b] = cls_targets[b].unsqueeze(0)
            pos = cls_targets[b] > 0  # [N,#anchors]
            num_pos = pos.data.long().sum()
            if num_pos==0:
                count+=1
                #print(batch_idx)
                continue
            loc, cls = net(inputs[b])
            loss = criterion(loc, loc_targets[b], cls, cls_targets[b])
            batch_loss+=loss

        if batch_loss.data.cpu().numpy() ==0:
            count+=2
            continue
        
        optimizer.zero_grad()
        #loc_preds, cls_preds = net(inputs)
        #loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        batch_loss.backward()
        optimizer.step()

        train_loss += batch_loss.data[0]
        print('train_loss: %.3f | avg_loss: %.3f' % (loss.data[0], train_loss/(batch_idx-count+1)))
    
    return train_loss/ len(trainloader)
# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    count=0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        #need to check 10_G272_15Nov2016_0019.JPG, batch_idx=189
        batch_loss=Variable(torch.zeros(1).cuda())
        for b in range(bs):
            inputs[b] = Variable(inputs[b].cuda())
            loc_targets[b] = Variable(loc_targets[b].cuda())
            cls_targets[b] = Variable(cls_targets[b].cuda())
            inputs[b] = inputs[b].unsqueeze(0)
            loc_targets[b] = loc_targets[b].unsqueeze(0)
            cls_targets[b] = cls_targets[b].unsqueeze(0)
            pos = cls_targets[b] > 0  # [N,#anchors]
            num_pos = pos.data.long().sum()
            if num_pos==0:
                count+=1
                #print(batch_idx)
                continue
            loc, cls = net(inputs[b])
            loss = criterion(loc, loc_targets[b], cls, cls_targets[b])
            batch_loss+=loss
        test_loss += batch_loss.data[0]
        print('test_loss: %.3f | avg_loss: %.3f' % (loss.data[0], test_loss/(batch_idx-count+1)))
        
    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        best_model_wts = copy.deepcopy(net.state_dict())
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        net.load_state_dict(best_model_wts)
        torch.save(net, './checkpoint/adam_e4_iou45_pre_fpn50_full_b2_vhr_50_s7.pkl' )
        best_loss = test_loss
    
    return test_loss

train_loss_list=[]
test_loss_list=[]
for epoch in range(start_epoch, start_epoch+50):
    loss=train(epoch)
    train_loss_list.append(loss)
    loss=test(epoch)
    test_loss_list.append(loss)
    if epoch % 10 ==0:
       torch.save(net, './checkpoint/adam_e4_iou45_pre_fpn50_full_b2_vhr_50_s7_epoch%d.pkl' % epoch )

torch.save(net, './checkpoint/adam_e4_iou45_pre_fpn50_full_b2_vhr_50_s7_final.pkl' )   
import csv

with open('/home/peng/Dropbox/retina/vhr/train_loss_full_adam_iou45_b2_fpn50_s7.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(train_loss_list)    
    
    
with open('/home/peng/Dropbox/retina/vhr/test_loss_full_adam_iou45_b2_fpn50_s7.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(test_loss_list)  
'''
