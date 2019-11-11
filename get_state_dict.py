'''Init RestinaNet50 with pretrained ResNet50 model.

Download pretrained ResNet50 params from:
  https://download.pytorch.org/models/resnet50-19c8e357.pth
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init

from fpn import FPN50
from fpn_vgg import FPN_VGG
from retinanet import RetinaNet


print('Loading pretrained ResNet50 model..')
d = torch.load('./model/vgg16_bn.pth')
new=list(d.items())
print('Loading into FPN50..')
#fpn = FPN50()###
fpn = FPN_VGG()
dd = fpn.state_dict()
for k in d.keys():
    if not k.startswith('fc'):  # skip fc layers
        dd[k] = d[k]


####vgg
count=0
for key,value in dd.items():
    if count >= len(new):
        break
    layer_name,weights=new[count]   
    if not layer_name.startswith('classifier'):
        dd[key]=weights
    count+=1


'''
n=len(dd.keys())
for ii in range(n):
    print (ii,dd.keys()[ii])
    break
n=len(d.keys())   
for i in range(n):
    print (i,d.keys()[i])
    break    
'''
print('Saving RetinaNet..')
net = RetinaNet(num_classes=10)
for m in net.modules():
    if isinstance(m, nn.Conv2d):
        init.normal(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            init.constant(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

pi = 0.01
init.constant(net.cls_head[-1].bias, -math.log((1-pi)/pi))

net.fpn.load_state_dict(dd)
torch.save(net.state_dict(), './model/vhr_vgg16.pth')
print('Done!')

