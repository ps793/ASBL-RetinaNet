'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from encoder import DataEncoder
from transform import resize, random_flip, random_flop, random_crop, center_crop


'''
list_file='/mnt/ssd_disk/naka247/peng/DOTA/val_retina_dota.txt'
train=False
transform=transform
input_size=1024
idx=5
batch=
'''


class ListDataset(data.Dataset):
    def __init__(self, list_file, train, transform):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        #self.root = root
        self.train = train
        self.transform = transform
        #self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder()

        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split(',')
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                xmin = splited[1+5*i]
                ymin = splited[2+5*i]
                xmax = splited[3+5*i]
                ymax = splited[4+5*i]
                c = splited[5+5*i]
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        fname = fname.split("'")[1]
        img = Image.open(fname)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        #size = self.input_size

        # Data augmentation.
        if self.train:
            img, boxes = random_flip(img, boxes)
            #img, boxes = random_flop(img, boxes)
            #img, boxes = random_crop(img, boxes)
            #img, boxes = resize(img, boxes, (size,size))
        #else:
            #img, boxes = resize(img, boxes, (size,size))
            #img, boxes = center_crop(img, boxes, (size,size))

        img = self.transform(img)
        return img, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        
        num_imgs = len(imgs)
        #inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        inputs=[]
        for i in range(num_imgs):
            inputs.append(imgs[i])
            c,w,h=inputs[i].shape
            #print(w,h,c)
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        #print(cls_targets)
            #print(cls_target.size())
        return inputs, loc_targets, cls_targets

    def __len__(self):
        return self.num_samples


def test():
    import torchvision

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    trainset = ListDataset(list_file='/mnt/ssd_disk/naka247/peng/NWPU VHR-10 dataset/train_retina_vhr.txt' , train=True, transform=transform)###may follow other papers
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=8, collate_fn=trainset.collate_fn)

    for images, loc_targets, cls_targets in trainloader:
        #print(images.size())
        print(loc_targets.size())
        print(cls_targets.size())
        grid = torchvision.utils.make_grid(images, 1)
        torchvision.utils.save_image(grid, 'a.jpg')
        break

'''    
torch.cat(loc_targets,dim=0)
h=4
w=4
num_imgs=2
inputs = torch.zeros(num_imgs, 3, h, w)
'''
# test()
