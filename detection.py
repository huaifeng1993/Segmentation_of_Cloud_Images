# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by SHEN HUIXIANG  (shhuixi@qq.com)
# Created On: 2018-12-01
# --------------------------------------------------------
from model.dfanet import xceptionAx3
from data import CloudDataset, ToTensor
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import datetime
from torch.optim import lr_scheduler
from config import Config
from train import Trainer
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import sys, time

cfig = Config()
net = xceptionAx3(num_classes=1)  #create CNN model.
criterion = nn.BCELoss()  #define the loss

optimizer = optim.SGD(
    net.parameters(), lr=0.001, momentum=0.9)  #select the optimizer
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
# create the train_dataset_loader and val_dataset_loader.
cloud_data = CloudDataset(
    img_dir='data/images/', labels_dir='data/GTmaps/')

trainer = Trainer('inference', optimizer, exp_lr_scheduler, net, cfig, './log')
trainer.load_weights(trainer.find_last())
#trainer.load_weights('log/renset20190102T1348/model_renset_0046.pt')

since = time.time()
for x in range(0,801,5):
    images = cloud_data[x]['image']
    gt_map = cloud_data[x]['gt_map']
    mask = trainer.detect(images)
    mask =np.round(mask*255)
    # images=cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('result/{}_image.png'.format(x),images)
    # cv2.imwrite('result/{}gt_map.png'.format(x),gt_map)
    #cv2.imwrite('result/{}sigmoid.png'.format(x),mask)

    #fig.set_size_inches(600/100.0,600/100.0)#输出width*height像素
    

    print(mask.shape)
    fig=plt.figure()
    fig.set_size_inches(600/100.0,300/100.0)#输出width*height像素
    plt.subplot(121)
    plt.xticks([])  #去掉横坐标值
    plt.yticks([])  #去掉纵坐标值
    
    plt.imshow(images)
    #plt.show()

    # plt.subplot()
    # plt.xticks([])  #去掉横坐标值
    # plt.yticks([])  #去掉纵坐标值

    # plt.imshow(gt_map)

    plt.subplot(122)
    plt.xticks([])  #去掉横坐标值
    plt.yticks([])  #去掉纵坐标值
    plt.imshow(mask)
    #plt.show()

    
    
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
    plt.margins(0,0)
    plt.savefig('result/{}sigmoid.png'.format(x))
    #input()
time_elapsed = time.time() - since
print('one epoch complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))