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
from loss import BCFocalLoss

cfig = Config()
net = xceptionAx3(num_classes=1)  #create CNN model.
criterion = nn.BCEWithLogitsLoss()  #define the los

optimizer = optim.SGD(
    net.parameters(), lr=0.0001, momentum=0.9,weight_decay=0.0001)  #select the optimizer

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
# create the train_dataset_loader and val_dataset_loader.

train_tarnsformed_dataset = CloudDataset(
    img_dir='data/images224',
    labels_dir='data/masks224/',
    transform=transforms.Compose([ToTensor()]))

val_tarnsformed_dataset = CloudDataset(
    img_dir='data/images224',
    labels_dir='data/masks224/',val=True,
    transform=transforms.Compose([ToTensor()]))

train_dataloader = DataLoader(
    train_tarnsformed_dataset, batch_size=8, shuffle=True, num_workers=4)

val_dataloader = DataLoader(
    val_tarnsformed_dataset, batch_size=8, shuffle=True, num_workers=4)

trainer = Trainer('training', optimizer,exp_lr_scheduler, net, cfig, './log')
trainer.load_weights(trainer.find_last()) 
trainer.train(train_dataloader, val_dataloader, criterion, 150)
trainer.evaluate(val_dataloader)
print('Finished Training')
