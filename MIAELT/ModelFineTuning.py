#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data
from torch.utils.data import SubsetRandomSampler
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets, models
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import random
import numpy as np
import torchnet as tnt
import torch.utils.data as data
from PIL import Image

from trainer import *

plt.ion()   # interactive mode


def getindlist(num, size):
    indices = []
    for i in range(int(size/64)):
        startind = 64*i
        endind = 64*(i) + num
        indices += list(range(startind, endind))
    return indices


def resnetfinetune(data_path, transfer_path, config, args, moedel=None):
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    use_cuda = config.general.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    encoder_path = transfer_path + "Pre_train.pth"

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'val': transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
    }


    data_train = datasets.ImageFolder(os.path.join(data_path, 'val'), data_transforms['train'])
    data_test = datasets.ImageFolder(os.path.join(data_path, 'val'), data_transforms['val'])
    image_datasets = {'train': data_train ,'val': data_test}
    assert image_datasets

    dataloader_sampler = {x: SubsetRandomSampler(getindlist(args.trainSamples, len(image_datasets[x]))) for x in
                          ['train', 'val']}
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config.learning.batch_size, pin_memory=True, shuffle=False,
                                       sampler=dataloader_sampler[x]) for x in
        ['train', 'val']}
    dataset_sizes = {x: len(dataloader_sampler[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes


    model_ft = torch.load(encoder_path)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 4)
    # model_ft.load(encoder_path)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)
    # Finetuning
    for param in model_ft.parameters():
        param.requires_grad = False
    for param in model_ft.layer4.parameters():
        param.requires_grad = True
    for param in model_ft.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    params_tofit = filter(lambda p: p.requires_grad, model_ft.parameters())
    optimizer = optim.SGD(params_tofit, lr=config.learning.learning_rate, momentum=config.learning.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=config.learning.learning_rate, weight_decay=0.01)
    # learning rate adopt
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_every,
                                           gamma=config.learning.decrease_lr_factor)
    # training target model
    model, ret_para, best_acc = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, dataloaders,
                                                          dataset_sizes,
                                                          num_epochs=config.learning.epochs)
    print("The best accuracy of model is: {}".format(best_acc))
    print("The accuracy of model(test) is: {}".format(ret_para[6][config.learning.epochs - 1]))
    print("The accuracy of model(train) is: {}".format(ret_para[1][config.learning.epochs - 1]))
    np.save(transfer_path + "Finetuning_res.npy", ret_para)
    torch.save(model, transfer_path + "Finetuning_model.pth")

    return model


def evaluate(data, data_path, args, config):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'val': transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
    }


    data_train = datasets.ImageFolder(os.path.join(data_path, data), data_transforms['train'])
    data_test = datasets.ImageFolder(os.path.join(data_path, data), data_transforms['val'])
    image_datasets = {'train': data_train ,'val': data_test}
    dataset_sizes = {'train': len(data_train), 'val': len(data_test)}
    dataloader_sampler = {x: SubsetRandomSampler(getindlist(args.trainSamples, len(image_datasets[x]))) for x in
                          ['train', 'val']}

    # dataloaders = {
    #     x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config.learning.batch_size, pin_memory=True, shuffle=True,) for x in ['train', 'val']}

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config.learning.batch_size, pin_memory=True, shuffle=False,
                                       sampler=dataloader_sampler[x]) for x in
        ['train', 'val']}
    # dataset_sizes = {x: len(dataloader_sampler[x]) for x in ['train', 'val']}

    model_val = torch.load('./TransferLearning/Finetuning_model.pth').to(device)
    criterion = nn.CrossEntropyLoss()
    retunr_value, best_acc = eval(model_val, criterion, dataloaders, dataset_sizes)
    np.save('./TransferLearning/'+data+'.npy', retunr_value)



