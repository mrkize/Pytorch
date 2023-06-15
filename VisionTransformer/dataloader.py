# -*- coding: utf-8 -*-
# @File : dataloader.py
# @Author: Runist
# @Time : 2021/10/28 10:26
# @Software: PyCharm
# @Brief:
import copy
from random import random

from torch.utils.data import Dataset
from torchvision import transforms, datasets
import os
import glob
import torch
import cv2 as cv
import numpy as np
from PIL import Image

class VITdataset(Dataset):

    def __init__(self, root_dir, spilt='nothing'):
        self.root_dir = root_dir
        self.Transform = transforms.Compose([
                transforms.Resize([32,32]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        # Output of pretransform should be PIL images
        self.data_dir = root_dir
        self.dataset = datasets.ImageFolder(self.data_dir+'train', self.Transform) + datasets.ImageFolder(self.data_dir+'val', self.Transform)
        train_set, val_set = dataset_split(self.dataset)
        if spilt == 'train':
            self.dataset = train_set
        elif spilt == 'val':
            self.dataset = val_set


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # if self.shuffle:
        #     return imgshuffle(self.dataset[idx][0].unsqueeze(0),alpha=0.2), self.dataset[idx][1]
        # else:
        return self.dataset[idx][0], self.dataset[idx][1]



data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}



def model_dataloader(root_dir, c_fn=None):
    train_set = VITdataset(root_dir=root_dir, spilt='train')
    val_set = VITdataset(root_dir=root_dir, spilt='val')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=c_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, collate_fn=c_fn)
    data_loader = {'train': train_loader, 'val': val_loader}
    data_size = {"train": len(train_set), "val": len(val_set)}
    return data_loader, data_size


def dataset_split(data):
    np.random.seed(101)
    idx =list(range(len(data)))
    np.random.shuffle(idx)
    train_set = torch.utils.data.Subset(data, idx[0:int(len(idx)/2)])
    val_set = torch.utils.data.Subset(data, idx[int(len(idx)/2):])
    return train_set, val_set

def para_dataloader(root_dir):
    dataset = VITdataset(root_dir=root_dir)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    return train_loader, len(dataset)

class imgshuffle():
    def __init__(self, patch_ratio, pixel_ratio, shf_dim = -1, seed=101, img_size = 32, patch_size = 8):
        self.ptr = patch_ratio
        self.pxr = pixel_ratio
        self.seed = seed
        np.random.seed(seed)
        self.img_size = img_size
        self.patch_size = patch_size
        self.shf_dim = shf_dim
        self.img2patchs =torch.nn.Unfold((patch_size,patch_size), stride=patch_size)
        self.patches_to_im = torch.nn.Fold(
            output_size=(img_size, img_size),
            kernel_size=(patch_size, patch_size),
            stride=patch_size
        )


    def __call__(self, data):
        # data = data.transpose(1,0)
        imgs = torch.stack([unit[0] for unit in data],dim=0)
        labels = torch.stack([torch.tensor(unit[1]) for unit in data],dim=0)
        if self.shf_dim == 2:
            to_patches = self.img2patchs(imgs)
            to_patches_copy = copy.deepcopy(to_patches)
            all_idx = np.arange(to_patches.shape[2])
            choice_idx = np.random.choice(all_idx, int(to_patches.shape[2] * self.ptr), replace=False)
            shu_choice_idx = np.random.permutation(choice_idx)
            to_patches[:, :, choice_idx] = to_patches_copy[:, :, shu_choice_idx]
            to_images = self.patches_to_im(to_patches)
        elif self.shf_dim == 1:
            to_patches = self.img2patchs(imgs)
            to_patches_copy = copy.deepcopy(to_patches)
            patch_idx = np.random.choice(np.arange(to_patches.shape[2]), int(to_patches.shape[2] * self.ptr), replace=False)
            choice_idx = np.random.choice(np.arange(to_patches.shape[1]), int(to_patches.shape[1] * self.pxr), replace=False)
            shu_choice_idx = np.random.permutation(choice_idx)
            for pi in patch_idx:
                to_patches[:, choice_idx, : pi] = to_patches_copy[:, shu_choice_idx, : pi]
            to_images = self.patches_to_im(to_patches)
        else:
            to_images = imgs
        return (to_images,labels)




