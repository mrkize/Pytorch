import copy
from random import random

import cv2
from torch.utils.data import Dataset,Subset,ConcatDataset
from torchvision import transforms, datasets
import os
import glob
import torch
import cv2 as cv
import numpy as np
from PIL import Image

class VITdataset(Dataset):

    def __init__(self, root_dir, split, num_class, nums_per_class,random_seed = 1001, is_target=True):
        self.Transform = transforms.Compose([transforms.Resize([32, 32]),
                                             # transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # Output of pretransform should be PIL images
        self.data_dir = root_dir
        self.dataset = datasets.ImageFolder(self.data_dir+'train', self.Transform) + datasets.ImageFolder(self.data_dir+'val', self.Transform)
        train_set_1, val_set_1 = dataset_split_2(self.dataset, num_class, nums_per_class[0], is_target, random_seed)
        train_set_2, val_set_2 = dataset_split_2(self.dataset, num_class, nums_per_class[1], is_target, random_seed)
        if split == 'train':
            self.dataset = train_set_1 + train_set_2
        elif split == 'val':
            self.dataset = val_set_1 + val_set_2


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # if self.shuffle:
        #     return imgshuffle(self.dataset[idx][0].unsqueeze(0),alpha=0.2), self.dataset[idx][1]
        # else:
        return self.dataset[idx][0], self.dataset[idx][1]



# data_transform = {
#     "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
#     "val": transforms.Compose([transforms.Resize(256),
#                                transforms.CenterCrop(224),
#                                transforms.ToTensor(),
#                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}



def cifar_dataloader(config, c_fn=None, is_target=True):
    seed = config.general.seed
    train_set = VITdataset(root_dir=config.path.data_path,
                           split='train',
                           num_class = config.patch.num_classes,
                           nums_per_class = config.patch.nums_per_class,
                           random_seed = seed,
                           is_target=is_target)

    val_set = VITdataset(root_dir=config.path.data_path,
                           split='val',
                           num_class = config.patch.num_classes,
                           nums_per_class = config.patch.nums_per_class,
                           random_seed = seed,
                           is_target=is_target)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.learning.batch_size, shuffle=True, num_workers=config.general.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.learning.batch_size, shuffle=False, num_workers=config.general.num_workers, pin_memory=True)
    data_loader = {'train': train_loader, 'val': val_loader}
    data_size = {"train": len(train_set), "val": len(val_set)}
    return data_loader, data_size


# def dataset_split(data, random_seed, is_target , split):
def dataset_split(dataset, num_class, nums_per_class, is_target=True, seed=101):
    np.random.seed(seed)
    idx = list(range(nums_per_class))
    np.random.shuffle(idx)
    idx_train = np.array(idx)[:int(2*nums_per_class / 3)] if is_target else np.array(idx)[int(nums_per_class / 3):]
    idx_val = np.array(idx)[int(2*nums_per_class / 3):] if is_target else np.array(idx)[:int(nums_per_class / 3)]
    # idx_train = np.array(idx)[:int(nums_per_class / 2)] if is_target else np.array(idx)[int(nums_per_class / 2):]
    # idx_val = np.array(idx)[int(nums_per_class / 2):] if is_target else np.array(idx)[:int(nums_per_class / 2)]
    index_train = []
    index_val = []
    for i in range(num_class):
        index_train += idx_train.tolist()
        idx_train += nums_per_class
        index_val += idx_val.tolist()
        idx_val += nums_per_class
    train_set = Subset(dataset, index_train)
    val_set = Subset(dataset, index_val)
    return train_set, val_set

def dataset_split_2(dataset, num_class, nums_per_class, is_target=True, seed=101):
    np.random.seed(seed)
    idx = list(range(nums_per_class))
    np.random.shuffle(idx)
    idx_train = np.array(idx)[:int(nums_per_class / 2)] if is_target else np.array(idx)[int(nums_per_class / 2):]
    idx_val = np.array(idx)[int(nums_per_class / 2):] if is_target else np.array(idx)[:int(nums_per_class / 2)]
    index_train = []
    index_val = []
    for i in range(num_class):
        index_train += idx_train.tolist()
        idx_train += nums_per_class
        index_val += idx_val.tolist()
        idx_val += nums_per_class
    train_set = Subset(dataset, index_train)
    val_set = Subset(dataset, index_val)
    return train_set, val_set


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
        return to_images,labels


# class pub_data():
#     def __init__(self, root_dir, img_size=224, patch_size=16):
#         self.root_dir = root_dir
#         self.Transform = transforms.Compose([
#                 transforms.Resize([img_size,img_size]),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#             ])
#         self.dataset = datasets.ImageFolder(self.root_dir+'train', self.Transform) + datasets.ImageFolder(self.root_dir+'val', self.Transform)
#         self.im_to_patches = torch.nn.Unfold((patch_size, patch_size), stride=patch_size)
#         self.idx = list(range(len(self.dataset)))
#
#
#     def __call__(self, length = 100, random_seed = 101):
#         np.random.seed(random_seed)
#         choice_idx = np.random.choice(self.idx, size=length, replace=False)
#         subset = torch.utils.data.Subset(self.dataset, choice_idx)
#         subdata = []
#         for i in range(len(subset)):
#             subdata.append(subset[i][0])
#         data = torch.stack(subdata,dim=0)
#         data = self.im_to_patches(data)
#         return data

def public_data(root_dir, img_size=224, patch_size=16, length = 256, random_seed = 101):
    np.random.seed(random_seed)
    im_to_patches = torch.nn.Unfold((patch_size, patch_size), stride=patch_size)
    Transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = datasets.ImageFolder(root_dir, Transform)
    idx = np.random.choice(np.arange(len(dataset)), size=length, replace=False)
    subset = torch.utils.data.Subset(dataset, idx)
    data = torch.stack([subset[i][0] for i in range(length)], dim=0)
    data = im_to_patches(data)
    return data



class imageNet100(Dataset):

    def __init__(self, root_dir, spilt='nothing', num_class=100, nums_per_class=1000, is_target = True, seed = 101):
        # Output of pretransform should be PIL images
        self.Transform = transforms.Compose([transforms.Resize([224,224]),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.root_dir = root_dir
        if spilt == 'train':
            self.dataset = datasets.ImageFolder(self.root_dir+'train', self.Transform)
            self.dataset = train_data_spilt(self.dataset, num_class, nums_per_class, is_target, seed)
        elif spilt == 'val':
            self.dataset = datasets.ImageFolder(self.root_dir+'val', self.Transform)
        elif spilt == 'test':
            self.dataset = datasets.ImageFolder(self.root_dir+'test', self.Transform)
        else:
            self.dataset = datasets.ImageFolder(self.root_dir+'val', self.Transform) + datasets.ImageFolder(self.root_dir+'test', self.Transform)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # if self.shuffle:
        #     return imgshuffle(self.dataset[idx][0].unsqueeze(0),alpha=0.2), self.dataset[idx][1]
        # else:
        return self.dataset[idx][0], self.dataset[idx][1]


def train_data_spilt(dataset, num_class, nums_per_class, is_target=True, seed=101):
    np.random.seed(seed)
    idx = list(range(nums_per_class))
    np.random.shuffle(idx)
    idx = np.array(idx)[:int(nums_per_class/2)] if is_target else np.array(idx)[int(nums_per_class/2):]
    index = []
    for i in range(num_class):
        index += idx.tolist()
        idx += nums_per_class
    train_set = Subset(dataset, index)
    return train_set


def ImageNet_loader(config, is_target=True):
    train_set = imageNet100(root_dir=config.path.data_path, spilt='train', is_target=is_target, seed=config.general.train_spilt_seed)
    val_set = imageNet100(root_dir=config.path.data_path, spilt='val')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.learning.batch_size, shuffle=True, num_workers=config.general.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.learning.batch_size, shuffle=False, num_workers=config.general.num_workers, pin_memory=True)
    data_loader = {'train': train_loader, 'val': val_loader}
    data_size = {"train": len(train_set), "val": len(val_set)}
    return data_loader, data_size


def ImageNet_MIA_loader(root_dir, config, is_target=True):
    train_set = imageNet100(root_dir=root_dir, spilt='train', is_target=is_target)
    nontrain_set = imageNet100(root_dir=root_dir, spilt='val+test')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.learning.batch_size, shuffle=True, pin_memory=True)
    nontrain_loader = torch.utils.data.DataLoader(nontrain_set, batch_size=config.learning.batch_size, shuffle=False, pin_memory=True)
    data_loader = {'train': train_loader, 'val': nontrain_loader}
    data_size = {"train": len(train_set), "val": len(nontrain_set)}
    return data_loader, data_size