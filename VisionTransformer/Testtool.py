import argparse

import numpy as np

from mymodel import ViT,ViT_mask,ViT_mask_plus
from dataloader import model_dataloader, VITdataset, imgshuffle
from masking_generator import JigsawPuzzleMaskedRegion
from trainer import mask_train_model, train_VIT, predict
import torch
import torch.nn.functional as F
from utils import config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#预测对比
def predict_cmp(loader, size, model_dir):
    torch.random.manual_seed(1001)
    model = ViT.load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth').to(device)
    acc1 = predict(model, loader, size)
    for type in ['mask', 'mask_plus']:
        if type == 'mask':
            model = ViT_mask.load_VIT(model_dir+ type+'.pth').to(device)
        else:
            model = ViT_mask_plus.load_VIT(model_dir+ type+'.pth').to(device)
        acc = predict(model, loader, size)
        print("model acc({}):{:.3f}".format(type,acc))
    return


# 使用MJP方法来评估模型的连续性
def test_mask_model(config, loader, size, model_dir):
    torch.random.manual_seed(1001)
    jigsaw_pullzer = JigsawPuzzleMaskedRegion(img_size=config.patch.img_size,
                                              patch_size=config.patch.patch_size
                                              )
    model = ViT.load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth').to(device)
    acc1 = predict(model, loader, size, jigsaw_pullzer)
    print("model acc(unmasked model):{:.3f}".format(acc1))
    for type in ['mask', 'mask_plus']:
        if type == 'mask':
            model = ViT_mask.load_VIT(model_dir+ type+'.pth').to(device)
        else:
            model = ViT_mask_plus.load_VIT(model_dir+ type+'.pth').to(device)
        acc = predict(model, loader, size, jigsaw_pullzer)
        print("model acc({}):{:.3f}".format(type,acc))
    return acc1


# 使用随机打乱图片的方法来评估模型的连续性
def test_mask_model_imgshuff(config, root_dir, model_dir, val='all'):
    torch.random.manual_seed(1001)
    imgshf = imgshuffle(patch_ratio=0.5, pixel_ratio=0.2, shf_dim=2)
    if val == 'all':
        dataset = VITdataset(root_dir=root_dir)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=imgshf)
        size = len(dataset)
    else:
        loader, size = model_dataloader(root_dir=root_dir, c_fn=imgshf)
        loader, size = loader['val'], size['val']
    model = ViT.load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth').to(device)
    acc1 = predict(model, loader, size)
    print("model acc(unmasked):{:.3f}".format(acc1))
    for type in ['mask', 'mask_plus']:
        if type == 'mask':
            model = ViT_mask.load_VIT(model_dir+ type+'.pth').to(device)
        else:
            model = ViT_mask_plus.load_VIT(model_dir+ type+'.pth').to(device)
        acc = predict(model, loader, size)
        print("model acc({}):{:.3f}".format(type,acc))
    return

# 评估模型位置编码的隐私泄露问题，使用
def Privacy_laekage(config, loader, size):
    model1 = ViT.load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth').to(device)
    model2 = ViT_mask_plus.load_VIT('./Network/VIT_Model_cifar10/VIT_mask_plus.pth').to(device)
    jigsaw = JigsawPuzzleMaskedRegion(img_size=config.patch.img_size,
                                              patch_size=config.patch.patch_size
                                              )
    for phase in ['train', 'val']:
        sum=0
        for batch_idx, (data, target) in enumerate(loader[phase]):
            inputs, labels = data.to(device), target.to(device)
            inputs_mask, _ = jigsaw(inputs)
            out = F.softmax(model1(inputs),dim=1)
            out_mask = F.softmax(model1(inputs_mask),dim=1)
            cross1 = torch.nn.functional.cross_entropy(out,out_mask)
            out = F.softmax(model2(inputs),dim=1)
            out_mask = F.softmax(model2(inputs_mask),dim=1)
            cross2 = torch.nn.functional.cross_entropy(out,out_mask)
            if cross1>cross2:
                sum+=1
        print('{} cross:{}'.format(phase,sum/(batch_idx+1)))
    return


def quick_sort():
    return






def shf_attack():
    return
