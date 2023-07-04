import argparse

import numpy as np

from mymodel import ViT,ViT_mask,ViT_mask_avg
from dataloader import VITdataset, imgshuffle, cifar_dataloader
from masking_generator import JigsawPuzzleMaskedRegion
from trainer import mask_train_model, train_VIT, predict
import torch
import torch.nn.functional as F
from utils import MyConfig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#预测对比
def predict_cmp(loader, size, model_dir):
    torch.random.manual_seed(1001)
    alpha = 0.2
    model = ViT.load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth').to(device)
    acc1 = predict(model, loader, size)
    print("model acc(unmask):{:.3f}".format(acc1))
    for type in ['mask_0.5', 'mask_plus', 'mask_avg']:
        if type == 'mask_0.5':
            model = ViT_mask.load_VIT(model_dir+ type+'.pth').to(device)
        # elif type == 'mask_plus':
        #     model = ViT_mask_plus.load_VIT(model_dir+ type+'.pth').to(device)
        #     pe_mean = model.pos_embedding.data[:17, ].mean(0)
        #     model.pos_embedding.data = (1 - alpha) * model.pos_embedding.data + alpha * pe_mean
        else:
            model = ViT_mask_avg.load_VIT(model_dir+ type+'.pth').to(device)
            pe_mean = model.pos_embedding.data[:17, ].mean(0)
            model.pos_embedding.data = (1 - alpha) * model.pos_embedding.data + alpha * pe_mean
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
    for type in ['mask_0.5', 'mask_plus', 'mask_avg']:
        if type == 'mask_0.5':
            model = ViT_mask.load_VIT(model_dir+ type+'.pth').to(device)
        # elif type == 'mask_plus':
        #     model = ViT_mask_plus.load_VIT(model_dir+ type+'.pth').to(device)
        else:
            model = ViT_mask_avg.load_VIT(model_dir+ type+'.pth').to(device)
        acc = predict(model, loader, size, jigsaw_pullzer)
        print("model acc({}):{:.3f}".format(type,acc))
    return


# 使用随机打乱图片的方法来评估模型的连续性
def test_mask_model_imgshuff(config, root_dir, model_dir, val='all'):
    torch.random.manual_seed(1001)
    alpha = 0.2
    imgshf = imgshuffle(patch_ratio=0.5, pixel_ratio=0.2, shf_dim=2)
    if val == 'all':
        dataset = VITdataset(root_dir=root_dir)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=imgshf)
        size = len(dataset)
    else:
        loader, size = cifar_dataloader(root_dir=root_dir, c_fn=imgshf)
        loader, size = loader['val'], size['val']
    model = ViT.load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth').to(device)
    acc1 = predict(model, loader, size)
    print("model acc(unmasked):{:.3f}".format(acc1))
    for type in ['mask_0.5', 'mask_plus', 'mask_avg']:
        if type == 'mask_0.5':
            model = ViT_mask.load_VIT(model_dir+ type+'.pth').to(device)
            # pe_mean = model.pos_embedding.data[:17, ].mean(0)
            # model.pos_embedding.data = (1 - alpha) * model.pos_embedding.data + alpha * pe_mean
        # elif type == 'mask_plus':
        #     model = ViT_mask_plus.load_VIT(model_dir+ type+'.pth').to(device)
        #     pe_mean = model.pos_embedding.data[:17, ].mean(0)
        #     model.pos_embedding.data = (1 - alpha) * model.pos_embedding.data + alpha * pe_mean
        else:
            model = ViT_mask_avg.load_VIT(model_dir+ type+'.pth').to(device)
            pe_mean = model.pos_embedding.data[:17, ].mean(0)
            model.pos_embedding.data = (1 - alpha) * model.pos_embedding.data + alpha * pe_mean
        acc = predict(model, loader, size)
        print("model acc({}):{:.3f}".format(type,acc))
    return

# 评估模型位置编码的隐私泄露问题，使用
def Privacy_laekage(config, loader, size):
    model1 = ViT.load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth').to(device)
    model2 = ViT_mask.load_VIT('./Network/VIT_Model_cifar10/VIT_mask.pth').to(device)
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


def pe_mix_up(loader, size, model_dir):
    torch.random.manual_seed(1001)
    model = ViT.load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth').to(device)
    acc1 = predict(model, loader, size)
    print("model acc(unmask):{:.3f}".format(acc1))
    model = ViT_mask.load_VIT(model_dir+'mask_plus.pth').to(device)

    acc = predict(model, loader, size)
    print("model acc({}):{:.3f}".format(type,acc))
    return






def shf_attack():
    return
