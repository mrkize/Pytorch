import argparse

from mymodel import ViT,ViT_mask,ViT_mask_plus
from dataloader import model_dataloader, VITdataset, imgshuffle
from masking_generator import JigsawPuzzleMaskedRegion
from trainer import mask_train_model, train_VIT, predict
import torch
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
def Privacy_laekage(model, loader):
    return


