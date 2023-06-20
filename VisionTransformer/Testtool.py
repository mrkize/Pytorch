import argparse
from mymodel import ViT,ViT_mask
from dataloader import model_dataloader, VITdataset, imgshuffle
from masking_generator import JigsawPuzzleMaskedRegion
from trainer import mask_train_model, train_VIT, predict
import torch
from utils import config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_mask_model(config, root_dir):
    torch.random.manual_seed(1001)
    dataset = VITdataset(root_dir=root_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    jigsaw_pullzer = JigsawPuzzleMaskedRegion(img_size=config.patch.img_size,
                                              patch_size=config.patch.patch_size)
    model = ViT.load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth').to(device)
    acc1 = predict(model, loader, len(dataset))
    acc2 = predict(model, loader, len(dataset), jigsaw_pullzer)
    print("model acc(unmasked):{:.3f}".format(acc1))
    print("model acc(unmasked, shuffled):{:.3f}".format(acc2))
    model = ViT_mask.load_VIT('./Network/VIT_Model_cifar10/VIT_mask.pth').to(device)
    acc1 = predict(model, loader, len(dataset))
    acc2 = predict(model, loader, len(dataset), jigsaw_pullzer)
    print("model acc(unmasked):{:.3f}".format(acc1))
    print("model acc(unmasked, shuffled):{:.3f}".format(acc2))
    return

def test_mask_model_imgshuff(config, root_dir):
    torch.random.manual_seed(1001)
    dataset = VITdataset(root_dir=root_dir)
    imgshf = imgshuffle(patch_ratio=0.5, pixel_ratio=0.2, shf_dim=2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=imgshf)
    model = ViT.load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth').to(device)
    acc1 = predict(model, loader, len(dataset))
    print("model acc(unmasked):{:.3f}".format(acc1))
    model = ViT_mask.load_VIT('./Network/VIT_Model_cifar10/VIT_mask.pth').to(device)
    acc2 = predict(model, loader, len(dataset))
    print("model acc(unmasked):{:.3f}".format(acc2))
    return acc1,acc2