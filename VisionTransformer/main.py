import argparse
from mymodel import ViT,ViT_mask
from dataloader import cifar_dataloader, VITdataset, imgshuffle, ImageNet_loader
from masking_generator import JigsawPuzzleMaskedRegion
from trainer import mask_train_model, train_VIT, predict
import torch
from utils import MyConfig
# from Testtool import test_mask_model_imgshuff, test_mask_model, predict_cmp, Privacy_laekage

parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--ordinary_train', action="store_true", help='whether use mask')
parser.add_argument('--epochs', type=int, default=1, help='training epoch')
parser.add_argument('--config', type=str, default='IN100Swin', help='dataset')
parser.add_argument('--model_type', type=str, default='Swin_mask_avg', help='model name')
parser.add_argument('--mask_type', type=str, default='pub_fill', help='if fill')
parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio')
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config_dict ={
    'cifar10': "config/cifar10/",
    'cifar100': "config/cifar100/",
    'ImageNet100': "config/ViT-T/",
    'IN100Swin': "config/Swin-T/"
}

torch.random.manual_seed(1001)
config_path = config_dict[opt.config]
config = MyConfig.MyConfig(path=config_path)
config.set_subkey('learning', 'epochs', opt.epochs)
def ordinary_train(model_type):
    if 'cifar' in config_path:
        data_loader, data_size = cifar_dataloader(config=config, is_target=True)
    else:
        data_loader, data_size = ImageNet_loader(config=config, is_target=True)
    train_VIT(model_type, data_loader, data_size, config)
    if 'cifar' in config_path:
        data_loader, data_size = cifar_dataloader(config=config, is_target=False)
    else:
        data_loader, data_size = ImageNet_loader(config=config, is_target=False)
    train_VIT(model_type, data_loader, data_size, config)

def mask_train(model_type):
    if 'cifar' in config_path:
        data_loader, data_size = cifar_dataloader(config=config, is_target=True)
    else:
        data_loader, data_size = ImageNet_loader(config=config, is_target=True)

    mask_train_model(model_type, config, data_loader, data_size, if_mixup=False, mask_ratio=opt.mask_ratio, mt=opt.mask_type)

    if 'cifar' in config_path:
        data_loader, data_size = cifar_dataloader(config=config, is_target=False)
    else:
        data_loader, data_size = ImageNet_loader(config=config, is_target=False)

    mask_train_model(model_type, config, data_loader, data_size, if_mixup=False, mask_ratio=opt.mask_ratio, mt=opt.mask_type)


if opt.ordinary_train:
    ordinary_train(opt.model_type)
else:
    mask_train(opt.model_type)







# loader = {'train': data_loader['val'], 'val': data_loader['train']}
# size = {'train': data_size['val'], 'val': data_size['train']}
# train_VIT('ape_shadow', loader, size, config, PE=True)
# mask_train_model('mask_avg_fill_shadow', config, loader, size, if_mixup=False, PEratio=0.5, mt='pub_fill')
# predict_cmp(data_loader['val'], data_size['val'], model_dir)

#训练一个shadow model，它的训练集是原始target model的验证集
# if opt.val == 'all':
#     dataset = VITdataset(root_dir=root_dir)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
#     size = len(dataset)
# else:
#     loader, size = model_dataloader(root_dir=root_dir)
#     loader, size = loader['val'], size['val']
# loader, size = model_dataloader(root_dir=root_dir)
# test_mask_model(config, loader, size, model_dir)
# Privacy_laekage(config, loader, size)
#
# test_mask_model_imgshuff(config, root_dir, model_dir)


# 训练使用PE的target model
# model, ret_para = train_VIT(data_loader, data_size, config, PE=True)
# torch.save(model.state_dict(),'./Network/VIT_Model_cifar10/VIT_PE.pth')
# np.save("./results/VIT_cifar10/VIT_pos_shadow.npy", ret_para)
# 训练不使用PE的target model
# model, ret_para = train_VIT(data_loader, data_size, config, PE=False)
# torch.save(model.state_dict(),'./Network/VIT_Model_cifar10/VIT_NoPE.pth')
# np.save("./results/VIT_cifar10/VIT_nopos_shadow.npy", ret_para)

# 训练shadow model
# loader = {'train': data_loader['val'], 'val': data_loader['train']}
# size = {'train': data_size['val'], 'val': data_size['train']}
# 训练使用PE的shadow model
# model, ret_para = train_VIT(loader, size, config, PE=True)
# torch.save(model.state_dict(),'./Network/VIT_Model_cifar10/VIT_PE_shadow.pth')
# np.save("./results/VIT_cifar10/VIT_pos_shadow.npy", ret_para)

# 训练不使用PE的shadow model
# model, ret_para = train_VIT(loader, size, config, PE=False)
# torch.save(model.state_dict(),'./Network/VIT_Model_cifar10/VIT_NoPE_shadow.pth')
# np.save("./results/VIT_cifar10/VIT_nopos_shadow.npy", ret_para)



# data_loader, data_size = para_dataloader(root_dir=root_dir)
# ret = fitting_para(model_root, data_loader, data_size, config)

# 分析加入位置编码与不加入位置编码时对相同输入的输出异同
# loader, loader_size = para_dataloader(root_dir)
# model_pos = load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth').to(device)
# model_nopos = load_VIT('./Network/VIT_Model_cifar10/VIT_NoPE.pth').to(device)
# model_nopos.PE = False
# model, ret = train_VIT(data_loader, data_size, config, PE=False)
# print(ret)
#
# for batch_idx, (data, target) in enumerate(loader):
#     inputs, labels = data.to(device), target.to(device)
#     a = model_pos(inputs)
#     b = model_nopos(inputs, pos=False)
#     print('ground truth:',target[0])
#     print(a[0])
#     print(b[0])
#
#     print('------------------------------------------------------------------------------------------------')
#     if batch_idx == 3:
#         break



# b = model_pos.pos_emb()
# pos_para = parameter().to(device)
# optimizer = optim.SGD(pos_para.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_every,
#                                        gamma=config.learning.decrease_lr_factor)
#
# ret_para = train_para(model_pos, pos_para, optimizer, exp_lr_scheduler, data_loader['train'], data_size['train'])

# preds = model_vit(shf_input, emb=False)

# print(preds.shape)  # (16, 1000)

#进行成员推理攻击