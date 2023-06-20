import argparse
from mymodel import ViT,ViT_mask
from dataloader import model_dataloader, VITdataset, imgshuffle
from masking_generator import JigsawPuzzleMaskedRegion
from trainer import mask_train_model, train_VIT, predict
import torch
from utils import config
from Testtool import test_mask_model_imgshuff

parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--no_PE', action="store_true", help='whether use PE')
opt = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.random.manual_seed(1001)
config = config()
root_dir = '../data/cifar-10/'
model_root = './Network/VIT_Model/'
# pos_emb_shuffle_test(root_dir)
mask_train_model(root_dir, config, if_mixup=False)

#训练一个shadow model，它的训练集是原始target model的验证集

# acc1, acc2 = test_mask_model_imgshuff(config, root_dir)




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