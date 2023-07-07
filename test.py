import numpy as np
import torch
import torchvision
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# def train_para(vit_pos, vit_nopos, para, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
#     soft = nn.Softmax(1)
#     retunr_value_train = np.zeros(num_epochs)
#     # dataset = torch.utils.data.ConcatDataset([dataloaders['train'], dataloaders['val']])
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         scheduler.step()
#         para.train()
#         running_loss = 0.0
#         for batch_idx, (data, target) in enumerate(dataloaders):
#             inputs, labels = data.to(device), target.to(device)
#             optimizer.zero_grad()
#
#             x_n = para(inputs)
#             outputs_1 = soft(vit_pos(inputs))
#             outputs_2 = soft(vit_nopos(x_n))
#
#             # _, preds = torch.max(outputs, 1)
#             # labels = torch.squeeze(labels)
#
#             loss = F.cross_entropy(outputs_1, outputs_2)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)
#         epoch_loss = running_loss / dataset_sizes
#         retunr_value_train[epoch] = epoch_loss
#         print('para:\n',para.para)
#         # print('real loss:',F.l1_loss(para.para,vit_pos.pos_embedding))
#     print("DONE TRAIN")
#     return retunr_value_train
#
#
# def fitting_para(model_root, dataloaders, dataset_sizes, config):
#     vit_pos = ViT.load_VIT(model_root + 'VIT_PE.pth')
#     vit_nopos = ViT.load_VIT(model_root + 'VIT_NoPE.pth')
#     para = ViT.parameter().to(device)
#     optimizer = optim.SGD(para.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_every,gamma=config.learning.decrease_lr_factor)
#     ret_val = train_para(vit_pos, vit_nopos, para, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=50)
#     torch.save(para.state_dict(), model_root+'para.pth')
#
#     print('para:\n',para.para)
#     print('PE:\n',vit_pos.pos_embedding)
#     return ret_val

torchvision.datasets.STL10(root='./data',
                           split='train',
                           download=True
                           )