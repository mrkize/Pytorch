import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.optim import lr_scheduler
import time


from masking_generator import JigsawPuzzleMaskedRegion
from models.vit_timm import VisionTransformer
from dataloader import model_dataloader
from mymodel import ViT, ViT_mask, ViT_mask_plus, ViT_ape



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,num_epochs=25):
    print("DATASET SIZE", dataset_sizes)
    since = time.time()
    #save the best model

    best_acc = 0
    retunr_value_train = np.zeros((10,num_epochs))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        #print('-' * 10)
        # print("Start training: epoch ",format(epoch))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                scheduler.step()
            # elif epoch%5 != 0 and epoch != num_epochs-1:
            #     continue   #every 5 epoch execute once
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            TP = 0
            FN = 0
            FP = 0
            TN = 0
            # Iterate over data.
            for batch_idx, (data, target) in enumerate(dataloaders[phase]):
                inputs, labels = data.to(device), target.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    labels = torch.squeeze(labels)
                    loss = criterion(outputs, labels)


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics

                running_loss += loss.item() * inputs.size(0)
                running_corrects += preds.eq(labels).sum().item()
                # RTPR = recall = TP/(TP + FN) precision = TP/(TP + FP) FPR = FP/(FP + TN)
                TP += torch.sum(preds & labels.data)
                FN += (torch.sum(preds) - torch.sum(preds & labels.data))
                FP += (torch.sum(labels.data) - torch.sum(preds & labels.data))
                TN += (preds.shape[0] - torch.sum(preds | labels.data))
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = 1.0 * running_corrects / dataset_sizes[phase]
            epoch_precision = TP/(TP + FP)
            epoch_recall = TP / (TP + FN)
            epoch_FPR = FP / (FP + TN)
            if phase == 'train':
                print('train acc:', epoch_acc, end=' ')
                retunr_value_train[0][epoch] = epoch_loss
                retunr_value_train[1][epoch] = epoch_acc
                retunr_value_train[2][epoch] = epoch_precision.item()
                retunr_value_train[3][epoch] = epoch_recall.item()
                retunr_value_train[4][epoch] = epoch_FPR.item()
            else:
                print('val acc:', epoch_acc)
                retunr_value_train[5][epoch] = epoch_loss
                retunr_value_train[6][epoch] = epoch_acc
                retunr_value_train[7][epoch] = epoch_precision.item()
                retunr_value_train[8][epoch] = epoch_recall.item()
                retunr_value_train[9][epoch] = epoch_FPR.item()



            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #    phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase != 'train' and epoch_acc > best_acc:
                best_acc = epoch_acc

        #print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("DONE TRAIN")

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model, retunr_value_train, best_acc

def predict(model, dataloaders, dataset_sizes, jigsaw=None):
    model.eval()  # Set model to evaluate mode
    running_corrects = 0
    # Iterate over data.
    for batch_idx, (data, target) in enumerate(dataloaders):
        inputs, labels = data.to(device), target.to(device)
        if jigsaw is not None:
            inputs, unk_mask = jigsaw(inputs)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            labels = torch.squeeze(labels)
        running_corrects += preds.eq(labels).sum().item()
    acc = 1.0 * running_corrects / dataset_sizes
    return acc


def train_VIT(model_type, data_loader, data_size, config, PE):
    if 'ape' in model_type:
        model_vit = ViT_ape.creat_VIT(PE).to(device)
    else:
        model_vit = ViT.creat_VIT(PE).to(device)
    # model_vit = ViT.load_VIT('./Network/VIT_Model_cifar10/VIT_NoPE.pth').to(device)
    # model_vit.PE = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_vit.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=config.learning.learning_rate, weight_decay=0.01)
    # learning rate adopt
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_every,gamma=config.learning.decrease_lr_factor)
    # training target model
    model, ret_para, best_acc = train_model(model_vit, criterion, optimizer, exp_lr_scheduler,data_loader,data_size, num_epochs=config.learning.epochs)
    torch.save(model.state_dict(), './Network/VIT_Model_cifar10/VIT_'+model_type+'.pth')
    np.save('./results/VIT_cifar10/VIT_'+model_type+'.pth', ret_para)
    # if not os.path.exists(config.path.model_path + 'VIT_Model/'):
    #     os.makedirs(config.path.model_path + 'VIT_Model/')
    # if PE==True:
    #     torch.save(model.state_dict(), config.path.model_path + 'VIT_Model/VIT_PE.pth')
    #     np.save(config.path.result_path + "VIT_pos.npy", ret_para)
    # else:
    #     torch.save(model.state_dict(), config.path.model_path + 'VIT_Model/VIT_NoPE.pth')
    #     np.save(config.path.result_path + "VIT_nopos.npy", ret_para)


    return model, ret_para

def train_para(vit_pos, vit_nopos, para, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    soft = nn.Softmax(1)
    retunr_value_train = np.zeros(num_epochs)
    # dataset = torch.utils.data.ConcatDataset([dataloaders['train'], dataloaders['val']])
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        scheduler.step()
        para.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloaders):
            inputs, labels = data.to(device), target.to(device)
            optimizer.zero_grad()

            x_n = para(inputs)
            outputs_1 = soft(vit_pos(inputs))
            outputs_2 = soft(vit_nopos(x_n))

            # _, preds = torch.max(outputs, 1)
            # labels = torch.squeeze(labels)

            loss = F.cross_entropy(outputs_1, outputs_2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / dataset_sizes
        retunr_value_train[epoch] = epoch_loss
        print('para:\n',para.para)
        # print('real loss:',F.l1_loss(para.para,vit_pos.pos_embedding))
    print("DONE TRAIN")
    return retunr_value_train


def fitting_para(model_root, dataloaders, dataset_sizes, config):
    vit_pos = ViT.load_VIT(model_root + 'VIT_PE.pth')
    vit_nopos = ViT.load_VIT(model_root + 'VIT_NoPE.pth')
    para = ViT.parameter().to(device)
    optimizer = optim.SGD(para.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_every,gamma=config.learning.decrease_lr_factor)
    ret_val = train_para(vit_pos, vit_nopos, para, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=50)
    torch.save(para.state_dict(), model_root+'para.pth')

    print('para:\n',para.para)
    print('PE:\n',vit_pos.pos_embedding)
    return ret_val

def mask_train(model, loader, size, criterion, scheduler, optimizer, mixup_fn, jigsaw_pullzer, config, inp_shf):
    print("DATASET SIZE", size)
    since = time.time()
    #save the best model
    ret_value = np.zeros((4, config.learning.epochs))
    # print(optimizer.state_dict()['param_groups'][0]['lr'])
    #print('-' * 10)
    # print("Start training: epoch ",format(epoch))
    for epoch in range(config.learning.epochs):
        print('Epoch {}/{}'.format(epoch, config.learning.epochs - 1))
    # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                scheduler.step()
            # elif epoch%5 != 0 and epoch != num_epochs-1:
            #     continue   #every 5 epoch execute once
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for batch_idx, (data, target) in enumerate(loader[phase]):
                inputs, labels = data.to(device), target.to(device)
                if mixup_fn is not None:
                    inputs, labels = mixup_fn(inputs, labels)

                unk_mask = None
                if epoch >= config.train.warmup_epoch and torch.rand(1) > config.train.jigsaw:
                    inp, unk_mask = jigsaw_pullzer(inputs)
                    unk_mask = torch.from_numpy(unk_mask).long().to(device)
                    inputs = inp if inp_shf else inputs
                # if phase == 'val':
                #     if torch.rand(1) > config.train.jigsaw:
                #         inputs, _ = jigsaw_pullzer(inputs)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, unk_mask=unk_mask)
                    _, preds = torch.max(outputs, 1)
                    # labels = torch.squeeze(labels)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics

                running_loss += loss.item() * inputs.size(0)
                running_corrects += preds.eq(target.to(device)).sum().item()
            epoch_loss = running_loss / size[phase]
            epoch_acc = 1.0 * running_corrects / size[phase]
            if phase == 'train':
                print('train acc:{:.3f}'.format(epoch_acc), end=' ')
                ret_value[0][epoch] = epoch_loss
                ret_value[1][epoch] = epoch_acc

            else:
                print('val acc:{:.3f}'.format(epoch_acc))
                ret_value[2][epoch] = epoch_loss
                ret_value[3][epoch] = epoch_acc

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("DONE TRAIN")

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model, ret_value

def mask_train_model(model_type, config, data_loader, data_size, if_mixup=False, PEratio=0.5):
    inp_shf = False
    if 'plus' in model_type:
        model = ViT_mask_plus.creat_VIT().to(device)
        inp_shf =False
    elif 'mask' in model_type:
        model = ViT_mask.creat_VIT().to(device)
        inp_shf = True
    else:
        model = ViT.creat_VIT().to(device)
    mixup_fn = None
    if if_mixup:
        mixup_fn = Mixup(
            mixup_alpha=0.8,num_classes=10)
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()


    jigsaw_pullzer = JigsawPuzzleMaskedRegion(img_size=config.patch.img_size,
                                              patch_size=config.patch.patch_size,
                                              num_masking_patches=int(PEratio*config.patch.num_patches),
                                              min_num_patches=config.train.min_num_patches)
    optimizer = optim.SGD(model.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=config.learning.learning_rate, weight_decay=0.01)
    # learning rate adopt
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_every,gamma=config.learning.decrease_lr_factor)
    model, ret = mask_train(model, data_loader, data_size, criterion, exp_lr_scheduler, optimizer, mixup_fn, jigsaw_pullzer, config, inp_shf)
    torch.save(model.state_dict(), './Network/VIT_Model_cifar10/VIT_'+model_type+'.pth')
    np.save('./results/VIT_cifar10/VIT_'+model_type+'.pth', ret)
    return
