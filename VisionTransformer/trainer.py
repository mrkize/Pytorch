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
from mymodel import ViT, ViT_mask, ViT_ape, ViT_mask_avg, Swin, Swin_mask_avg, Swin_mask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,config):
    print("DATASET SIZE", dataset_sizes)
    since = time.time()
    #save the best model
    num_epochs = config.learning.epochs
    best_acc = 0
    retunr_value_train = np.zeros((4,num_epochs))

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
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = 1.0 * running_corrects / dataset_sizes[phase]
            if phase == 'train':
                print('train acc:', epoch_acc, end=' ')
                retunr_value_train[0][epoch] = epoch_loss
                retunr_value_train[1][epoch] = epoch_acc
            else:
                print('val acc:', epoch_acc)
                retunr_value_train[2][epoch] = epoch_loss
                retunr_value_train[3][epoch] = epoch_acc
            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #    phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase != 'train' and epoch_acc > best_acc:
                best_acc = epoch_acc
        # if (epoch+1)%10 ==0:
        #     torch.save(model.state_dict(), config.path.model_path + config.general.type + '.pth')
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


def train_VIT(model_type, data_loader, data_size, config):
    config.set_subkey('general', 'type', model_type)
    if 'Swin' in model_type:
        if 'ape' in model_type:
            model_vit = Swin.creat_Swin(config).to(device)
        else:
            model_vit = Swin.creat_Swin(config).to(device)
    else:
        if 'ape' in model_type:
            model_vit = ViT_ape.creat_VIT(config).to(device)
        else:
            model_vit = ViT.creat_VIT(config).to(device)
    # model_vit = ViT.load_VIT('./Network/VIT_Model_cifar10/VIT_NoPE.pth').to(device)
    # model_vit.PE = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_vit.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=config.learning.learning_rate, weight_decay=0.01)
    # learning rate adopt
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_every,gamma=config.learning.decrease_lr_factor)
    # training target model
    model, ret_para, best_acc = train_model(model_vit, criterion, optimizer, exp_lr_scheduler,data_loader,data_size, config)
    torch.save(model.state_dict(), config.path.model_path+model_type+'.pth')
    np.save(config.path.result_path+model_type+'.npy', ret_para)
    # if not os.path.exists(config.path.model_path + 'VIT_Model/'):
    #     os.makedirs(config.path.model_path + 'VIT_Model/')
    # if PE==True:
    #     torch.save(model.state_dict(), config.path.model_path + 'VIT_Model/VIT_PE.pth')
    #     np.save(config.path.result_path + "VIT_pos.npy", ret_para)
    # else:
    #     torch.save(model.state_dict(), config.path.model_path + 'VIT_Model/VIT_NoPE.pth')
    #     np.save(config.path.result_path + "VIT_nopos.npy", ret_para)
    return model, ret_para


def mask_train(model, loader, size, criterion, scheduler, optimizer, mixup_fn, jigsaw_pullzer, config):
    print("DATASET SIZE", size)
    since = time.time()
    #save the best model
    ret_value = np.zeros((4, config.learning.epochs))
    # print(optimizer.state_dict()['param_groups'][0]['lr'])
    #print('-' * 10)
    # print(config.learning.epochs)

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
                if phase == 'train:':
                    if epoch >= config.mask.warmup_epoch and torch.rand(1) > config.mask.jigsaw:
                        inputs, unk_mask = jigsaw_pullzer(inputs)
                        unk_mask = torch.from_numpy(unk_mask).long().to(device)

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
        # if (epoch+1)%10 ==0:
        #     torch.save(model.state_dict(), config.path.model_path + config.general.type + str(mask_ratio)   + '.pth')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("DONE TRAIN")

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model, ret_value

def mask_train_model(model_type, config, data_loader, data_size, if_mixup=False, mask_ratio=0.5, mt='mjp'):
    config.set_subkey('general', 'type', model_type)
    if 'Swin' in model_type:
        if 'avg' in model_type:
            model = Swin_mask_avg.creat_Swin(config).to(device)
        else:
            model = Swin_mask.creat_Swin(config).to(device)
    else:
        if 'avg' in model_type:
            model = ViT_mask_avg.creat_VIT(config).to(device)
        else:
            model = ViT_mask.creat_VIT().to(device)
    mixup_fn = None
    if if_mixup:
        mixup_fn = Mixup(
            mixup_alpha=0.8,num_classes=10)
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()


    jigsaw_pullzer = JigsawPuzzleMaskedRegion(img_size=config.patch.image_size,
                                              patch_size=config.patch.patch_size,
                                              num_masking_patches=int(mask_ratio*config.patch.num_patches),
                                              min_num_patches=config.mask.min_num_patches,
                                              mask_type = mt)
    optimizer = optim.SGD(model.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=config.learning.learning_rate, weight_decay=0.01)
    # learning rate adopt
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_every,gamma=config.learning.decrease_lr_factor)
    model, ret = mask_train(model, data_loader, data_size, criterion, exp_lr_scheduler, optimizer, mixup_fn, jigsaw_pullzer, config)
    torch.save(model.state_dict(), config.path.model_path + model_type + str(mask_ratio) + '.pth')
    np.save(config.path.model_path +model_type + str(mask_ratio)  + '.pth', ret)
    return
