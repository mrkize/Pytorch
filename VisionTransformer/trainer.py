import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from dataloader import *
from model import creat_VIT, load_VIT, parameter



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

def predict(model, dataloaders, dataset_sizes):
    retunr_value_train = np.zeros((10))
    model.eval()  # Set model to evaluate mode
    running_corrects = 0
    # TP = 0
    # FN = 0
    # FP = 0
    # TN = 0
    # Iterate over data.
    for batch_idx, (data, target) in enumerate(dataloaders):
        inputs, labels = data.to(device), target.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            labels = torch.squeeze(labels)
        running_corrects += preds.eq(labels).sum().item()
        # RTPR = recall = TP/(TP + FN) precision = TP/(TP + FP) FPR = FP/(FP + TN)
        # TP += torch.sum(preds & labels.data)
        # FN += (torch.sum(preds) - torch.sum(preds & labels.data))
        # FP += (torch.sum(labels.data) - torch.sum(preds & labels.data))
        # TN += (preds.shape[0] - torch.sum(preds | labels.data))
    acc = 1.0 * running_corrects / dataset_sizes
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # FPR = FP / (FP + TN)
    retunr_value_train[6] = acc
    # retunr_value_train[7] = epoch_precision.item()
    # retunr_value_train[8] = epoch_recall.item()
    # retunr_value_train[9] = epoch_FPR.item()
    return acc


def train_VIT(data_loader, data_size, config, PE):
    model_vit = creat_VIT(PE).to(device)
    model_vit = load_VIT('./Network/VIT_Model_cifar10/VIT_NoPE.pth').to(device)
    model_vit.PE = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_vit.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=config.learning.learning_rate, weight_decay=0.01)
    # learning rate adopt
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_every,gamma=config.learning.decrease_lr_factor)
    # training target model
    model, ret_para, best_acc = train_model(model_vit, criterion, optimizer, exp_lr_scheduler,data_loader,data_size, num_epochs=config.learning.epochs)
    print("The best accuracy of model is: {}".format(best_acc))
    print("The accuracy of model(test) is: {}".format(ret_para[6][config.learning.epochs - 1]))
    print("The accuracy of model(train) is: {}".format(ret_para[1][config.learning.epochs - 1]))

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
    retunr_value_train = np.zeros((num_epochs))
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
    vit_pos = load_VIT(model_root + 'VIT_PE.pth')
    vit_nopos = load_VIT(model_root + 'VIT_NoPE.pth')
    para = parameter().to(device)
    optimizer = optim.SGD(para.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_every,gamma=config.learning.decrease_lr_factor)
    ret_val = train_para(vit_pos, vit_nopos, para, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=50)
    torch.save(para.state_dict(), model_root+'para.pth')

    print('para:\n',para.para)
    print('PE:\n',vit_pos.pos_embedding)
    return ret_val
