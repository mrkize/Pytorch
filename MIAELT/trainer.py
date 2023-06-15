import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from dataloaders import *
from model import *



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, step=['train', 'val'],num_epochs=25):
    print("DATASET SIZE", dataset_sizes)
    since = time.time()
    #save the best model

    best_acc = 0
    retunr_value_train = np.zeros((10,num_epochs))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)
        # print("Start training: epoch ",format(epoch))

        # Each epoch has a training and validation phase
        for phase in step:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
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
                running_corrects += torch.sum(preds == labels.data)
                # RTPR = recall = TP/(TP + FN) precision = TP/(TP + FP) FPR = FP/(FP + TN)
                TP += torch.sum(preds & labels.data)
                FN += (torch.sum(preds) - torch.sum(preds & labels.data))
                FP += (torch.sum(labels.data) - torch.sum(preds & labels.data))
                TN += (preds.shape[0] - torch.sum(preds | labels.data))
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_precision = TP/(TP + FP)
            epoch_recall = TP / (TP + FN)
            epoch_FPR = FP / (FP + TN)
            if phase == 'train':
                retunr_value_train[0][epoch] = epoch_loss
                retunr_value_train[1][epoch] = epoch_acc
                retunr_value_train[2][epoch] = epoch_precision.item()
                retunr_value_train[3][epoch] = epoch_recall.item()
                retunr_value_train[4][epoch] = epoch_FPR.item()
            else:
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

def eval(model, criterion, dataloaders, dataset_sizes, step=['train', 'val']):
    best_acc = 0
    retunr_value_train = np.zeros((10))
    for phase in step:
        model.eval()  # Set model to evaluate mode
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

            # forward
            # track history if only in train

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            labels = torch.squeeze(labels)
            loss = criterion(outputs, labels)

                # backward + optimize only if in training phase


            # statistics

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            # RTPR = recall = TP/(TP + FN) precision = TP/(TP + FP) FPR = FP/(FP + TN)
            TP += torch.sum(preds & labels.data)
            FN += (torch.sum(preds) - torch.sum(preds & labels.data))
            FP += (torch.sum(labels.data) - torch.sum(preds & labels.data))
            TN += (preds.shape[0] - torch.sum(preds | labels.data))
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        epoch_precision = TP / (TP + FP)
        epoch_recall = TP / (TP + FN)
        epoch_FPR = FP / (FP + TN)
        if phase == 'train':
            retunr_value_train[0] = epoch_loss
            retunr_value_train[1] = epoch_acc
            retunr_value_train[2] = epoch_precision.item()
            retunr_value_train[3] = epoch_recall.item()
            retunr_value_train[4] = epoch_FPR.item()
        else:
            retunr_value_train[5] = epoch_loss
            retunr_value_train[6] = epoch_acc
            retunr_value_train[7] = epoch_precision.item()
            retunr_value_train[8] = epoch_recall.item()
            retunr_value_train[9] = epoch_FPR.item()

        # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        #    phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase != 'train' and epoch_acc > best_acc:
            best_acc = epoch_acc
    return retunr_value_train, best_acc


def get_model(model_type, data_type, istarget, not_train, config, model_path, res_path, num_train = 1):
    # train or val
    step = ['train', 'val']

    model_name = 'target' if istarget else 'shadow'
    if data_type == 'cifar10':
        data_train = custum_CIFAR10(model_name, train=True, config = config).dataset
        data_test = custum_CIFAR10(model_name, train=False, config = config).dataset
        # data_train = CIFAR10("./data", train=True, download=True,
        #            transform=transforms.Compose([
        #                transforms.ToTensor(),
        #                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #            ]))
        # data_test = CIFAR10("./data", train=False, download=True,
        #            transform=transforms.Compose([
        #                transforms.ToTensor(),
        #                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #            ]))
        num_classes = 10
    elif data_type == 'cifar100':
        data_train = custum_CIFAR100(model_name, train=True, config = config).dataset
        data_test = custum_CIFAR100(model_name, train=False, config = config).dataset
        num_classes = 100
    elif data_type == 'cinic10':
        data_train = custum_CINIC10(model_name, train=True, config = config).dataset
        data_test = custum_CINIC10(model_name, train=False, config = config).dataset
        num_classes = 10
    elif data_type == 'rot_data':
        data_train = RotationDataset('../data/cinic-10/train')
        data_test = RotationDataset('../data/cinic-10/test')
        num_classes = 4
        model_type = 'res34'
        model_name = 'Pre_train'
        # step = ['train']

    # loss function
    criterion = nn.CrossEntropyLoss()
    # use dataloader
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=config.learning.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=config.learning.batch_size, shuffle=True)
    # dataloaders information data and datasize
    if not_train:
        model = torch.load(model_path + model_name + ".pth")
    else:
        print("Start training {} model".format(model_name))
        dataloaders = {"train": train_loader, "val": test_loader}
        dataset_sizes = {"train": len(data_train), "val": len(data_test)}
        # create model
        if model_type == 'vgg':
            model = models.vgg16(pretrained=False).to(device)
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
            # model = VGG16(num_classes).to(device)
        elif model_type == 'resnet':
            model = models.resnet18(pretrained=False).to(device)
            # model = resnet50(num_classes).to(device)
        elif model_type == 'res34':
            # Model Initialization
            model = models.resnet34(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 4)
            model = model.to(device)
        # optimizer
        optimizer = optim.SGD(model.parameters(), lr=config.learning.learning_rate, momentum=config.learning.momentum)
        # optimizer = optim.Adam(model.parameters(), lr=config.learning.learning_rate, weight_decay=0.01)
        # learning rate adopt
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_every,gamma=config.learning.decrease_lr_factor)
        # training target model
        model, ret_para, best_acc = train_model(model, criterion, optimizer, exp_lr_scheduler,dataloaders,dataset_sizes, step,
                           num_epochs=config.learning.epochs)

        torch.save(model, model_path + model_name + ".pth")
        print("The best accuracy of model is: {}".format(best_acc))
        print("The accuracy of model(test) is: {}".format(ret_para[6][config.learning.epochs-1]))
        print("The accuracy of model(train) is: {}".format(ret_para[1][config.learning.epochs-1]))
        np.save(res_path + "res_" + model_name +".npy", ret_para)

    return model, train_loader, test_loader


