import argparse
import datetime
import os
import time
import torch.utils.data
import torchvision.models

import dataset
import numpy as np
import scipy.io as scio
from model import *
import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def spilt_dataset(data):

    return data


def train(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, step=['train', 'val'],num_epochs=25):
    since = time.time()

    print("Start training")
    best_acc = 0
    retunr_value_train = np.zeros((4,num_epochs))

    for epoch in range(num_epochs):
        if epoch%10 == 0:
            print("Training Epoch:",epoch)
        for phase in step:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (data,label) in enumerate(dataloaders[phase]):
                inputs, labels = data.to(device), label.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    labels = torch.squeeze(labels)
                    loss = criterion(outputs, labels)


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                retunr_value_train[0][epoch] = epoch_loss
                retunr_value_train[1][epoch] = epoch_acc

            else:
                retunr_value_train[2][epoch] = epoch_loss
                retunr_value_train[3][epoch] = epoch_acc
        # if abs(retunr_value_train[1][epoch]-retunr_value_train[1][epoch-1]) < 0.005:
        #     break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("DONE TRAIN")

    return model, retunr_value_train

parser = argparse.ArgumentParser(description='RotLearning')
parser.add_argument("--pretrain", action="store_true", help="preterain a model")
parser.add_argument("--finetune", action="store_true", help="finetune")
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--step_size', type=int, default=75)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_class', type=int, default=10)

args = parser.parse_args()


train_data = dataset.TrainSet('../TZBDATA/TRAIN_DATA/DATA_01/', 'train')
val_data = dataset.TrainSet('../TZBDATA/TRAIN_DATA/DATA_01/', 'val')

# model = torchvision.models.vgg16(weights=None)
# con1 = torch.nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1,1), bias=False)
# weight = con1.weight
# bias = con1.bias
# model.features[0] = con1
# model.features[0].weight = nn.Parameter(weight)
# model.features[0].bias = nn.Parameter(bias)
# model.classifier.add_module("add_linear",torch.nn.Linear(1000,10))
# model = model.to(device)

model = torchvision.models.resnet18(weights=None)
con1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias=False)
weight = con1.weight
model.conv1 = con1
model.conv1.weight = nn.Parameter(weight)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model = model.to(device)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batchsize, shuffle=True)

dataloaders = {'train': train_loader, 'val':val_loader}
dataset_sizes = {'train':len(train_data), 'val':len(val_data)}
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
model, ret_para = train(model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes ,step=['train', 'val'], num_epochs=args.epochs)

now = str(datetime.datetime.now())[:19]
now = now.replace(":","-")
# now = now.replace("-","")
# now = now.replace(" ","")

save_path = './Res/'+str(now)
config = {'batchsize':args.batchsize, 'lr':args.lr, 'momentum': args.momentum, 'step_size':args.step_size, 'gamma':args.gamma, 'num_epochs':args.epochs,
          'datasize_train':dataset_sizes['train'], 'datasize_val':dataset_sizes['val']}
if not os.path.exists(save_path):
    os.makedirs(save_path)
if ret_para[1][-1]>0.6:
    torch.save(model.state_dict(),save_path+'/model.pth')
np.save(save_path+'/config.npy',config)
np.save(save_path+'/res.npy',ret_para)

