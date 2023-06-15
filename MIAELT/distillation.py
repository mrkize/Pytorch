import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F
from model import *
from dataloaders import *

class DistillationLoss:
    def __init__(self, config):
        self.student_loss = nn.CrossEntropyLoss()
        self.distillation_loss = nn.KLDivLoss()
        self.temperature = config.distillation.temperature
        self.alpha = config.distillation.alpha

    def __call__(self, student_logits, student_target_loss, teacher_logits):

        distillation_loss = self.distillation_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                                   F.softmax(teacher_logits / self.temperature, dim=1))

        loss = (1 - self.alpha) * student_target_loss + self.alpha * distillation_loss
        return loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def distill_model(teacher_mode, dataloaders, dataset_sizes, config, istarget, args):
    save_path = "./Network/Model_" + args.data + "/"
    save_path = save_path + "target_dis/" if istarget == True else save_path + "shadow_dis/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.model_distill=='vgg':
        init_model = VGG16(args.num_class).to(device)
    # elif args.model_distill == 'resnet':
    elif args.model_distill=='resnet':
        init_model = models.resnet34(args.num_class).to(device)
    else:
        init_model = models.resnet34(args.num_class).to(device)

    optimizer = optim.SGD(init_model.parameters(), lr=config.learning.learning_rate,momentum=config.learning.momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_factor,gamma=config.learning.decrease_lr_every)
    print("Distillation dataset size", dataset_sizes)
    since = time.time()
    crit = nn.CrossEntropyLoss()

    epoch_gap = config.learning.epochs - config.distillation.distill_epoch
    disloss = DistillationLoss(config)
    for epoch in range(config.learning.epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)
        # for phase in ['train', 'val']:
        scheduler.step()
        init_model.train()  # Set model to training mode
        # Iterate over data.
        for batch_idx, (data, target) in enumerate(dataloaders):
            inputs, labels = data.to(device), target.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = init_model(inputs)
            # _, preds = torch.max(outputs, 1)
            # calculate the student-ground truth gap loss2target, but in fact we don't use it
            # in <DistillationLoss> ,the alpha = 1, means we caculate this gap but we don't use it
            loss2target = crit(outputs, labels)
            #caculate the hard label teacher_out

            # _, preds = torch.max(teacher_mode(inputs).data, 1)
            # hard_label = F.one_hot(preds).to(device)
            hard_label =soft2onehot(teacher_mode(inputs)).to(device)
            # hard_label =
            loss = disloss(outputs, loss2target, hard_label)
            loss.backward()
            optimizer.step()

        if epoch >= epoch_gap:
            torch.save(init_model, save_path + "epoch_" + str(epoch-epoch_gap) + ".pth")

    time_elapsed = time.time() - since
    print('Distill complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("DONE Distillation Target model" if istarget else "DONE Distillation Shadow model")
    return

def get_trajectory(model_path, data_train, data_test, epochs, istarget):
    member_test = None
    member_train = None
    trajectory_train = None
    trajectory_test = None
    trajectory = None
    # member_train = [[]]
    # member_test = [[]]
    # load_path = "./Network/Model_CIFAR10/"
    if istarget:
        load_path = model_path + "target_dis/epoch_"
    else:
        load_path = model_path + "shadow_dis/epoch_"
    for i in range(epochs):
        path = load_path + str(i) + ".pth"

        #the last epoch use target/shadow model
        if i == epochs - 1:
            path = model_path + "/target.pth" if istarget else model_path + "shadow.pth"
        model = torch.load(path)
        model.eval()
        trajectory_current = None
        for batch_idx, (data, target) in enumerate(data_train):
            inputs, labels = data.to(device), target.to(device)
            logit_target = model(inputs)
            loss = [F.cross_entropy(logit_target_i.unsqueeze(0), label_i.unsqueeze(0)) for (logit_target_i, label_i)in zip(logit_target, labels)]
            loss = np.array([loss_i.detach().cpu().numpy() for loss_i in loss]).reshape(-1, 1)
            trajectory_current = loss if batch_idx == 0 else np.concatenate((trajectory_current, loss), 0)
            if i == 0:
                member_train = np.repeat([[1]], inputs.shape[0], 0) if batch_idx == 0 else np.concatenate([np.repeat([[1]], inputs.shape[0], 0), member_train])

        trajectory_train = trajectory_current if i == 0 else np.concatenate((trajectory_train, trajectory_current), 1)
    # print(trajectory_train.shape,member_train.shape)
    for i in range(epochs):
        path = load_path + str(i) + ".pth"
        if i == epochs-1:
            path = model_path + "/target.pth" if istarget else model_path + "shadow.pth"
        model = torch.load(path)
        model.eval()
        trajectory_current = None
        for batch_idx, (data, target) in enumerate(data_test):
            inputs, labels = data.to(device), target.to(device)
            logit_target = model(inputs)
            loss = [F.cross_entropy(logit_target_i.unsqueeze(0), label_i.unsqueeze(0)) for (logit_target_i, label_i)in zip(logit_target, labels)]
            loss = np.array([loss_i.detach().cpu().numpy() for loss_i in loss]).reshape(-1, 1)
            trajectory_current = loss if batch_idx == 0 else np.concatenate((trajectory_current, loss), 0)
            if i == 0:
                member_test = np.repeat([[0]], inputs.shape[0], 0) if batch_idx == 0 else np.concatenate([np.repeat([[0]], inputs.shape[0], 0), member_test])
        trajectory_test = trajectory_current if i == 0 else np.concatenate((trajectory_test, trajectory_current), 1)

    # print(trajectory_test.shape, member_test.shape)

    trajectory = np.concatenate([trajectory_train, trajectory_test])
    member = np.concatenate([member_train, member_test])
    return trajectory,member

def soft2onehot(soft_label):
    onehot = torch.zeros(soft_label.shape)
    _, preds = torch.max(soft_label.data, 1)
    onehot[range(soft_label.shape[0]), preds] = 1
    return onehot

