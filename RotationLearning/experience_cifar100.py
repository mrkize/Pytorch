import torch
import torch.nn as nn
import torch.optim as optim
from dataloaders import *
from utils import config
import numpy as np
from model import *
from torch.optim import lr_scheduler
from trainer import *
from distillation import *
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import lightgbm as lgb
import os


def experience_cifar100(config, path, args):
    print("Start data: {} , mode {}",format(args.data, args.model))
    # get device
    print(config)
    use_cuda = config.general.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # set random seed
    torch.manual_seed(config.general.seed)

    mode_path = config.path.model_path
    mode_path = config.path.model_path + "Model_" + args.data +"/"
    print(mode_path)
    if not os.path.exists(mode_path):
        os.makedirs(mode_path)

    model_target, train_loader_target, test_loader_target = get_model(args.model, args.data, istarget=True, train=args.train, config=config ,model_path=mode_path, res_path=path)
    model_shadow, train_loader_shadow, test_loader_shadow = get_model(args.model, args.data, istarget=False, train=args.train, config=config ,model_path=mode_path, res_path=path)

    # distillation
    if args.data == "cifar10":
        data_train_distill = custum_CIFAR10('distill', train=True, config = config).dataset
        data_test_distill = custum_CIFAR10('distill', train=True, config = config).dataset
        args.num_class = 10
    elif args.data == "cifar100":
        data_train_distill = custum_CIFAR100('distill', train=True, config = config).dataset
        data_test_distill = custum_CIFAR100('distill', train=True, config = config).dataset
        args.num_class = 100

    train_loader_distill = torch.utils.data.DataLoader(data_train_distill, batch_size=config.learning.batch_size, shuffle=True)
    test_loader_distill = torch.utils.data.DataLoader(data_test_distill, batch_size=config.learning.batch_size, shuffle=True)
    dataloaders_distill = {"train": train_loader_distill, "val": test_loader_distill}
    if args.distill:
        #distill shadow model, save in directory model_path, get training set
        distill_model(model_shadow, train_loader_distill, len(data_train_distill), config, False, args)

    trajectory_shadow, member_shadow = get_trajectory(mode_path, train_loader_shadow, test_loader_shadow, config.distillation.distill_epoch+1, False)
    attack_train_data = torch.utils.data.TensorDataset(torch.tensor(trajectory_shadow), torch.tensor(member_shadow))
    attack_train_loader_data = torch.utils.data.DataLoader(attack_train_data, batch_size=128, shuffle=True)

    np.save(path + "/trajectory_shadow" + ".npy", trajectory_shadow)
    np.save(path + "/member_shadow" + ".npy", member_shadow)
    if args.distill:
        # distill target model, save in directory model_path, get test set
        distill_model(model_target, train_loader_distill, len(data_train_distill), config, True)

    trajectory_target, member_target = get_trajectory(mode_path, train_loader_target, test_loader_target, config.distillation.distill_epoch+1, True)
    attack_validate_data = torch.utils.data.TensorDataset(torch.tensor(trajectory_target), torch.tensor(member_target))
    attack_validate_loader_data = torch.utils.data.DataLoader(attack_validate_data, batch_size=config.distillation.attack_batch_size, shuffle=True)

    np.save(path + "/trajectory_target" + ".npy", trajectory_target)
    np.save(path + "/member_target" + ".npy", member_target)
    #dataset
    dataloaders_attack = {"train": attack_train_loader_data, "val": attack_validate_loader_data}
    dataset_sizes_attack = {"train": config.general.train_target_size + config.general.test_target_size, "val": config.general.train_target_size + config.general.test_target_size}


    # create attack model
    attack_model = ATTACK(config.distillation.distill_epoch + 1).to(device)
    optimizer = optim.SGD(attack_model.parameters(), lr=config.learning.learning_rate,momentum=config.learning.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config.learning.decrease_lr_factor,gamma=config.learning.decrease_lr_every)
    criterion = nn.CrossEntropyLoss()
    attack_model, ret_para_attack, best_acc_attack = train_model(attack_model, criterion, optimizer, exp_lr_scheduler, dataloaders_attack, dataset_sizes_attack, config.learning.epochs)
    torch.save(attack_model, mode_path + "attack_model.pth")
    print("The best accuracy of model is: {}".format(best_acc_attack))
    print("The accuracy of model is: {}".format(ret_para_attack[5][config.learning.epochs-1]))
    np.save(path + "/res_train_attack" + ".npy", ret_para_attack)


    print("End ",args.data)
    return
