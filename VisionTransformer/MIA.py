import datetime
import shfattacktrain
import torch

from utils import MyConfig
from atk_models.resnet_simclr import ResNetSimCLR, LinearClassifier
from atk_models.attack_model import MLP_CE
from utils.mia.attackTraining import attackTraining
from utils.mia.metric_based_attack import AttackTrainingMetric
from utils.mia.label_only_attack import AttackLabelOnly
from mymodel import ViT,ViT_mask,ViT_ape,ViT_mask_avg,Swin,Swin_mask, Swin_mask_avg,Swin_ape
from dataloader import imgshuffle, cifar_dataloader, ImageNet_loader

import numpy as np
import os
import argparse
torch.manual_seed(0)
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--maskPE', action = "store_true",
                        help='whether use PE')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of training epochs')
    parser.add_argument('--shf_dim', type=int, default=0,
                        help='shuffle dim for image, <=0 means donot shuffle')
    parser.add_argument('--ptr', type=float, default=0.1,
                        help='patches shuffle ratio')
    parser.add_argument('--pxr', type=float, default=0.2,
                        help='pixel shuffle ratio')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')

    # model dataset
    parser.add_argument('--model', type=str, default='ViT_mask_avg_0.5')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset')
    parser.add_argument('--data_path', type=str, default='data/',
                        help='data_path')
    parser.add_argument('--mode', type=str, default='target',
                        help='control using target dataset or shadow dataset (for membership inference attack)')


    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SimCLR', "CE"], help='choose method')


    # linear training epochs
    parser.add_argument('--linear_epoch', type=int, default=0,
                        help='conduct MIA over a specific models 0 means the original target model, other numbers like 10 or 20 means target model linear layer only trained for 10 or 20 epochs')
    # label
    parser.add_argument('--original_label', type=str, default='Gender')
    parser.add_argument("--aux_label", type=str, default='Race')
    parser.add_argument('--single_label_dataset', type=list, default=["cifar10", "cifar100", "STL10", "ImageNet10", "ImageNet100", "cinic10", "Fashion-Mnist"],
                        help="single_label_dataset")
    parser.add_argument('--multi_label_dataset', type=list, default=["UTKFace", "CelebA", "Place365", "Place100", "Place50", "Place20"],
                        help="multi_label_dataset")
    parser.add_argument('--mia_type', type=str, default="nn-based",
                        help="nn-based, lebel-only, metric-based")
    parser.add_argument('--select_posteriors', type=int, default=-1,
                        help='how many posteriors we select, if -1, we remains the original setting')

    parser.add_argument('--mia_defense', type=str,
                        default="None", help='None or memGuard')

    opt = parser.parse_args()

    # model_encoder_dim_dict = {
    #     "resnet18": 512,
    #     "resnet50": 2048,
    #     "alexnet": 4096,
    #     "vgg16": 4096,
    #     "vgg11": 4096,
    #     "mobilenet": 1280,
    #     "cnn": 512,
    # }
    dataset_class_dict = {
        "STL10": 10,
        "cifar10": 10,
        "cifar100": 100,
        "UTKFace": 2,
        "CelebA": 2,
        "Place365": 2,
        "Place100": 2,
        "Place50": 2,
        "Place20": 2,
        "ImageNet100": 100,
        "ImageNet10": 10,
    }
    opt.n_class = dataset_class_dict[opt.dataset]
    # opt.encoder_dim = model_encoder_dim_dict[opt.model]

    return opt




def write_res(wf, attack_name, res):
    # line = "%s,%s,%s,%d," % (
    #     opt.dataset, opt.model, opt.no_PE, opt.shf_dim)

    line = "%s:\t" % attack_name

    line += ",".join(["%.3f" % (row) for row in res])
    line += "\n"
    wf.write(line)

def write_spilt(wf):
    wf.write('--------------------------------------------------------------------------------------------------------------------\n')

def write_time(wf):
    wf.write(str(datetime.datetime.now())[:19]+'\n')

def write_config(wf, opt):
    line = "%s, %s, use_PE: %s, shf_dim: %d, ptr: %f, pxr: %f\n" % (
        opt.dataset, opt.model, not opt.maskPE, opt.shf_dim, opt.ptr, opt.pxr)
    wf.write(line)


opt = parse_option()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_dict = { 'ViT': {
                'cifar10': "config/cifar10/",
                'cifar100': "config/cifar100/",
                'ImageNet10': "config/ImageNet10/",
                'ImageNet100': "config/ImageNet100/"
                },
                'Swin': {
                'cifar10': "config/Swin-cifar10/",
                'cifar100': "config/Swin-cifar100/",
                'ImageNet10': "config/Swin-ImageNet10/",
                'ImageNet100': "config/Swin-ImageNet100/"
                }
}

# torch.random.manual_seed(1001)
config_path = config_dict['Swin'][opt.dataset] if 'Swin' in opt.model else config_dict['ViT'][opt.dataset]


config = MyConfig.MyConfig(path=config_path)
if 'cifar' in opt.dataset:
    target_loader, target_size = cifar_dataloader(config, is_target=True)
    shadow_loader, shadow_size = cifar_dataloader(config, is_target=False)
    target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader = target_loader['train'], target_loader['val'], shadow_loader['train'], shadow_loader['val']
else:
    target_loader, target_size = ImageNet_loader(config, is_target=True)
    shadow_loader, shadow_size = ImageNet_loader(config, is_target=False)
    target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader = target_loader['train'], target_loader['val'], shadow_loader['train'], shadow_loader['val']

def load_model(opt, config):
    if "Swin" in opt.model:
        if 'mask_avg' in opt.model:
            path = config.path.model_path+opt.model[:13]
            target_model = Swin_mask_avg.load_Swin(path+opt.model[13:]+'.pth', config)
            shadow_model = Swin_mask_avg.load_Swin(path+'_shadow'+opt.model[13:]+'.pth', config)
        elif 'ape' in opt.model:
            path = config.path.model_path + opt.model + '.pth'
            target_model = Swin_ape.load_Swin(path + '.pth', config)
            shadow_model = Swin_ape.load_Swin(path + '_shadow.pth', config)
        elif 'mask' in opt.model:
            path = config.path.model_path + opt.model[:9]
            target_model = Swin_mask.load_Swin(path+opt.model[9:]+'.pth', config)
            shadow_model = Swin_mask.load_Swin(path + '_shadow'+opt.model[9:]+'.pth', config)
        else:
            path = config.path.model_path + opt.model
            target_model = Swin.load_Swin(path + '.pth', config)
            shadow_model = Swin.load_Swin(path + '_shadow.pth', config)
    else:
        if 'ViT_mask_avg' in opt.model:
            path = config.path.model_path+opt.model[:12]
            target_model = ViT_mask_avg.load_VIT(path+opt.model[12:]+'.pth', config)
            shadow_model = ViT_mask_avg.load_VIT(path+'_shadow'+opt.model[12:]+'.pth', config)
        elif 'ViT_ape' in opt.model:
            path = config.path.model_path + opt.model + '.pth'
            target_model = ViT_ape.load_VIT(path + '.pth', config)
            shadow_model = ViT_ape.load_VIT(path + '_shadow.pth', config)
        elif 'mask' in opt.model:
            path = config.path.model_path + opt.model[:8]
            target_model = ViT_mask.load_VIT(path+opt.model[8:]+'.pth', config)
            shadow_model = ViT_mask.load_VIT(path + '_shadow'+opt.model[8:]+'.pth', config)
        else :
            path = config.path.model_path + opt.model
            target_model = ViT.load_VIT(path + '.pth', config)
            shadow_model = ViT.load_VIT(path + '_shadow.pth', config)

    if opt.maskPE:
        target_model.PE = False
        shadow_model.PE = False

    return target_model, shadow_model


# if opt.no_PE == False:
#     target_model = load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth')
#     shadow_model = load_VIT('./Network/VIT_Model_cifar10/VIT_PE_shadow.pth')
# else:
#     target_model = load_VIT('./Network/VIT_Model_cifar10/VIT_NoPE.pth')
#     shadow_model = load_VIT('./Network/VIT_Model_cifar10/VIT_NoPE_shadow.pth')



if opt.select_posteriors == -1:
    attack_model = MLP_CE()
else:
    attack_model = MLP_CE(selected_posteriors=opt.select_posteriors)

ratio = [0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375]


os.makedirs("log/model/exp_attack/", exist_ok=True)
for i in ratio:
    opt.model = 'ViT' if i==0 else 'ViT_mask_' + str(i)
    target_model, shadow_model = load_model(opt, config)
    if opt.mia_type == "nn-based":
        attack = attackTraining(opt, target_train_loader, target_test_loader,
                                shadow_train_loader, shadow_test_loader, target_model, shadow_model, attack_model, device)

        attack.parse_dataset()

        acc_train = 0
        acc_test = 0
        epoch_train = opt.epochs
        train_acc, test_acc = attack.train(epoch_train)  # train 100 epoch
        target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc = attack.original_performance

        if opt.linear_epoch == 0:
            with open("log/model/exp_attack/mia_update.txt", "a") as wf:
                res = [epoch_train, target_train_acc, target_test_acc,
                       shadow_train_acc, shadow_test_acc, train_acc, test_acc]
                write_time(wf)
                write_config(wf, opt)
                write_res(wf, "NN-ATK-based", res)
                write_spilt(wf)

        else:
            with open("log/model/exp_attack/mia_update_different_linear_epoch.txt", "a") as wf:
                res = [opt.linear_epoch, epoch_train, target_train_acc, target_test_acc,
                       shadow_train_acc, shadow_test_acc, train_acc, test_acc]
                write_time(wf)
                write_config(wf, opt)
                write_res(wf, "NN-ATK-based", res)
                write_spilt(wf)
        print("Finish")

    elif opt.mia_type == "shf-nn-based":
        attack_model = shfattacktrain.MLP_CE()
        attack = shfattacktrain.attackTraining(opt, target_train_loader, target_test_loader,
                                shadow_train_loader, shadow_test_loader, target_model, shadow_model, attack_model, device)

        attack.parse_dataset()

        acc_train = 0
        acc_test = 0
        epoch_train = opt.epochs
        train_acc, test_acc = attack.train(epoch_train)  # train 100 epoch
        target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc = attack.original_performance

        if opt.linear_epoch == 0:
            with open("log/model/exp_attack/mia_update.txt", "a") as wf:
                res = [epoch_train, target_train_acc, target_test_acc,
                       shadow_train_acc, shadow_test_acc, train_acc, test_acc]
                write_time(wf)
                write_config(wf, opt)
                write_res(wf, "shf-nn-based", res)
                write_spilt(wf)
        print("Finish")


    elif opt.mia_type == "metric-based":
        attack = AttackTrainingMetric(opt, target_train_loader, target_test_loader,
                                      shadow_train_loader, shadow_test_loader, target_model, shadow_model, attack_model, device)

        attack.parse_dataset()

        acc_train = 0
        acc_test = 0
        epoch_train = opt.epochs

        train_tuple0, test_tuple0, test_results0, train_tuple1, test_tuple1, test_results1, train_tuple2, test_tuple2, test_results2, train_tuple3, test_tuple3, test_results3 = attack.train()
        target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc = attack.original_performance
        if opt.linear_epoch == 0:
            with open("log/model/exp_attack/mia_update.txt", "a") as wf:
                res0 = [epoch_train, target_train_acc, target_test_acc,
                        shadow_train_acc, shadow_test_acc, train_tuple0[0], test_tuple0[0]]
                res1 = [epoch_train, target_train_acc, target_test_acc,
                        shadow_train_acc, shadow_test_acc, train_tuple1[0], test_tuple1[0]]
                res2 = [epoch_train, target_train_acc, target_test_acc,
                        shadow_train_acc, shadow_test_acc, train_tuple2[0], test_tuple2[0]]
                res3 = [epoch_train, target_train_acc, target_test_acc,
                        shadow_train_acc, shadow_test_acc, train_tuple3[0], test_tuple3[0]]
                write_time(wf)
                write_config(wf, opt)
                write_res(wf, "Metric-corr", res0)
                write_res(wf, "Metric-conf", res1)
                write_res(wf, "Metric-entr", res2)
                write_res(wf, "Metric-ment", res3)
                write_spilt(wf)


        else:
            with open("log/model/exp_attack/mia_update_different_linear_epoch.txt", "a") as wf:
                res0 = [opt.linear_epoch, epoch_train, target_train_acc, target_test_acc,
                        shadow_train_acc, shadow_test_acc, train_tuple0[0], test_tuple0[0]]
                res1 = [opt.linear_epoch, epoch_train, target_train_acc, target_test_acc,
                        shadow_train_acc, shadow_test_acc, train_tuple1[0], test_tuple1[0]]
                res2 = [opt.linear_epoch, epoch_train, target_train_acc, target_test_acc,
                        shadow_train_acc, shadow_test_acc, train_tuple2[0], test_tuple2[0]]
                res3 = [opt.linear_epoch, epoch_train, target_train_acc, target_test_acc,
                        shadow_train_acc, shadow_test_acc, train_tuple3[0], test_tuple3[0]]
                write_time(wf)
                write_config(wf, opt)
                write_res(wf, "Metric-corr", res0)
                write_res(wf, "Metric-conf", res1)
                write_res(wf, "Metric-entr", res2)
                write_res(wf, "Metric-ment", res3)
                write_spilt(wf)
        print("Finish")

    # Note that we change train acc to threshold in label-only attack
    elif opt.mia_type == "label-only":
        attack = AttackLabelOnly(opt, target_train_loader, target_test_loader,
                                 shadow_train_loader, shadow_test_loader, target_model, shadow_model, attack_model, device)

        acc_train = 0
        acc_test = 0
        epoch_train = opt.epochs

        attack.searchThreshold(num_samples=500)
        test_tuple = attack.test(num_samples=500)
        target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc, threshold = attack.original_performance

        #!!!!!!!!!!!we replace train acc with the threshold!!!!!!
        if opt.linear_epoch == 0:
            res = [epoch_train, target_train_acc, target_test_acc, shadow_train_acc,
                   shadow_test_acc, threshold, test_tuple[0]]
            with open("log/model/exp_attack/mia_update.txt", "a") as wf:
                write_config(wf, opt)
                write_res(wf, "Label-only", res)
                write_spilt(wf)
        else:
            with open("log/model/exp_attack/mia_update_different_linear_epoch.txt", "a") as wf:
                res = [opt.linear_epoch, epoch_train, target_train_acc, target_test_acc, shadow_train_acc,
                       shadow_test_acc, threshold, test_tuple[0]]
                write_config(wf, opt)
                write_res(wf, "Label-only", res)
                write_spilt(wf)

        print("Finish")
