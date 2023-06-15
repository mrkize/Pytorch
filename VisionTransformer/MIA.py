import datetime

import torch
from models.resnet_simclr import ResNetSimCLR, LinearClassifier, CombineModel
from models.attack_model import MLP_CE
from utils.dataset_parser.dataset_loader import GetDataLoader
from utils.mia.attackTraining import attackTraining
from utils.mia.metric_based_attack import AttackTrainingMetric
from utils.mia.label_only_attack import AttackLabelOnly
from model import load_VIT
from dataloader import model_dataloader, imgshuffle
import numpy as np
import time
import os
import argparse
torch.manual_seed(0)
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--no_PE', action = "store_true",
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
    parser.add_argument('--model', type=str, default='VIT')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='dataset')
    parser.add_argument('--data_path', type=str, default='data/',
                        help='data_path')
    parser.add_argument('--mode', type=str, default='target',
                        help='control using target dataset or shadow dataset (for membership inference attack)')

    parser.add_argument('--mean', type=str,
                        help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str,
                        help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str,
                        default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32,
                        help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SimCLR', "CE"], help='choose method')
    parser.add_argument('--projection_head_out_dim', type=int, default=256,
                        help='xxx')

    # linear training epochs
    parser.add_argument('--linear_epoch', type=int, default=0,
                        help='conduct MIA over a specific models 0 means the original target model, other numbers like 10 or 20 means target model linear layer only trained for 10 or 20 epochs')
    # label
    parser.add_argument('--original_label', type=str, default='Gender')
    parser.add_argument("--aux_label", type=str, default='Race')
    parser.add_argument('--single_label_dataset', type=list, default=["CIFAR10", "CIFAR100", "STL10"],
                        help="single_label_dataset")
    parser.add_argument('--multi_label_dataset', type=list, default=["UTKFace", "CelebA", "Place365", "Place100", "Place50", "Place20"],
                        help="multi_label_dataset")
    parser.add_argument('--mia_type', type=str, default="metric-based",
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
        "CIFAR10": 10,
        "CIFAR100": 100,
        "UTKFace": 2,
        "CelebA": 2,
        "Place365": 2,
        "Place100": 2,
        "Place50": 2,
        "Place20": 2,
    }
    opt.n_class = dataset_class_dict[opt.dataset]
    # opt.encoder_dim = model_encoder_dim_dict[opt.model]

    return opt


def _load_encoder_model(opt):

    model = ResNetSimCLR(
        base_model=opt.model, encoder_dim=opt.encoder_dim, out_dim=opt.projection_head_out_dim)
    model = model.to(device)
    return model


def _load_classifier_model(opt):
    n_features = opt.encoder_dim
    n_classes = opt.n_class

    model = LinearClassifier(n_features, n_classes)
    model = model.to(device)
    return model


def _load_model(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


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
        opt.dataset, opt.model, not opt.no_PE, opt.shf_dim, opt.ptr, opt.pxr)
    wf.write(line)


opt = parse_option()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

imgshuffle = imgshuffle(patch_ratio=opt.ptr, pixel_ratio=opt.pxr, shf_dim=opt.shf_dim)
data_loader, data_size = model_dataloader(root_dir='../data/cifar-10/',c_fn=imgshuffle)
target_train_loader, target_test_loader, shadow_train_loader, shadow_test_loader = data_loader['train'], data_loader['val'], data_loader['val'], data_loader['train']
if opt.no_PE == False:
    target_combine_model = load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth')
    shadow_combine_model = load_VIT('./Network/VIT_Model_cifar10/VIT_PE_shadow.pth')
else:
    target_combine_model = load_VIT('./Network/VIT_Model_cifar10/VIT_NoPE.pth')
    shadow_combine_model = load_VIT('./Network/VIT_Model_cifar10/VIT_NoPE_shadow.pth')
    target_combine_model.PE = False
    shadow_combine_model.PE = False

# target_combine_model.PE = False if opt.without_PE else True
# shadow_combine_model.PE = False if opt.without_PE else True

if opt.select_posteriors == -1:
    attack_model = MLP_CE()
else:
    attack_model = MLP_CE(selected_posteriors=opt.select_posteriors)


os.makedirs("log/model/exp_attack/", exist_ok=True)
if opt.select_posteriors == -1:
    if opt.mia_type == "nn-based":
        attack = attackTraining(opt, target_train_loader, target_test_loader,
                                shadow_train_loader, shadow_test_loader, target_combine_model, shadow_combine_model, attack_model, device)

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

    elif opt.mia_type == "metric-based":
        attack = AttackTrainingMetric(opt, target_train_loader, target_test_loader,
                                      shadow_train_loader, shadow_test_loader, target_combine_model, shadow_combine_model, attack_model, device)

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
                                 shadow_train_loader, shadow_test_loader, target_combine_model, shadow_combine_model, attack_model, device)

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

else:
    if opt.mia_type == "nn-based":
        attack = attackTraining(opt, target_train_loader, target_test_loader,
                                shadow_train_loader, shadow_test_loader, target_combine_model, shadow_combine_model, attack_model, device)

        attack.parse_dataset()

        acc_train = 0
        acc_test = 0
        epoch_train = opt.epochs
        train_acc, test_acc = attack.train(epoch_train)  # train 100 epoch
        target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc = attack.original_performance

        if opt.linear_epoch == 0:
            with open("log/model/exp_attack/mia_update_diff_posteriors.txt", "a") as wf:
                res = [opt.select_posteriors, epoch_train, target_train_acc,
                       target_test_acc, shadow_train_acc, shadow_test_acc, train_acc, test_acc]
                write_time(wf)
                write_config(wf, opt)
                write_res(wf, "NN-ATK-based", res)
                write_spilt(wf)

        else:
            pass
        print("Finish")

    elif opt.mia_type == "metric-based":
        attack = AttackTrainingMetric(opt, target_train_loader, target_test_loader,
                                      shadow_train_loader, shadow_test_loader, target_combine_model, shadow_combine_model, attack_model, device)

        attack.parse_dataset()

        acc_train = 0
        acc_test = 0
        epoch_train = opt.epochs

        train_tuple0, test_tuple0, test_results0, train_tuple1, test_tuple1, test_results1, train_tuple2, test_tuple2, test_results2, train_tuple3, test_tuple3, test_results3 = attack.train()
        target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc = attack.original_performance
        if opt.linear_epoch == 0:
            with open("log/model/exp_attack/mia_update_diff_posteriors.txt", "a") as wf:
                res0 = [opt.select_posteriors, epoch_train, target_train_acc, target_test_acc,
                        shadow_train_acc, shadow_test_acc, train_tuple0[0], test_tuple0[0]]
                res1 = [opt.select_posteriors, epoch_train, target_train_acc, target_test_acc,
                        shadow_train_acc, shadow_test_acc, train_tuple1[0], test_tuple1[0]]
                res2 = [opt.select_posteriors, epoch_train, target_train_acc, target_test_acc,
                        shadow_train_acc, shadow_test_acc, train_tuple2[0], test_tuple2[0]]
                res3 = [opt.select_posteriors, epoch_train, target_train_acc, target_test_acc,
                        shadow_train_acc, shadow_test_acc, train_tuple3[0], test_tuple3[0]]
                write_time(wf)
                write_config(wf, opt)
                write_res(wf, "Metric-corr", res0)
                write_res(wf, "Metric-conf", res1)
                write_res(wf, "Metric-entr", res2)
                write_res(wf, "Metric-ment", res3)
                write_spilt(wf)

        else:
            pass
        print("Finish")

    # Note that we change train acc to threshold in label-only attack
    elif opt.mia_type == "label-only":
        attack = AttackLabelOnly(opt, target_train_loader, target_test_loader,
                                 shadow_train_loader, shadow_test_loader, target_combine_model, shadow_combine_model, attack_model, device)

        acc_train = 0
        acc_test = 0
        epoch_train = opt.epochs

        attack.searchThreshold(num_samples=-1)
        test_tuple = attack.test(num_samples=-1)
        target_train_acc, target_test_acc, shadow_train_acc, shadow_test_acc, threshold = attack.original_performance

        #!!!!!!!!!!!we replace train acc with the threshold!!!!!!
        if opt.linear_epoch == 0:
            res = [epoch_train, target_train_acc, target_test_acc, shadow_train_acc,
                   shadow_test_acc, threshold, test_tuple[0]]

            with open("log/model/exp_attack/mia_update_diff_posteriors.txt", "a") as wf:
                write_time(wf)
                write_config(wf, opt)
                write_res(wf, "Label-only", res)
                write_spilt(wf)
        else:
            pass

        print("Finish")
