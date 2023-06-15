# from experience_mnist import *
from experience_cifar10 import *
import os
import shutil
import datetime
import argparse
import sys
sys.path.append( "path" )
config = config()
now = str(datetime.datetime.now())[:19]
now = now.replace(":","")
now = now.replace("-","")
now = now.replace(" ","")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TrajectoryMIA')
    parser.add_argument("--not-train", action="store_true", help="is training state or not")
    parser.add_argument("--not-distill", action="store_true", help="is training state or not")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='target', help=['target', 'shadow', 'distill_target', 'distill_shadow'])
    parser.add_argument('--model', type=str, default='vgg', help=['resnet', 'mobilenet', 'vgg', 'wideresnet'])
    parser.add_argument('--data', type=str, default='cifar10', help=['cinic10', 'cifar10', 'cifar100', 'gtsrb'])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model_distill', type=str, default='vgg', help=['resnet', 'mobilenet', 'vgg', 'wideresnet'])
    parser.add_argument('--target_path', type=str, default='', help="attack other model's path")
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--mia_type', type=str, help=['build-dataset', 'black-box'])

    args = parser.parse_args()
    config.set_subkey('general', 'seed', args.seed)
    src_dir = config.path.data_path

    path = config.path.result_path + args.data + "-" +args.model + "/" + str(now) + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    dst_dir = path + "/config.yaml"
    shutil.copy(src_dir, dst_dir)
    print(args)
    experience_cifar10(config, path, args)





