from utils import config
import os
import shutil
import datetime
import argparse
import sys
from trainer import *
from ModelFineTuning import *
sys.path.append( "path" )

config = config()
now = str(datetime.datetime.now())[:19]
now = now.replace(":","")
now = now.replace("-","")
now = now.replace(" ","")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RotLearning')
    parser.add_argument("--pretrain", action="store_true", help="preterain a model")
    parser.add_argument("--finetune", action="store_true", help="finetune")
    parser.add_argument("--evaluate", action="store_true", help="finetune")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='res34', help=['resnet', 'mobilenet', 'vgg', 'wideresnet'])
    parser.add_argument('--data', type=str, default='rot_data', help=['cinic10', 'cifar10', 'cifar100', 'gtsrb', 'rot_data'])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--trainSamples', type=int, default=16, help='number of training samples to use')

    args = parser.parse_args()
    config.set_subkey('general', 'seed', args.seed)
    src_dir = config.path.data_path

    path = './TransferLearning/'
    if not os.path.exists(path):
        os.makedirs(path)
    dst_dir = path + "/config.yaml"
    shutil.copy(src_dir, dst_dir)
    print(args)
    if args.pretrain:
        model, train_loader, test_loader = get_model(args.model, args.data, istarget=True, not_train=args.not_train, config=config, model_path=path, res_path=path)
    elif args.finetune:
        model = resnetfinetune("../data/cinic-10/", "./TransferLearning/", config, args)
    elif evaluate:
        # evaluate train/val/test
        evaluate('train', '../data/cinic-10/', args, config)
        evaluate('val', '../data/cinic-10/', args, config)
        evaluate('test', '../data/cinic-10/', args, config)








