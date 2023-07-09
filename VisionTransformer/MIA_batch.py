import argparse
import os


parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--maskPE', action = "store_true",
                    help='whether use PE')
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch_size')
parser.add_argument('--mia_type', type=str, default='nn-based',
                    help='num of workers to use')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='number of training epochs')
opt = parser.parse_args()


cifar10_ratio = ['0', '0.0625', '0.125', '0.1875', '0.25', '0.3125', '0.375', '0.4375', '0.5', '0.5625', '0.625', '0.6875', '0.75', '0.8125', '0.875', '0.9375']
ImageNet10_ratio = ['{:.3f}'.format(i/196) for i in range(6,102,12)]
if opt.dataset == 'cifar10':
    ratio = cifar10_ratio
else:
    ratio = ImageNet10_ratio
cmd = 'python MIA.py --mia_type {} --dataset {} --model '.format(opt.mia_type,opt.dataset)
model_name = "ViT_mask_"
for i in ratio:
    commond = cmd + 'ViT' if i==0 else cmd + model_name + i
    os.system(commond)

