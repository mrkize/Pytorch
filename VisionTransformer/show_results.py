import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter


path = "./results/VIT_cifar10/"
files = [f for f in os.listdir(path)]
print(files)
num = int(input())

read_path = path + str(files[num])

res = np.load(read_path)
print(res.shape)
print('train result:')
print(res[1][-5:-1])
print('val result:')
print(res[6][-5:-1])
# for i in range (config.1
# learning.epochs):
#     print(res_attack[1][i])
#     print(res_attack[3][i])