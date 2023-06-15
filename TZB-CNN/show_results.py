import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter


path = "./Res/"
files = [f for f in os.listdir(path)]
print(files)
num = int(input())

read_path = path + str(files[num])
config = np.load(read_path+'/config.npy', allow_pickle='TRUE').item()
res = np.load(read_path+'/res.npy')

print(config)
print('train result:')
print(res[1])
print('val result:')
print(res[3])
# for i in range (config.learning.epochs):
#     print(res_attack[1][i])
#     print(res_attack[3][i])