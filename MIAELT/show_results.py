import numpy as np
import os
from utils import config
from torch.utils.tensorboard import SummaryWriter


config = config()
path = "./transformer_testcifar10-vgg/20230322114734/"
# files = [f for f in os.listdir(path)]
# print(files)
# read_path = path + str(files[2]) + "/"
read_path = path
shadow_path = read_path + "res_train_shadow.npy"
target_path = read_path + "res_train_target.npy"
attack_path = read_path + "res_train_attack.npy"
trajectory_shadow_path = read_path + "trajectory_shadow.npy"
trajectory_target_path = read_path + "trajectory_target.npy"
member_shadow_path = read_path + "member_shadow.npy"
res_shadow = np.load(shadow_path)
res_target = np.load(target_path)
res_attack = np.load(attack_path)
trajectory_shadow = np.load(trajectory_shadow_path)
trajectory_target = np.load(trajectory_target_path)
member_shadow = np.load(member_shadow_path)
# print(res_target[1][config.learning.epochs-1])

writer = SummaryWriter("logs")

print(member_shadow.shape)

# for i in range(config.distillation.distill_epoch+1):
#     for j in range(trajectory_shadow.shape[0]):
#         if j%100 == 0:
#             writer.add_scalar("trajectory_shadow_"+str(j)+"_"+str(member_shadow[i]), trajectory_shadow[j][i], i)

for i in range(config.distillation.distill_epoch + 1):
    for j in range(trajectory_target.shape[0]):
        if j % 100 == 0:
            writer.add_scalar("trajectory_target_" + str(j),
                              trajectory_shadow[j][i], i)

            # writer.add_scalar("trajectory_target", trajectory_target[j][i], i)

writer.close()

# for i in range (config.learning.epochs):
#     print(res_attack[1][i])
#     print(res_attack[3][i])