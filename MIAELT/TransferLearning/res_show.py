import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

train_path = "train.npy"
val_path = "val.npy"
test_path = "test.npy"
finetune_path = "Finetuning_res.npy"


res_train = np.load(train_path)
res_val = np.load(val_path)
res_test = np.load(test_path)
res_finetune = np.load(finetune_path)

print("model pre-trian:train , finetun: val")
print("training set result:\n",res_train[6])
print("Val set result:\n",res_val[6])
print("test set result:\n",res_test[6])
print("Finetuning result:\n",res_finetune[1][24],res_finetune[6][24])