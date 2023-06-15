import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


dataset = TensorDataset(torch.tensor(list(range(20))))  # 构造一个数据集（0到19）
idx = list(range(len(dataset)))  # 创建索引，SubsetRandomSampler会自动乱序
# idx = torch.zeros(len(dataset)).long()  # 传入相同的索引，SubsetRandomSampler只会采样相同结果
n = len(dataset)
split = n//5
train_sampler = SubsetRandomSampler(idx[split::])  # 随机取80%的数据做训练集
test_sampler = SubsetRandomSampler(idx[::split])  # 随机取20%的数据做测试集
train_loader = DataLoader(dataset, sampler=train_sampler)
test_loader = DataLoader(dataset, sampler=test_sampler)

print('data for training:')
for i in train_loader:
    print(i)
print('data for testing:')
for i in test_loader:
    print(i)