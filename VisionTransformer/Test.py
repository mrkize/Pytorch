from trainer import *

data_dir = '../data/dataset/'
model_pos = torch.load('./Network/VIT_pos.pth').to(device)
model_nopos = torch.load('./Network/VIT_nopos.pth').to(device)
train_data, train_loader = model_dataloader(root_dir=data_dir, spilt='train')
val_data, val_loader = model_dataloader(root_dir=data_dir, spilt='val')
soft = nn.Softmax(1)
cro = nn.CrossEntropyLoss()
sum_train = 0


