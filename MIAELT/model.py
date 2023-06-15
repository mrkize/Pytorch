import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            #1
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #2
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #3
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            #4
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #5
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #6
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #7
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #8
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #9
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #10
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #11
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #12
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #13
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.AvgPool2d(kernel_size=1,stride=1),
            )
        self.classifier = nn.Sequential(
            #14
            nn.Linear(512,4096),
            nn.ReLU(True),
            nn.Dropout(),
            #15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #16
            nn.Linear(4096,num_classes),
            )
        #self.classifier = nn.Linear(512, 10)
 
    def forward(self, x):
        out = self.features(x)
        #        print(out.shape)
        out = out.view(out.size(0), -1)
        #        print(out.shape)
        out = self.classifier(out)
        #        print(out.shape)
        return out


class Residual(nn.Module):
    def __init__(self, input_channel, out_channel, use_conv1x1=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, int(out_channel / 4), kernel_size=1, stride=strides)
        self.conv2 = nn.Conv2d(int(out_channel / 4), int(out_channel / 4), kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(int(out_channel / 4), out_channel, kernel_size=1)
        if use_conv1x1:
            self.conv4 = nn.Conv2d(input_channel, out_channel, kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(int(out_channel / 4))
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.relu(self.bn1(self.conv2(Y)))
        Y = self.bn2(self.conv3(Y))
        if self.conv4:
            X = self.conv4(X)
        Y += X
        return F.relu(Y)


# 多个残差单元组成的残差块
def res_block(input_channel, out_channel, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channel, out_channel, use_conv1x1=True, strides=2))
            input_channel = out_channel
        if i == 0 and first_block:
            blk.append(Residual(input_channel, out_channel, use_conv1x1=True, strides=1))
            input_channel = out_channel
        else:
            blk.append(Residual(input_channel, out_channel))
    return blk


def resnet50(classes, num_channel = 3):
    b1 = nn.Sequential(
        nn.Conv2d(num_channel, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b2 = nn.Sequential(*res_block(64, 256, 3, first_block=True))
    b3 = nn.Sequential(*res_block(256, 512, 4))
    b4 = nn.Sequential(*res_block(512, 1024, 6))
    b5 = nn.Sequential(*res_block(1024, 2048, 3))
    model = nn.Sequential(b1, b2, b3, b4, b5,
                          nn.AdaptiveAvgPool2d((1, 1)),
                          nn.Flatten(),
                          nn.Linear(2048, classes)
                          )
    return model

# class pertrain_vgg16():
#     def __init__(self, num_class):
#         vgg16 = torchvision.models.vgg16(pretrained=True)
#         vgg16.




class ATTACK(nn.Module):
    def __init__(self, dim_in):
        super(ATTACK, self).__init__()
        self.dim_in = dim_in
        self.fc1 = nn.Linear(self.dim_in, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, self.dim_in)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


