# -*- coding: UTF-8 -*-

import glob
import os
from PIL import Image
from torchvision import datasets


dataset = datasets.ImageFolder('../data/cifar-100/val')
for i in range(4):
    outpath = "../data/cifar-100_rotate/val/"+str(i)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    j = 0
    for img,label in dataset:
        im_rotate = img.rotate(90*i, expand=1)  # 逆时针旋转90度,expand=1表示原图直接旋转
        im_rotate.save(outpath + '/' + str(j) +'_rot_'+'i'+'.jpg')
        j+=1
print('所有图片均已旋转完毕，并存入输出文件夹')
