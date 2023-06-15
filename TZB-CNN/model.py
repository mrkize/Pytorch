import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class RCS_rec(nn.Module):
    def __init__(self, num_classes=10):
        self.classifier = nn.Sequential(

            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, num_classes),
        )


    def forward(self, x):
        out = self.classifier(x)
        return out