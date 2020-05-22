import torch
import torch.nn as nn
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)

class msNet(nn.Module):
    def __init__(self):
        self.conv = nn.Sequential(
            nn.Conv2d(512, 100, 7, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.class=nn.Sequential(
            nn.Linear(7*7*100,49),
            nn.Softmax(1))

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.class(x)
        return x


spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                 Scale(7),ToTensor()])


def msLoss(x,m):
    ok
    
