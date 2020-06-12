from attentionmodel_ml import attentionModel_ml
import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *
from objectAttentionModelConvLSTM import attentionModel
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from spatial_transforms import Normalize

class residual_block(nn.Module):
    def __init__(self):
        super(residual_block,self).__init__()
        self.conv1 = nn.Conv2d(64,64, kernel_size=3, stride=1,padding= 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(negative_slope=0.02, inplace=True)
        self.conv2 = nn.Conv2d(64,64, kernel_size=3, stride=1,padding= 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(negative_slope=0.02, inplace=True)
        

    def forward(self,x):
        x_p=x
        x= self.conv1(x)     
        x= self.bn1(x)
        x=self.relu1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x= x_p + x 
        x=self.relu2(x)
        return x


class colorization(nn.Module):
    def __init__(self,num_classes=61, mem_size=512, regressor=False):
        
        super(colorization, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.residual_block=[]
        for i in range(7):
            self.residual_block.append(residual_block())
        self.conv2 = nn.Conv2d(64, 3, kernel_size= 1, stride=1, padding=0, bias=False)
        self.deconv= nn.ConvTranspose2d(3, 3, 8, stride=4, padding=2, groups=3, bias=False)
        self.attML = attentionModel_ml(num_classes, mem_size, regressor)
    
    def forward(self,inputVariable):
        normalize = Normalize(mean=255*[0.485, 0.456, 0.406], std=255*[0.229, 0.224, 0.225])
        flow_list =[]
        for t in range(inputVariable.size(0)): 
            x=self.conv1(inputVariable[t])
            
            x=self.bn1(x) 
            x=self.relu(x) 
            x=self.maxpool(x)

            for i in range(7):
                x=self.residual_block[i](x)

            x=self.conv2(x) 
            x=self.deconv(x)
            '''image_list=[]
            for tensor in x:
                tensor=normalize(tensor=tensor,inv=False, flow=False)
                image_list.append(tensor)
            image_list = torch.stack(image_list, 0)
            print(image_list.size())'''
            flow_list.append(x)
        flow_list = torch.stack(flow_list, 0)
        T=flow_list[0][0].data
        T=normalize(T, False, False)
        save_image(inputVariable[0][0][0], 'color.jpg')
        save_image(inputVariable[0][0][1], 'color.jpg')
        save_image(T, "color.jpg")
        Out=self.attML(flow_list)
        return Out
