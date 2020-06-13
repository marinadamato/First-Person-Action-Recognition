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
    def __init__(self,num_classes=61, mem_size=512):
        
        super(colorization, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.i=0
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.residual_block=[]
        for i in range(4):
            self.residual_block.append(residual_block())
        self.residual_block = nn.Sequential(*self.residual_block)
        self.conv2 = nn.Conv2d(64, 3, kernel_size= 1, stride=1, padding=0, bias=False)
        self.deconv= nn.ConvTranspose2d(3, 3, 8, stride=4, padding=2, groups=3, bias=False)
        self.attML = attentionModel(num_classes, mem_size)
    
    def forward(self,inputVariable,entropy=None):
        flow_list =[]
        for t in range(inputVariable.size(0)): 
            x=self.conv1(inputVariable[t])
            
            x=self.bn1(x) 
            x=self.relu(x) 
            x=self.maxpool(x)
            
            
            x=self.residual_block(x)

            x=self.conv2(x) 
            x=self.deconv(x)
            flow_list.append(x)
        flow_list = torch.stack(flow_list, 0)
        
        if self.i==25:
            T=flow_list[0][0].data
            save_image(inputVariable[0][0][0], 'x.jpg')
            save_image(inputVariable[0][0][1], 'y.jpg')
            save_image(255*T, "color.jpg")
            print('new image')
            self.i=0
        self.i+=1
        if entropy == None :
            Out=self.attML(flow_list)
            return Out
        else:
            tmp=[]
            for t in range(flow_list.size(0)):
                y,_,_=self.attML.resNet(flow_list[t])
                tmp.append(y)
            return torch.stack(tmp, 0)
        
