import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *

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



class attentionModel_ml(nn.Module):
    def __init__(self, num_classes=61, mem_size=512):
        super(attentionModel, self).__init__()
        self.num_classes = num_classes
        self.msNet= msNet()
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputVariable):
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h*w)
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
            output_msnet=self.msNet.forward(attentionFeat)
            state = self.lstm_cell(attentionFeat, state)
        
        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        return feats, output_msnet
