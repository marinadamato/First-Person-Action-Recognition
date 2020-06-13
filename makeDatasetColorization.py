import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import sys
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip, Binary)



def gen_split(root_dir, stackSize, phase):
    DatasetX = []
    DatasetY = []
    Labels = []
    NumFrames = []
    
    for original_dir in sorted(os.listdir(root_dir)):
        if (original_dir=='.DS_Store' or original_dir=='processed_frames2'): continue
        root_dir1 = os.path.join(root_dir, original_dir) #GTEA61/flow_x_processed/
        for dir_user in sorted(os.listdir(root_dir1)):
            if dir_user=='.DS_Store': continue
            if (phase=='train') ^ (dir_user=="S2"):
                class_id = 0
                dir = os.path.join(root_dir1, dir_user) #GTEA61/processed_frames2/S1/
                for target in sorted(os.listdir(dir)):
                    if target=='.DS_Store': continue
                    dir1 = os.path.join(dir, target) #GTEA61/processed_frames2/S1/close_choco/
                    insts = sorted(os.listdir(dir1))
                    if insts != []:
                        for inst in insts:
                            if inst=='.DS_Store': continue
                            inst_dir = os.path.join(dir1, inst) #GTEA61/processed_frames2/S1/close_choco/1/
                            numFrames = len(glob.glob1(inst_dir, '*.png'))
                            if numFrames >= stackSize:
                                if original_dir=='flow_x_processed':
                                    DatasetX.append(inst_dir)
                                    Labels.append(class_id)
                                    NumFrames.append(numFrames)
                                else:DatasetY.append(inst_dir)
                               
                                   
                                
                    class_id += 1
    return DatasetX, DatasetY, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, sequence=False, stackSize=5,
                 train=True, numSeg=5, fmt='.png', phase='train', seqLen = 25):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.imagesX, self.imagesY, self.labels, self.numFrames = gen_split(
            root_dir, stackSize, phase)
        self.spatial_transform = spatial_transform        
         
        self.train = train
        self.numSeg = numSeg
        self.sequence = sequence
        self.stackSize = stackSize
        self.fmt = fmt
        self.phase = phase
        self.seqLen = seqLen

    def __len__(self):
        return len(self.imagesX)

    def __getitem__(self, idx):
        vid_nameX = self.imagesX[idx]
        vid_nameY = self.imagesY[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeqSegs = []
        self.spatial_transform.randomize_parameters()
    
        inpSeqX = []
        inpSeqY = []
     
        inpSeqF = []
        mapSeq = []
        for i in np.linspace(1, numFrame, self.stackSize, endpoint=False):
            fl_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + '.png'
            img = Image.open(fl_name)
            inpSeqX.append(self.spatial_transform(img.convert('L'), inv=True, flow=True))
            # fl_names.append(fl_name)
            f1_name = vid_nameY + '/flow_y_' + str(int(round(i))).zfill(5) + '.png'
            img2 = Image.open(f1_name)
            inpSeqY.append(self.spatial_transform(img2.convert('L'), inv=False, flow=True))

        inpSeqSegs = torch.stack([torch.stack(inpSeqX, 0).squeeze(1),torch.stack(inpSeqY, 0).squeeze(1)],0).permute(1,0,2,3)

        return inpSeqSegs, label
