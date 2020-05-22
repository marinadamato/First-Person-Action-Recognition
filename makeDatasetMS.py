import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import random


def gen_split(root_dir, stackSize, phase):
    RGB = []
    Labels = []
    Maps = []
    NumFrames = []
    root_dir = os.path.join(root_dir, 'processed_frames2') #GTEA61/processed_frames2/
    
    for dir_user in sorted(os.listdir(root_dir)): #S1/
        if dir_user=='.DS_Store': continue
        if (phase=='train') ^ (dir_user=="S2"):
            dir = os.path.join(root_dir, dir_user) #GTEA61/processed_frames2/S1/
            class_id=0
            
            for target in sorted(os.listdir(dir)): #close_choco/
                if target=='.DS_Store': continue
                dir1 = os.path.join(dir, target) #GTEA61/processed_frames2/S1/close_choco/
                
                insts = sorted(os.listdir(dir1)) #1/
                if insts != []:
                    for inst in insts:
                        if inst=='.DS_Store': continue
                        
                        inst_dir = os.path.join(dir1, inst+"/rgb") #GTEA61/processed_frames2/S1/close_choco/1/rgb/
                        numFrames = len(glob.glob1(inst_dir, '*.png'))
                        
                        if numFrames >= stackSize:
                            RGB.append(inst_dir)
                            Labels.append(class_id)
                            NumFrames.append(numFrames)
                            
                        inst_dir = os.path.join(dir1, inst+"/mmaps") #GTEA61/processed_frames2/S1/close_choco/1/mmaps/
                        numFrames = len(glob.glob1(inst_dir, '*.png'))
                        
                        if numFrames >= stackSize:
                            Maps.append(inst_dir)
                class_id += 1
    return RGB, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform_rgb=None, spatial_transform_map=None, seqLen=20,
                 train=True, mulSeg=False, numSeg=1, fmt='.png',phase='train'):

        self.images, self.labels, self.numFrames = gen_split(root_dir, 5,phase)
        self.spatial_transform_rgb = spatial_transform_rgb
        self.spatial_transform_map = spatial_transform_map
        self.train = train
        self.mulSeg = mulSeg
        self.numSeg = numSeg
        self.seqLen = seqLen
        self.fmt = fmt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeq = []
        mapSeq = []
        self.spatial_transform.randomize_parameters()
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_name + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            maps_name = vid_name + '/' + 'map' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            mappa = Image.open(f1_name)
            inpSeq.append(self.spatial_transform_rgb(img.convert('RGB')))
            mapSeq.append(self.spatial_transform_map(mappa.convert('L')) #Grayscale
        inpSeq = torch.stack(inpSeq, 0)
        mapSeq = torch.stack(mapSeq, 0)
        return inpSeq, mapSeq, label
