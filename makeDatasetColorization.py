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
    DatasetF = []
    Labels = []
    Maps = []
    NumFrames = []
    
    for original_dir in sorted(os.listdir(root_dir)):
        if (original_dir=='.DS_Store'): continue
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
                            if (original_dir == 'processed_frames2'):
                                numFrames = len(glob.glob1(inst_dir+'/rgb', '*.png'))
                            if numFrames >= stackSize:

                                if (original_dir == 'flow_x_processed'):
                                    DatasetX.append(inst_dir)
                                if (original_dir == 'flow_y_processed'):
                                    DatasetY.append(inst_dir)
                                if (original_dir == 'processed_frames2'):   
                                    DatasetF.append(inst_dir+'/rgb')
                                    Labels.append(class_id)
                                    Maps.append(inst_dir+'/mmaps')
                                NumFrames.append(numFrames)
                                   
                                
                    class_id += 1
    return DatasetX, DatasetY, DatasetF, Maps, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, sequence=False, stackSize=5,
                 train=True, numSeg=5, fmt='.png', phase='train', seqLen = 25):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.imagesX, self.imagesY, self.imagesF, self.maps, self.labels, self.numFrames = gen_split(
            root_dir, stackSize, phase)
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.spatial_transform0 = spatial_transform
        self.spatial_rgb= Compose([self.spatial_transform0, ToTensor()])
        
        
        self.spatial_transform_map = Compose([self.spatial_transform0, Scale(7), ToTensor(), Binary(0.4)])
        
         
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
        vid_nameF = self.imagesF[idx]
        map_folder = self.maps[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeqSegs = []
        self.spatial_transform0.randomize_parameters()
    
        inpSeqX = []
        inpSeqY = []
     
        inpSeqF = []
        mapSeq = []
        for i in np.linspace(1, numFrame, self.stackSize, endpoint=False):
            fl_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + '.png'
            img = Image.open(fl_name)
            inpSeqX.append(self.spatial_rgb(img.convert('L'), inv=True, flow=True))
            # fl_names.append(fl_name)
            f1_name = vid_nameY + '/flow_y_' + str(int(round(i))).zfill(5) + '.png'
            img2 = Image.open(f1_name)
            inpSeqY.append(self.spatial_rgb(img2.convert('L'), inv=False, flow=True))

            fl_name = vid_nameF + '/' + 'rgb' + str(int(np.floor(i))).zfill(4) + self.fmt
            img = Image.open(fl_name)
            flag=1
            j=i
            while(flag):
                
                maps_name =  map_folder + '/' + 'map' + str(int(np.floor(j))).zfill(4) + self.fmt
                try:
                    mappa = Image.open(maps_name)
                    flag=0
                except:
                    if j<=i:
                        j= 2*i-j+1 #j=i --> j=i +1 ; j=i-1 j-i=-1 --> j=i-(-1)+1
                    else:
                        j= 2*i-j #j=i+1 j-i=1 --> j=i-1
                    continue

            
            inpSeqF.append(self.spatial_rgb(img.convert('RGB')))
            mapSeq.append(self.spatial_transform_map(mappa.convert('L'))) #Grayscale

        inpSeqSegs = torch.stack([torch.stack(inpSeqX, 0).squeeze(1),torch.stack(inpSeqY, 0).squeeze(1)],0).permute(1,0,2,3)

        inpSeqF = torch.stack(inpSeqF, 0)
        mapSeq = torch.stack(mapSeq, 0)
        return inpSeqSegs, inpSeqF, mapSeq, label#, vid_nameF#, fl_name
