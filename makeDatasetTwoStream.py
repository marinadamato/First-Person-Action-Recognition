import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import glob
import sys


def gen_split(root_dir, stackSize, phase):
    DatasetX = []
    DatasetY = []
    DatasetF = []
    Labels = []
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
                                NumFrames.append(numFrames)
                    class_id += 1
    return DatasetX, DatasetY, DatasetF, Labels, NumFrames

class makeDataset(Dataset):
    def __init__(self, root_dir, spatial_transform=None, sequence=False, stackSize=5,
                 train=True, numSeg=5, fmt='.png', phase='train', seqLen = 25):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.imagesX, self.imagesY, self.imagesF, self.labels, self.numFrames = gen_split(
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
        vid_nameF = self.imagesF[idx]
        label = self.labels[idx]
        numFrame = self.numFrames[idx]
        inpSeqSegs = []
        self.spatial_transform.randomize_parameters()
        if self.sequence is True:
            if numFrame <= self.stackSize:
                frameStart = np.ones(self.numSeg)
            else:
                frameStart = np.linspace(1, numFrame - self.stackSize, self.numSeg)
            for startFrame in frameStart:
                inpSeq = []
                for k in range(self.stackSize):
                    i = k + int(startFrame)
                    fl_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + '.png'
                    img = Image.open(fl_name)
                    inpSeq.append(self.spatial_transform(img.convert('L'), inv=True, flow=True))
                    # fl_names.append(fl_name)
                    fl_name = vid_nameY + '/flow_y_' + str(int(round(i))).zfill(5) + '.png'
                    img = Image.open(fl_name)
                    inpSeq.append(self.spatial_transform(img.convert('L'), inv=False, flow=True))
                inpSeqSegs.append(torch.stack(inpSeq, 0).squeeze())
            inpSeqSegs = torch.stack(inpSeqSegs, 0)
        else:
            if numFrame <= self.stackSize:
                startFrame = 1
            else:
                if self.phase == 'train':
                    startFrame = random.randint(1, numFrame - self.stackSize)
                else:
                    startFrame = np.ceil((numFrame - self.stackSize)/2)
            inpSeq = []
            for k in range(self.stackSize):
                i = k + int(startFrame)
                fl_name = vid_nameX + '/flow_x_' + str(int(round(i))).zfill(5) + '.png'
                img = Image.open(fl_name)
                inpSeq.append(self.spatial_transform(img.convert('L'), inv=True, flow=True))
                # fl_names.append(fl_name)
                fl_name = vid_nameY + '/flow_y_' + str(int(round(i))).zfill(5) + '.png'
                img = Image.open(fl_name)
                inpSeq.append(self.spatial_transform(img.convert('L'), inv=False, flow=True))
            inpSeqSegs = torch.stack(inpSeq, 0).squeeze(1)
        inpSeqF = []
        for i in np.linspace(1, numFrame, self.seqLen, endpoint=False):
            fl_name = vid_nameF + '/' + 'image_' + str(int(np.floor(i))).zfill(5) + self.fmt
            img = Image.open(fl_name)
            inpSeqF.append(self.spatial_transform(img.convert('RGB')))
        inpSeqF = torch.stack(inpSeqF, 0)
        return inpSeqSegs, inpSeqF, label#, vid_nameF#, fl_name
