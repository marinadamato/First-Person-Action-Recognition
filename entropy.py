import torch
import torch.nn as nn
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from attentionmodel_ml import *
from colorization_block import colorization
from makeDatasetColorization import makeDataset
import argparse
import sys
import os
from tensorboardX import SummaryWriter


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = torch.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


def main_run(dataset, train_data_dir, stage1_dict, out_dir, seqLen, trainBatchSize,
             numEpochs, lr1, decay_factor, decay_step, memSize):

    if dataset == 'gtea61':
        num_classes = 61
    elif dataset == 'gtea71':
      num_classes = 71
    elif dataset == 'gtea_gaze':
        num_classes = 44
    elif dataset == 'egtea':
        num_classes = 106
    else:
        print('Dataset not found')
        sys.exit()

    
    model_folder= os.path.join('./', out_dir, dataset, 'Color','entropy')
    # Create the dir
    if os.path.exists(model_folder):
        print('Directory {} exists!'.format(model_folder))
        sys.exit()
    os.makedirs(model_folder)

    # Log files
    
    train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')


    # Data loader
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224)])

    vid_seq_train = makeDataset(train_data_dir,
                                spatial_transform=spatial_transform, seqLen=seqLen, fmt='.png',phase='train')

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                            shuffle=True, num_workers=4, pin_memory=True)
    

    train_params = []

    model = colorization(num_classes=num_classes, mem_size=memSize)
    model.attML.load_state_dict(torch.load(stage1_dict))
    model.train(True)
    model.attML.train(False)

    for params in model.bn1.parameters():
        params.requires_grad = True
        train_params += [params]
    for params in model.relu.parameters():
        params.requires_grad = True
        train_params += [params]
    for params in model.maxpool.parameters():
        params.requires_grad = True
        train_params += [params]
   
    for params in model.residual_block.parameters():
        params.requires_grad = True
        train_params += [params]
    for params in model.conv2.parameters():
        params.requires_grad = True
        train_params += [params]
    for params in model.deconv.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.attML.parameters():
        params.requires_grad = False

    model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer_fn = torch.optim.Adam(train_params, lr=lr1, weight_decay=4e-5, eps=1e-4)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=decay_step,
                                                           gamma=decay_factor)
    min_loss=float('inf')
    train_iter = 0
    

    for epoch in range(numEpochs):
        model.train(True)
        model.attML.train(False)
        
        epoch_loss = 0
        iterPerEpoch = 0
        
        
        for i, (flow, _ ,binary_map, targets) in enumerate(train_loader):
            
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            flow = flow.permute(1, 0, 2, 3, 4).cuda()
            logit=model(flow,True)
            loss=HLoss()(logit)
            loss.backward()
            optimizer_fn.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss/iterPerEpoch
        
        
        if loss<min_loss:
            min_loss=loss
            save_path_model = (model_folder + '/model_rgb_state_dict.pth')
            torch.save(model.state_dict(), save_path_model)
        print('Train: Epoch = {} | Loss = {} '.format(epoch+1, avg_loss))
       
        train_log_loss.write('Train Loss after {} epochs = {}\n'.format(epoch + 1, avg_loss))

        optim_scheduler.step()

    train_log_loss.close()


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--trainDatasetDir', type=str, default='./dataset/gtea_warped_flow_61/split2/train',
                        help='Train set directory')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--stage1Dict', type=str, default='./experiments/gtea61/rgb/stage1/best_model_state_dict.pth',
                        help='Stage 1 model path')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--numEpochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--stepSize', type=float, default=[25, 75, 150], nargs="+", help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')
    

    args = parser.parse_args()

    dataset = args.dataset
    trainDatasetDir = args.trainDatasetDir
    outDir = args.outDir
    stage1Dict = args.stage1Dict
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    numEpochs = args.numEpochs
    lr1 = args.lr
    stepSize = args.stepSize
    decayRate = args.decayRate
    memSize = args.memSize

    main_run(dataset, trainDatasetDir, stage1Dict, outDir, seqLen, trainBatchSize,
             numEpochs, lr1, decayRate, stepSize, memSize)

__main__()
    
