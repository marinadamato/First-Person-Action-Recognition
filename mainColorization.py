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

def main_run(dataset, stage, train_data_dir, val_data_dir, stage1_dict, out_dir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decay_factor, decay_step, memSize):

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

    model_folder = os.path.join('./', out_dir, dataset, 'Color')  # Dir for saving models and log files
    # Create the dir
    if os.path.exists(model_folder):
        print('Directory {} exists!'.format(model_folder))
        sys.exit()
    os.makedirs(model_folder)

    # Log files
    writer = SummaryWriter(model_folder)
    train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')
    train_log_acc = open((model_folder + '/train_log_acc.txt'), 'w')
    val_log_loss = open((model_folder + '/val_log_loss.txt'), 'w')
    val_log_acc = open((model_folder + '/val_log_acc.txt'), 'w')
    train_log_loss_ms= open((model_folder + '/train_log_loss_ms.txt'), 'w')
    val_log_loss_ms = open((model_folder + '/val_log_loss_ms.txt'), 'w')

    # Data loader
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224)])

    vid_seq_train = makeDataset(train_data_dir,
                                spatial_transform=spatial_transform, seqLen=seqLen, fmt='.png',phase='train')

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                            shuffle=True, num_workers=4, pin_memory=True)
    if val_data_dir is not None:

        vid_seq_val = makeDataset(val_data_dir,
                                   spatial_transform=Compose([Scale(256), CenterCrop(224)]),
                                   seqLen=seqLen, fmt='.png',phase='test')

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                shuffle=False, num_workers=2, pin_memory=True)
        valInstances = vid_seq_val.__len__()


    trainInstances = vid_seq_train.__len__()

    train_params = []
    if stage == 1:
        raise "Stage must be 2"
    else:

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

    train_iter = 0
    min_accuracy = 0

    for epoch in range(numEpochs):
        model.train(True)
        model.attML.train(False)
        optim_scheduler.step()
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        epoch_loss_ms = 0
        
        writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'], epoch+1)
        for i, (flow, _ ,binary_map, targets) in enumerate(train_loader):
            
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            inputVariable = Variable(flow.permute(1, 0, 2, 3, 4).cuda())
            labelVariable = Variable(targets.cuda())
            trainSamples += flow.size(0)
            output_label, output_ms = model(inputVariable)
            
            loss = loss_fn(output_label, labelVariable)
            
            if stage==2 :
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            binary_map = Variable(binary_map.permute(1, 0, 2, 3, 4).type(torch.LongTensor).cuda())
            binary_map =binary_map.contiguous().view(-1)
            output_ms = output_ms.contiguous().view(-1,2)            
            
            if stage==2:
                loss_ms=loss_fn(output_ms, binary_map)
                loss_ms.backward()
                epoch_loss_ms+=loss_ms.item()
        
            optimizer_fn.step()
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += torch.sum(predicted == labelVariable.data).data.item()
            epoch_loss += loss.item()
        avg_loss = epoch_loss/iterPerEpoch
        if stage ==2:
            avg_loss_ms= epoch_loss_ms/iterPerEpoch
            #avg_loss = avg_loss + avg_loss_ms
            train_log_loss_ms.write('Train Loss MS after {} epochs = {}\n'.format(epoch + 1, avg_loss_ms))
        trainAccuracy = (numCorrTrain / trainSamples) * 100

        print('Train: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch+1, avg_loss, trainAccuracy))
        writer.add_scalar('train/epoch_loss', avg_loss, epoch+1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)
        train_log_loss.write('Train Loss after {} epochs = {}\n'.format(epoch + 1, avg_loss))
        
        train_log_acc.write('Train Accuracy after {} epochs = {}%\n'.format(epoch + 1, trainAccuracy))
        if val_data_dir is not None:
            if (epoch+1) % 1 == 0:
                model.train(False)
                val_loss_epoch = 0
                val_iter = 0
                val_samples = 0
                numCorr = 0
                epoch_loss_ms_val = 0
                for j, (flow, _, binary_map, targets) in enumerate(val_loader):
                    val_iter += 1
                    val_samples += flow.size(0)
                    
                    inputVariable = Variable(flow.permute(1, 0, 2, 3, 4).cuda(), volatile=True)
                    labelVariable = Variable(targets.cuda(async=True), volatile=True)
                    output_label, output_ms = model(inputVariable)
                    val_loss = loss_fn(output_label, labelVariable)
                    val_loss_epoch += val_loss.data[0]
                    binary_map = Variable(binary_map.permute(1, 0, 2, 3, 4).type(torch.LongTensor).cuda())
                    binary_map = binary_map.contiguous().view(-1)
                    output_ms = output_ms.contiguous().view(-1,2)
                    
                    if stage==2:
                        loss_ms=loss_fn(output_ms, binary_map)
                        
                        epoch_loss_ms_val+=loss_ms.item()
                                
                    _, predicted = torch.max(output_label.data, 1)
                    numCorr += torch.sum(predicted == labelVariable.data).data.item()
                val_accuracy = (numCorr / val_samples) * 100
                avg_val_loss = val_loss_epoch / val_iter
                if stage ==2:
                    avg_loss_ms= epoch_loss_ms/iterPerEpoch
                    #avg_loss = avg_loss + avg_loss_ms 
                    val_log_loss_ms.write('Val Loss MS after {} epochs = {}\n'.format(epoch + 1, avg_loss_ms))
                print('Val: Epoch = {} | Loss {} | Accuracy = {}'.format(epoch + 1, avg_val_loss, val_accuracy))
                writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
                writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
                
                val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))
                val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))
                if val_accuracy > min_accuracy:
                    save_path_model = (model_folder + '/model_rgb_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)
                    min_accuracy = val_accuracy
            else:
                if (epoch+1) % 10 == 0:
                    save_path_model = (model_folder + '/model_rgb_state_dict_epoch' + str(epoch+1) + '.pth')
                    torch.save(model.state_dict(), save_path_model)

    train_log_loss.close()
    train_log_acc.close()
    val_log_acc.close()
    val_log_loss.close()
    train_log_loss_ms.close()
    val_log_loss_ms.close()
    writer.export_scalars_to_json(model_folder + "/all_scalars.json")
    writer.close()


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--stage', type=int, default=1, help='Training stage')
    parser.add_argument('--trainDatasetDir', type=str, default='./dataset/gtea_warped_flow_61/split2/train',
                        help='Train set directory')
    parser.add_argument('--valDatasetDir', type=str, default=None,
                        help='Val set directory')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--stage1Dict', type=str, default='./experiments/gtea61/rgb/stage1/best_model_state_dict.pth',
                        help='Stage 1 model path')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--valBatchSize', type=int, default=64, help='Validation batch size')
    parser.add_argument('--numEpochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--stepSize', type=float, default=[25, 75, 150], nargs="+", help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')

    args = parser.parse_args()

    dataset = args.dataset
    stage = args.stage
    trainDatasetDir = args.trainDatasetDir
    valDatasetDir = args.valDatasetDir
    outDir = args.outDir
    stage1Dict = args.stage1Dict
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    numEpochs = args.numEpochs
    lr1 = args.lr
    stepSize = args.stepSize
    decayRate = args.decayRate
    memSize = args.memSize
    

    main_run(dataset, stage, trainDatasetDir, valDatasetDir, stage1Dict, outDir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decayRate, stepSize, memSize)

__main__()
    
