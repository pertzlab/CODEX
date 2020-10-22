###############################################################
# Train CNN for classification, output logs with tensorboard  #
###############################################################
#TODO: Use FixedCrop for test loader?

import torch
import numpy as np
from torch.utils.data import DataLoader
from load_data import DataProcesser
from torchvision import transforms
from models import ConvNetCam, ConvNetCamBi
from class_dataset import myDataset, ToTensor, Subtract, RandomShift, RandomNoise, RandomCrop, FixedCrop
from train_utils import accuracy, AverageMeter
import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import zipfile
import time

# %% Hyperparameters
nepochs = 5
myseed = 7
torch.manual_seed(myseed)
torch.cuda.manual_seed(myseed)
np.random.seed(myseed)

batch_size = 128
length = 750
nclass = 4
nfeatures = 10
lr = 1e-2
selected_classes = []


# %% Load and process Data
data_file = 'data/synthetic_len750.zip'
meas_var = ['FRST', 'SCND']
data = DataProcesser(data_file)
data.subset(sel_groups=meas_var, start_time=0, end_time=750)
if selected_classes:
    data.dataset = data.dataset[data.dataset[data.col_class].isin(selected_classes)]
data.get_stats()
# data.process(method='center_train', independent_groups=True)
data.split_sets()
data_train = myDataset(dataset=data.train_set, transform=transforms.Compose([
    #RandomCrop(output_size=length, ignore_na_tails=True),
    transforms.RandomApply([RandomNoise(mu=0, sigma=0.02)]),
    #Subtract([data.stats['mu']['ERK']['train']]),
    ToTensor()
]))
data_test = myDataset(dataset=data.validation_set, transform=transforms.Compose([
    #RandomCrop(output_size=length, ignore_na_tails=True),
    #Subtract([data.stats['mu']['ERK']['train']]),
    ToTensor()
]))
load_model = None
# load_model = 'path/to/file.pytorch'

# %% Tensorboard logs and model save
file_logs = os.path.splitext(os.path.basename(data_file))[0]  # file name without extension
logs_str = 'logs/' + '_'.join(meas_var) + '/' + datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + \
           '_' + file_logs + '/'
writer = SummaryWriter(logs_str)
save_model = 'models/' + logs_str.lstrip('logs/').rstrip('/') + '.pytorch'

if not os.path.exists(file_logs):
    os.makedirs(file_logs)
if not os.path.exists('models/' + '_'.join(meas_var)):
    os.makedirs('models/' + '_'.join(meas_var))

#%%
def TrainModel(train_loader, test_loader, nepochs, nclass=nclass, load_model=load_model,
               save_model=save_model, logs=True, save_pyfiles=True, lr=lr):
    # ------------------------------------------------------------------------------------------------------------------
    # Model, loss, optimizer
    model = ConvNetCamBi(batch_size=batch_size, nclass=nclass, length=length, nfeatures=nfeatures)
    if load_model:
        model.load_state_dict(torch.load(load_model))
    model.double()
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6, 8], gamma=0.5)
    top1 = AverageMeter()
    top2 = AverageMeter()

    # Create zip archive with all python file at execution time
    if save_pyfiles:
        lpy = [i for i in os.listdir(".") if i.endswith(".py")]
        with zipfile.ZipFile(logs_str + "AllPyFiles.zip", mode='w') as zipMe:
            for file in lpy:
                zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)

    # ------------------------------------------------------------------------------------------------------------------
    # Get adequate size of sample for nn.Conv layers
    # Add a dummy channel dimension for conv1D layer (if multivariate, treat as a 2D plane with 1 channel)
    assert len(train_loader.dataset[0]['series'].shape) == 2
    nchannel, univar_length = train_loader.dataset[0]['series'].shape
    if nchannel == 1:
        view_size = (batch_size, 1, univar_length)
    elif nchannel >= 2:
        view_size = (batch_size, 1, nchannel, univar_length)

    # ------------------------------------------------------------------------------------------------------------------
    # Training loop
    for epoch in range(nepochs):
        scheduler.step()
        model.train()
        top1.reset()
        top2.reset()

        loss_train = []
        for i_batch, sample_batch in enumerate(train_loader):
            series, label = sample_batch['series'], sample_batch['label']
            if cuda_available:
                series, label = series.cuda(), label.cuda()
            series = series.view(view_size)

            prediction = model(series)

            loss = criterion(prediction, label)
            loss_train.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_batch % 25 == 0:
                print('Training epoch: [{0}/{4}][{1}/{2}]; Loss: {3}'.format(epoch + 1, i_batch + 1, len(train_loader),
                                                                             loss, nepochs))

            prec1, prec2 = accuracy(prediction, label, topk=(1, 2))
            top1.update(prec1[0], series.size(0))
            top2.update(prec2[0], series.size(0))

            if i_batch % 100 == 0:
                print('Training Accuracy Epoch: [{0}]\t'
                      'Prec@1 {top1.val.data:.3f} ({top1.avg.data:.3f})\t'
                      'Prec@2 {top2.val.data:.3f} ({top2.avg.data:.3f})'.format(
                    epoch, top1=top1, top2=top2))
            if logs:
                writer.add_scalar('Train/Loss', loss, epoch * len(train_loader) + i_batch + 1)
                writer.add_scalar('Train/Top1', top1.val, epoch * len(train_loader) + i_batch + 1)
                writer.add_scalar('Train/Top2', top2.val, epoch * len(train_loader) + i_batch + 1)
        if logs:
            writer.add_scalar('MeanEpoch/Train_Loss', np.mean(loss_train), epoch)
            writer.add_scalar('MeanEpoch/Train_Top1', top1.avg, epoch)
            writer.add_scalar('MeanEpoch/Train_Top2', top2.avg, epoch)

        # --------------------------------------------------------------------------------------------------------------
        # Evaluation loop
        model.eval()
        top1.reset()
        top2.reset()
        loss_eval = []
        for i_batch, sample_batch in enumerate(test_loader):
            series, label = sample_batch['series'], sample_batch['label']
            if cuda_available:
                series, label = series.cuda(), label.cuda()
            series = series.view(view_size)
            label = torch.autograd.Variable(label)

            prediction = model(series)
            loss = criterion(prediction, label)
            loss_eval.append(loss.cpu().detach().numpy())

            prec1, prec2 = accuracy(prediction, label, topk=(1, 2))
            top1.update(prec1[0], series.size(0))
            top2.update(prec2[0], series.size(0))

        # For validation loss, report only after the whole batch is processed
        if logs:
            writer.add_scalar('Val/Loss', loss, epoch * len(train_loader) + i_batch + 1)
            writer.add_scalar('Val/Top1', top1.val, epoch * len(train_loader) + i_batch + 1)
            writer.add_scalar('Val/Top2', top2.val, epoch * len(train_loader) + i_batch + 1)
            writer.add_scalar('MeanEpoch/Val_Loss', np.mean(loss_eval), epoch)
            writer.add_scalar('MeanEpoch/Val_Top1', top1.avg, epoch)
            writer.add_scalar('MeanEpoch/Val_Top2', top2.avg, epoch)


        print('===>>>\t'
              'Prec@1 ({top1.avg.data:.3f})\t'
              'Prec@2 ({top2.avg.data:.3f})'.format(top1=top1, top2=top2))

    if save_model:
        torch.save(model, save_model)
    return model


# Define train and loader
# Dataloaders
train_loader = DataLoader(dataset=data_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          drop_last=True)
test_loader = DataLoader(dataset=data_test,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4,
                         drop_last=True)

t0 = time.time()
mymodel = TrainModel(train_loader, test_loader, nepochs, nclass)
t1 = time.time()

print('Elapsed time: {}'.format(t1 - t0))
