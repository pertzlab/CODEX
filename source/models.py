import torch
import torch.nn as nn
from torch.nn.functional import softmax
import pytorch_lightning as pl
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial

#######################################################################################
# Up-to-date architectures. Uses pytorch lightning. Original implementations in pure Pytorch at the end of this file

class LitConvNetCam(pl.LightningModule):

    def __init__(self, batch_size, lr_scheduler_milestones, lr_gamma, nclass, nfeatures, length, lr=1e-2, L2_reg=1e-3, top_acc=1, loss=torch.nn.CrossEntropyLoss()):
        super().__init__()

        self.batch_size = batch_size
        self.nclass = nclass
        self.nfeatures = nfeatures
        self.length = length

        self.loss = loss
        self.lr = lr
        self.lr_scheduler_milestones = lr_scheduler_milestones
        self.lr_gamma = lr_gamma
        self.L2_reg = L2_reg

        # Log hyperparams (all arguments are logged by default)
        self.save_hyperparameters(
            'length',
            'nfeatures',
            'L2_reg',
            'lr',
            'lr_gamma',
            'lr_scheduler_milestones',
            'batch_size',
            'nclass'
        )

        # Metrics to log
        if not top_acc < nclass:
            raise ValueError('`top_acc` must be strictly smaller than `nclass`.')
        self.train_acc = torchmetrics.Accuracy(top_k=top_acc)
        self.val_acc = torchmetrics.Accuracy(top_k=top_acc)
        self.train_f1 = torchmetrics.F1(nclass, average='macro')
        self.val_f1 = torchmetrics.F1(nclass, average='macro')

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Conv1d(in_channels=20, out_channels=nfeatures, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(nfeatures),
            nn.ReLU(True)
        )
        self.pool = nn.AvgPool1d(kernel_size=self.length)
        self.classifier = nn.Sequential(
            nn.Linear(1*nfeatures, nclass),  # 1 because global pooling reduce length of features to 1
            #nn.Softmax(1)  # Already included in nn.CrossEntropy
        )

    @property
    def input_size(self):
        # Add a dummy channel dimension for conv1D layer
        return (self.batch_size, 1, self.length)

    def forward(self, x):
        # (batch_size x length TS)
        x = self.features(x)

        # (batch_size x nfeatures x length_TS)
        # Average pooling for CAM: global pooling so set kernel size to all data
        x = self.pool(x)

        # (batch_size x nfeatures x length_pool; length_pool=1 if global pooling)
        # Flatten features (size batch, lengthpool * nchannels)
        x = x.view(self.batch_size, x.size(2)*self.nfeatures)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.L2_reg)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_scheduler_milestones, gamma=self.lr_gamma),
            'name': 'LearningRate'
        }
        return [optimizer], [lr_scheduler]

    def on_train_start(self):
        # Add mean accuracies per epoch to the hyperparam Tensorboard's tab
        self.logger.log_hyperparams(self.hparams, {'hp/train_acc': 0, 'hp/val_acc': 0})

    def training_step(self, batch, batch_idx):
        series, label = batch['series'], batch['label']
        series = series.view(self.input_size)
        prediction = self(series)
        train_loss = self.loss(prediction, label)
        train_acc = self.train_acc(softmax(prediction, dim=1), label)
        train_f1 = self.train_f1(softmax(prediction, dim=1), label)
        # Add logs of the batch to tensorboard
        self.log('train_loss', train_loss, on_step=True)
        self.log('train_acc', train_acc, on_step=True)
        self.log('train_f1', train_f1, on_step=True)
        return {'loss': train_loss, 'preds': prediction, 'target': label}

    # This hook receive the outputs of all training steps as a list of dictionaries
    def training_epoch_end(self, train_outputs):
        mean_loss = torch.stack([x['loss'] for x in train_outputs]).mean()
        train_acc = self.train_acc.compute()
        train_f1 = self.train_f1.compute()
        self.log('MeanEpoch/train_loss', mean_loss)
        # The .compute() of Torchmetrics objects compute the average of the epoch and reset for next one
        self.log('MeanEpoch/train_acc', train_acc)
        self.log('MeanEpoch/train_f1', train_f1)
        self.log('hp/train_acc', train_acc)

    def validation_step(self, batch, batch_idx):
        series, label = batch['series'], batch['label']
        series = series.view(self.input_size)
        prediction = self(series)
        val_loss = self.loss(prediction, label)
        val_acc = self.val_acc(softmax(prediction, dim=1), label)
        val_f1 = self.val_f1(softmax(prediction, dim=1), label)
        self.log('MeanEpoch/val_acc', val_acc, on_epoch=True, prog_bar=True)
        self.log('MeanEpoch/val_f1', val_f1, on_epoch=True)
        self.log('hp/val_acc', val_acc)
        return {'loss': val_loss, 'preds': prediction, 'target': label}

    # This hook receive the outputs of all validation steps as a list of dictionaries
    def validation_epoch_end(self, val_outputs):
        mean_loss = torch.stack([x['loss'] for x in val_outputs]).mean()
        # Create a figure of the confmat that is loggable
        preds = torch.cat([softmax(x['preds'], dim=1) for x in val_outputs])
        target = torch.cat([x['target'] for x in val_outputs])
        confmat = torchmetrics.functional.confusion_matrix(preds, target, normalize=None, num_classes=self.nclass)
        df_confmat = pd.DataFrame(confmat.cpu().numpy(), index = range(self.nclass), columns = range(self.nclass))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_confmat, annot=True, cmap='Blues').get_figure()
        plt.close(fig_)

        self.log('MeanEpoch/val_loss', mean_loss)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


class LitConvNetCamBi(pl.LightningModule):

    def __init__(self, batch_size, lr_scheduler_milestones, lr_gamma, nclass, nfeatures, length, lr=1e-2, L2_reg=1e-3, top_acc=1, loss=torch.nn.CrossEntropyLoss()):
        super().__init__()

        self.batch_size = batch_size
        self.nclass = nclass
        self.nfeatures = nfeatures
        self.length = length

        self.loss = loss
        self.lr = lr
        self.lr_scheduler_milestones = lr_scheduler_milestones
        self.lr_gamma = lr_gamma
        self.L2_reg = L2_reg

        # Log hyperparams (all arguments are logged by default)
        self.save_hyperparameters(
            'length',
            'nfeatures',
            'L2_reg',
            'lr',
            'lr_gamma',
            'lr_scheduler_milestones',
            'batch_size',
            'nclass'
        )

        # Metrics to log
        if not top_acc < nclass:
            raise ValueError('`top_acc` must be strictly smaller than `nclass`.')
        self.train_acc = torchmetrics.Accuracy(top_k=top_acc)
        self.val_acc = torchmetrics.Accuracy(top_k=top_acc)
        self.train_f1 = torchmetrics.F1(nclass, average='macro')
        self.val_f1 = torchmetrics.F1(nclass, average='macro')

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=nfeatures, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(nfeatures),
            nn.ReLU(True)
        )
        self.pool = nn.AvgPool2d(kernel_size=(2, self.length))
        self.classifier = nn.Sequential(
            nn.Linear(1*nfeatures, nclass),  # 1 because global pooling reduce length of features to 1
            #nn.Softmax(1)  # Already included in nn.CrossEntropy
        )

    @property
    def input_size(self):
        # Add a dummy channel dimension for conv1D layer
        return (self.batch_size, 1, 2, self.length)

    def forward(self, x):
        # (batch_size x length TS)
        x = self.features(x)

        # (batch_size x nfeatures x length_TS)
        # Average pooling for CAM: global pooling so set kernel size to all data
        x = self.pool(x)

        # (batch_size x nfeatures x length_pool; length_pool=1 if global pooling)
        # Flatten features (size batch, lengthpool * nchannels)
        x = x.view(self.batch_size, x.size(2)*self.nfeatures)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.L2_reg)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_scheduler_milestones, gamma=self.lr_gamma),
            'name': 'LearningRate'
        }
        return [optimizer], [lr_scheduler]

    def on_train_start(self):
        # Add mean accuracies per epoch to the hyperparam Tensorboard's tab
        self.logger.log_hyperparams(self.hparams, {'hp/train_acc': 0, 'hp/val_acc': 0})

    def training_step(self, batch, batch_idx):
        series, label = batch['series'], batch['label']
        series = series.view(self.input_size)
        prediction = self(series)
        train_loss = self.loss(prediction, label)
        train_acc = self.train_acc(softmax(prediction, dim=1), label)
        train_f1 = self.train_f1(softmax(prediction, dim=1), label)
        # Add logs of the batch to tensorboard
        self.log('train_loss', train_loss, on_step=True)
        self.log('train_acc', train_acc, on_step=True)
        self.log('train_f1', train_f1, on_step=True)
        return {'loss': train_loss, 'preds': prediction, 'target': label}

    # This hook receive the outputs of all training steps as a list of dictionaries
    def training_epoch_end(self, train_outputs):
        mean_loss = torch.stack([x['loss'] for x in train_outputs]).mean()
        train_acc = self.train_acc.compute()
        train_f1 = self.train_f1.compute()
        self.log('MeanEpoch/train_loss', mean_loss)
        # The .compute() of Torchmetrics objects compute the average of the epoch and reset for next one
        self.log('MeanEpoch/train_acc', train_acc)
        self.log('MeanEpoch/train_f1', train_f1)
        self.log('hp/train_acc', train_acc)

    def validation_step(self, batch, batch_idx):
        series, label = batch['series'], batch['label']
        series = series.view(self.input_size)
        prediction = self(series)
        val_loss = self.loss(prediction, label)
        val_acc = self.val_acc(softmax(prediction, dim=1), label)
        val_f1 = self.val_f1(softmax(prediction, dim=1), label)
        self.log('MeanEpoch/val_acc', val_acc, on_epoch=True, prog_bar=True)
        self.log('MeanEpoch/val_f1', val_f1, on_epoch=True)
        self.log('hp/val_acc', val_acc)
        return {'loss': val_loss, 'preds': prediction, 'target': label}

    # This hook receive the outputs of all validation steps as a list of dictionaries
    def validation_epoch_end(self, val_outputs):
        mean_loss = torch.stack([x['loss'] for x in val_outputs]).mean()
        # Create a figure of the confmat that is loggable
        preds = torch.cat([softmax(x['preds'], dim=1) for x in val_outputs])
        target = torch.cat([x['target'] for x in val_outputs])
        confmat = torchmetrics.functional.confusion_matrix(preds, target, normalize=None, num_classes=self.nclass)
        df_confmat = pd.DataFrame(confmat.cpu().numpy(), index = range(self.nclass), columns = range(self.nclass))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_confmat, annot=True, cmap='Blues').get_figure()
        plt.close(fig_)

        self.log('MeanEpoch/val_loss', mean_loss)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items



#######################################################################################
# Original implementation, in pure Pytorch. Use the Pytorch-lightning versions instead

class ConvNetCam(nn.Module):

    def __init__(self, batch_size, nclass=7, nfeatures=20, length=120):
        super(ConvNetCam, self).__init__()

        self.batch_size = batch_size
        self.nclass = nclass
        self.nfeatures = nfeatures
        self.length = length

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(20),
            nn.ReLU(True),
            nn.Conv1d(in_channels=20, out_channels=nfeatures, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(nfeatures),
            nn.ReLU(True)
        )
        self.pool = nn.AvgPool1d(kernel_size=self.length)
        self.classifier = nn.Sequential(
            nn.Linear(1*nfeatures, nclass),  # 1 because global pooling reduce length of features to 1
            #nn.Softmax(1)  # Already included in nn.CrossEntropy
        )

    def forward(self, x):
        # (batch_size x length TS)
        x = self.features(x)

        # (batch_size x nfeatures x length_TS)
        # Average pooling for CAM: global pooling so set kernel size to all data
        x = self.pool(x)

        # (batch_size x nfeatures x length_pool; length_pool=1 if global pooling)
        # Flatten features (size batch, lengthpool * nchannels)
        x = x.view(self.batch_size, x.size(2)*self.nfeatures)
        x = self.classifier(x)
        return x


class ConvNetCamBi(nn.Module):
    # Consider the bivariate series as a 2D image

    def __init__(self, batch_size, nclass=7, nfeatures=20, length=120):
        super(ConvNetCamBi, self).__init__()

        self.batch_size = batch_size
        self.nclass = nclass
        self.nfeatures = nfeatures
        self.length = length

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3,5), stride=1, padding=(1,2)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.Conv2d(in_channels=20, out_channels=nfeatures, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(nfeatures),
            nn.ReLU(True)
        )
        self.pool = nn.AvgPool2d(kernel_size=(2, self.length))
        self.classifier = nn.Sequential(
            nn.Linear(1*nfeatures, nclass),  # 1 because global pooling reduce length of features to 1
            #nn.Softmax(1)  # Already included in nn.CrossEntropy
        )

    def forward(self, x):
        # (batch_size x number_pixel_row x length_TS)
        x = self.features(x)

        # (batch_size x nfeatures x number_pixel_row x length_TS)
        # Average pooling for CAM: global pooling so set kernel size to all data
        x = self.pool(x)

        # (batch_size x nfeatures x length_pool; length_pool=1 if global)
        # Flatten features (batch_size, length_pool * nchannels)
        x = x.view(self.batch_size, x.size(2)*self.nfeatures)
        x = self.classifier(x)
        return x


#######################################################################################
# ResNet architectures, pure Pytorch

# Adapted from https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278 and https://www.bigrabbitdata.com/pytorch-17-residual-network-resnet-explained-in-detail-with-implementation-cifar10/
# -----------------------------------------------------------------------------
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=stride, padding=1, bias=False)
def conv3x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3,5), stride=stride, padding=1, bias=False)

# -----------------------------------------------------------------------------
class ResidualBlockSkip2Bi(nn.Module):
    '''
        Conv--> Batchnorm-->ReLu-->Conv-->Batchnorm--> 
        Only downsample if needed
    '''
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlockSkip2Bi, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# -----------------------------------------------------------------------------
class ResNetBi(nn.Module):
    def __init__(self, block, batch_size, nclass=7, nfeatures=20, length=120, nResblocks_layer1=2, nResblocks_layer2=2, nResblocks_layer3=2):
        super(ResNetBi, self).__init__()
        self.batch_size = batch_size
        self.nclass = nclass
        self.length = length
        self.nfeatures = nfeatures

        self.in_channels = 20
        self.layers = [nResblocks_layer1, nResblocks_layer2, nResblocks_layer3]
        # For the initial convolution layer, without residuals. Consider one channel with 2 rows of pixels
        self.conv = conv3x3(in_channels=1, out_channels=20)
        self.bn = nn.BatchNorm2d(20)
        self.relu = nn.ReLU(inplace=True)
        # Build the residual layers, each layer comprises several residual blocks
        self.layer1 = self.make_layer(block, out_channels=20, nResblocks=self.layers[0], stride=1)
        self.layer2 = self.make_layer(block, out_channels=20, nResblocks=self.layers[1], stride=1)
        self.layer3 = self.make_layer(block, out_channels=nfeatures, nResblocks=self.layers[2], stride=1)

        # Put all residual blocks into one container
        self.features = nn.Sequential(
            self.conv,
            self.bn,
            self.relu,
            self.layer1,
            self.layer2,
            self.layer3
        )

        # Flatten the output of the residuals blocks into a 1d vector 
        self.pool = nn.AvgPool2d(kernel_size=(2, self.length))
        self.classifier = nn.Sequential(
            nn.Linear(1*nfeatures, nclass),  # 1 because global pooling reduce length of features to 1
            #nn.Softmax(1)  # Already included in nn.CrossEntropy
        )
        
    def make_layer(self, blocktype, out_channels, nResblocks, stride=1):
        downsample = None
        # Only downsample when stride is not 2 
        # or when input channel doesn't match output channel
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(conv3x3(self.in_channels, out_channels, stride=stride),
                                       nn.BatchNorm2d(out_channels))
        layers = []
        # append the first residual block for each layer
        # downsample the image if needed
        layers.append(blocktype(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        # append the suceeding residual block in each layer
        for ii in range(1, nResblocks):
            layers.append(blocktype(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.features(x)
        out = self.pool(out)
        out = out.view(self.batch_size, out.size(2)*self.nfeatures)
        out = self.classifier(out)
        return out

