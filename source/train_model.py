###############################################################
# Train CNN for classification, output logs with tensorboard  #
###############################################################
# Standard libraries
import argparse
import datetime
import os
import sys
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
from torchvision import transforms

# Custom functions/classes
path_to_module = '../source'  # Path where all the .py files are, relative to the notebook folder
sys.path.append(path_to_module)
from models import LitConvNetCam, LitConvNetCamBi
from class_dataset import (FixedCrop, RandomCrop, RandomNoise, RandomShift,
                           Subtract, ToTensor, myDataset)
from load_data import DataProcesser
from train_utils import AverageMeter, accuracy, even_intervals

# %% Train
def makeParser():
    parser = argparse.ArgumentParser(description='Train a ConvNetCam model with Adam optimizer and MultiStep learning rate reduction.')

    # For the model
    parser.add_argument('-c', '--nclass', help='Number of classes in the data. If None, automatically detects the number of unique values in the class column of the dataset. Default: None.', type=int, default=None)
    parser.add_argument('-l', '--length', help='Length of the input time-series. If the input is multivariate, each channel will have the specified length. If None, automatically detects the longest common length across al trajectories. Default: None.', type=int, default=None)
    parser.add_argument('-f', '--nfeatures', help='Number of features of the input representation before classification. Hint: [5-10], increase if model underfits, decrease if model overfits. Default: 10.', type=int, default=10)
    parser.add_argument('-b', '--batch', help='Batch size. Hint: usually a power of 2. Default: 128.', type=int, default=128)
    parser.add_argument('-r', '--lr', help='Initial learning rate. Default: 0.01', type=float, default=0.01)
    parser.add_argument('-s', '--schedule', help='Scheduled epochs at which the learning rate will be reduced. If None, will evenly divide the number of epochs into 3. Default: None.', type=int, nargs='+', default=None)
    parser.add_argument('-g', '--gamma', help='Factor by which the learning rate is multiplied at the specified epochs. Default: 0.1.', type=float, default=0.1)
    parser.add_argument('-p', '--penalty', help='L2 penalty weigth. Hint: increase if model overfits. Default: 0.001.', type=float, default=0.001)
    # For the data
    parser.add_argument('-d', '--data', help='Path to the data zip file.', type=str, default=None)
    parser.add_argument('-m', '--measurement', help='Names of the measurement variables. In DataProcesser convention, this is the prefix in a column name that contains a measurement (time being the suffix). Pay attention to the order since this is how the dimensions of a sample of data will be ordered (i.e. 1st in the list will form 1st row of measurements in the sample, 2nd is the 2nd, etc...) If None, DataProcesser will extract automatically the measurement names and use the order of appearance in the column names. Default: None.', type=str, nargs='+')
    parser.add_argument('--startTime', help='Use to subset data to a specific time range. If None, DataProcesser will extract automatically the range of time in the dataset columns. Useful to completely exclude some acquisition times where irrelevant measures are acquired. Default: None.', type=int, default=None)
    parser.add_argument('--endTime', help='Use to subset data to a specific time range. If None, DataProcesser will extract automatically the range of time in the dataset columns. Useful to completely exclude some acquisition times where irrelevant measures are acquired. Default: None.', type=int, default=None)
    # For the trainer
    parser.add_argument('-e', '--nepochs', help='Number of training epochs. Default: 3.', type=int, default=3)
    parser.add_argument('--ngpu', help='Number of GPU to use for training. 0 means CPU-only; -1 means all available GPU. Default: -1.', type=int, default=-1)
    parser.add_argument('--ncpuLoad', help='Number of CPU cores to use for loading the data before passing to the model. Default: 1.', type=int, default=1)
    # Logs and reproducibility
    parser.add_argument('--logdir', help='Path to directory where to store the logs and the models. Default: "./logs/"', type=str, default='./logs/')
    parser.add_argument('--seed', help='Seed random numbers for reproducibility. Default: 7.', type=int, default=7)
    # Handle boolean argument
    imba_parser = parser.add_mutually_exclusive_group(required=False)
    imba_parser.add_argument('--imba', dest='imba', action='store_true', help='If the option is passed, correct for imbalanced data (i.e. a dataset where the number of individuals between the classes is not well-balanced). This will over-sample the underrepresented class and under-sample the overrepresented classes. See `https://github.com/ufoym/imbalanced-dataset-sampler`. Default: False.')
    imba_parser.add_argument('--no-imba', dest='imba', action='store_false')
    parser.set_defaults(imba=False)

    # Add the flags of pytorch_lightning trainer
    parser = pl.Trainer.add_argparse_args(parser)

    return parser


def makeLogger(args, dir_logs='logs/', subdir_logs='sublogs/', file_logs=None):
    file_data = os.path.splitext(os.path.basename(args.data))[0]  # file name without extension
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H__%M__%S')

    dir_logs = dir_logs
    if subdir_logs is None:
        subdir_logs = '_'.join(args.measurement)
    if file_logs is None:
        file_logs =  '_'.join([timestamp, file_data])

    if not os.path.exists(dir_logs):
        os.makedirs(dir_logs)

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=dir_logs, name=subdir_logs, version=file_logs, default_hp_metric=False)

    return tb_logger


def makeConfigs(args):
    # Convert args to dictionary
    dargs = vars(args)

    config_model = {
    'length': dargs['length'],
    'nclass': dargs['nclass'],
    'nfeatures': dargs['nfeatures'],
    'batch_size': dargs['batch'],
    'lr': dargs['lr'],
    'lr_scheduler_milestones': dargs['schedule'],
    'lr_gamma': dargs['gamma'],
    'L2_reg': dargs['penalty']
    }

    config_data = {
    'data_file': dargs['data'],
    'meas_var': dargs['measurement'],
    'start_time': dargs['startTime'],
    'end_time': dargs['endTime']
    }

    # Add the flags of pytorch_lightning trainer, so can use any option of pl
    custom_keys = ['length', 'nclass', 'nfeatures','batch', 'lr', 'schedule',
     'gamma', 'penalty', 'data', 'measurement', 'startTime', 'endTime', 'ngpu', 'nepochs', 'logdir', 'seed', 'imba', 'no-imba', 'ncpuLoad']
    pl_keys = set(dargs.keys()).difference(custom_keys)
    config_trainer = {k:dargs[k] for k in pl_keys}
    # Overwrite the default with manually passed values
    config_trainer['max_epochs'] = dargs['nepochs']
    config_trainer['min_epochs'] = dargs['nepochs']
    config_trainer['gpus'] = dargs['ngpu']

    return config_model, config_data, config_trainer


def makeLoaders(args, return_nclass=False, return_length=False, return_measurement=False, return_start=False, return_end=False):
    data = DataProcesser(args.data, datatable=False)

    # Select measurements and times, subset classes and split the dataset
    meas_var = data.detect_groups_times()['groups'] if args.measurement is None else args.measurement
    start_time = data.detect_groups_times()['times'][0] if args.startTime is None else args.startTime
    end_time = data.detect_groups_times()['times'][1] if args.endTime is None else args.endTime
    # Auto detect
    nclass = data.dataset[data.col_class].nunique() if args.nclass is None else args.nclass
    length = data.get_max_common_length() if args.length is None else args.length

    data.subset(sel_groups=meas_var, start_time=start_time, end_time=end_time)
    data.get_stats()
    data.split_sets()

    # Input preprocessing, this is done sequentially, on the fly when the input is passed to the network
    average_perChannel = [data.stats['mu'][meas]['train'] for meas in meas_var]
    ls_transforms = transforms.Compose([
        RandomCrop(output_size=length, ignore_na_tails=True),
        Subtract(average_perChannel),
        ToTensor()])

    # Define the dataset objects that associate data to preprocessing and define the content of a batch
    # A batch of myDataset contains: the trajectories, the trajectories identifier and the trajecotires class identifier
    data_train = myDataset(dataset=data.train_set, transform=ls_transforms)
    data_validation = myDataset(dataset=data.validation_set, transform=ls_transforms)

    if args.batch > len(data_train) or args.batch > len(data_validation):
        raise ValueError('Batch size ({}) must be smaller than the number of trajectories in the training ({}) and the validation ({}) sets.'.format(args.batch, len(data_train), len(data_validation)))

    # Quick recap of the data content
    print('Channels order: {} \nTime range: ({}, {}) \nClasses: {}'.format(meas_var, start_time, end_time, list(data.dataset[data.col_class].unique())))
    nclass_data = len(list(data.dataset[data.col_class].unique()))
    if nclass != nclass_data:
        warnings.warn('The number of classes in the model output ({}) is not equal to the number of classes in the data ({}).'.format(nclass, nclass_data))

    if args.imba:
        print('Attempting to handle classes imbalance.')
        train_loader = DataLoader(
            dataset=data_train,
            batch_size=args.batch,
            sampler=ImbalancedDatasetSampler(data_train),
            num_workers=args.ncpuLoad,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            dataset=data_train,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.ncpuLoad,
            drop_last=True
        )

    validation_loader = DataLoader(
        dataset=data_validation,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.ncpuLoad,
        drop_last=True
    )

    out = {
        'train_loader': train_loader,
        'validation_loader': validation_loader
    }
    if return_measurement:
        out['measurement'] = meas_var
    if return_start:
        out['start_time'] = start_time
    if return_end:
        out['end_time'] = end_time
    if return_nclass:
        out['nclass'] = nclass
    if return_length:
        out['length'] = length
    return out


def main(config_model, config_trainer, train_loader, validation_loader, nmeasurement, file_model):
    if nmeasurement == 1:
        model = LitConvNetCam(**config_model)
    elif nmeasurement == 2:
        model = LitConvNetCamBi(**config_model)
    else:
        raise NotImplementedError('This script is intended for monovariate and bivariate measurements only.\
             To extend to higher dimensions, create the appropriate model and call it the main() function.')
    model.double()
    trainer = pl.Trainer(**config_trainer)
    trainer.fit(model, train_loader, validation_loader)
    torch.save(model, file_model)


if __name__ == '__main__':
    parser = makeParser()
    args = parser.parse_args()
    pl.utilities.seed.seed_everything(args.seed)
    config_model, config_data, config_trainer = makeConfigs(args)

    loaders = makeLoaders(
        args,
        return_measurement=True,
        return_start=False,
        return_end=False,
        return_length=True,
        return_nclass=True
    )
    train_loader = loaders['train_loader']
    validation_loader = loaders['validation_loader']
    measurement = loaders['measurement']
    max_common_length = loaders['length']
    nclass = loaders['nclass']

    mylogger = makeLogger(
        args,
        dir_logs=args.logdir,
        subdir_logs='_'.join(measurement)
    )
    # Save the final model in a pytorch format
    file_model = mylogger.log_dir + '.pytorch'

    # Update the defaults
    update_model = {}
    if args.length is None:
        update_model['length'] = max_common_length
        print('Max common length detected: {}'.format(max_common_length))
    if args.nclass is None:
        update_model['nclass'] = nclass
        print('Number of classes detected: {}'.format(nclass))
    if args.schedule is None:
        update_model['lr_scheduler_milestones'] = even_intervals(args.nepochs, ninterval=3)
    config_model.update(update_model)

    update_trainer = {
        'callbacks': [LearningRateMonitor(logging_interval='epoch')],
        'log_every_n_steps': 1,
        'logger': mylogger,
        'benchmark': True
    }
    config_trainer.update(update_trainer)

    t0 = time.time()
    main(config_model, config_trainer, train_loader, validation_loader, nmeasurement=len(measurement), file_model=file_model)
    t1 = time.time()
    print('Elapsed time: {:.2f} min'.format((t1 - t0)/60))
    print('Model saved at: {}'.format(file_model))
