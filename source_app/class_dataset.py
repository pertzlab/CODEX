###################################################################
# Define dataset classes and preprocessing for Pytorch DataLoader #
###################################################################

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from copy import deepcopy
import warnings
from re import search
from collections import OrderedDict


#TODO: make a single dataset class for both bi and univariate
#TODO: when exporting random crop positions, ToTensor() must be called right after RandomCrop(), modify this to have it in __getitem__
class myDataset(Dataset):
    """Standard dataset object with ID, class"""

    def __init__(self, dataset=None, csv_file=None, transform=None, col_id='ID', col_class='class', groups=None):
        """
        General Dataset class for arbitrary uni and multivariate time series.
        Args:
            data (pandas object): observations (series) in rows, measurements in columns. Names of columns must have the
             format: A_1, A_2, A_3,..., C_1, C_2,... where A and C are groups (channels) and 1,2,3... measurement time.
             It is possible to pass an arbitrary number of channels, but all channels must have same length! Channels
             will be vertically stacked in the order specified in 'channels'.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            col_id (string): Name of the column containing the ID of the series.
            col_class (string): Name of the column containing the class of the series.
            groups (list of strings): Order in which channels will be stacked. First on top, last at the bottom. If
             None, infers the order from the order of the columns in the dataset.
        """
        # Read dataset
        if dataset is not None and csv_file:
            raise ValueError('Only one of data and csv_file can be provided.')
        if dataset is not None:
            self.dataset = dataset
        elif csv_file:
            self.dataset = pd.read_csv(csv_file, header=None)

        self.transform = transform
        self.col_id = col_id
        self.col_class = col_class
        self.check_datasets()
        self.col_id_idx = self.dataset.columns.get_loc(self.col_id)
        self.col_class_idx = self.dataset.columns.get_loc(self.col_class)

        self.groups = groups
        if self.groups is None:
            self.detect_groups()

        self.groups_indices = {}
        self.get_groups_indices()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        identifier = self.dataset.iloc[idx, self.col_id_idx]
        label = np.array(self.dataset.iloc[idx, self.col_class_idx], dtype='int64')
        sequence_arrays = [self.dataset.iloc[idx, self.groups_indices[group][0] : self.groups_indices[group][1]].values
                           for group in self.groups]  # .values turns into numpy
        series = np.vstack(sequence_arrays)
        series = series.astype('float')
        sample = {'identifier': identifier, 'series': series, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    # Callback function used in imbalanced dataset loader
    def get_labels(self):
        return list(self.dataset[self.col_class].astype(int))

    def detect_groups(self):
        colnames = list(self.dataset.columns.values)
        colnames.remove(self.col_id)
        colnames.remove(self.col_class)
        groups = list(OrderedDict.fromkeys([i.split('_')[0] for i in colnames]))
        self.groups = groups

    def get_groups_indices(self):
        """
        Get indices of measurement groups for fast indexing with __getitem__ and .iloc
        """
        # Get unique groups
        colnames = list(self.dataset.columns.values)
        colnames.remove(self.col_id)
        colnames.remove(self.col_class)
        for group in self.groups:
            group_columns = [i for i in colnames if search('^{0}_'.format(group), i)]
            group_indices = [self.dataset.columns.get_loc(c) for c in group_columns]
            group_indices.sort()
            self.groups_indices[group] = [group_indices[0], group_indices[-1] + 1]  # +1 to include last element

    def check_datasets(self):
        """
        Check that dataset is properly formatted.
        """
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        colnames_dataset = list(self.dataset.columns.values)
        colnames_dataset.remove(self.col_id)
        colnames_dataset.remove(self.col_class)

        if not self.col_id in self.dataset.columns.values:
            warnings.warn('ID column is missing in dataset.')
        if not self.col_class in self.dataset.columns.values:
            warnings.warn('Class column not present in dataset.')
        if self.dataset.select_dtypes(numerics).empty:
            warnings.warn('No numerical columns in dataset.')
        if (not all([search('^\w+_', i) for i in colnames_dataset])) or (not all([search('_\d+$', i) for i in colnames_dataset])):
            ill_col = [i for i in colnames_dataset if not search('^\w+_', i)]
            ill_col += [i for i in colnames_dataset if not search('_\d+$', i)]
            warnings.warn('At least some column names of dataset are ill-formatted. Should follow "Group_Time" format. '
                          'List of ill-formatted: {0}'.format(ill_col))
        if any(self.dataset[self.col_id].duplicated()):
            warnings.warn('Found duplicated ID in dataset.')
        return None


# ======================================================================================================================
# Transformation and augmentation

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        series, label, identifier = sample['series'], sample['label'], sample['identifier']
        if 'crop' in sample.keys():
            return {'series': torch.from_numpy(series),
                    'label': torch.from_numpy(label),
                    'identifier': identifier,
                    'crop': sample['crop']}
        else:
            return {'series': torch.from_numpy(series),
                    'label': torch.from_numpy(label),
                    'identifier': identifier}


class Subtract(object):
    """Subtract fixed value from series. Useful to remove mean of training set.

    Args:
        value (float, int): value to subtract
    """

    def __init__(self, value):
        assert isinstance(value, (int, float, list))
        self.value = value

    def __call__(self, sample):
        series, label, identifier = sample['series'], sample['label'], sample['identifier']
        if isinstance(self.value, (int, float)):
            self.value = np.array([self.value] * series.shape[0])
        elif isinstance(self.value, list):
            self.value = np.array(self.value)
        value_array = np.ones_like(series) * self.value[:, None]  # multiply each row by a different scalar
        series -= value_array
        return {'series': series,
                'label': label,
                'identifier': identifier}


# TODO: to test with multivariate
class RandomNoise(object):
    """Add Gaussian noise.

    Args:
        mu (int, float): Mean of the noise
        sigma (int, float): Standard deviation of the noise
    """

    def __init__(self, mu, sigma):
        assert isinstance(mu, (int, float))
        assert isinstance(sigma, (int, float))
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        series, label, identifier = sample['series'], sample['label'], sample['identifier']
        noise = np.random.normal(self.mu, self.sigma, series.shape)
        series += noise
        return {'series': series,
                'label': label,
                'identifier': identifier}


# TODO: to test with multivariate
class RandomShift(object):
    """Shift series left or right and pad with background.

    Args:
        max_shift (int): maximum shift, shift uniformly picked in [0, max_shift]
        direction (str): shift to the 'left', 'right' or 'random'
        method (float): method used for padding. Can be any valid argument for numpy.pad(mode=) other than function.
    """

    def __init__(self, max_shift, direction='random', pad='edge', **kwargs):
        assert isinstance(max_shift, int)
        assert isinstance(direction, str)
        assert isinstance(pad, str)
        assert direction in ['left', 'right', 'random']
        assert pad in ['constant', 'edge', 'linear_ramp', 'maximum', 'mean',
                       'median', 'minimum', 'reflect', 'symmetric', 'wrap']
        self.max_shift = max_shift
        self.direction = direction
        self.pad = pad
        self.extra_args = kwargs

    def __call__(self, sample):
        series, label, identifier = sample['series'], sample['label'], sample['identifier']
        shift_width = np.random.randint(0, self.max_shift+1)

        # Pad on both sides then trim
        if(shift_width > 0):
            direction = deepcopy(self.direction)
            if direction == 'random':
                direction = np.random.choice(['left', 'right'], 1)[0]
            series = np.pad(series, pad_width=shift_width, mode=self.pad, **self.extra_args)
            # Trim 2*shift_width because na.pad on both sides
            if direction == 'left':
                series = series[0:(len(series)-2*shift_width)]
            elif direction == 'right':
                series = series[2*shift_width:]  # becomes -1 if shift is 0

        return {'series': series,
                'label': label,
                'identifier': identifier}


class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (int): Desired output size.
        ignore_na_tails (bool): If input has NA borders, ignore and crop in central region without NA. Only the first
        channel will be used to determine the tails, so it WILL RETURN ERROR if NA tails are not aligned across channels
    """

    def __init__(self, output_size, ignore_na_tails=True, export_crop_pos=False):
        assert isinstance(output_size, int)
        self.output_size = output_size
        self.ignore_na_tails = ignore_na_tails
        self.export_crop_pos = export_crop_pos

    def __call__(self, sample):
        series, label, identifier = sample['series'], sample['label'], sample['identifier']
        univar_series = series[0, :].squeeze()  # use only first channel to determine where to crop
        length = len(univar_series)
        new_length = self.output_size

        if self.ignore_na_tails:
            pos_non_na = np.where(~np.isnan(univar_series))
            start = pos_non_na[0][0]
            end = pos_non_na[0][-1]
            left = np.random.randint(start, end - new_length + 2)  # +1 to include last in randint; +1 for slction span
        else:
            left = np.random.randint(0, length - new_length)
        series = series[:, left : left + new_length]

        if self.export_crop_pos:
            return {'series': series,
                    'label': label,
                    'identifier': identifier,
                    'crop': (left, left + new_length)}
        else:
            return {'series': series,
                    'label': label,
                    'identifier': identifier}


class FixedCrop(object):
    """Crop a series between fixed indexes.

    Args:
        start (int): Start index
        end (int): End index (exclusive)
    """

    def __init__(self, start, end):
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert end > start
        self.start = start
        self.end = end

    def __call__(self, sample):
        series, label, identifier = sample['series'], sample['label'], sample['identifier']
        series = series[:, self.start:self.end]

        return {'series': series,
                'label': label,
                'identifier': identifier}


class scale(object):
    """Scale to zero mean and unit standard deviation

    Args:
        with_mean (cool): Whether to subtract the mean
        with_std (bool): Whether to divide by the standard deviation
        per_channel (bool): Whether to do the scaling globally or idependantly per channel (i.e. per row)
    """

    def __init__(self, with_mean=True, with_std=True, per_channel=True):
        assert isinstance(with_mean, bool)
        assert isinstance(with_std, bool)
        self.with_mean = with_mean
        self.with_std = with_std
        self.per_channel = per_channel

    def __call__(self, sample):
        series, label, identifier = sample['series'], sample['label'], sample['identifier']
        if self.per_channel:
            if self.with_mean:
                series -= series.mean(axis=1, keepdims=True)
            if self.with_std:
                series /= series.std(axis=1, keepdims=True)
        else:
            if self.with_mean:
                series -= series.mean()
            if self.with_std:
                series /= series.std()

        return {'series': series,
                'label': label,
                'identifier': identifier}
