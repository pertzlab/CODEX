from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from functools import reduce
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import scipy
import warnings
import matplotlib.pyplot as plt


def model_output(model, dataloader, export_prob=True, export_feat=True, softmax=True, device=None,
                 feature_layer='pool', progress_bar=True):
    """
    Function to get representation and probability of each trajectory in a loader. Increasing the batch_size of
    dataloader can greatly reduce computation time.

    :param model: str or pytorch model. If str, path to the model file.
    :param dataloader: pytorch Dataloader, classification output will be created for each element in the loader. Pay
    attention to the attribute drop_last, if True last batch would not be processed. Increase to lower computation time.
    :param export_prob: bool, whether to export classification output.
    :param export_feat: bool, whether to export latent features. feature_layer defines the layer output to hook.
    :param softmax: bool, whether to apply softmax to the classification output.
    :param device: str, pytorch device. If None will try to use cuda, if not available will use cpu.
    :param feature_layer: str, name of the model module from which to hook output if export_feat is True.
    :return: A pandas DataFrame with columns: ID, Class. If export_prob, one column for each class of the output named:
    'Prob_XXX' where XXX is the class name. If export_feat, one column for each element in the hooked layer output
    named: 'Feat_I' where I is an increasing integer starting at 0.
    """
    # Default arguments and checks
    batch_size = dataloader.batch_size
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dataloader.drop_last:
        warnings.warn('dataloader.drop_last is True, some data might be discarded.')

    # path to model file
    if isinstance(model, str):
        model = torch.load(model)
        model.eval()
        model.double()
        model.batch_size = batch_size
        model = model.to(device)
    else:
        model.eval()
        model.double()
        model.batch_size = batch_size
        model = model.to(device)

    # Lists with results of every batch
    lidt = []
    llab = []
    if export_prob:
        lprob = []
    if export_feat:
        lfeat = []
        def hook_feature(module, input, output):
            lfeat.append(output.data.squeeze().cpu().numpy())
        model._modules.get(feature_layer).register_forward_hook(hook_feature)

    if progress_bar:
        pbar = tqdm(total=len(dataloader))
    nbatch = len(dataloader)
    for ibatch, sample in enumerate(iter(dataloader)):
        # Flag last batch, can have different size from the others
        if ibatch + 1 == nbatch:
            model.batch_size = len(sample['label'])
        image_tensor, label, identifier = sample['series'], sample['label'], sample['identifier']
        image_tensor = image_tensor.to(device)
        # uni: batch, 1 dummy channel, length TS
        # (1,1,length) for uni; (1,1,2,length) for bi
        assert len(dataloader.dataset[0]['series'].shape) == 2
        nchannel, univar_length = dataloader.dataset[0]['series'].shape
        if nchannel == 1:
            view_size = (model.batch_size, 1, univar_length)
        elif nchannel >= 2:
            view_size = (model.batch_size, 1, nchannel, univar_length)
        image_tensor = image_tensor.view(view_size)

        scores = model(image_tensor)
        if softmax:
            scores = F.softmax(scores, dim=1).data.squeeze()
        # Store batch results
        llab.append(label.data.cpu().numpy())
        lidt.append(identifier)
        if export_prob:
            lprob.append(scores.data.squeeze().cpu().numpy())
        if progress_bar:
            pbar.update(1)

    frames = []
    # 1) Frame with ID
    lidt = np.concatenate(lidt)
    llab = np.concatenate(llab)
    frames.append(pd.DataFrame(list(zip(lidt, llab)), columns=['ID', 'Class']))
    # 2) Frame with proba
    if export_prob:
        nclass = lprob[0].shape[1] if batch_size > 1 else len(lprob[0])
        lprob = np.vstack(lprob)
        colnames = ['Prob_' + str(i) for i in range(nclass)]
        frames.append(pd.DataFrame(lprob, columns=colnames))
    # 3) Frame with features
    if export_feat:
        nfeats = lfeat[0].shape[1] if batch_size > 1 else len(lfeat[0])
        lfeat = np.vstack(lfeat)
        colnames = ['Feat_' + str(i) for i in range(nfeats)]
        frames.append(pd.DataFrame(lfeat, columns=colnames))
    df_out = pd.concat(frames, axis=1)
    # Remove hook
    model._modules[feature_layer]._forward_hooks = OrderedDict()
    return df_out


def tensorboard_to_df(path_to_logs, tags):
    """
    Read tensorboard logs and return a DataFrame with values. Useful for further plotting.
    :param path_to_logs: path to directory with log files
    :param tag: string name of the tag to read and export. Can pass a string or a list of strings. Tags must have
    common step vector in the logs.
    :return: A pandas Dataframe in wide format with one column for each tag
    """

    if isinstance(tags, str):
        tags = [tags]
    dfList = []
    event_acc = EventAccumulator(path_to_logs)
    event_acc.Reload()
    for tag in tags:
        w_times, step_nums, vals = zip(*event_acc.Scalars(tag))
        df = pd.DataFrame(data={'step': step_nums, tag: vals})
        dfList.append(df)

    # Merge all dataframes
    out = reduce(lambda x, y: pd.merge(x, y, on='step'), dfList)
    return out


def visualize_layer(model, layer_idx=0, linkage='average'):
    weights = model.features[layer_idx].cpu().weight.detach().numpy().squeeze()
    nfilt, h ,w = weights.shape
    link = scipy.cluster.hierarchy.linkage(weights.reshape(nfilt, -1), method=linkage)
    order = scipy.cluster.hierarchy.dendrogram(link)['leaves']
    for i in range(nfilt):
        plt.subplot(nfilt, 1, i+1)
        plt.imshow(weights[order[i]])
    plt.tight_layout()
    plt.show()
    return None
