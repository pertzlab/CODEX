import argparse
import base64
import io
import json
import os
import random
import re
import warnings
from copy import deepcopy
from functools import reduce

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import torch
import torch.nn.functional as F
from dash.dependencies import Input, Output, State
from skimage.filters import threshold_li, threshold_mean
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from class_dataset import RandomCrop, Subtract, ToTensor, myDataset
from load_data import DataProcesser
from plot_maps import create_cam, create_gbackprop, select_series
from tooltips import make_tooltips
from utils import frange, get_label_forImbaSampler, model_output_app


#TODO: selection hover/click mode
def parseArguments_overlay():
    parser = argparse.ArgumentParser(description='Project the CNN features with tSNE and browse interactively.')
    parser.add_argument('-m', '--model', help='str, path to the model file', type=str)
    parser.add_argument('-d', '--data', help='str, path to the model file', type=str)
    parser.add_argument('-s', '--set', help='str, set to project. Must be one of ["all", "train", "validation", "test"]. Default to "all".',
                        type=str, default='all')
    parser.add_argument('-b', '--batch', help='int, size of the batch when passing data to the model. '
                                              'Increase as high as GPU memory allows for speed up.'
                                              ' Must be smaller than the number of series selected,'
                                              ' Default is set to 1/10 of the dataset size.',
                        type=int, default=-1)
    parser.add_argument('--measurement', help='list of str, names of the measurement variables. In DataProcesser convention,'
                                              ' this is the prefix in a column name that contains a measurement'
                                              ' (time being the suffix). Pay attention to the order since this is'
                                              ' how the dimensions of a sample of data will be ordered (i.e. 1st in'
                                              ' the list will form 1st row of measurements in the sample,'
                                              ' 2nd is the 2nd, etc...). Leave empty for automatic detection.',
                        type=str, default='', nargs='*')
    parser.add_argument('--seed', help='int, seed for random, ensures reproducibility. Default to 7.',
                        type=int, default=7)
    parser.add_argument('--start', help='int, start time range for selecting data. Useful to ignore part of the '
                                        'data were irrelevant measurement were acquired. Set to -1 for automatic detection.',
                        type=int, default=-1)
    parser.add_argument('--end', help='int, end time range for selecting data. Useful to ignore part of the '
                                        'data were irrelevant measurement were acquired. Set to -1 for automatic detection.',
                        type=int, default=-1)
    parser.add_argument('-p', '--port', help='int, port on which to start the application.', type=int, default=8050)
    args = parser.parse_args()
    return(args)

args = parseArguments_overlay()
# ----------------------------------------------------------------------------------------------------------------------
# Inputs
myseed = args.seed
np.random.seed(myseed); random.seed(myseed); torch.manual_seed(myseed)
# Parameters
data_file = args.data

start_time = None if args.start==-1 else args.start
end_time = None if args.end==-1 else args.end
measurement = None if args.measurement=='' else args.measurement
selected_classes = None
perc_selected_ids = 1  # Select only percentile of all trajectories, not always useful to project them all and slow
batch_sz = args.batch  # set as high as GPU memory can handle for speed up

rand_crop = True
set_to_project = args.set  # one of ['all', 'train', 'validation', 'test']

# ----------------------------------------------------------------------------------------------------------------------
# Model Loading
model_file = args.model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load(model_file) if torch.cuda.is_available() else torch.load(model_file, map_location='cpu')
net.eval()
net.double()
net = net.to(device)
length = net.length

# ----------------------------------------------------------------------------------------------------------------------
# Data Loading, Subsetting, Preprocessing
data = DataProcesser(data_file, datatable=False)
measurement = data.detect_groups_times()['groups'] if measurement is None else measurement
start_time = data.detect_groups_times()['times'][0] if start_time is None else start_time
end_time = data.detect_groups_times()['times'][1] if end_time is None else end_time
data.subset(sel_groups=measurement, start_time=start_time, end_time=end_time)

# Check that the measurements columns are numeric, if not try to convert to float64
cols_to_check = '^(?:{})'.format('|'.join(measurement))  # ?: for non-capturing group
cols_to_check = data.dataset.columns.values[data.dataset.columns.str.contains(cols_to_check)]
cols_to_change = [(s,t) for s,t in zip(cols_to_check, data.dataset.dtypes[cols_to_check]) if not pd.api.types.is_numeric_dtype(data.dataset[s])]
if len(cols_to_change) > 0:
    warnings.warn('Some measurements columns are not of numeric type. Attempting to convert the columns to float64 type. List of problematic columns: {}'.format(cols_to_change))
    try:
        cols_dict = {s[0]:'float64' for s in cols_to_change}
        data.dataset = data.dataset.astype(cols_dict)
    except ValueError:
        warnings.warn('Conversion to float failed for at least one column.')

data.get_stats()
if selected_classes is not None:
    data.dataset = data.dataset[data.dataset[data.col_class].isin(selected_classes)]
# Suppress the warning that data were not processed, irrelevant for the app
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    data.split_sets(which='dataset')

print('Start time: {}; End time: {}; Measurement: {}'.format(start_time, end_time, measurement))

# ----------------------------------------------------------------------------------------------------------------------
# df used to plot measurements
assert set_to_project in ['all', 'train', 'validation', 'test']
if set_to_project == 'all':
    df = deepcopy(data.dataset)
elif set_to_project == 'train':
    df = deepcopy(data.train_set)
elif set_to_project == 'validation':
    df = deepcopy(data.validation_set)
elif set_to_project == 'test':
    df = deepcopy(data.test_set)
print('Size of raw dataframe: {}'.format(df.shape))
df.rename(columns={data.col_id: 'ID', data.col_class: 'Class'}, inplace = True)
selected_ids = np.random.choice(df.loc[:,'ID'].unique(), round(perc_selected_ids * df.shape[0]), replace=False)
df = df[df['ID'].isin(selected_ids)]
print('Size of selected dataframe: {}'.format(df.shape))

if batch_sz == -1:
    batch_sz = round(df.shape[0]/10)
print('Batch size: {}'.format(batch_sz))
assert batch_sz <= df.shape[0]
# Split for each measurement, melt and append.
ldf = []
for meas in measurement:
    col_meas = [i for i in df.columns if re.match('^{}_'.format(meas), i)]
    temp = df[['ID', 'Class'] + col_meas].melt(['ID', 'Class'])
    temp['Time'] = temp['variable'].str.extract('([0-9]+$)')
    temp['variable'] = temp['variable'].str.replace('_[0-9]+$', '', regex=True)
    ldf.append(temp)
df = pd.concat(ldf)
del temp
del ldf
df.sort_values(['ID', 'Time'])
max_slider = df['value'].max()
min_slider = df['value'].min()
default_slider = df['value'].quantile([0.005, 0.995])

# ----------------------------------------------------------------------------------------------------------------------
# Prepare dataloader for t-SNE, set high batch_size to speed up
subtract_numbers = [data.stats['mu'][meas]['train'] for meas in measurement]
if rand_crop:
    ls_transforms = transforms.Compose([
        Subtract(subtract_numbers),
        RandomCrop(output_size=length, ignore_na_tails=True, export_crop_pos=True),
        ToTensor()])
else:
    ls_transforms = transforms.Compose([Subtract(subtract_numbers),
                                        ToTensor()])

mydataset = myDataset(dataset=data.dataset[data.dataset['ID'].isin(selected_ids)], transform=ls_transforms)
mydataloader = DataLoader(dataset=mydataset,
                          batch_size=batch_sz,
                          shuffle=False,
                          drop_last=False)

# ----------------------------------------------------------------------------------------------------------------------
# Classes object definition
classes = tuple(data.classes[data.col_classname])
classes_col = data.classes[data.col_classname]
if selected_classes is not None:
    classes = [i for i in classes if i
               in list(data.classes[data.classes[data.col_class].isin(selected_classes)][data.col_classname])]
    classes = tuple(classes)
    classes_dict = data.classes[data.classes[data.col_class].isin(selected_classes)].to_dict()[data.col_classname]
else:
    classes_dict = data.classes.to_dict()[data.col_classname]

net.batch_size = batch_sz  # Learn representations over the whole dataset at once if equal to dataset length
# ----------------------------------------------------------------------------------------------------------------------
# Copy with unit batch for creating CAM and backprop
net_unit_batch = deepcopy(net)
net_unit_batch.batch_size = 1
# Redefine without random crop for CAM plotting
mydataset_unit_batch = myDataset(dataset=data.dataset[data.dataset[data.col_id].isin(selected_ids)],
                                 transform=transforms.Compose([ToTensor()]))
mydataloader_unit_batch = DataLoader(dataset=mydataset_unit_batch,
                          batch_size=1,
                          shuffle=False,
                          drop_last=False)
del data  # Free some memory

# ----------------------------------------------------------------------------------------------------------------------
# Compute tSNE coords, model output, store cropping info for plotting
# No grad to save GPU memory, no backward pass is done here
@torch.no_grad()
def tsne(model, dataloader, device, layer_feature='pool', ncomp=2, ini='pca', perplex=30.0, lr=200.0, niter=250, exag=12, metric='euclidean'):
    df_out = model_output_app(model, dataloader, export_prob=True, export_feat=True, device=device,
                              feature_layer=layer_feature, export_crop_pos=rand_crop)
    df_out['Class'].replace(classes_col, inplace=True)
    prob_cols = [i for i in df_out.columns if i.startswith('Prob_')]
    feat_cols = [i for i in df_out.columns if i.startswith('Feat_')]

    label = np.array(df_out['Class'])
    identifier = np.array(df_out['ID'])

    # Df with classification output, used to build html table when hovering points and export of selection
    global DF_PROBS
    DF_PROBS = deepcopy(df_out[['Class'] + prob_cols])  # copy avoids pandas warning about modifying copy
    DF_PROBS.index = df_out['ID']
    old_labels = {col: re.search('\d+$', col) for col in DF_PROBS.columns.values}
    new_labels = {col: re.sub('\d+$', classes_col[int(old_labels[col].group())], col)
                  for col in old_labels if old_labels[col]}
    DF_PROBS.rename(new_labels, axis='columns', inplace=True)
    DF_PROBS.columns = [re.sub('Prob_', '', col) for col in DF_PROBS.columns]

    # Df with CNN features, used to export of selection
    global DF_FEATS
    DF_FEATS = df_out[feat_cols]
    DF_FEATS.index = df_out['ID']

    # Df with positions of crop, used to plot CAM maps at right positions, highlight part seen by model
    if rand_crop:
        global DF_CROP
        DF_CROP = df_out[['ID', 'Crop_start', 'Crop_end']]

    # Reshape in a single big array and compute TSNE coord
    # One row for each trajectory, columns according to the number of features
    feature_blobs_array = np.array(df_out[feat_cols])
    feature_embedded = TSNE(n_components=ncomp, init=ini, perplexity=perplex,
                            learning_rate=lr, n_iter=niter, verbose=2, early_exaggeration=exag, metric=metric).fit_transform(feature_blobs_array)

    return feature_embedded, label, identifier

# ----------------------------------------------------------------------------------------------------------------------
# Color palettes
def hex_to_rgb(value, add_alpha=-1):
    assert (add_alpha <= 1 and add_alpha >= 0) or (add_alpha == -1)
    value = value.lstrip('#')
    lv = len(value)
    out = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    if add_alpha != -1:
        out += (add_alpha,)
    return out

def recycle_palette(palette, size):
    """Cycle through a palette (list) until it reaches desired length (int)."""
    length = len(palette)
    nrepeat = (size // length) + 1
    out = palette * nrepeat
    return out[0:size]


global PALETTES
PALETTES = {
    'D3': list(map(hex_to_rgb, px.colors.qualitative.D3)),
    'Plotly': list(map(hex_to_rgb, px.colors.qualitative.Plotly)),
    'G10': list(map(hex_to_rgb, px.colors.qualitative.G10)),
    'T10': list(map(hex_to_rgb, px.colors.qualitative.T10)),
    'Alphabet': list(map(hex_to_rgb, px.colors.qualitative.Alphabet)),
    'Dark24': list(map(hex_to_rgb, px.colors.qualitative.Dark24))
}

# ----------------------------------------------------------------------------------------------------------------------
# App layout
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

card_tsne = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label('Layer to project:'),
                dcc.Dropdown(
                    id = 'drop-layer',
                    options = [{'label': i, 'value': i} for i in list(net._modules.keys())],
                    clearable = False,
                    value = 'pool'
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label('Initialization:'),
                dcc.Dropdown(
                    id = 'drop-init',
                    options=[{'label': i, 'value': i} for i in ['random', 'pca']],
                    clearable = False,
                    value = 'pca'
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label('N dimensions:'),
                dcc.Dropdown(
                    id = 'drop-ndim',
                    options=[{'label':i, 'value':j} for i,j in [('2D', 2), ('3D', 3)]],
                    clearable = False,
                    value = 2
                ),
            ]
        ),
    ],
    body = True,
    id = 'card-tsne'
)

card_tsne_params = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label('Learning rate:'),
                dbc.Input(
                    id='input-lr',
                    value=600,
                    type='number',
                    placeholder='Learning Rate (default = 600)',
                    min=1
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label('Perplexity:'),
                dbc.Input(
                    id='input-perp',
                    value=50,
                    type='number',
                    placeholder='Perplexity (default = 50)',
                    min=1
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label('N iterations:'),
                dbc.Input(
                    id='input-niter',
                    value=250,
                    type='number',
                    placeholder='Max number of iteration',
                    min=250
                ),
            ]
        ),
    ],
    body = True,
    id = 'card-tsne-params'
)

card_tsne_params_advanced = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label('Early exaggeration:'),
                dbc.Input(
                    id='input-exaggeration',
                    value=12,
                    type='number',
                    placeholder='Early exxageration for 250 iterations',
                    min=1
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label('Distance:'),
                dcc.Dropdown(
                    id='drop-distance',
                    options=[{'label': i, 'value': i} for i in ['euclidean', 'cityblock', 'cosine']],
                    clearable = False,
                    value = 'euclidean'
                )
            ]
        )
    ],
    body=True
)

card_overlay = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label('Color overlay for trajectories:'),
                dcc.Dropdown(
                    id = 'drop-overlay',
                    options = [{'label': i, 'value': i} for i in ['None', 'CAM', 'Guided Backprop']],
                    clearable = False,
                    value = 'None'
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label('Target class overlay:'),
                dcc.Dropdown(
                    id = 'drop-target',
                    options=[{'label':'Prediction', 'value': 'prediction'}] +
                            [{'label': v, 'value': k} for k,v in classes_dict.items()],
                    clearable = False,
                    value = 'prediction',
                    placeholder='Select overlay first'
                ),
            ],
            id='div-target'
        ),
        dbc.FormGroup(
            [
                dbc.Label('Binarization overlay:'),
                dcc.Dropdown(
                    id = 'drop-bin',
                    options=[{'label': i, 'value': i} for i in ['No bin',
                                                                'Li Threshold',
                                                                'Mean Threshold']],
                    clearable = False,
                    value = 'No bin',
                    placeholder='Select overlay first'
                ),
            ],
            id='div-bin'
        ),
    ],
    body = True,
)

card_scatterplot = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label('Color palette:'),
                dcc.Dropdown(
                    id = 'drop-palette',
                    options=[{'label': i, 'value': i} for i in PALETTES.keys()],
                    clearable = False,
                    value = 'D3',
                ),
            ],
        ),
        dbc.FormGroup(
            [
                dbc.Label('Point alpha'),
                dcc.Slider(
                    id='slider-alpha',
                    min=0,
                    max=1,
                    step=0.01,
                    value=1,
                    updatemode='mouseup',
                    marks={i: {'label': i} for i in [0, 0.25, 0.5, 0.75, 1]}
                )
            ]
        ),
        dcc.Checklist(
            id='check-density',
            value=[],
            options=[{'label': ' Plot density', 'value': True}]
        ),
        dbc.FormGroup(
            [
                dbc.Label('Number bins density'),
                dcc.Slider(
                    id='slider-density',
                    min=1,
                    max=20,
                    step=1,
                    value=5,
                    updatemode='mouseup',
                    marks={i: {'label': i} for i in range(1, 21, 5)}
                )
            ],
            id='formgroup-slider-density'
        )
    ],
    body = True
)

card_plot = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label('Highlight prototypes:'),
                dcc.Dropdown(
                    id = 'drop-proto',
                    options = [
                        {'label': 'None', 'value': 'None'},
                        {'label': 'Top', 'value': 'Top'}
                    ],
                    clearable = False,
                    value = 'None',
                ),
            ],
        ),
        dbc.FormGroup(
            [
                dbc.Label('Number prototypes:'),
                dcc.Slider(
                    id='slider-proto',
                    min=1,
                    max=20,
                    step=1,
                    value=5,
                    updatemode='mouseup',
                    marks={i:{'label': i} for i in range(1, 21, 5)}
                )
            ],
            id = 'formgroup-slider-proto'
        ),
        dcc.Checklist(
            id = 'check-xrange',
            value = [],
            options=[{'label': ' Center x-axis on network input', 'value': True}]
        ),
        dcc.Checklist(
            id = 'check-xshow',
            value = [True],
            options=[{'label': ' Show borders of network input', 'value': True}]
        ),
        dbc.FormGroup(
            [
                dbc.Label('Range of y-axis'),
                dcc.RangeSlider(
                    id='slider-yrange',
                    min=min_slider,
                    max=max_slider,
                    step=(max_slider-min_slider)/100,
                    value=default_slider,
                    allowCross=False,
                    updatemode='drag',
                    marks={i: {'label': str(round(i, 2))} for i in
                     frange(min_slider, max_slider - 1e-9 + (max_slider - min_slider) / 5, (max_slider - min_slider) / 5)}
                )
            ]
        )
    ],
    body = True
)

card_load_precomputed = dbc.Card(
    [   
        dcc.Upload(
            id='upload-precomputed',
            children=dbc.Button(children='Upload .csv file',
                                outline=True,
                                color='primary',
                                id='button-upload-precomputed'),
            # TODO: add "xlrd" and "odfpy" dependencies to conda env to read Excel and ODT files
            # accept=['.csv', '.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt']
            accept=['.csv']
        ),
        dbc.FormGroup(
            [
                dbc.Label('X coordinates column:'),
                dcc.Dropdown(
                    id='drop-x-column',
                    options=[{'label': 'placeholder', 'value': '__'}],
                    clearable=False,
                    disabled=True,
                    placeholder='Upload first'
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label('Y coordinates column:'),
                dcc.Dropdown(
                    id='drop-y-column',
                    options=[{'label': 'placeholder', 'value': '__'}],
                    clearable=False,
                    disabled=True,
                    placeholder='Upload first'
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label('ID column:'),
                dcc.Dropdown(
                    id='drop-id-column',
                    options=[{'label': 'placeholder', 'value': '__'}],
                    clearable=False,
                    disabled=True,
                    placeholder='Upload first'
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label('Group column:'),
                dcc.Dropdown(
                    id='drop-group-column',
                    options=[{'label': 'placeholder', 'value': '__'}],
                    clearable=False,
                    disabled=True,
                    placeholder='Upload first'
                )
            ]
        )
    ],
    body=True,
    style={'display': 'none'},  # hide upon booting up
    id = 'card-load-precomputed'
)


button_collapse = dbc.Button(
                    'Fold parameters menu \u25BE',
                    id='collapse-button',
                    color='primary'
                  )

check_collapse_advanced = dcc.Checklist(
    id='check-advanced',
    value = [],
    options=[{'label': ' Show advanced tSNE options', 'value': True}]
)

check_collapse_exportPlot = dcc.Checklist(
    id='check-exportPlot',
    value=[],
    options=[{'label': ' Show figure export options', 'value': True}]
)

check_precomputed = dcc.Checklist(
    id='check-precomputed',
    value = [],
    options=[{'label': ' Load precomputed tSNE','value': True},]
)

button_submit = html.Div(
    [
        dbc.Spinner(
            html.Div(id='loading-tsne'),
            size='md',
            color='primary'
        ),
        dbc.Button(
            '\u21BB Run t-SNE',
            id='submit-tsne',
            n_clicks=0,
            color='primary',
        )
    ],
    id='div-submit-button',
    style={'display': 'inline-block'}  # center the button and the spinner
)

button_load = html.Div(
    [
        dbc.Button(
            '\u21BB Load t-SNE',
            id='button-load-tsne',
            n_clicks=0,
            color='success',
        )
    ],
    id='div-load-button',
    style={'display': 'none'}  # center the button and the spinner
)

button_export = dbc.Button(
    '\u2913 Export Selection',
    id='button-export',
    n_clicks=0,
    color='primary'
)

popover_export = dbc.Popover(
    [
        dbc.PopoverHeader('Empty export'),
        dbc.PopoverBody('No data selected for export. First, select the points to export with the box or the lasso selection tool in the plot banner.')
    ],
    id='popover-export',
    is_open=False,
    target='button-export'
)

dropdown_export = dcc.Dropdown(
    id = 'drop-export',
    options = [
        {'label': 'Class', 'value': 'Class'},
        {'label': 'tSNE Coordinates', 'value': 'Coord'},
        {'label': 'CNN Output', 'value': 'Probability'},
        {'label': 'CNN Features', 'value': 'Feature'},
        {'label': 'Crop Info', 'value': 'Crop'},
        {'label': 'Input Data', 'value': 'Input'}
    ],
    multi = True,
    searchable = False,
    value = ['Class', 'Coord', 'Probability', 'Feature', 'Crop'],
    placeholder = 'Select elements to export'
)

button_exportPlot = dbc.Button(
            '\u2913 Export Plot',
            id='button-exportPlot',
            n_clicks=0,
            color='primary'
)

card_exportPlot = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label('Height:'),
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.Input(
                                id='input-exportPlot-height',
                                value=500,
                                type='number',
                                placeholder='Height in pixel (default = 500)',
                                min=1
                            ),
                            dbc.InputGroupAddon('px', addon_type='append')
                        ]
                    ),
                )
            ],
            row=True
        ),
        dbc.FormGroup(
            [
                dbc.Label('Width:'),
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.Input(
                                id='input-exportPlot-width',
                                value=700,
                                type='number',
                                placeholder='Width in pixel (default = 700)',
                                min=1
                            ),
                            dbc.InputGroupAddon('px', addon_type='append')
                        ]
                    ),
                )
            ],
            row=True
        ),
        dbc.FormGroup(
            [
                dbc.Label('Scale:'),
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.Input(
                                id='input-exportPlot-scale',
                                value=float(1.0),
                                type='number',
                                placeholder='Scale (default = 1.0)',
                                min=float(0.0),
                                step='any'
                            )
                        ]
                    ),
                )
            ],
            row=True
        )
    ],
    body=True
)

card_exportPlot_filetype = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label('File format:'),
                dbc.RadioItems(
                    options=[
                        {'label': 'pdf', 'value': 'pdf'},
                        {'label': 'png', 'value': 'png'},
                        {'label': 'jpg', 'value': 'jpg'},
                        {'label': 'svg', 'value': 'svg'}
                    ],
                    value='pdf',
                    id='radio-exportPlot'
                )
            ]
        )
    ],
    body=True
)

tooltips = make_tooltips()

app.layout = dbc.Container(
    [
        # Hidden division used to store the tSNE coordinates, this way can
        # update the plot appearance without recomputing the tSNE
        html.Div(id='hidden-tsne', style={'display': 'none'}),
        # Hidden division used to store upload content
        html.Div(id='hidden-upload-raw', style={'display': 'none'}),
        html.Div(id='hidden-upload-ready', style={'display': 'none'}),
        dbc.Row(
            [
                dbc.Col(button_collapse, width=2),
                dbc.Col(check_collapse_exportPlot, width=2),
                dbc.Col(check_collapse_advanced, width=2),
                dbc.Col(check_precomputed, width=2)
            ],
            align='center',
            justify='start',
        ),
        dbc.Collapse(
            dbc.Row(
                [
                    dbc.Col(
                        dbc.CardDeck(
                            [
                                card_load_precomputed,
                                card_tsne,
                                card_tsne_params,
                                card_overlay,
                                card_scatterplot,
                                card_plot
                            ],
                            id='main_deck'
                        ),
                        width = 10
                    ),
                    dbc.Col(
                        [
                            button_submit,
                            button_load
                        ],
                        width = 2
                    ),
                ],
                align='end',
                justify='start',
                style={'background-color': '#f8f9fa', 'padding': '15px 5px 20px 20px', 'borderBottom': 'thin lightgrey solid'}
            ),
        id ='collapse',
        is_open=True
        ),
        dbc.Collapse(
            dbc.Row(
                [
                    dbc.Col(
                        card_tsne_params_advanced,
                        width=2
                    )
                ],
                align='end',
                justify='around',
                style={'background-color': '#f8f9fa', 'padding': '15px 5px 20px 20px', 'borderBottom': 'thin lightgrey solid'}
            ),
            id='collapse-advanced',
            is_open=False
        ),
        dbc.Collapse(
            dbc.Row(
                [
                    dbc.Col(
                        dbc.CardDeck(
                            [
                                card_exportPlot_filetype,
                                card_exportPlot                                
                            ],
                            id='deck-exportFile'
                        ),
                        width=4
                    )
                ],
                align='end',
                justify='around',
                style={'background-color': '#f8f9fa', 'padding': '15px 5px 20px 20px', 'borderBottom': 'thin lightgrey solid'}
            ),
            id='collapse-exportPlot',
            is_open=False
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='plot-tsne'), width = 6),
                dbc.Col(dcc.Graph(id='plot-meas'), width = 6)
            ],
            no_gutters=True
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dropdown_export,
                            button_export,
                            popover_export,
                            dcc.Download(id='download-export')
                        ]
                    ),
                    width = 3
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            button_exportPlot,
                            dcc.Download(id='download-exportPlot')
                        ]
                    ),
                    width = 1
                ),
                dbc.Col(
                    html.Div(id='table-proba'),
                    width = 6
                )
            ],
            justify = 'between',
            align = 'end',
            no_gutters = True
        ),
    ] + tooltips,
    fluid=True,
)

# ----------------------------------------------------------------------------------------------------------------------
# Collapse menu with parameters
@app.callback(
    Output('collapse', 'is_open'),
    [Input('collapse-button', 'n_clicks')],
    [State('collapse', 'is_open')],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# ----------------------------------------------------------------------------------------------------------------------
# Collapse menu with advanced options
@app.callback(
    Output('collapse-advanced', 'is_open'),
    [Input('check-advanced', 'value')]
)
def toggle_collapse_advanced(checkval):
    if checkval:
        return True
    else:
        return False

# ----------------------------------------------------------------------------------------------------------------------
# Collapse menu with export plot options
@app.callback(
    Output('collapse-exportPlot', 'is_open'),
    [Input('check-exportPlot', 'value')]
)
def toggle_collapse_exportPlot(checkval):
    if checkval:
        return True
    else:
        return False

# ----------------------------------------------------------------------------------------------------------------------
# Change menu of parameters for loading precomputed tSNE
@app.callback(
    [
        Output('card-tsne', 'style'),
        Output('card-tsne-params', 'style'),
        Output('div-submit-button', 'style'),
        Output('card-load-precomputed', 'style'),
        Output('div-load-button', 'style')
    ],
    [Input('check-precomputed', 'value')]
)
def toggle_precomputed(checkval):
    if checkval:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {}, {'display': 'inline-block'}
    else:
        return {}, {}, {'display': 'inline-block'}, {'display': 'none'}, {'display': 'none'}


# ----------------------------------------------------------------------------------------------------------------------
# Disable dropdowns of CAMs/backprop if no overlay selected
@app.callback(
    [Output('drop-target', 'disabled'), Output('drop-bin', 'disabled')],
    [Input('drop-overlay', 'value')]
)
def disable_drops(overlay_value):
    if overlay_value == 'None':
        return True, True
    else:
        return False, False

@app.callback(
    [Output('div-target', 'style'), Output('div-bin', 'style')],
    [Input('drop-overlay', 'value')],
)
def grey_out(overlay_value):
    if overlay_value == 'None':
        #return {'display': 'none'}, {'display': 'none'}
        return {'color': 'rgb(158, 158, 158)'}, {'color': 'rgb(158, 158, 158)'}
    else:
        return {}, {}

@app.callback(
    [Output('drop-target', 'value'), Output('drop-bin', 'value')],
    [Input('drop-overlay', 'value')],
    [State('drop-target', 'value'), State('drop-bin', 'value')]
)
def show_placeholder(overlay_value, curr_target, curr_bin):
    if overlay_value == 'None':
        #return {'display': 'none'}, {'display': 'none'}  # to disappear completely
        return 'placeholder', 'placeholder'
    else:
        new_target = 'prediction' if curr_target == 'placeholder' else curr_target
        new_bin = 'No bin' if curr_bin == 'placeholder' else curr_bin
        out = (new_target, new_bin)
        return out

# ----------------------------------------------------------------------------------------------------------------------
# Disable sliders number prototypes if no prototype method selected
@app.callback(
    Output('slider-proto', 'disabled'),
    [Input('drop-proto', 'value')]
)
def disable_slider_proto(droproto_value):
    if droproto_value == 'None':
        return True
    else:
        return False

@app.callback(
    Output('formgroup-slider-proto', 'style'),
    [Input('drop-proto', 'value')],
)
def grey_out_proto(droproto_value):
    if droproto_value == 'None':
        return {'display': 'none'}
        #return {'color': 'rgb(158, 158, 158)'}
    else:
        return {}

# ----------------------------------------------------------------------------------------------------------------------
# Disable sliders number bins if no density plotted
@app.callback(
    Output('slider-density', 'disabled'),
    [Input('check-density', 'value')]
)
def disable_slider_density(checkdensity_value):
    if checkdensity_value:
        return False
    else:
        return True

@app.callback(
    Output('formgroup-slider-density', 'style'),
    [Input('check-density', 'value')],
)
def grey_out_density(checkdensity_value):
    if checkdensity_value:
        return {}
    else:
        return {'display': 'none'}
        #return {'color': 'rgb(158, 158, 158)'}

# ----------------------------------------------------------------------------------------------------------------------
# Change alpha of points upon checking prototypes for better highlight
@app.callback(
    Output('slider-alpha', 'value'),
    [Input('drop-proto', 'value')],
)
def change_alpha(prototypes):
    if prototypes == 'None':
        return 1
    else:
        return 0.25


# ----------------------------------------------------------------------------------------------------------------------
# Read t-SNE coordinates from a file and store the table in a hidden division, also set values in dropdowns
@app.callback(
    [Output('hidden-upload-raw', 'children'),
    Output('drop-x-column', 'options'),
    Output('drop-y-column', 'options'),
    Output('drop-id-column', 'options'),
    Output('drop-group-column', 'options'),
    Output('drop-x-column', 'disabled'),
    Output('drop-y-column', 'disabled'),
    Output('drop-id-column', 'disabled'),
    Output('drop-group-column', 'disabled'),
    Output('drop-x-column', 'placeholder'),
    Output('drop-y-column', 'placeholder'),
    Output('drop-id-column', 'placeholder'),
    Output('drop-group-column', 'placeholder'),
    Output('button-upload-precomputed', 'outline'),
    Output('button-upload-precomputed', 'color'),
    Output('button-upload-precomputed', 'children'),
    Output('drop-x-column', 'value'),
    Output('drop-y-column', 'value'),
    Output('drop-id-column', 'value'),
    Output('drop-group-column', 'value')],
    [Input('upload-precomputed', 'contents')],
    [State('upload-precomputed', 'filename'),
    State('drop-x-column', 'options'),
    State('drop-y-column', 'options'),
    State('drop-id-column', 'options'),
    State('drop-group-column', 'options')]
)
def read_tsne(contents, filename, xcol, ycol, idcol, grcol):
    # Do nothing when no file loaded (avoid error at initialization)
    if contents is None:
        plchldr = 'Upload first'
        return (
            None,
            xcol,
            ycol,
            idcol,
            grcol,
            True,
            True,
            True,
            True,
            plchldr,
            plchldr,
            plchldr,
            plchldr,
            True,
            'primary',
            'Upload .csv file',
            '__placeholder',
            '__placeholder'
            '__placeholder',
            '__placeholder',
            '__placeholder'
        )

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    file_name, file_extension = os.path.splitext(filename)
    try:
        if file_extension == '.csv':
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        # TODO: add "xlrd" and "odfpy" dependencies to conda env to read Excel and ODT files
        # elif file_extension in ['.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt']:
        #     # Assume that the user uploaded an excel file
        #     print('try to read excel')
        #     df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    toStore = df.to_dict(orient='list')
    colnames = list(df.columns)
    dict_colnames = [{'label':col, 'value': col} for col in colnames]

    # Automatically fill in recognized column names
    # recognized_colnames = {
    #     'xcol': ['xTSNE'],
    #     'ycol': ['yTSNE'],
    #     'idcol': ['ID'],
    #     'grcol': ['Class']
    # }
    prefilled_columns = {}
    prefilled_columns['xcol'] = 'xTSNE' if 'xTSNE' in colnames else ''
    prefilled_columns['ycol'] = 'yTSNE' if 'yTSNE' in colnames else ''
    prefilled_columns['idcol'] = 'ID' if 'ID' in colnames else ''
    prefilled_columns['grcol'] = 'Class' if 'Class' in colnames else ''

    return (
        json.dumps(toStore),
        dict_colnames,
        dict_colnames,
        dict_colnames,
        dict_colnames,
        False,
        False,
        False,
        False,
        'Select column',
        'Select column',
        'Select column',
        'Select column',
        False,
        'success',
        filename,
        prefilled_columns['xcol'],
        prefilled_columns['ycol'],
        prefilled_columns['idcol'],
        prefilled_columns['grcol']
    )

@app.callback(
    Output('hidden-upload-ready', 'children'),
    [Input('hidden-upload-raw', 'children'),
    Input('drop-x-column', 'value'),
    Input('drop-y-column', 'value'),
    Input('drop-id-column', 'value'),
    Input('drop-group-column', 'value')]
)
def prepare_for_tsne(contents, xcol, ycol, idcol, grcol):
    if (contents is None) or any(map(lambda x: x is None, [xcol, ycol, idcol, grcol])):
        return None
    dict_load = json.loads(contents)
    tsne_coord = [[x,y] for x,y in zip(dict_load[xcol], dict_load[ycol])]
    labels, ids = dict_load[grcol], dict_load[idcol]
    toStore = {'tsne_coord': tsne_coord, 'labels': labels, 'ids': ids}
    return json.dumps(toStore)

# ----------------------------------------------------------------------------------------------------------------------
# Compute t-SNE coordinates and store them in a hidden division for sharing between callbacks
@app.callback(
    [Output('hidden-tsne', 'children'),
     Output('loading-tsne', 'children')],
    [Input('submit-tsne', 'n_clicks'),
    Input('button-load-tsne', 'n_clicks')],
    [State('drop-layer', 'value'),
     State('drop-init', 'value'),
     State('drop-ndim', 'value'),
     State('input-lr', 'value'),
     State('input-perp', 'value'),
     State('input-niter', 'value'),
     State('input-exaggeration', 'value'),
     State('drop-distance', 'value'),
     State('hidden-upload-ready', 'children')]
)
def compute_tsne(n_clicks, n_clicks_load, layer, init, ndim, lrate, perp, n_iter, exagg, metr, loaded_tsne):
    # Determine which button was clicked
    ctx = dash.callback_context
    button_triggered = ctx.triggered[0]['prop_id'].split('.')[0]
    # Upon initilization, run as if clicked submit-tsne button
    if (not ctx.triggered) or (button_triggered == 'submit-tsne'):
        globals()
        tsne_coord, labels, ids = tsne(model=net, dataloader=mydataloader, device=device, layer_feature=layer,
                                    ncomp=ndim, ini=init, perplex=perp, lr=lrate, niter=n_iter, exag=exagg, metric=metr)
        # Need to convert numpy arrays to list for JSON conversion
        toStore = {'tsne_coord': tsne_coord.tolist(), 'labels': labels.tolist(), 'ids': ids.tolist()}
        out = json.dumps(toStore)
    elif button_triggered == 'button-load-tsne':
        out = loaded_tsne
    return out, None

# ----------------------------------------------------------------------------------------------------------------------
# Plot t-SNE embedding
@app.callback(
    Output('plot-tsne', 'figure'),
    [Input('hidden-tsne', 'children'),
     Input('slider-alpha', 'value'),
     Input('check-density', 'value'),
     Input('slider-density', 'value'),
     Input('drop-palette', 'value'),
     Input('drop-proto', 'value'),
     Input('slider-proto', 'value')]
)
def plot_tsne(dict_tsne, alpha, density, nbins, palette, prototypes, nprototypes):
    globals()
    dict_load = json.loads(dict_tsne)
    tsne_coord, labels, ids = dict_load['tsne_coord'], dict_load['labels'], dict_load['ids']
    # Revert the lists in the JSON back to numpy arrays
    tsne_coord = np.array(tsne_coord)
    labels = np.array(labels)
    ids = np.array(ids)
    # Create a different trace for each class
    ndim = tsne_coord.shape[1]
    traces = []
    marker_palette = recycle_palette(PALETTES[palette], len(classes))
    for classe, marker_color in zip(classes, marker_palette):
        marker_color_str = 'rgb({0}, {1}, {2})'.format(*marker_color)
        density_color_hi_str = 'rgba({0}, {1}, {2}, 0.5)'.format(*marker_color)
        density_color_lo_str = 'rgba({0}, {1}, {2}, 0)'.format(*marker_color)
        idx = np.where(labels==classe)

        if ndim==2:
            if prototypes == 'Top':
                df_class = DF_PROBS.loc[DF_PROBS['Class']==classe]
                nproto = min(nprototypes, len(df_class.index))
                ids_proto = df_class.sort_values(by=classe).tail(nproto).index.to_numpy()
                idx_proto = [np.where(ids==i) for i in ids_proto]
                idx_proto = np.concatenate(idx_proto, axis=1)[0]
                idx_proto = (np.array(idx_proto), )
                traces.append(go.Scattergl(x = tsne_coord[idx_proto, 0].squeeze(),
                                           y = tsne_coord[idx_proto, 1].squeeze(),
                                           mode='markers',
                                           legendgroup='prototypes',
                                           # opacity=alpha,
                                           marker=dict(color=marker_color_str,
                                                       symbol='diamond',
                                                       size=8),
                                           name=classe,
                                           text=ids[idx_proto]
                                          ))
            traces.append(go.Scattergl(x = tsne_coord[idx, 0].squeeze(),
                                       y = tsne_coord[idx, 1].squeeze(),
                                       mode='markers',
                                       opacity=alpha,
                                       marker=dict(color=marker_color_str),
                                       name=classe,
                                       text=ids[idx]
                                      ))
            if density:
                traces.append(go.Histogram2dContour(
                    x = tsne_coord[idx, 0].squeeze(),
                    y = tsne_coord[idx, 1].squeeze(),
                    name = classe,
                    ncontours = nbins,
                    hoverinfo = 'skip',
                    colorscale = [[0, density_color_lo_str], [1, density_color_hi_str]],
                    showscale = False
                ))



        elif ndim==3:
            if prototypes == 'Top':
                df_class = DF_PROBS.loc[DF_PROBS['Class']==classe]
                nproto = min(nprototypes, len(df_class.index))
                ids_proto = df_class.sort_values(by=classe).tail(nproto).index.to_numpy()
                idx_proto = [np.where(ids==i) for i in ids_proto]
                idx_proto = np.concatenate(idx_proto, axis=1)[0]
                idx_proto = (np.array(idx_proto), )
                traces.append(go.Scatter3d(x = tsne_coord[idx_proto, 0].squeeze(),
                                           y = tsne_coord[idx_proto, 1].squeeze(),
                                           z = tsne_coord[idx_proto, 2].squeeze(),
                                           mode='markers',
                                           legendgroup='prototypes',
                                           # opacity=alpha,
                                           marker=dict(color=marker_color_str,
                                                       symbol='diamond',
                                                       size=5),
                                           name=classe,
                                           text=ids[idx_proto]
                                          ))
            traces.append(go.Scatter3d(x = tsne_coord[idx, 0].squeeze(),
                                       y = tsne_coord[idx, 1].squeeze(),
                                       z = tsne_coord[idx, 2].squeeze(),
                                       mode='markers',
                                       opacity=alpha,
                                       marker=dict(size=3,
                                                   color = marker_color_str),
                                       name=classe,
                                       text=ids[idx]
                                    ))

    if ndim==2:
        return {
            'data': traces,
            'layout': go.Layout(
                xaxis={'title': 't-SNE 1'},
                yaxis={'title': 't-SNE 2'},
                hovermode='closest',
                uirevision=dict_tsne  # Reset zoom, pan, etc, only when tSNE is updated
            ),
        }
    if ndim==3:
        return {
            'data': traces,
            'layout': go.Layout(
                go.Scene(
                  xaxis={'title': 't-SNE 1'},
                  yaxis={'title': 't-SNE 2'},
                  #zaxis={'title': 't-SNE 3'}
                ),
                hovermode='closest',
                uirevision=dict_tsne  # Reset zoom, pan, etc, only when tSNE is updated
            )
        }

# ----------------------------------------------------------------------------------------------------------------------
# Plot measurement of hovered trajectory
@app.callback(
    Output('plot-meas', 'figure'),
    [Input('plot-tsne', 'hoverData'),
     Input('slider-yrange', 'value'),
     Input('check-xshow', 'value'),
     Input('check-xrange', 'value'),
     Input('drop-overlay', 'value'),
     Input('drop-target', 'value'),
     Input('drop-bin', 'value')]
)
def update_plot_meas(hoverData, yrange, xshow, xcheck, overlay, target, bin):
    if hoverData is not None:
        hovered_cell = hoverData['points'][0]['text']
        dff = df[df['ID'] == hovered_cell]
        if overlay == 'None':
        # Create one trace per measurement
            plot = {
                'data': [go.Scattergl(
                    x=dff[dff['variable'] == meas]['Time'],
                    y=dff[dff['variable'] == meas]['value'],
                    mode='lines+markers',
                    name=meas,
                    marker={'opacity': 0.75}
                ) for meas in measurement],
                'layout': {
                    'title': hovered_cell,
                    'xaxis':{'title': 'Time'},
                    'yaxis':{'title': measurement,
                             'range': [yrange[0],yrange[1]]},
                    'uirevision': 'foo'  # Never reset zoom, pan, etc.
                }
            }

        else:
            # Create overlay of colors
            # Rebuild measurement array from melted dataframe, much faster than retrieving in dataloader in big dataset
            hovered_series = dff.pivot(index='variable', columns='Time', values='value')
            hovered_series = hovered_series.reindex(measurement, axis=0)
            hovered_series = hovered_series.reindex([str(i) for i in sorted(hovered_series.columns.astype('int'))], axis=1)
            hovered_series = np.array(hovered_series)
            # Subtract preprocessing before network (same as in class_dataset)
            if isinstance(subtract_numbers, (int, float)):
                subtract_array = np.array([subtract_numbers] * hovered_series.shape[0])
            elif isinstance(subtract_numbers, list):
                subtract_array = np.array(subtract_numbers)
            subtract_array = np.ones_like(hovered_series) * subtract_array[:, None]
            hovered_series -= subtract_array
            hovered_series = torch.tensor(hovered_series).to(device)

            if 'DF_CROP' in globals():
                start = int(DF_CROP.loc[DF_CROP['ID']==hovered_cell]['Crop_start'])
                end = int(DF_CROP.loc[DF_CROP['ID'] == hovered_cell]['Crop_end'])
                hovered_series = hovered_series[:, start:end]

            if overlay == 'CAM':
                ov_array = create_cam(model=net_unit_batch, device=device, array_series=hovered_series, target_class=target)
            elif overlay == 'Guided Backprop':
                ov_array = create_gbackprop(model=net_unit_batch, device=device, array_series=hovered_series, target_class=target)

            cmap = 'Jet'
            cmin, cmax = np.min(ov_array), np.max(ov_array)
            if bin == 'Li Threshold':
                thresh = threshold_li(ov_array)
                ov_array = np.where(ov_array >= thresh, 1, 0).astype('float')
                cmap = [[0,'grey'], [1,'red']]
                cmin, cmax = 0, 1
            elif bin == 'Mean Threshold':
                thresh = threshold_mean(np.abs(ov_array))
                ov_array = np.where(np.abs(ov_array) >= thresh, 1, 0).astype('float')
                cmap = [[0,'grey'], [1,'red']]
                cmin, cmax = 0, 1

            # Padding with NAs to compensate cropped time points and have overlay start and end in the right x positions, no pad in y
            if len(ov_array.shape) == 1:
                ov_array = np.reshape(ov_array, (1,-1)) # add dummy dimension for pad function for univariate series
            ov_array = np.pad(ov_array, [(0,0),
                                (int(DF_CROP.loc[DF_CROP['ID']==hovered_cell]['Crop_start']),
                                 int(len(dff)/len(measurement)) - int(DF_CROP.loc[DF_CROP['ID']==hovered_cell]['Crop_end']))],
                         'constant', constant_values=np.NaN)

            plot = {
                'data': [go.Scattergl(
                    x=dff[dff['variable'] == meas]['Time'],
                    y=dff[dff['variable'] == meas]['value'],
                    mode='lines+markers',
                    name=meas,
                    marker={'color': ov_array[imeas, :], 'colorscale': cmap, 'cmin':cmin, 'cmax':cmax,
                            'showscale': True, 'opacity': 0.75,}
                ) for imeas, meas in enumerate(measurement)],
                'layout': {
                    'title': hovered_cell,
                    'xaxis':{'title': 'Time'},
                    'yaxis':{'title': measurement,
                             'range': [yrange[0],yrange[1]]},
                    'uirevision': 'foo'  # Never reset zoom, pan, etc.
                }
            }

        # Add vertical lines to bound part of the signal seen by the network
        if 'DF_CROP' in globals():
            if xshow:
                x_start = int(DF_CROP.loc[DF_CROP['ID']==hovered_cell]['Crop_start'] + start_time)
                x_end =  int(DF_CROP.loc[DF_CROP['ID'] == hovered_cell]['Crop_end'] + start_time) - 1
                plot['layout']['shapes'] = [
                    {
                        'type': 'line',
                        'xref': 'x',
                        'yref': 'paper',
                        'x0': x_start,
                        'y0': 0,
                        'x1': x_start,
                        'y1': 1,
                        'opacity': 0.7,
                        'line': {'color': 'red', 'width': 2.5, 'dash': 'dash'}
                    },
                    {
                        'type': 'line',
                        'xref': 'x',
                        'yref': 'paper',
                        'x0': x_end,
                        'y0': 0,
                        'x1': x_end,
                        'y1': 1,
                        'opacity': 0.7,
                        'line': {'color': 'red', 'width': 2.5, 'dash': 'dash'}
                    }
                ]
            if xcheck:
                plot['layout']['xaxis'].update({'range': [x_start, x_end]})
    else:
        # Dummy placeholder plot for initialization, otherwise doesn't update upon subsequent hover
        plot = {
            'data': [go.Scattergl(
                x=[0],
                y=[min_slider],
                mode='lines+markers',
                name=meas,
                marker={'opacity': 0.75}
            ) for meas in measurement],
            'layout': {
                'title': '',
                'xaxis':{'title': 'Time'},
                'yaxis':{'title': measurement,
                         'range': [yrange[0],yrange[1]]}
            }
        }
    return plot



# ----------------------------------------------------------------------------------------------------------------------
# Display probability of classification of hovered trajectory
@app.callback(
    Output('table-proba', 'children'),
    [Input('plot-tsne', 'hoverData')]
)
def update_table_proba(selected_id):
    def generate_table(dataframe, hilight_class, hilight_pred):
        dict_styles = {
            col:{}
            for col in dataframe.columns
        }
        if hilight_class != hilight_pred:
            dict_styles.update({
                hilight_class: {'background-color':'#96ceb4'},
                hilight_pred: {'background-color':'#ff6f69'}
            })
        else:
            dict_styles.update({
                hilight_class: {'background-color':'#ffcc5c'}
            })

        table_header = [
            html.Tr([
                    html.Th(col, style=dict_styles[col])
                    for col in dataframe.columns
                ])
        ]

        table_body = [
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ])
            for i in range(len(dataframe))
        ]

        return dbc.Table(
            table_header + table_body,
            striped=True,
            responsive=True  # horizontal scroll possible on the table
        )

    if selected_id is not None:
        cell_id = selected_id['points'][0]['text']
        cell_class = classes_dict[df.loc[df['ID'] == cell_id]['Class'].unique()[0]]
        cell_probs = DF_PROBS.loc[[cell_id]]
        cell_probs.drop('Class', axis=1, inplace=True)
        cell_probs = round(cell_probs, 4)
        cell_predict = cell_probs.idxmax(1)[0]  # Name of the predicted class
        return generate_table(cell_probs, cell_class, cell_predict)


# ----------------------------------------------------------------------------------------------------------------------
# Export selected trajectories
@app.callback(
    Output('download-export', 'data'),
    [Input('button-export', 'n_clicks')],
    [State('plot-tsne', 'selectedData'),
     State('drop-export', 'value'),
     State('download-export', 'data')],
    prevent_initial_call = True
)
def export_selection(nclicks, selected_points, export_options, current_data):
    if selected_points is None:
        # Return the current value if no point is selected
        print('No data point selected for export.')
        return current_data
    selected_points = selected_points['points']
    selected_ids = [point['text'] for point in selected_points]
    dff = deepcopy(df[df['ID'].isin(selected_ids)])  # copy avoids pandas warning about modifying copy
    dff['column_wide'] = dff['variable'] + '_' + dff['Time'].astype('object')
    col_order = [meas + '_' + str(ti) for meas in measurement for ti in sorted(dff['Time'].unique().astype('int'))]
    # Rebuild measurement array from melted dataframe, much faster than retrieving in dataloader in big dataset
    dff_wide = dff.pivot(index='ID', columns='column_wide', values='value')
    dff_wide = dff_wide.reindex(col_order, axis=1)

    # ----------------------------------------------
    # Add crop, tsne coords and model output
    frame_class = dff[['ID', 'Class']].groupby('ID').first().reset_index()
    frame_class.rename(columns={'Class': 'Class_ID'}, inplace=True)
    frame_class['Class'] = frame_class['Class_ID']
    frame_class.replace({'Class': classes_dict}, inplace=True)

    frame_coord = {point['text']: (round(point['x'], 4), round(point['y'], 4)) for point in selected_points}
    frame_coord = pd.DataFrame.from_dict(frame_coord, orient='index', columns=['xTSNE', 'yTSNE'])
    frame_coord.reset_index(inplace=True)
    frame_coord.rename(columns={'index': 'ID'}, inplace=True)

    frame_probs = deepcopy(DF_PROBS.loc[selected_ids])
    frame_probs.drop('Class', axis=1, inplace=True)
    frame_probs = round(frame_probs, 4)
    frame_probs.columns = ['Prob_' + col for col in frame_probs.columns]
    frame_probs.reset_index(inplace=True)
    frame_probs.rename(columns={'index': 'ID'}, inplace=True)

    frame_feats = deepcopy(DF_FEATS.loc[selected_ids])
    frame_feats = round(frame_feats, 4)
    frame_feats.reset_index(inplace=True)
    frame_feats.rename(columns={'index': 'ID'}, inplace=True)

    frame_crop = deepcopy(DF_CROP[DF_CROP['ID'].isin(selected_ids)])
    # ----------------------------------------------
    # Merge the requested frames
    df_out = [frame_class['ID']]
    if 'Class' in export_options:
        df_out.append(frame_class)
    if 'Coord' in export_options:
        df_out.append(frame_coord)
    if 'Probability' in export_options:
        df_out.append(frame_probs)
    if 'Feature' in export_options:
        df_out.append(frame_feats)
    if 'Crop' in export_options:
        df_out.append(frame_crop)
    if 'Input' in export_options:
        df_out.append(dff_wide)
    if len(df_out) == 1:  # if only the ID column
        df_out = df_out[0].to_frame()
    else:
        df_out = reduce(lambda left,right: pd.merge(left, right, on='ID'), df_out)
    df_out.set_index('ID', inplace=True)

    return dcc.send_data_frame(df_out.to_csv, filename='export_table.csv')

# ----------------------------------------------------------------------------------------------------------------------
# Open popover if export to csv is empty
@app.callback(
    Output('popover-export', 'is_open'),
    [Input('button-export', 'n_clicks')],
    [State('plot-tsne', 'selectedData')],
    prevent_initial_call = True
)
def toggle_popover_export(nclicks, selected_points):
    if selected_points is None:
        return True

# ----------------------------------------------------------------------------------------------------------------------
# Export Plot of the tSNE
@app.callback(
    Output('download-exportPlot', 'data'),
    [Input('button-exportPlot', 'n_clicks')],
    [State('plot-tsne', 'figure'),
     State('input-exportPlot-height', 'value'),
     State('input-exportPlot-width', 'value'),
     State('input-exportPlot-scale', 'value'),
     State('radio-exportPlot', 'value')],
    prevent_initial_call = True
)
def export_tSNEPDF(nclicks, fig, height, width, scale, fiformat):
    fig_exp = go.Figure()
    fig_data = fig['data']
    fig_layout = fig['layout']

    # Scattergl is rendered as blurry raster
    for trace in fig_data:
        if trace['type'] == 'scattergl':
            trace['type'] = 'scatter'
    fig_exp.add_traces(fig_data)
    fig_exp.update_layout(fig_layout)
    # Bytes string encoding the figure
    stringFig = fig_exp.to_image(format=fiformat, engine='kaleido', width=width, height=height, scale=float(scale))

    return dcc.send_bytes(stringFig, filename='export_figure.{}'.format(fiformat))


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=False, port=args.port)
