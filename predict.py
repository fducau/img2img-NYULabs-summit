'''
Base code from https://github.com/pytorch/examples/tree/master/dcgan
'''

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from itertools import cycle
from itertools import izip
import time
from model import netModel
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import copy
import numpy as np
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', help='experiment name', required=True)
parser.add_argument('--transform_path', help='path to dataset', default='./data/test/edges/')
parser.add_argument('--reload_model', help='model to be used for prediction')
parser.add_argument('--save_folder', help='Path to save predictions')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--outf', default='./runs/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')


opt = parser.parse_args()
opt.no_lsgan = True
opt.cuda=True
print(opt)

try:
    opt_experiment = pkl.load(open(opt.outf + opt.exp_name + '/options_dictionary.pkl', 'r'))
except:
    print('Options dictionary could not found in experiment folder {}'.format(opt.outf + opt.exp_name))

opt_generate = opt
opt = opt_experiment
if not isinstance(opt, dict):
    try:
        opt = vars(opt)
    except:
        raise ValueError('Reloaded opttions dictionary could not be read.')

opt['update'](vars(opt_generate))
opt['batchSize'] = 1

if opt['save_folder'] is None:
    opt['save_folder'] = opt['outf'] + opt['exp_name'] + '/generation/'

try:
    os.makedirs(opt['save_folder'])
except OSError:
    pass

# Reload last model found in experiment folder if not defined
if opt['reload_model'] is None:
    try:
        experiment_files = os.listdir(opt['outf'] + opt['exp_name'])
        netG_files = [e for e in experiment_files if 'netG_epoch' in e]
        candidate_files = [e.split('_')[-1] for e in netG_files]
        candidate_files = np.array([int(e.split('.')[0]) for e in candidate_files])

        reload_model = opt['outf'] + opt['exp_name'] + '/' + netG_files[candidate_files.argmax()]
        opt.reload_model = reload_model
    except:
        raise UnboundLocalError('No candidate model found to reload.')

if not os.path.isdir(opt['transform_path']):
    raise ValueError('Not a valid samples folder to transform {}'.format(opt['transform_path']))
if not os.path.isfile(opt['reload_model']):
    raise ValueError('Model {} not found'.format(opt['reload_model']))




if opt['manualSeed'] is None:
    opt['manualSeed'] = random.randint(1, 10000)
print("Random Seed: ", opt['manualSeed'])
random.seed(opt['manualSeed'])
torch.manual_seed(opt['manualSeed'])
if opt['cuda']:
    torch.cuda.manual_seed_all(opt['manualSeed'])

cudnn.benchmark = True

if torch.cuda.is_available() and not opt['cuda']:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# folder dataset
dataset_transform = dset.ImageFolder(
    root=opt['transform_path'],
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)

dataset_size = len(dataset_transform)
dataloader = torch.utils.data.DataLoader(dataset_transform, batch_size=opt['batchSize'],
                                         shuffle=False, num_workers=int(opt['workers']))



model = netModel()
model.initialize(opt)
print("model was created")

if os.path.isfile(opt['reload_model']):
    print("=> loading checkpoint '{}'".format(opt['reload_model']))
    checkpoint = torch.load(opt['reload_model'])
    model.netG.load_state_dict(checkpoint)
    print("=> loaded checkpoint {}".format(opt['reload_model']))
else:
    print("=> no checkpoint found at '{}'".format(opt['reload_model']))

model.train_mode = False

for i, data_edges in enumerate(dataloader):
    data_edges[0] = data_edges[0][None:]
    model.set_input(data_edges)
    model.forward()

    visuals = model.get_current_visuals()
    visuals_concat = np.concatenate([visuals['fake_in'].data, visuals['fake_out'].data], 0)

    vutils.save_image(visuals_concat,
                      '{}/generation_{}.png'.format(opt['outf'] + opt['exp_name'], i),
                      normalize=True)
