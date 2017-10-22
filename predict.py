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

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', help='experiment name', required=True)
parser.add_argument('--transform_path', help='path to dataset', default='./data/test/edges/', required=True)
parser.add_argument('--reload_model', help='model to be used for prediction', required=True)
parser.add_argument('--save_folder', help='Path to save predictions')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--outf', default='./runs/', help='folder to output images and model checkpoints')


opt = parser.parse_args()
opt.no_lsgan = True
opt.cuda=True
print(opt)

if opt.save_folder is None:
    opt.save_folder = opt.outf + exp_name

try:
    os.makedirs(opt.save_folder)
except OSError:
    pass

if not os.path.isdir(opt.transform_path):
    raise ValueError('Not a valid samples folder to transform {}'.format(opt.transform_path))
if not os.path.isfile(opt.reload_model):
    raise ValueError('Model {} not found'.format(opt.reload_model))


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# folder dataset
dataset_transform = dset.ImageFolder(
    root=opt.transform_path,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)

dataset_size = len(dataset_transform)
dataloader = torch.utils.data.DataLoader(dataset_transform, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))


model = netModel()
model.initialize(opt)
print("model was created")

if os.path.isfile(opt.reload_model):
    print("=> loading checkpoint '{}'".format(opt.reload_model))
    checkpoint = torch.load(opt.reload_model)
    model.netG.load_state_dict(checkpoint)
    print("=> loaded checkpoint {}".format(opt.reload_model))
else:
    print("=> no checkpoint found at '{}'".format(opt.resume))


