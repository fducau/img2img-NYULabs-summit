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
parser.add_argument('--transform_path', help='path to dataset', default='/home/paperspace/Desktop/data/test/hed_faces_wild/')
parser.add_argument('--reload_model', help='model to be used for prediction')
parser.add_argument('--save_folder', help='Path to save predictions')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--outf', default='./runs/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')


opt = parser.parse_args()
opt.no_lsgan = True
opt.cuda=True
print(opt)

def generation_loop(dataloader, model, opt, save_imgs=False, normalize=True):
    visuals_output = []
    for i, data_edges in enumerate(dataloader):
        data_edges[0] = data_edges[0][None:]
        model.set_input(data_edges)
        model.forward()

        visuals = model.get_current_visuals()
        visuals_concat = torch.cat([visuals['fake_in'].data, visuals['fake_out'].data], 0)
        visuals_output.append(visuals_concat)

        if save_imgs:

            vutils.save_image(visuals_concat,
                              '{}/generation_{}.png'.format(opt['save_folder'], i),
                              normalize=True)

    return visuals_output

def generate_through_models(dataloader, model, opt):
    model_names = get_all_model_names(opt)
    model_paths = [opt['outf'] + opt['exp_name'] + '/' + model_name for model_name in model_names]

    for model_path, model_name in zip(model_paths, model_names):
        checkpoint = torch.load(model_path)
        model.netG.load_state_dict(checkpoint)
        print("=> loaded checkpoint {}".format(model_name))

        imgs_output =  generation_loop(dataloader, model, save_imgs=False)
        for img in imgs_output:
            vutils.save_image(img,
                              '{}/{}_{}.png'.format(opt['save_folder'], model_name, i),
                              normalize=True)



def get_all_model_names(opt):

    experiment_files = os.listdir(opt['outf'] + opt['exp_name'])
    netG_files = [e for e in experiment_files if 'netG_epoch' in e]

    return netG_files


def get_latest_model_name(opt):

    netG_files = get_all_model_names(opt)
    candidate_files = [e.split('_')[-1] for e in netG_files]
    candidate_files = np.array([int(e.split('.')[0]) for e in candidate_files])

    return netG_files[candidate_files.argmax()]


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

opt.update(vars(opt_generate))
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
        reload_model = get_latest_model_name(opt)
        reload_model_path = opt['outf'] + opt['exp_name'] + '/' + reload_model
        opt['reload_model'] = reload_model
        opt['reload_model_path'] = reload_model_path
    except:
        raise UnboundLocalError('No candidate model found to reload.')

if not os.path.isdir(opt['transform_path']):
    raise ValueError('Not a valid samples folder to transform {}'.format(opt['transform_path']))
if not os.path.isfile(opt['reload_model_path']):
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

if os.path.isfile(opt['reload_model_path']):
    print("=> loading checkpoint '{}'".format(opt['reload_model']))
    checkpoint = torch.load(opt['reload_model_path'])
    model.netG.load_state_dict(checkpoint)
    print("=> loaded checkpoint {}".format(opt['reload_model']))
else:
    print("=> no checkpoint found at '{}'".format(opt['reload_model']))

model.train_mode = False

generation_loop(dataloader, model, opt, save_imgs=True)





