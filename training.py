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
parser.add_argument('--dataroot_faces', help='path to dataset', default='./data/train/faces/')
parser.add_argument('--dataroot_adv', help='path to dataset', default='./data/train/adversarial/')
parser.add_argument('--dataroot_edges', help='path to dataset', default='./data/train/edges')

parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')

parser.add_argument('--B_width', type=int, default=256, help='the width of the Face input image to network')
parser.add_argument('--B_height', type=int, default=256, help='the height of the Face input image to network')
parser.add_argument('--A_width', type=int, default=256, help='the width of the Edge input image to network')
parser.add_argument('--A_height', type=int, default=256, help='the height of the Edge input image to network')


parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='Number of generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lr_update_every', type=int, default=50, help='Number of epochs to update learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--L1lambda', type=float, default=0.01, help='Loss in generator')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./runs/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--which_model_netG', type=str, default='unet_128', help='selects model to use for netG')
parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
parser.add_argument('--norm', type=str, default='batch', help='batch normalization or instance normalization')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
parser.add_argument('--display_freq', type=int, default=100, help='Save images frequency')
parser.add_argument('--print_freq', type=int, default=50, help='Screen output frequency')

opt = parser.parse_args()
opt.no_lsgan = True
opt.cuda=True
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

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
dataset_faces = dset.ImageFolder(
    root=opt.dataroot_faces,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)

assert dataset_faces
dataset_size = len(dataset_faces)
dataloader_faces = torch.utils.data.DataLoader(dataset_faces,
                                               batch_size=opt.batchSize,
                                               shuffle=False,
                                               num_workers=int(opt.workers))

# folder dataset
dataset_edges = dset.ImageFolder(
    root=opt.dataroot_edges,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)

dataset_adv = dset.ImageFolder(
    root=opt.dataroot_adv,
    transform=transforms.Compose([transforms.ToTensor()])
)

assert dataset_edges
assert dataset_adv
dataloader_edges = torch.utils.data.DataLoader(dataset_edges,
                                               batch_size=opt.batchSize,
                                               shuffle=False,
                                               num_workers=int(opt.workers))

dataloader_adv = torch.utils.data.DataLoader(dataset_adv,
                                             batch_size=opt.batchSize,
                                             shuffle=True,
                                             num_workers=int(opt.workers))
model = netModel()
model.initialize(opt)
print("model was created")
# Add visualizer?

# Create output folder
if not os.path.isdir(opt.outf + opt.exp_name):
    if not os.path.isdir(opt.outf):
        os.mkdir(opt.outf)
    os.mkdir(opt.outf + opt.exp_name)

pkl.dump(opt, open('{}/options_dictionary.pkl'.format(opt.outf + opt.exp_name), 'wb'))


total_steps = 0
for epoch in range(opt.niter):
    epoch_start_time = time.time()
    i = -1
    for data_faces, data_adv, data_edges in izip(dataloader_faces, dataloader_adv, dataloader_edges):
        i += 1
        iter_start_time = time.time()
        total_steps += opt.batchSize

        model.set_input((data_faces, data_adv, data_edges))
        model.optimize_parameters()

        if i % opt.display_freq == 0:
            visuals = model.get_current_visuals()

            vutils.save_image(visuals['real_out'].data,
                              '%s/real_samples.png' % (opt.outf + opt.exp_name),
                              normalize=True)
            vutils.save_image(visuals['fake_out'].data,
                              '%s/fake_samples_epoch_%03d.png' % (opt.outf + opt.exp_name, epoch),
                              normalize=True)
            vutils.save_image(visuals['fake_in'].data,
                              '%s/input_samples.png' % (opt.outf + opt.exp_name),
                              normalize=True)

        if total_steps % (opt.print_freq * opt.batchSize) == 0:
            errors = model.get_current_errors()
            print('[{}/{}] Epoch: {}, G_GAN: {:.4f}, G_L1: {:.4f}, D_real: {:.4f}, D_fake: {:.4f}'.format(
                  total_steps, dataset_size, epoch,
                  errors['G_GAN'], errors['G_L1'], errors['D_real'],
                  errors['D_fake']))

            # do model checkpoint
            torch.save(model.netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf + opt.exp_name, epoch))
            torch.save(model.netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf + opt.exp_name, epoch))

        if (epoch != 0) and (epoch % opt.lr_update_every == 0):
            model.update_learning_rate()
