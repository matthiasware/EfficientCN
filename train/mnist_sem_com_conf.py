import sys
sys.path.append("./..")
sys.path.append('.')

# default libraries
import time
import datetime
import pickle
import pprint
from pathlib import Path
import math


# third party libraries
import torch
import torchvision
import torchvision.transforms as T


import numpy as np
from dotted_dict import DottedDict

# local imports
from train.mnist_train_sem_com import train


def conf():
    #Tranformations
    transform_train = T.Compose([
        T.RandomAffine(
            degrees=(-30, 30),
            shear=(-30, 30),
            # translate=(0.9, 0.9),
        ),
        T.RandomResizedCrop(
            28,
            scale=(0.8, 1.2),
            ratio=(1, 1),
        ),
        T.ToTensor()
    ])
    transform_valid = T.Compose([
        T.ToTensor()
    ])

    batch_size = 256
    num_epochs = 150
    num_workers = 2
    leraning_rate = 5e-4
    model = 'MnistCNN_R' #MnistEffCapsNet, MnistCNN_CR_SF, MnistCNN_CR, MnistCNN_R

    config = {
        'model': model,
        'device': 'cuda:0',
        'debug': False,
        'train': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_workers': num_workers,
            'num_vis': 16,
            'pin_memory': True,
            'transform' : transform_train,
        },
        'valid': {
            'num_workers': num_workers,       # Either set num_worker high or pin_memory=True
            'batch_size': batch_size,
            'num_vis': 16,
            'pin_memory': True,
            'transform' : transform_valid,
        },
        'optimizer': 'adam',
        'optimizer_args': {
            'lr': leraning_rate,
            'weight_decay': 0.,
        },
        'scheduler': 'exponential_decay',
        'scheduler_burnin': 10,  # [epochs]
        'scheduler_args': {
            'gamma': 0.97
        },
        'freqs': {
            'valid': 1,   # [epochs]
            'rec': 1,     # [epochs] show reconstructions
            'ckpt': 10,   # [epochs]
        },
        'paths': {
            'data': '/mnt/data/datasets',
            'experiments': '/mnt/data/experiments/EfficientCN/mnist',
        },
        'names': {
            'model_dir': 'effcn_mnist_{a}_{b}'.format(a = model, b = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')),
            'ckpt_dir': 'ckpts',
            'img_dir': 'imgs',
            'log_dir': 'logs',
            'model_file': 'model_{}.ckpt',
            'stats_file': 'stats.pkl',
            'config_file': 'config.pkl',
            'acc_plot': 'acc.png',
            'loss_plot': 'loss.png',
        },
        'loss': {
            'margin': {
                'lbd': 0.5,
                'm_plus': 0.9,
                'm_minus': 0.1,
                'weight': 1.0
            },
            'rec': {
                'weight': 0.392,
                'by_class': True
            }
        },
        'stop_acc': 0.9973
    }

    config = DottedDict(config)
    return config



if __name__ == '__main__':
    #print(conf())

    #train(conf())
    """
    c1 = conf()
    c1.train.batch_size = 16
    c1.valid.batch_size = 16
    c1.optimizer_args.lr = 1e-4
    c1.model = 'MnistEffCapsNet' #MnistEffCapsNet, MnistCNN_CR_SF, MnistCNN_CR, MnistCNN_R
    c1.names.model_dir = 'effcn_mnist_{a}_{b}'.format(a = c1.model, b = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S'))
    train(c1)
    time.sleep(1)

    c2 = conf()
    c2.train.batch_size = 16
    c2.valid.batch_size = 16
    c2.optimizer_args.lr = 1e-4
    c2.model = 'MnistCNN_CR_SF' #MnistEffCapsNet, CNN_CR_SF, CNN_CR, CNN_R
    c2.names.model_dir = 'effcn_mnist_{a}_{b}'.format(a = c2.model, b = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S'))
    train(c2)
    time.sleep(1)

    c3 = conf()
    c3.train.batch_size = 16
    c3.valid.batch_size = 16
    c3.optimizer_args.lr = 1e-4
    c3.model = 'MnistCNN_CR' #MnistEffCapsNet, CNN_CR_SF, CNN_CR, CNN_R
    c3.names.model_dir = 'effcn_mnist_{a}_{b}'.format(a = c3.model, b = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S'))
    train(c3)
    time.sleep(1)
    """
    c4 = conf()
    c4.train.batch_size = 16
    c4.valid.batch_size = 16
    c4.optimizer_args.lr = 1e-4
    c4.model = 'MnistCNN_R' #MnistEffCapsNet, CNN_CR_SF, CNN_CR, CNN_R
    c4.names.model_dir = 'effcn_mnist_{a}_{b}'.format(a = c4.model, b = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S'))
    train(c4)




