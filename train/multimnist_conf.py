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
from train.multimnist_effcn_train_test import train


def conf():
    #Tranformations
    transform_train = None
    transform_valid = None

    batch_size = 64
    num_epochs = 10 #150
    num_workers = 2
    leraning_rate = 5e-4

    config = {
        'device': 'cuda:0',
        'debug': True,
        'train': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_workers': num_workers,
            'num_vis': 8,
            'pin_memory': True,
            'transform' : transform_train,
        },
        'valid': {
            'num_workers': num_workers,       # Either set num_worker high or pin_memory=True
            'batch_size': batch_size,
            'num_vis': 8,
            'pin_memory': True,
            'transform' : transform_valid,
        },
        'gen': {
            'generate': False,       
            'num': [1000,1000]
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
            'data': '/mnt/data/datasets/multimnist',
            'experiments': '/mnt/data/experiments/EfficientCN/multimnist',
        },
        'names': {
            'model_dir': 'effcn_multimnist_{}'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')),
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
    c1.train.batch_size = 64
    c1.valid.batch_size = 64
    c1.optimizer_args.lr = 5e-5
    train(c1)
    time.sleep(1)

    c2 = conf()
    c2.train.batch_size = 64
    c2.valid.batch_size = 64
    c2.optimizer_args.lr = 1e-4
    c2.freqs.ckpt = 1
    train(c2)
    """
    c3 = conf()
    c3.train.batch_size = 1064
    c3.valid.batch_size = 1064
    c3.optimizer_args.lr = 5e-4
    c3.freqs.ckpt = 1
    train(c3)

    #time.sleep(1)
    
    #c4 = conf()
    #c4.train.batch_size = 4000
    #c4.valid.batch_size = 4000
    #train(c4)
    
