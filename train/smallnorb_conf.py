import sys
sys.path.append("./..")
sys.path.append('../')
sys.path.append(".")

# default libraries
import time
import datetime
import pickle
import pprint
from pathlib import Path

# third party libraries
import torch
import torchvision
import torchvision.transforms as T

import numpy as np
from dotted_dict import DottedDict

# local imports
from train.smallnorb_effcn_train import train


def conf():
    #Tranformations
    transform_train = T.Compose([
        T.ColorJitter(brightness= [0.5,1.], contrast=[0.5,1.], saturation=0, hue=0),
        T.Resize(64),
        T.RandomCrop(48),
        T.Normalize(mean=[191.7811/255,193.0594/255,0],std=[45.2232/255, 44.2558/255,1]),
    ])
    transform_valid = T.Compose([
        T.Resize(64),
        T.CenterCrop(48),        
        T.Normalize(mean=[191.0684/255,192.0952/255,0],std=[45.4354/255, 44.3388/255,1]),        
    ])  



    batch_size = 16
    num_epochs = 200
    num_workers = 2

    config = {
        'device': 'cuda:0',
        'debug': False,
        'train': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_workers': num_workers,
            'num_vis': 16,
            'pin_memory': True,
            'transform' : transform_train,
            'mode' : "pseudo"
        },
        'valid': {
            'num_workers': num_workers,       # Either set num_worker high or pin_memory=True
            'batch_size': batch_size,
            'num_vis': 16,
            'pin_memory': True,
            'transform' : transform_valid,
            'mode' : "pseudo"
        },
        'optimizer': 'adam',
        'optimizer_args': {
            'lr': 5e-4,
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
            'data': 'data/SmallNORB',
            'experiments': 'experiments',
        },
        'names': {
            'model_dir': 'effcn_smallnorb_{}'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')),
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
                'by_class': False
            }
        },
        'stop_acc': 0.9973
    }

    return DottedDict(config)

def conf_debug():
    #Tranformations
    transform_train = T.Compose([
        T.ColorJitter(brightness= [0.5,1.], contrast=[0.5,1.], saturation=0, hue=0),
        T.Resize(64),
        T.RandomCrop(48),
        T.Normalize(mean=[191.7811/255,193.0594/255,0],std=[45.2232/255, 44.2558/255,1]),
    ])
    transform_valid = T.Compose([
        T.Resize(64),
        T.CenterCrop(48),        
        T.Normalize(mean=[191.0684/255,192.0952/255,0],std=[45.4354/255, 44.3388/255,1]),        
    ])  



    batch_size = 256
    num_epochs = 1
    num_workers = 2

    config = {
        'device': 'cuda:0',
        'debug': True,
        'train': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_workers': num_workers,
            'num_vis': 16,
            'pin_memory': True,
            'transform' : transform_train,
            'mode' : "pseudo"
        },
        'valid': {
            'num_workers': num_workers,       # Either set num_worker high or pin_memory=True
            'batch_size': batch_size,
            'num_vis': 16,
            'pin_memory': True,
            'transform' : transform_valid,
            'mode' : "pseudo"
        },
        'optimizer': 'adam',
        'optimizer_args': {
            'lr': 5e-4,
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
            'data': 'data/SmallNORB',
            'experiments': 'experiments',
        },
        'names': {
            'model_dir': 'effcn_smallnorb_{}'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')),
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

    return DottedDict(config)



if __name__ == '__main__':

    train(conf_debug())
    
    """
    i = 0
    while i < 10:
        train(conf1())
        i += 1
    """