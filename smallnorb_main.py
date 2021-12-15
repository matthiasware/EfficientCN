import sys
sys.path.append("./..")

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
from smallnorb_effcn_train_conf import train
from smallnorb.jitter import ColorJitter


def conf1():
    #Tranformations
    transform_train = T.Compose([
        #.Normalize(mean=[191.7811,193.0594],std=[45.2232, 44.2558]),
        T.Normalize(mean=[127.5, 127.5],std=[127.5, 127.5]),
        T.Resize(64),
        T.RandomCrop(48),
        ColorJitter(brightness= [0.,2.], contrast=[0.5,1.5], saturation=0, hue=0),
    ])
    transform_valid = T.Compose([
        #T.Normalize(mean=[191.0684,192.0952],std=[45.4354, 44.3388]),
        T.Normalize(mean=[127.5, 127.5],std=[127.5, 127.5]),
        T.Resize(64),
        T.CenterCrop(48),
    ])  

    batch_size = 16
    num_epochs = 200
    num_workers = 2

    config = {
        'device': 'cuda:0',
        'debug': True,
        'train': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_workers': num_workers,
            'num_vis': 8,
            'pin_memory': True,
            'transform' : transform_train
        },
        'valid': {
            'num_workers': num_workers,       # Either set num_worker high or pin_memory=True
            'batch_size': batch_size,
            'num_vis': 8,
            'pin_memory': True,
            'transform' : transform_valid
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
        'stop_acc': 1.0 #0.9973
    }

    return DottedDict(config)

def conf2():
    #Tranformations
    transform_train = T.Compose([
        T.Resize(64),
        T.RandomCrop(48),
        ColorJitter(brightness= [0.,2.], contrast=[0.5,1.5], saturation=0, hue=0),
        T.Normalize(mean=[127.5, 127.5],std=[127.5, 127.5]),
    ])
    transform_valid = T.Compose([
        T.Resize(64),
        T.CenterCrop(48),
        T.Normalize(mean=[127.5, 127.5],std=[127.5, 127.5]),
    ])     

    batch_size = 16
    num_epochs = 200
    num_workers = 2

    config = {
        'device': 'cuda:0',
        'debug': True,
        'train': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_workers': num_workers,
            'num_vis': 8,
            'pin_memory': True,
            'transform' : transform_train
        },
        'valid': {
            'num_workers': num_workers,       # Either set num_worker high or pin_memory=True
            'batch_size': batch_size,
            'num_vis': 8,
            'pin_memory': True,
            'transform' : transform_valid
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
        'stop_acc': 1.0 #0.9973
    }

    return DottedDict(config)

def conf3():
    #Tranformations
    transform_train = T.Compose([
        T.Normalize(mean=[191.7811,193.0594],std=[45.2232, 44.2558]),
        T.Resize(64),
        T.RandomCrop(48),
        ColorJitter(brightness= [0.,2.], contrast=[0.5,1.5], saturation=0, hue=0),
    ])
    transform_valid = T.Compose([
        T.Normalize(mean=[191.0684,192.0952],std=[45.4354, 44.3388]),
        T.Resize(64),
        T.CenterCrop(48),
    ])     

    batch_size = 16
    num_epochs = 200
    num_workers = 2

    config = {
        'device': 'cuda:0',
        'debug': True,
        'train': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_workers': num_workers,
            'num_vis': 8,
            'pin_memory': True,
            'transform' : transform_train
        },
        'valid': {
            'num_workers': num_workers,       # Either set num_worker high or pin_memory=True
            'batch_size': batch_size,
            'num_vis': 8,
            'pin_memory': True,
            'transform' : transform_valid
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
        'stop_acc': 1.0 #0.9973
    }

    return DottedDict(config)

def conf4():
    #Tranformations
    transform_train = T.Compose([
        T.Normalize(mean=[127.5, 127.5],std=[127.5, 127.5]),
        T.Resize(64),
        T.RandomCrop(48),
        ColorJitter(brightness= [0.,2.], contrast=[0.5,1.5], saturation=0, hue=0),
    ])
    transform_valid = T.Compose([
        T.Normalize(mean=[127.5, 127.5],std=[127.5, 127.5]),
        T.Resize(64),
        T.CenterCrop(48),
    ])  

    batch_size = 16
    num_epochs = 200
    num_workers = 2

    config = {
        'device': 'cuda:0',
        'debug': True,
        'train': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_workers': num_workers,
            'num_vis': 8,
            'pin_memory': True,
            'transform' : transform_train
        },
        'valid': {
            'num_workers': num_workers,       # Either set num_worker high or pin_memory=True
            'batch_size': batch_size,
            'num_vis': 8,
            'pin_memory': True,
            'transform' : transform_valid
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

    train(conf1())
    train(conf2())
    train(conf3())
    train(conf4())
