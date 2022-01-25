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
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dotted_dict import DottedDict

# local imports
from effcn.models_smallnorb import SmallNorbEffCapsNet
from effcn.functions import create_margin_loss
from effcn.utils import count_parameters
from datasets.smallnorb import SmallNORB
from misc.optimizer import get_optimizer, get_scheduler

# will most likely result in a 30% speed up
torch.backends.cudnn.benchmark = True

def default():
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
        'debug': True,
        'train': {
            'batch_size': batch_size,
            'num_epochs': 1,
            'num_workers': num_workers,
            'num_vis': 8,
            'pin_memory': True,
            'transform' : transform_train,
            'mode' : "pseudo"
        },
        'valid': {
            'num_workers': num_workers,       # Either set num_worker high or pin_memory=True
            'batch_size': batch_size,
            'num_vis': 8,
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
            'data': '/mnt/data/datasets/smallnorb',
            'experiments': '/mnt/data/experiments/EfficientCN/smallnorb',
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

    config = DottedDict(config)
    return config

def eval_model(model, device, data_loader, config, func_margin, func_rec):
    model.eval()
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    for x, y_true, _ in data_loader:
        x = x.to(device)
        y_true = y_true.to(device)

        with torch.no_grad():
            if config.loss.rec.by_class == True:
                u_h, x_rec = model.forward(x, y_true)
            else:
                u_h, x_rec = model.forward(x)

            # LOSS
            y_one_hot = F.one_hot(y_true, num_classes=5)
            loss_margin = func_margin(u_h, y_one_hot)
            loss_rec = func_rec(x, x_rec)

            # total loss
            loss = (loss_margin * config.loss.margin.weight) + \
                (loss_rec * config.loss.rec.weight)

            # validate batch
            y_pred = torch.argmax(torch.norm(u_h, dim=2), dim=1)

            epoch_loss += loss.item()
            epoch_correct += (y_true == y_pred).sum().item()
            epoch_total += x.shape[0]
    epoch_acc = epoch_correct / epoch_total
    return epoch_loss, epoch_acc

def create_reconstruction_grid_img(model, device, x, y, permute=False):
    model.eval()
    with torch.no_grad():
        _, x_rec = model.forward(x.to(device))
        _, x_rec_y = model.forward(x.to(device),y.to(device))
    x_rec = x_rec.cpu()
    x_rec_y = x_rec_y.cpu()
    scal = lambda x: (x-x.min())/(x.max()-x.min())
    img = torchvision.utils.make_grid(
        torch.cat([scal(x[:,:1,:,:]), 
                    scal(x_rec[:,:1,:,:]),
                    scal(x_rec_y[:,:1,:,:]),
                    scal((x_rec[:,:1,:,:] - x_rec_y[:,:1,:,:])),
                    scal(x[:,1:2,:,:]), 
                    scal(x_rec[:,1:2,:,:]),
                    scal(x_rec_y[:,1:2,:,:]), 
                    scal((x_rec[:,1:2,:,:]-x_rec_y[:,1:2,:,:]))], dim=0), nrow=x.shape[0])
    if permute:
        img = img.permute(1, 2, 0)
    return img

def plot_acc_from_stats(stats, p_file):
    train_max = max(stats["train"]["acc"])
    valid_max = max(stats["valid"]["acc"])

    plt.figure(figsize=(10, 10))
    plt.plot(stats["train"]["epoch"], stats["train"]["acc"],
             label="train {:.5f}".format(train_max), color='b')
    plt.plot(stats["valid"]["epoch"], stats["valid"]["acc"],
             label="valid {:.5f}".format(valid_max), color='red')

    #
    plt.axhline(y=train_max, color='b', linestyle='dotted')
    plt.axhline(y=valid_max, color='red', linestyle='dotted')
    plt.title("ACC")
    plt.legend()
    plt.tight_layout()

    plt.savefig(p_file)
    plt.close()

def plot_loss_from_stats(stats, p_file):
    plt.figure(figsize=(10, 10))
    plt.plot(stats["train"]["epoch"], stats["train"]
             ["loss"], label="train", color='b')
    plt.plot(stats["valid"]["epoch"], stats["valid"]
             ["loss"], label="valid", color='red')
    #
    plt.title("LOSS")
    plt.legend()
    plt.tight_layout()

    plt.savefig(p_file)
    plt.close()

def mkdir_directories(dirs, parents, exist_ok):
    for director in dirs:
        Path(director).mkdir(parents=parents, exist_ok=exist_ok)


def train(config=None):

    if config == None:
        config = default()
        print("#" * 100)
        print("Config isn't set. \n Default Settings are choosen!")
        print("#" * 100)

    print("#"* 100)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    
    p_data = config.paths.data
    p_experiment = Path(config.paths.experiments) / config.names.model_dir
    p_ckpts = p_experiment / config.names.ckpt_dir
    p_logs = p_experiment / config.names.log_dir
    p_config = p_experiment / config.names.config_file
    p_stats = p_experiment / config.names.stats_file
    p_imgs = p_experiment / config.names.img_dir
    p_acc_plot = p_experiment / config.names.acc_plot
    p_loss_plot = p_experiment / config.names.loss_plot    

    
    device = torch.device(config.device)  
    
    ##################################
    #Get & Preprocess data

    #Tranformations
    transform_train = config.train.transform
    transform_valid = config.valid.transform
    mode_train = config.train.mode
    mode_valid = config.valid.mode

    #load Dataset
    ds_train = SmallNORB(root=p_data,train=True, download=True, transform=transform_train, mode=mode_train)
    ds_valid = SmallNORB(root=p_data,train=False, download=True, transform=transform_valid, mode=mode_valid)
    
    #stack data to batches
    dl_train = torch.utils.data.DataLoader(ds_train, 
                                        batch_size=config.train.batch_size, 
                                        shuffle=True, 
                                        persistent_workers=True,
                                        pin_memory=config.train.pin_memory,
                                        num_workers=config.train.num_workers)
    dl_valid = torch.utils.data.DataLoader(ds_valid, 
                                        batch_size=config.valid.batch_size, 
                                        shuffle=True, 
                                        persistent_workers=True,
                                        pin_memory=config.valid.pin_memory,
                                        num_workers=config.valid.num_workers)
    

    # Data for visualization of the img reconstructions
    x, y, _ = next(iter(dl_train))
    x_vis_train = x[:config.train.num_vis]
    y_vis_train = y[:config.train.num_vis]
    
    x, y, _ = next(iter(dl_valid))
    x_vis_valid = x[:config.valid.num_vis]
    y_vis_valid = y[:config.valid.num_vis]

  

    ##################################
    #Train Model

    #Model
    model = SmallNorbEffCapsNet()
    model = model.to(device)

    # optimizer
    optimizer = get_optimizer(
        config.optimizer, model.parameters(), config.optimizer_args)
    if config.scheduler is not None:
        scheduler = get_scheduler(
            config.scheduler, optimizer, config.scheduler_args)
    else:
        scheduler = None 


    # create directories
    if config.debug:
        # remove dir and recreate it if in debug mode
        if p_experiment.exists():
            shutil.rmtree(p_experiment)
        mkdir_directories([p_experiment, p_ckpts, p_logs,
                           p_imgs], parents=True, exist_ok=True)
    else:
        mkdir_directories([p_experiment, p_ckpts, p_logs,
                           p_imgs], parents=True, exist_ok=False)


    print("p_experiment:  {}".format(p_experiment))
    # summary writer
    sw = SummaryWriter(p_logs)
    print("tensorboard --logdir={}".format(str(p_logs)))
    print("#" * 100)

    # save configs
    with open(p_config, "wb") as file:
        pickle.dump(config, file)


    # custom training stats
    stats = {
        "train": {
            'acc': [],
            'loss': [],
            'epoch': [],
        },
        "valid": {
            'acc': [],
            'loss': [],
            'epoch': [],
        },
        "notes": [],
        "params": [count_parameters(model)]
    }

    # print params
    print("#params:            {:,}".format(count_parameters(model)))
    print("#" * 100)


    start = time.time()
    stop_run = False  # set if some event occurs

    # LOSS FUNCTIONS [create in advance for speed]
    func_margin_loss = create_margin_loss(
        lbd=config.loss.margin.lbd,
        m_plus=config.loss.margin.m_plus,
        m_minus=config.loss.margin.m_minus
    )
    func_rec_loss = torch.nn.MSELoss()



    #best result
    best_acc = 0.
    best_epoch = None

    ##################################
    # Train Loop
    for epoch_idx in range(1, config.train.num_epochs + 1, 1):

        model.train()
        desc = "Train [{:3}/{:3}]:".format(epoch_idx, config.train.num_epochs)
        pbar = tqdm(dl_train, bar_format=desc + '{bar:10}{r_bar}{bar:-10b}')
        
        epoch_loss = 0
        epoch_correct = 0


        for x, y_true, _ in pbar:
            x = x.to(device)
            y_true = y_true.to(device)

            #optimizer.zero_grad()
            # way faster than optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
            
            if config.loss.rec.by_class == True:
                u_h, x_rec = model.forward(x, y_true)
            else:
                u_h, x_rec = model.forward(x)

            # Margin & Reconstruction Loss
            y_one_hot = F.one_hot(y_true, num_classes=5)
            loss_margin = func_margin_loss(u_h, y_one_hot)
            loss_margin = loss_margin * config.loss.margin.weight
            loss_rec = func_rec_loss(x, x_rec)
            loss_rec = loss_rec * config.loss.rec.weight

            # Total Loss
            loss = loss_margin + loss_rec
            loss.backward()
            
            optimizer.step()
            
            # validate batch
            y_pred = torch.argmax(torch.norm(u_h, dim=2), dim=1)

            correct = (y_true == y_pred).sum()
            acc = correct / x.shape[0]

            epoch_correct += correct.item()
            epoch_loss += loss.item()

            pbar.set_postfix(
                    {'loss': loss.item(),
                     'mar': loss_margin.item(),
                     'rec': loss_rec.item(),
                     'acc': acc.item()
                    }
            )

        # TRAIN STAS
        sw.add_scalar("train/loss", epoch_loss, epoch_idx)
        sw.add_scalar("train/acc", epoch_correct /
                      len(ds_train), epoch_idx)

        stats["train"]["epoch"].append(epoch_idx)
        stats["train"]["acc"].append(epoch_correct / len(ds_train))
        stats["train"]["loss"].append(epoch_loss)

        if scheduler is not None and (epoch_idx > config.scheduler_burnin):
            scheduler.step()

        if math.isnan(epoch_loss):
            print_str = "Stopping epoch {}: epoch_loss={}".format(
                epoch_idx, epoch_loss)
            print(print_str)
            stats["notes"].append(print_str)
            stop_run = True


        ##################################
        # Model Eval
        model.eval()

        #Save Model ckpt
        if (epoch_idx % config.freqs.ckpt == 0) or (config.train.num_epochs == epoch_idx):
            p_ckpt = p_ckpts / config.names.model_file.format(epoch_idx)
            torch.save(model.state_dict(), p_ckpt)    
        
        #Generate and save grids
        if (epoch_idx % config.freqs.rec == 0) or (config.train.num_epochs == epoch_idx):

            img_train = create_reconstruction_grid_img(
                model, device, x_vis_train, y_vis_train)
            img_valid = create_reconstruction_grid_img(
                model, device, x_vis_valid, y_vis_valid)

            plt.imshow(img_train.permute(1,2,0))
            plt.tight_layout()
            plt.savefig(p_imgs / "img_train_{:03d}.png".format(epoch_idx))
            plt.tight_layout()
            plt.close()

            plt.imshow(img_valid.permute(1,2,0))
            plt.savefig(p_imgs / "img_valid_{:03d}.png".format(epoch_idx))
            plt.close()


            sw.add_image("train/rec", img_train, epoch_idx)
            sw.add_image("valid/rec", img_valid, epoch_idx)

        #Validation loop
        if (epoch_idx % config.freqs.valid == 0) or (config.train.num_epochs == epoch_idx):
            loss_valid, acc_valid = eval_model(
                model, device, dl_valid, config, func_margin_loss, func_rec_loss)

            sw.add_scalar("valid/loss", loss_valid, epoch_idx)
            sw.add_scalar("valid/acc", acc_valid, epoch_idx)

            stats["valid"]["epoch"].append(epoch_idx)
            stats["valid"]["acc"].append(acc_valid)
            stats["valid"]["loss"].append(loss_valid)

            print_str = "Valid: loss: {:.5f}, acc: {:.5f} "
            print(print_str.format(loss_valid, acc_valid))

            if acc_valid >= config.stop_acc:
                print_str = "Stopping epoch {}: acc_valid {:.5f} > {:.5f}".format(
                    epoch_idx, acc_valid, config.stop_acc)
                print(print_str)
                stats["notes"].append(print_str)
                stop_run = True

        if stop_run:
            break

    end = time.time()
    stats["train_time"] = "{:.1f}".format(end - start)
    print("Training time: {:.1f}".format(end - start))

    with open(p_stats, "wb") as file:
        pickle.dump(stats, file)

    sw.close()

    plot_acc_from_stats(stats, p_acc_plot)
    plot_loss_from_stats(stats, p_loss_plot)
    return stats
        

if __name__ == '__main__':
    
    #run train with default settings
    train()
    