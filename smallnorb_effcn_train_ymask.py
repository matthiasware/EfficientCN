# default libraries
import time
import datetime
import sys
sys.path.append("./..")

# third party libraries
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path

# local imports
from effcn.models import SmallNorbEcnBackbone, SmallNorbEcnDecoder, SmallNorbEffCapsNetYMask
from effcn.layers import PrimaryCaps, FCCaps
from effcn.functions import margin_loss
from effcn.utils import count_parameters
from smallnorb.smallnorb import SmallNORB

# will most likely result in a 30% speed up
torch.backends.cudnn.benchmark = True

#########################
#  CONFIG
#########################

#  using params from paper
BATCH_SIZE = 16
NUM_EPOCHS = 200
LEARNING_RATE = 5e-4
SCHEDULER_GAMMA = 0.97
REC_LOSS_WEIGHT = 0.392
NUM_WORKERS = 6

# Dataset SmallNORB
NUM_CLASSES = 5


if torch.cuda.is_available():
    dev = "cuda:0" 
else:  
    dev = "cpu"  
DEVICE = torch.device(dev)

# paths
P_DATA = "./data"
P_CKTPS = "./data/ckpts"


def main():
    ##################################
    #Get & Preprocess data

    #Tranformations
    transform_train = T.Compose([
        T.Resize(64),
        T.RandomResizedCrop(48, scale=(0.8, 1.0), ratio=(1, 1)),
        T.ToTensor()
    ])
    transform_valid = T.Compose([
        T.Resize(64),
        T.RandomResizedCrop(48, scale=(0.8, 1.0), ratio=(1, 1)),
        T.ToTensor()
    ])   
    
    #load Dataset
    ds_train = SmallNORB(root='data/SmallNORB',train=True, download=True, transform=transform_train, mode="nopil")
    ds_valid = SmallNORB(root='data/SmallNORB',train=False, download=True, transform=transform_valid, mode="nopil")
    
    #stack data to batches
    dl_train = torch.utils.data.DataLoader(ds_train, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        pin_memory=True,
                                        num_workers=NUM_WORKERS)
    dl_valid = torch.utils.data.DataLoader(ds_valid, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        pin_memory=True,
                                        num_workers=NUM_WORKERS)   
    

    # Data for visualization of the img reconstructions
    x_vis, y_vis, _ = next(iter(dl_valid))    
    
    
    ##################################
    #Train Model

    #Model
    model = SmallNorbEffCapsNetYMask()
    model = model.to(DEVICE)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=SCHEDULER_GAMMA)    

    # checkpointing
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    p_run = Path(P_CKTPS) / "smallnorb_run_{}".format(st)
    p_run.mkdir(exist_ok=True, parents=True)

    # training statistics
    stats = {
        'acc_train': [],
        'acc_valid': [],
    }

    # print stuff
    print("#" * 100)
    print("#params:            {:,}".format(count_parameters(model)))
    print("Using device:       {}".format(DEVICE))
    print("Learning rate:      {}".format(LEARNING_RATE))
    print("Batch size:         {}".format(BATCH_SIZE))
    print("Writing results to: {}".format(p_run))
    print("#" * 100)


    ##################################
    # Train Loop
    for epoch_idx in range(1, NUM_EPOCHS +1):

        model.train()
        epoch_correct = 0
        epoch_total = 0
        desc = "Train [{:3}/{:3}]:".format(epoch_idx, NUM_EPOCHS)
        pbar = tqdm(dl_train, bar_format=desc + '{bar:10}{r_bar}{bar:-10b}')
        
        for x, y_true, _ in pbar:
            x = x.to(DEVICE)
            y_true = y_true.to(DEVICE)

            #optimizer.zero_grad()

            # way faster than optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
            
            u_h, x_rec, x_rec_y = model.forward(x, y_true)
            
            # Margin Loss & Reconstruction Loss
            y_one_hot = F.one_hot(y_true, num_classes=NUM_CLASSES)
            loss_margin = margin_loss(u_h, y_one_hot)
            loss_rec = torch.nn.functional.mse_loss(x, x_rec_y)
            loss_rec = REC_LOSS_WEIGHT * loss_rec
            
            # Total Loss
            loss = loss_margin + loss_rec
            loss.backward()
            
            optimizer.step()
            
            # validate batch
            y_pred = torch.argmax(torch.norm(u_h, dim=2), dim=1)
            
            batch_correct = (y_true == y_pred).sum()
            batch_total = y_true.shape[0]
            acc = batch_correct / batch_total

            epoch_correct += batch_correct
            epoch_total += batch_total
            
            pbar.set_postfix(
                    {'loss': loss.item(),
                    'mar': loss_margin.item(),
                    'rec': loss_rec.item(),
                    'acc': acc.item()
                    }
            )
        
        stats["acc_train"].append((epoch_correct/epoch_total).item())

        #lr_scheduler.step()

        ##################################
        # Evaluate Loop
        model.eval()
        
        epoch_correct = 0
        epoch_total = 0

        for x,y_true, _ in dl_valid:
            x = x.to(DEVICE)
            y_true = y_true.to(DEVICE)
            
            with torch.no_grad():
                u_h, x_rec, x_rec_y = model.forward(x, y_true)  

                # Margin Loss & Reconstruction Loss
                y_one_hot = F.one_hot(y_true, num_classes=NUM_CLASSES)
                loss_margin = margin_loss(u_h, y_one_hot)
                loss_rec = torch.nn.functional.mse_loss(x, x_rec)
                loss_rec = REC_LOSS_WEIGHT * loss_rec
                # Total Loss
                loss = loss_margin + loss_rec

                y_pred = torch.argmax(torch.norm(u_h, dim=2), dim=1)
                epoch_correct += (y_true == y_pred).sum()
                epoch_total += y_true.shape[0]

        print("loss_valid: {:.5f}".format(loss))
        print("acc_valid: {:.5f}".format(epoch_correct / epoch_total))
        stats["acc_valid"].append((epoch_correct/epoch_total).item())

        #  save reconstructions
        with torch.no_grad():
            _, x_rec, x_rec_y = model.forward(x_vis.to(DEVICE), y_vis.to(DEVICE))
        x_rec = x_rec.cpu()
        x_rec_y = x_rec_y.cpu()
        # channel 1
        img = torchvision.utils.make_grid(torch.cat([x_vis[:16,:1,:,:], x_rec[:16,:1,:,:], x_rec_y[:16,:1,:,:]], dim=0), nrow=16)
        img = img.permute(1,2,0)
        plt.figure(figsize=(16, 3))
        plt.tight_layout()
        plt.axis('off')
        plt.imshow(img)
        plt.savefig(p_run / "smallnorb_c1_rec_{}.png".format(epoch_idx))
        plt.close()        
        # channel 2
        img = torchvision.utils.make_grid(torch.cat([x_vis[:16,1:2,:,:], x_rec[:16,1:2,:,:], x_rec_y[:16,1:2,:,:]], dim=0), nrow=16)
        img = img.permute(1,2,0)
        plt.figure(figsize=(16, 3))
        plt.tight_layout()
        plt.axis('off')
        plt.imshow(img)
        plt.savefig(p_run / "smallnorb_c2_rec_{}.png".format(epoch_idx))
        plt.close()


        #append learning rate
        lr_scheduler.step()

        # save model during training each epoch
        model_name = "ecn_smallnorb_epoch_{}.ckpt".format(epoch_idx)
        p_model = p_run / model_name
        torch.save(model.state_dict(), p_model)  

    ##################################
    # Visualize Stats
    xx = list(range(1, epoch_idx + 1))
    plt.plot(xx, stats["acc_train"], label="train")
    plt.plot(xx, stats["acc_valid"], label="valid")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(p_run / "acc.png")
    plt.close()


if __name__ == '__main__':
    main()




