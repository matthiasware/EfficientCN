import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from tqdm import tqdm
from pathlib import Path
import datetime
import time

# local imports
from effcn.models import MnistEcnBackbone, MnistEcnDecoder, MnistEffCapsNet
from effcn.layers import PrimaryCaps, FCCaps
from effcn.functions import margin_loss, max_norm_masking
from effcn.utils import count_parameters


"""
    DISCLAIMER: WIP

    Ideas:
    - better parametrization of training procedure via config file 
    - use aggressive code optimizatiosn to speed up training
    - change model slightly to speed up training and get better results
    - advanced logging training statistics
    - tensorboard support
    - advanced augmentation techniques
    
    Some features might not work yet ;)

    Also we drop CPU support here and focus on GPU!
"""

def main():
    #########################
    #  DEVICE SETTINGS
    #########################
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    device = torch.device(dev)
    print("Using device: {}".format(device))

    #########################
    #  PREPROCESSING & DATA
    #########################
    transform_train = T.Compose([
        T.RandomAffine(
            degrees=(-35, 35),
            translate=(0.1, 0.1)
        ),
        T.RandomResizedCrop(
            28,
            scale=(0.8, 1.0),
            ratio=(1, 1),
        ),
        T.ToTensor()
    ])
    transform_valid = T.Compose([
        T.ToTensor()
    ])


    ds_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    ds_valid = datasets.MNIST(root="./data", train=False, download=True, transform=transform_valid)

    dl_train = torch.utils.data.DataLoader(ds_train, 
                                        batch_size=256, 
                                        shuffle=True, 
                                        num_workers=8)
    dl_valid = torch.utils.data.DataLoader(ds_valid, 
                                        batch_size=256, 
                                        shuffle=True, 
                                        num_workers=8)
    

    #
    # Data for visualization of the rec.
    #
    x_vis, y_vis = next(iter(dl_valid))

    #########################
    #  TRAIN MODEL
    #########################
    model = MnistEffCapsNet()
    model = model.to(device)

    print("#params: {}".format(count_parameters(model)))
    optimizer = optim.Adam(model.parameters(), lr = 5e-4 * 8)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)

    # checkpointing
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    p_ckpts = Path("./data/ckpts/run_{}".format(st))
    p_ckpts.mkdir(exist_ok=True, parents=True)
    print("\nWriting results to: {}\n".format(p_ckpts))

    # training statistics
    stats = {
        'acc_train': [],
        'acc_valid': [],
    }

    num_epochs = 10000
    for epoch_idx in range(1, num_epochs+1):
        # 
        # TRAIN LOOP
        #
        model.train()
        epoch_correct = 0
        epoch_total = 0
        desc = "Train [{:3}/{:3}]:".format(epoch_idx, num_epochs)
        pbar = tqdm(dl_train, bar_format=desc + '{bar:10}{r_bar}{bar:-10b}')
        for x,y_true in pbar:
            x = x.to(device)
            y_true = y_true.to(device)
            optimizer.zero_grad()
            
            u_h, x_rec = model.forward(x)
            
            # LOSS
            y_one_hot = F.one_hot(y_true, num_classes=10)
            loss_margin = margin_loss(u_h, y_one_hot)
            loss_rec = torch.nn.functional.mse_loss(x, x_rec)
            
            # param from paper
            loss = loss_margin + 0.392 * loss_rec
            loss.backward()
            
            optimizer.step()
            
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

        #
        #  EVAL LOOP
        #
        model.eval()
            
        epoch_correct = 0
        epoch_total = 0

        for x,y_true in dl_valid:
            x = x.to(device)
            y_true = y_true.to(device)
                
            with torch.no_grad():
                u_h, x_rec = model.forward(x)
                y_pred = torch.argmax(torch.norm(u_h, dim=2), dim=1)
                epoch_correct += (y_true == y_pred).sum()
                epoch_total += y_true.shape[0]
        
        print("   acc_valid: {:.3f}".format(epoch_correct / epoch_total))
        stats["acc_valid"].append((epoch_correct/epoch_total).item())
    
        #
        #  save reconstructions as imgs
        #
        with torch.no_grad():
            _, x_rec = model.forward(x_vis.to(device))
        x_rec = x_rec.cpu()
        img = torchvision.utils.make_grid(torch.concat([x_vis[:16], x_rec[:16]], dim=0), nrow=16)
        img = img.permute(1,2,0)
        plt.figure(figsize=(16, 2))
        plt.tight_layout()
        plt.axis('off')
        plt.imshow(img)
        plt.savefig(p_ckpts / "rec_{}.png".format(epoch_idx))
        plt.close()
        
        
        # I guess this is done once per epoch
        lr_scheduler.step()

        # save model during training in each epoch
        if epoch_idx % 100 == 0:
            print("Save ckpt!")
            model_name = "ecn_mnist_epoch_{}.ckpt".format(epoch_idx)
            p_model = p_ckpts / model_name
            torch.save(model.state_dict(), p_model)

    #########################
    #  VIS STATS
    #########################
    xx = list(range(1, epoch_idx + 1))

    plt.plot(xx, stats["acc_train"], label="train")
    plt.plot(xx, stats["acc_valid"], label="valid")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(p_ckpts / "acc.png")
    plt.close()
    

if __name__ == '__main__':
    main()