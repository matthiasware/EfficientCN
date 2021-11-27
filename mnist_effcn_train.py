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


def main():
    #########################
    #  DEVICE SETTINGS
    #########################
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    device = torch.device(dev)

    #########################
    #  PREPROCESSING & DATA
    #########################
    transform_train = T.Compose([
        T.RandomRotation(degrees=(-30, 30)),
        T.RandomResizedCrop(
            28,
            scale=(0.8, 1.0),
            ratio=(1, 1),
        ),
        T.RandomAffine(
            degrees=(-30, 30),
            #translate=(0.1, 0.1)
        ),
        T.ToTensor()
    ])
    transform_valid = T.Compose([
        T.ToTensor()
    ])


    ds_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    ds_valid = datasets.MNIST(root="./data", train=False, download=True, transform=transform_valid)

    dl_train = torch.utils.data.DataLoader(ds_train, 
                                        batch_size=16, 
                                        shuffle=True, 
                                        num_workers=4)
    dl_valid = torch.utils.data.DataLoader(ds_valid, 
                                        batch_size=16, 
                                        shuffle=True, 
                                        num_workers=4)
    

    #########################
    #  TRAIN MODEL
    #########################
    model = MnistEffCapsNet()
    optimizer = optim.Adam(model.parameters(), lr = 5e-4)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)


    num_epochs = 2

    for epoch_idx in range(num_epochs):
        model.train()
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
            
            loss = loss_margin + 0.392 * loss_rec
            loss.backward()
            
            optimizer.step()
            
            y_pred = torch.argmax(torch.norm(u_h, dim=2), dim=1)
            acc = (y_true == y_pred).sum() / y_true.shape[0]
            
            pbar.set_postfix(
                    {'loss': loss.item(),
                    'mar': loss_margin.item(),
                    'rec': loss_rec.item(),
                    'acc': acc.item()
                    }
            )
        # I guess this is done once per epoch
        lr_scheduler.step()


    #########################
    #  EVAL MODEL
    #########################
    model.eval()
        
    total_correct = 0
    total = 0

    for x,y_true in dl_valid:
        x = x.to(device)
        y_true = y_true.to(device)
            
        with torch.no_grad():
            u_h, x_rec = model.forward(x)
            y_pred = torch.argmax(torch.norm(u_h, dim=2), dim=1)
            total_correct += (y_true == y_pred).sum()
            total += y_true.shape[0]
    print("   acc_valid: {:.3f}".format(total_correct / total))


    #########################
    #  VIS RECONSTRUCTIONS
    #########################
    img = torchvision.utils.make_grid(torch.concat([x[:16], x_rec[:16]], dim=0), nrow=16)
    img = img.permute(1,2,0)
    plt.figure(figsize=(16, 2))
    plt.tight_layout()
    plt.axis('off')
    plt.imshow(img)
    plt.savefig("rec.png")

    #########################
    #  SAVE PARAMS
    #########################
    p_ckpts = Path("./data/ckpts")
    p_ckpts.mkdir(exist_ok=True, parents=True)
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    model_name = "ecn_mnist_epoch_{}_{}.ckpt".format(epoch_idx, st)
    p_model = p_ckpts / model_name
    torch.save(model.state_dict(), p_model)
    

if __name__ == '__main__':
    main()