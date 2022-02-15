import sys
sys.path.append("./../..")

# default libraries
import time
import datetime

# third party libraries
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

# local imports
from effcn.models_mnist import EffCapsNet
from effcn.functions import margin_loss
from misc.utils import count_parameters

#########################
#  CONFIG
#########################

#  using params from paper
BATCH_SIZE = 16
NUM_EPOCHS = 150
LEARNING_RATE = 5e-4 * 2**0
SCHEDULER_GAMMA = 0.96
REC_LOSS_WEIGHT = 0.392
NUM_WORKERS = 2

if torch.cuda.is_available():
    dev = "cuda" 
else:  
    dev = "cpu"  
DEVICE = torch.device(dev)

# paths
P_DATA = "/mnt/data/datasets"
P_CKTPS = "/mnt/data/experiments/EfficientCN/mnist"


def main():
    #########################
    #  PREPROCESSING & DATA
    #########################

    # does not use the complete preprocessing transformations from the paper
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

    ds_train = datasets.MNIST(root=P_DATA, train=True,
                              download=True, transform=transform_train)
    ds_valid = datasets.MNIST(root=P_DATA, train=False,
                              download=True, transform=transform_valid)

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

    #
    # Data for visualization of the img reconstructions
    #
    x_vis, y_vis = next(iter(dl_valid))

    #########################
    #  TRAIN MODEL
    #########################
    model = EffCapsNet()
    model = model.to(DEVICE)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=SCHEDULER_GAMMA)

    # checkpointing
    st = datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    p_run = Path(P_CKTPS) / "run_{}".format(st)
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

    for epoch_idx in range(1, NUM_EPOCHS + 1):
        #
        # TRAIN LOOP
        #
        model.train()
        epoch_correct = 0
        epoch_total = 0
        desc = "Train [{:3}/{:3}]:".format(epoch_idx, NUM_EPOCHS)
        pbar = tqdm(dl_train, bar_format=desc + '{bar:10}{r_bar}{bar:-10b}')
        for x, y_true in pbar:
            x = x.to(DEVICE)
            y_true = y_true.to(DEVICE)

            # way faster than optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

            u_h, x_rec = model.forward(x)

            # LOSS
            y_one_hot = F.one_hot(y_true, num_classes=10)
            loss_margin = margin_loss(u_h, y_one_hot)
            loss_rec = torch.nn.functional.mse_loss(x, x_rec)

            # param from paper
            loss = loss_margin + REC_LOSS_WEIGHT * loss_rec
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
        stats["acc_train"].append((epoch_correct / epoch_total).item())

        #
        #  EVAL LOOP
        #
        model.eval()

        epoch_correct = 0
        epoch_total = 0

        for x, y_true in dl_valid:
            x = x.to(DEVICE)
            y_true = y_true.to(DEVICE)

            with torch.no_grad():
                u_h, x_rec = model.forward(x)
                y_pred = torch.argmax(torch.norm(u_h, dim=2), dim=1)
                epoch_correct += (y_true == y_pred).sum()
                epoch_total += y_true.shape[0]

        print("   acc_valid: {:.5f}".format(epoch_correct / epoch_total))
        stats["acc_valid"].append((epoch_correct / epoch_total).item())

        #
        #  save reconstructions
        #
        with torch.no_grad():
            _, x_rec = model.forward(x_vis.to(DEVICE))
        x_rec = x_rec.cpu()
        img = torchvision.utils.make_grid(
            torch.cat([x_vis[:16], x_rec[:16]], dim=0), nrow=16)
        img = img.permute(1, 2, 0)
        plt.figure(figsize=(16, 2))
        plt.tight_layout()
        plt.axis('off')
        plt.imshow(img)
        plt.savefig(p_run / "rec_{}.png".format(epoch_idx))
        plt.close()

        # I guess this is done once per epoch
        lr_scheduler.step()

        # save model during training each epoch
        model_name = "ecn_mnist_epoch_{}.ckpt".format(epoch_idx)
        p_model = p_run / model_name
        torch.save(model.state_dict(), p_model)

    print("best acc train: {:.5f}".format(max(stats["acc_train"])))
    print("best acc valid: {:.5f}".format(max(stats["acc_valid"])))

    #########################
    #  VIS STATS
    #########################
    xx = list(range(1, epoch_idx + 1))
    plt.plot(xx, stats["acc_train"], label="train")
    plt.plot(xx, stats["acc_valid"], label="valid")
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(p_run / "acc.png")
    plt.close()


if __name__ == '__main__':
    main()
