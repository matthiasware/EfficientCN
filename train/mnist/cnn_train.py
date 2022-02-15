#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("./../..")

import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.nn as nn
from torch import optim

import numpy as np
from effcn.models_mnist import BaselineCNN
from misc.utils import count_parameters

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

print("Using device: {}".format(device))


def main():
    ds_train = datasets.MNIST(
        root='./../data', train=True, download=True, transform=T.ToTensor())
    ds_valid = datasets.MNIST(
        root="./../data", train=False, download=True, transform=T.ToTensor())

    dl_train = torch.utils.data.DataLoader(ds_train,
                                           batch_size=64,
                                           shuffle=True,
                                           num_workers=8)
    dl_valid = torch.utils.data.DataLoader(ds_valid,
                                           batch_size=64,
                                           shuffle=True,
                                           num_workers=8)

    model = BaselineCNN()
    model = model.to(device)

    print("#params: {}".format(count_parameters(model)))

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=0.96)

    print("Train ...", flush="True")
    num_epochs = 150
    model.train()
    for epoch in range(num_epochs):
        for idx, (x, y_true) in enumerate(dl_train):
            x = x.to(device)
            y_true = y_true.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_func(y_pred, y_true)
            loss.backward()
            optimizer.step()

            if idx % 1000 == 0:
                print("Epoch[{}/{}] - step {} loss: {:.6f}".format(epoch,
                                                                   num_epochs, idx, loss.item()))
        lr_scheduler.step()

    print("Eval ...", flush="True")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y_true in dl_valid:
            x = x.to(device)
            y_true = y_true.to(device)
            y_pred = model(x)
            y_pred = torch.max(y_pred, 1)[1]
            correct += (y_pred == y_true).sum().item()
            total += y_true.shape[0]
        acc = correct / total

    print("Accuracy @Test:      {:.3f}".format(acc))
    print("num total err @Test: {}".format(total - correct))


if __name__ == '__main__':
    main()
