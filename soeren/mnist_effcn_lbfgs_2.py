#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("./..")

import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from lbfgs import LBFGS
#from torch.optim import LBFGS

from effcn.models import MnistEffCapsNet
from effcn.functions import margin_loss
from effcn.utils import count_parameters


REC_LOSS_WEIGHT = 0.392

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
dev = 'cpu'
device = torch.device(dev)

print("Using device: {}".format(device))


if __name__ == '__main__':
    ds_train = datasets.MNIST(root='./../data', train=True, download=True, transform=T.ToTensor())
    ds_valid = datasets.MNIST(root="./../data", train=False, download=True, transform=T.ToTensor())

    batch_size_train = ds_train.__len__() // 10
    batch_size_valid = ds_valid.__len__() // 1
    dl_train = torch.utils.data.DataLoader(ds_train, 
                                           batch_size=batch_size_train, 
                                           shuffle=False, 
                                           num_workers=1)
    dl_valid = torch.utils.data.DataLoader(ds_valid, 
                                           batch_size=batch_size_valid, 
                                           shuffle=False, 
                                           num_workers=1)

    y_total_train = []
    dl_train_2 = []
    for x_train, y_train in dl_train:
        y_total_train += y_train
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        dl_train_2.append((x_train, y_train))
    y_total_train = torch.stack(y_total_train)
    y_total_train = y_total_train.to(device)

    dl_valid_2 = []
    for x_valid, y_valid in dl_valid:
        x_valid = x_valid.to(device)
        y_valid = y_valid.to(device)
        dl_valid_2.append((x_valid, y_valid))

    
    y_one_hot = F.one_hot(y_train, num_classes=10)

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    model = MnistEffCapsNet()
    model = model.to(device)

    print("#params: {}".format(count_parameters(model)))

    loss_func = nn.CrossEntropyLoss()
    optimizer = LBFGS(model.parameters(), history_size=10, max_iter=1, 
                      line_search_fn="strong_wolfe")

    print("Train ...", flush="True")
    num_epochs = 150
    model.train()
    for epoch in range(num_epochs):
        def closure():
            print('.')
            optimizer.zero_grad()
            y_pred_list = []
            total_loss = torch.tensor(0., requires_grad=False, device=device)

            for idx, (x_train, y_train) in enumerate(dl_train_2):
                u_h, x_rec = model.forward(x_train)
            
                # LOSS
                loss_margin = margin_loss(u_h, y_one_hot)
                loss_rec = torch.nn.functional.mse_loss(x_train, x_rec)
            
                # param from paper
                loss = loss_margin + REC_LOSS_WEIGHT * loss_rec
                loss.backward()
                       
                y_pred = torch.argmax(torch.norm(u_h, dim=2), dim=1)

                total_loss += loss
                y_pred_list += y_pred

            y_pred = torch.stack(y_pred_list)

            return total_loss, y_pred, idx + 1

        loss, y_predict = optimizer.step(closure)
        
        with torch.no_grad():
            y_pred_train = y_predict
            correct_train = (y_pred_train == y_total_train).sum().item()
            acc_train = correct_train / y_total_train.shape[0]
            
            correct_valid = 0
            total_valid = 0
            for x_valid, y_valid in dl_valid_2:
                u_h, _ = model.forward(x_valid)
                y_pred_valid = torch.argmax(torch.norm(u_h, dim=2), dim=1)
                correct_valid += (y_pred_valid == y_valid).sum().item()
                total_valid += y_valid.shape[0]
            acc_valid = correct_valid / total_valid
            print("Epoch[{}/{}]  loss: {:.6f}  train_acc: {:.4f}  test_acc: {:.4f}".format(epoch, num_epochs, loss, acc_train, acc_valid))
