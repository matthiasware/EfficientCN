#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("./..")

import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.nn as nn
from lbfgs import LBFGS
#from torch.optim import LBFGS

from effcn.models import MnistBaselineCNN
from effcn.utils import count_parameters

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
# dev = 'cpu'
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
    
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    model = MnistBaselineCNN()
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
            for idx, (x_train, y_train) in enumerate(dl_train_2): # train):
                y_pred = model(x_train)
                loss = loss_func(y_pred, y_train)         
                loss.backward()
                total_loss += loss
                y_pred_list += y_pred
                
            y_pred = torch.stack(y_pred_list)
            return total_loss, y_pred, idx + 1

        loss, y_predict = optimizer.step(closure)
        print('#')
        
        with torch.no_grad():
#            y_pred_train = model(x_train)
            y_pred_train = y_predict
#            loss = loss_func(y_pred_train, y_train)         
            y_pred_train = torch.max(y_pred_train, 1)[1]
            correct_train = (y_pred_train == y_total_train).sum().item()
            acc_train = correct_train / y_total_train.shape[0]
            
            correct_valid = 0
            total_valid = 0
            for x_valid, y_valid in dl_valid_2:
                y_pred_valid = model(x_valid)
                y_pred_valid = torch.max(y_pred_valid, 1)[1]
                correct_valid += (y_pred_valid == y_valid).sum().item()
                total_valid += y_valid.shape[0]
            acc_valid = correct_valid / total_valid
            print("Epoch[{}/{}]  loss: {:.6f}  train_acc: {:.4f}  test_acc: {:.4f}".format(epoch, num_epochs, loss, acc_train, acc_valid))
