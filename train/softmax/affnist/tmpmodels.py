import sys
sys.path.append("./../../..")

import torch
import torch.nn as nn
from effcn.layers import Squash


class CustomBB(nn.Module):
    """
        Custom backbone
    """

    def __init__(self, ch_in=3, n_classes=10):
        super().__init__()
        self.ch_in = ch_in
        self.n_classes = n_classes

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=128,
                      kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=9, groups=256,
                      stride=1, padding="valid"),
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.convs(x)
        # -> (b, 256), remove 1 X 1 grid and make vector of tensor shape
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class FCCaps(nn.Module):
    """
        Attributes
        ----------
        n_l ... number of lower layer capsules
        d_l ... dimension of lower layer capsules
        n_h ... number of higher layer capsules
        d_h ... dimension of higher layer capsules

        W   (n_l, n_h, d_l, d_h) ... weight tensor
        B   (n_l, n_h)           ... bias tensor
    """

    def __init__(self, n_l, n_h, d_l, d_h, attention_scaling):
        super().__init__()
        print(attention_scaling)
        self.n_l = n_l
        self.d_l = d_l
        self.n_h = n_h
        self.d_h = d_h

        self.W = torch.nn.Parameter(torch.rand(
            n_l, n_h, d_l, d_h), requires_grad=True)
        self.B = torch.nn.Parameter(torch.rand(n_l, n_h), requires_grad=True)
        self.squash = Squash(eps=1e-20)

        # init custom weights
        # i'm relly unsure about this initialization scheme
        # i don't think it makes sense in our case, but the paper says so ...
        torch.nn.init.kaiming_normal_(
            self.W, a=0, mode='fan_in', nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(
            self.B, a=0, mode="fan_in", nonlinearity="leaky_relu")

        self.attention_scaling = attention_scaling

    def forward(self, U_l):
        """
        einsum convenventions:
          n_l = i | h
          d_l = j
          n_h = k
          d_h = l

        Data tensors:
            IN:  U_l ... lower layer capsules
            OUT: U_h ... higher layer capsules
            DIMS:
                U_l (n_l, d_l)
                U_h (n_h, d_h)
                W   (n_l, n_h, d_l, d_h)
                B   (n_l, n_h)
                A   (n_l, n_l, n_h)
                C   (n_l, n_h)
        """
        U_hat = torch.einsum('...ij,ikjl->...ikl', U_l, self.W)
        A = torch.einsum("...ikl, ...hkl -> ...hik", U_hat, U_hat)
        
        # I removed the scaling, to enforce stronger couplings
        #A = A / self.attention_scaling)
        
        
        A_sum = torch.einsum("...hij->...hj", A)
        C = torch.softmax(A_sum / self.attention_scaling, dim=-1)
        
        # I removed the Bias term
        #CB = C + B
        
        U_h = torch.einsum('...ikl,...ik->...kl', U_hat, C)
        return self.squash(U_h)

    def forward_debug(self, U_l):
        """
            Same as forward() but returns more stuff to analyze routing
        """
        U_hat = torch.einsum('...ij,ikjl->...ikl', U_l, self.W)
        A = torch.einsum("...ikl, ...hkl -> ...hik", U_hat, U_hat)
        
        # I removed the scaling, to create stronger couplings
        #A = A / self.attention_scaling)
        
        
        A_sum = torch.einsum("...hij->...hj", A)
        C = torch.softmax(A_sum / self.attention_scaling, dim=-1)
        
        # I removed the Bias term
        #CB = C + B
        
        U_h = torch.einsum('...ikl,...ik->...kl', U_hat, C)
        return self.squash(U_h), C

class DeepCapsNet(nn.Module):
    """
        A Deeper CN that allows
    """
    def __init__(self, ns, ds, attention_scaling):
        super().__init__()
        self.ns = ns
        self.ds = ds
        
        self.backbone = CustomBB(ch_in=1)
        self.backbone.fc = nn.Identity()
        
        self.squash = Squash(eps=1e-20)
        layers = []
        for idx in range(1, len(ns), 1):
            n_l = ns[idx - 1]
            n_h = ns[idx]
            d_l = ds[idx - 1]
            d_h = ds[idx]
            layers.append(FCCaps(n_l, n_h, d_l, d_h, attention_scaling))
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        x = self.backbone(x)
        
        # primecaps
        x = self.squash(x.view(-1, self.ns[0], self.ds[0]))
        
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_debug(self, x):
        x = self.backbone(x)
        
        # primecaps
        x = self.squash(x.view(-1, self.ns[0], self.ds[0]))
        
        us = [torch.clone(x)]
        cc = []
        # fccaps
        for layer in self.layers:
            x, c = layer.forward_debug(x)
            cc.append(c.detach())
            us.append(torch.clone(x).detach())
        return x, cc, us