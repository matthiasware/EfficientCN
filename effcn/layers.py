import numpy as np
import torch
import torch.nn as nn
# from .functions import squash_func


class Squash(nn.Module):
    def __init__(self, eps=10e-21):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """
         IN:  (b, n, d)
         OUT: squash(x(b,n,d))
        """
        x_norm = torch.norm(x, dim=2, keepdim=True)
        return (1 - 1 / (torch.exp(x_norm) + self.eps)) * (x / (x_norm + self.eps))


class View(nn.Module):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class PrimaryCaps(nn.Module):
    """
        Attributes
        ----------
        F: int depthwise conv number of features
        K: int depthwise conv kernel dimension 
        N: int number of primary capsules
        D: int primary capsules dimension (number of properties)
        s: int depthwise conv strides
    """

    def __init__(self, F, K, N, D, s=1):
        super().__init__()
        self.F = F
        self.K = K
        self.N = N
        self.D = D
        self.s = s
        #
        self.dw_conv2d = nn.Conv2d(
            F, F, kernel_size=K, stride=s, groups=F, padding="valid")
        #
        self.squash = Squash(eps=1e-20)

    def forward(self, x):
        """
        IN:  (B,C,H,W)
        OUT: (B, N, D)

        therefore for x, we have the following constraints:
            (B,C,H,W) = (B, F,F,K)
        """
        # (B,C,H,W) -> (B,C,H,W)
        x = self.dw_conv2d(x)

        # (B,C,H,W) -> (B, N, D)
        x = x.view((-1, self.N, self.D))
        x = self.squash(x)
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

    def __init__(self, n_l, n_h, d_l, d_h):
        super().__init__()
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

        self.attention_scaling = np.sqrt(self.d_l)

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
        A = A / self.attention_scaling
        A_sum = torch.einsum("...hij->...hj", A)
        C = torch.softmax(A_sum, dim=-1)
        CB = C + self.B
        U_h = torch.einsum('...ikl,...ik->...kl', U_hat, CB)
        return self.squash(U_h)


class FCCaps2(nn.Module):
    """
    Fully-connected caps layer. It exploites the routing mechanism, explained in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing', 
    to create a parent layer of capsules. 
    
    nl: number of input capsuls          (nl...i)(nl...h)
    dl: dimension of input capsuls       (dl...j)
    nh: number of output capsuls         (nh...k)
    dh: dimension of output capsuls      (dh...l)
    b: batch size                        (...)

    W: weigth tensor                     (W->ikjl)
    B: bias matrix                       (B->1ik)
    U_l: input capsuls matrix            (U->...ij)    
    U_hat: weigthed input capsuls matrix (U_hat->...ikl)
    A: covariance tensor                 (A->...hik)
    C: couplimg coefficients             (C->...ik)
    
    input: nl, dl, nh, dh
    
    """
    
    def __init__(self, nl, nh, dl, dh):
        super().__init__()
        self.nl = nl
        self.dl = dl
        self.nh = nh
        self.dh = dh
        
        #
        self.W = torch.nn.Parameter(torch.rand([self.nl,self.nh,self.dl,self.dh]), requires_grad=True)
        self.B = torch.nn.Parameter(torch.rand([self.nl,self.nh]), requires_grad=True)
        #self.B = torch.nn.Parameter(torch.rand([1,self.nl,self.nh]), requires_grad=True)   #dim for B from tensorflow code    -> #Difference in Dimension definition => difference if batch >1 ...
        self.squash = Squash()         #eps in function predefind
        #
        
        # init custom weights -> not implemented
        torch.nn.init.kaiming_normal_(
            self.W, a=0, mode='fan_in', nonlinearity='leaky_relu')        #MM
        torch.nn.init.kaiming_normal_(
            self.B, a=0, mode="fan_in", nonlinearity="leaky_relu")        #MM    
        
        
        
    def forward(self, U_l):
        """
        Data tensors:
            Input:  U_l ... lower layer capsules
            Ouput: U_h ... higher layer capsules
        """
        
        U_hat = torch.einsum("...ij,ikjl->...ikl",U_l,self.W)
        A = torch.einsum("...hkl,...ikl->...hik",U_hat, U_hat)
        A = A / torch.sqrt(torch.Tensor([self.dl]))
        A_hat = torch.einsum("...hik->...ik",A)
        C = torch.softmax(A_hat,dim=-1)
        CB = C+self.B
        U_h = torch.einsum("...ikl,...ik->...kl",U_hat,CB)
        U_h = self.squash(U_h)
        return U_h
