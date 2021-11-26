import torch
import torch.nn as nn
from .functions import squash_func

class Squash(nn.Module):
    def __init__(self, eps=10e-21):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        """
         IN:  (b, n, d)
         OUT: (b, n, d)
        """
        return squash_func(x, self.eps)


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
        self.dw_conv2d = nn.Conv2d(F, F, kernel_size=K, stride=s, groups=F, padding="valid")
        #
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

        self.W = torch.nn.Parameter(torch.rand(n_l, n_h, d_l, d_h), requires_grad=True)
        self.B = torch.nn.Parameter(torch.rand(n_l, n_h), requires_grad=True)
        self.squash = Squash(eps=1e-20)

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
        A = A / torch.sqrt(torch.Tensor([self.d_l]))
        A_sum = torch.einsum("...hij->...hj",A)
        C = torch.softmax(A_sum,dim=-1)
        CB = C + self.B
        U_h = torch.einsum('...ikl,...ik->...kl', U_hat, CB)
        return self.squash(U_h)