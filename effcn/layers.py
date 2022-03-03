import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .functions import squash_hinton


class Squash(nn.Module):
    def __init__(self, eps=1e-21):
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

    def forward_debug(self, U_l):
        U_hat = torch.einsum('...ij,ikjl->...ikl', U_l, self.W)
        A = torch.einsum("...ikl, ...hkl -> ...hik", U_hat, U_hat)
        A_scaled = A / self.attention_scaling
        A_sum = torch.einsum("...hij->...hj", A_scaled)
        C = torch.softmax(A_sum, dim=-1)
        CB = C + self.B
        U_h_fin = torch.einsum('...ikl,...ik->...kl', U_hat, CB)
        U_h_sq = self.squash(U_h_fin)
        return U_hat, A, A_scaled, A_sum, C, CB, U_h_fin, U_h_sq


class FCCapsWOBias(nn.Module):
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
        # self.B = torch.nn.Parameter(torch.rand(n_l, n_h), requires_grad=True)
        self.squash = Squash(eps=1e-20)

        # init custom weights
        # i'm relly unsure about this initialization scheme
        # i don't think it makes sense in our case, but the paper says so ...
        torch.nn.init.kaiming_normal_(
            self.W, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # torch.nn.init.kaiming_normal_(
        #     self.B, a=0, mode="fan_in", nonlinearity="leaky_relu")

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
        # CB = C + self.B
        U_h = torch.einsum('...ikl,...ik->...kl', U_hat, C)
        return self.squash(U_h)

    def forward_debug(self, U_l):
        U_hat = torch.einsum('...ij,ikjl->...ikl', U_l, self.W)
        A = torch.einsum("...ikl, ...hkl -> ...hik", U_hat, U_hat)
        A_scaled = A / self.attention_scaling
        A_sum = torch.einsum("...hij->...hj", A_scaled)
        C = torch.softmax(A_sum, dim=-1)
        # CB = C + self.B
        U_h_fin = torch.einsum('...ikl,...ik->...kl', U_hat, C)
        U_h_sq = self.squash(U_h_fin)
        return U_hat, A, A_scaled, A_sum, C, U_h_fin, U_h_sq



class AgreementRouting(nn.Module):
    def __init__(self, n_l, n_h, n_iter):
        super(AgreementRouting, self).__init__()
        self.n_iter = n_iter
        self.b = nn.Parameter(torch.zeros((n_l, n_h)))

    def forward(self, u_predict):
        v, _ = self.forward_debug(u_predict)
        return v

    def forward_debug(self, u_predict):
        batch_size, n_l, n_h, output_dim = u_predict.size()

        c = F.softmax(self.b / 1, dim=-1)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash_hinton(s)

        if self.n_iter > 0:
            b_batch = self.b.expand((batch_size, n_l, n_h))
            for r in range(self.n_iter):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, n_h) / 1, dim=-1).view(-1, n_l, n_h, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash_hinton(s)
        return v, c.squeeze()


class CapsLayer(nn.Module):
    def __init__(self, n_l, d_l, n_h, d_h, n_iter=3):
        super(CapsLayer, self).__init__()
        self.d_l = d_l
        self.n_l = n_l
        self.d_h = d_h
        self.n_h = n_h
        self.weights = nn.Parameter(torch.Tensor(n_l, d_l, n_h * d_h))
        self.routing_module = AgreementRouting(n_l, n_h, n_iter)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.n_l)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        v, _ = self.forward_debug(caps_output)
        return v

    def forward_debug(self, caps_output):
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.n_l, self.n_h, self.d_h)
        v, c = self.routing_module.forward_debug(u_predict)
        return v, c
