# default libraries
import math
# third party libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
import numpy as np
# local imports
# ...


class PrimaryCaps(nn.Module):
    """
    Args:
        ch_in:  output of the normal conv layer
        ch_out: number of types of capsules
        K:      kernel size of convolution
        P:      size of pose matrix is P*P
        stride: stride of convolution
    Shape:
        input:       (bs, ch_in, h, w)             (bs, 32, 14, 14)
        output: p -> (bs, ch_out, h', w', P, P)    (bs, 32, 14, 14, 4, 4)
                a -> (bs, ch_out, h', w')          (bs, 32, 14, 14)
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """

    def __init__(self, ch_in=32, ch_out=32, K=1, P=4, stride=1, padding="valid"):
        super().__init__()
        self.pose = nn.Conv2d(in_channels=ch_in, out_channels=ch_out*P*P, kernel_size=K, stride=stride, bias=True)
        self.acti = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=K, stride=stride, bias=True),
            nn.Sigmoid()
        )
        self.P = P

    def forward(self, x):
        p = self.pose(x)
        a = self.acti(x)
        p = p.view(p.shape[0],-1,p.shape[2],p.shape[3],self.P,self.P)

        return p, a


class ConvCaps(nn.Module):
    """Create a convolutional capsule layer
    that transfer capsule layer L to capsule layer L+1
    by EM routing.
    Args:
        ch_in: input number of types of capsules
        ch_out: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iter: number of EM iterations
        hw_out: height and width of the output conv
        final_lambda: lambda value for EM interation scheduler
        class_caps: share transformation matrix across w*h and use scaled coordinate addition
        device : divice for computing
    Shape:
        input:  (bs, ch_in, h_in, w_in, P, P)      (bs, 32, 14, 14, 4, 4)
                (bs, ch_in, h_in, w_in, 1)         (bs, 32, 14, 14)
        output: (bs, ch_out, h_out, wh_out, P, P)      (bs, 32, 6, 6, 4, 4)
                (bs, ch_out, hh_out, wh_out, 1)         (bs, 32, 6, 6)
        h_out, w_out is computedin an convolution layer with staic kernel
    Parameter size:
                init:                                       computing:
        b_u:    (ch_out)                                    (1, 1, ch_out, 1)
        b_a:    (ch_out)                                    (1, 1, ch_out, 1)
        w:      (1, ch_in, h_out, w_out, P, P, ch_out)      (bs, ch_in, h_out, w_out, P, P, ch_out)


    """   

    def __init__(self, ch_in=32, ch_out=32, K=3, P=4, stride=2, iter=3, hw_out=(1,1), final_lambda=1e-02, class_caps=False, device="cpu"):
        super().__init__()
        # init vars
        self.ch_in  = ch_in
        self.ch_out = ch_out
        self.K = K
        self.P = P
        self.psize = P*P
        self.stride = stride
        self.iter = iter
        self.hw_out = hw_out
        self.class_caps = class_caps
        self.final_lambda = final_lambda
        self.device = torch.device(device)

        # constants
        self.eps = 1e-8
        
        # params
        self.b_u = nn.Parameter(torch.zeros(ch_out), requires_grad=True)
        self.b_a = nn.Parameter(torch.zeros(ch_out), requires_grad=True)
        self.w = nn.Parameter(torch.rand([1, ch_in, hw_out[0], hw_out[0], P, P, ch_out]), requires_grad=True)

        #torch.nn.init.kaiming_normal_(self.w)

        #conv with static kernel
        self.conv_stat = nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=K, stride=stride, bias=False, padding=0)
        self.conv_stat.weight = torch.nn.Parameter((torch.ones_like(self.conv_stat.weight)/K**2),requires_grad=False)

        # activations
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def voting(self, x):
        """
                Input:     (bs, ch_in, h_in, w_in, p, p)
                Output:    (bs, ch_in, h_ out, w_out, p, p, ch_out)
        """        
        #conv & shaping
        sh_in = x.shape
        x = x.view(sh_in[0]*self.P*self.P, self.ch_in,sh_in[2], sh_in[3])
        x = self.conv_stat(x)
        x = x.view(-1, x.shape[1], x.shape[2], x.shape[3], self.P,self.P)
        
        #expand x and w to number of out channels
        x = x.unsqueeze(-1).repeat([1, 1, 1, 1, 1, 1, self.ch_out])
        w = self.w.repeat([sh_in[0], 1, 1, 1, 1, 1, 1])
        
        assert x.shape == w.shape

        # compute v
        x = x.view(-1, x.shape[1], x.shape[2], x.shape[3], self.ch_out, self.P,self.P)
        w = w.view(-1, w.shape[1], w.shape[2], w.shape[3], self.ch_out, self.P,self.P)
        v = torch.matmul(x, w)
        v = v.view(-1, v.shape[1], v.shape[2], v.shape[3], self.P,self.P, self.ch_out)

        return v
    
    def add_cord(self, v):
        """
            Input:
                v:         (bs, ch_in, h, w, p, p, ch_out)
            Output:
                v:         (bs, ch_in, h, w, p, p, ch_out)
        """        
        #split shapes
        s_bs, s_ch_in, s_h, s_w, s_p1, s_p2, s_ch_out = v.shape
        v = v.view(s_bs, s_ch_in, s_h, s_w, s_ch_out, s_p1* s_p2)
        #coordinate addition
        ar_h = torch.arange(s_h, dtype=torch.float32) / s_h
        ar_w = torch.arange(s_w, dtype=torch.float32) / s_w
        coor_h = torch.FloatTensor(1, 1, s_h, 1, 1, s_p1* s_p2).fill_(0.).to(self.device)  
        coor_w = torch.FloatTensor(1, 1, 1, s_w, 1, s_p1* s_p2).fill_(0.).to(self.device)    
        coor_h[0, 0, :, 0, 0, 0] = ar_h
        coor_w[0, 0, 0, :, 0, 1] = ar_w
        v = v + coor_h + coor_w
        v = v.view(s_bs, s_ch_in, s_h, s_w, s_p1, s_p2, s_ch_out)     
        return v
    
    def _inv_temp(self,it):
        # AG 18/07/2018: modified schedule for inverse_temperature (lambda) based
        # on Hinton's response to questions on OpenReview.net: 
        # https://openreview.net/forum?id=HJWLfGWRb
        return (self.final_lambda * (1. - 0.95**(1 + it)))

    def em_routing(self, v, a):
        """
            Input:
                v:         (bs, ch_in, h, w, p, p, ch_out)
                a_in:      (bs, ch_in, h, w)
            
            For ConvCaps:
            Output:
                mu:        (bs, ch_out, h, w, p, p)
                a_out:     (bs, ch_out, h, w)
            Note that some dimensions are merged
            for computation convenient, that is
                v:         (bs*h*w, ch_in, p*p, ch_out)
                a_in:      (bs*h*w, ch_in, 1)
            
            For ClassCaps:
            Output:
                mu:        (bs, ch_out, p, p)
                a_out:     (bs, ch_out)
            Note that some dimensions are merged
            for computation convenient, that is
                v:         (bs, ch_in*h*w, p*p, ch_out)
                a_in:      (bs, ch_in*h*w, 1)
        """

        # split shapes
        s_bs, s_ch_in, s_h, s_w, s_p1, s_p2, s_ch_out = v.shape

        if self.class_caps == False:
            # reshape for conv caps
            v = v.view(s_bs*s_h*s_w, s_ch_in, s_ch_out, s_p1*s_p2)
            a = a.view(s_bs*s_h*s_w, s_ch_in).unsqueeze(-1)
            #declare r
            r = torch.FloatTensor(s_bs*s_h*s_w, s_ch_in, s_ch_out).fill_(1./s_ch_out).to(self.device)
        else:
            # cood add
            v = self.add_cord(v)
            # reshape for class caps
            v = v.view(s_bs, s_ch_in*s_h*s_w, s_ch_out, s_p1*s_p2)
            a = a.view(s_bs, s_ch_in*s_h*s_w).unsqueeze(-1)            
            # declare r
            r = torch.FloatTensor(s_bs, s_ch_in*s_h*s_w, s_ch_out).fill_(1./s_ch_out).to(self.device)


        #iteration
        for it in range(self.iter):
            # M-Step (with inverse temperatur schedulder lambda)
            lambd=self._inv_temp(it)
            a_out, mu, sig_sq = self.m_step(a, r, v, lambd=lambd)
            
            # E-Step
            if it < self.iter - 1:
                r = self.e_step(mu, sig_sq, a_out, v)

        #reshape from M and a as output
        if self.class_caps == False:
            mu = mu.view(s_bs, s_ch_out, s_h, s_w, s_p1, s_p2)
            a_out = a_out.view(s_bs, s_ch_out, s_h, s_w)
        else:
            mu = mu.view(s_bs, s_ch_out, s_p1, s_p2)
            a_out = a_out.view(s_bs, s_ch_out)

        return mu, a_out

    def e_step(self, mu, sig_sq, a_out, v):
        """
            ln_p_j = sum_h \dfrac{(\V^h_{ij} - \mu^h_j)^2}{2 \sigma^h_j}
                    - sum_h ln(\sigma^h_j) - 0.5*\sum_h ln(2*\pi)
            r = softmax(ln(a_j*p_j))
              = softmax(ln(a_j) + ln(p_j))
            Input:
                mu:        (bs*h*w, 1, ch_out, P*P)
                sig_sq:    (bs*h*w, 1, ch_out, P*P)
                a_out:     (bs*h*w, 1, ch_out, 1)
                v:         (bs*h*w, ch_in, ch_out, p*p)
            Local:
                p_ln:  (bs*h*w, ch_in, ch_out, p*p)
                ap_ln:     (bs*h*w, ch_in, ch_out, 1)
            Output:
                r:         (bs*h*w, ch_in, ch_out)
        """
        p_ln = -1. * (v - mu)**2 / (2 * sig_sq) - torch.log(sig_sq.sqrt()) - 0.5*torch.log(torch.tensor(2*math.pi))
        ap_ln = (p_ln.sum(dim=3, keepdim=True) + torch.log(a_out)).squeeze(-1)
        r = self.softmax(ap_ln)

        return r

    def m_step(self, a, r, v, lambd):
        """
            \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}
            (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}
            cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}
            a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))
            Input:
                a_in:      (bs*h*w, ch_in, 1)
                r:         (bs*h*w, ch_in, ch_out)
                v:         (bs*h*w, ch_in, ch_out, p*p)
            Local:
                cost_h:    (bs*h*w, 1, ch_out, P*P)
                r_sum:     (bs*h*w, 1, ch_out, 1)
            Output:
                a_out:     (bs*h*w, 1, ch_out, 1)
                mu:        (bs*h*w, 1, ch_out, P*P)
                sig_sq:    (bs*h*w, 1, ch_out, P*P)
        """
        s_st, s_ch_in, s_ch_out, p = v.shape 
        r = (r * a).unsqueeze(-1) + self.eps
        r_sum = r.sum(dim=1, keepdim=True)
        mu = torch.sum(r * v, dim=1, keepdim=True) / r_sum
        sig_sq = (torch.sum(r * (v - mu)**2, dim=1, keepdim=True) / r_sum)  + self.eps
        cost = (self.b_u.view(1,1,s_ch_out,1) + torch.log(sig_sq.sqrt())) * r_sum
        a_out = self.sigmoid(lambd*(self.b_a.view(1,1,s_ch_out,1) - cost.sum(dim=3, keepdim=True)))
        
        return a_out, mu, sig_sq

    def forward(self, x):
        #split pose and activation
        x, a = x

        # conv of poses to get votes v
        x = self.voting(x) 
        
        # conv activations
        a = self.conv_stat(a)

        #routing
        x, a = self.em_routing(x, a)


        return x, a


class CapsNetEM(nn.Module):
    """
    Genrate CapsNet with EM routing
    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...

        input: (bs, 1, 28, 28)
    """

    def __init__(self, A=32, B=32, C=32, D=32,E=10, K=3, P=4, iter=3, hw_out=(28,28), device="cpu"):
        super().__init__()
        hw_out = self.hw_cal(hw_out, kernel=5, padding=2, dilatation=1, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=A, kernel_size=(5, 5), stride=2, padding=2),
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(num_features=A),
        )
        hw_out = self.hw_cal(hw_out, kernel=1, padding=0, dilatation=1, stride=1)
        self.prime_caps = PrimaryCaps(ch_in=A, ch_out=B, K=1, P=P, stride=1, padding="valid")
        #
        hw_out = self.hw_cal(hw_out, kernel=K, padding=0, dilatation=1, stride=2)
        self.conv_caps1 = ConvCaps(ch_in=B, ch_out=C, K=K, P=P, stride=2, iter=iter, hw_out=hw_out, class_caps=False, device=device)
        #
        hw_out = self.hw_cal(hw_out, kernel=K, padding=0, dilatation=1, stride=1)
        self.conv_caps2 = ConvCaps(ch_in=C, ch_out=D, K=K, P=P, stride=1, iter=iter, hw_out=hw_out, class_caps=False, device=device)
        #
        self.class_caps = ConvCaps(ch_in=D, ch_out=E, K=1, P=P, stride=1, iter=iter, hw_out=hw_out, class_caps=True, device=device)

    def hw_cal(self, hw_in, kernel, padding=0, dilatation=1, stride=1):
        if type(hw_in) == type(int()):
            hw_out = math.floor((hw_in + 2*padding - dilatation * (kernel - 1) - 1) / stride + 1)
        elif type(hw_in) == type(tuple()):
            h_out = math.floor((hw_in[0] + 2*padding - dilatation * (kernel - 1) - 1) / stride + 1)
            w_out = math.floor((hw_in[1] + 2*padding - dilatation * (kernel - 1) - 1) / stride + 1)
            hw_out = (h_out, w_out)
        return hw_out

    def forward(self, x):
        x = self.conv1(x)
        x = self.prime_caps(x)
        x = self.conv_caps1(x)
        x = self.conv_caps2(x)
        x = self.class_caps(x)
        return x


if __name__ == '__main__':
    device = "cpu"
    CN = CapsNetEM(A=32, B=32, C=32, D=32, E=10, K=3, P=4, iter=3, hw_out=(28,28), device=device)
    CN.to(torch.device(device))
    x = torch.rand(1,1,28,28).to(torch.device(device))
    
    p, a = CN(x)

    print("Output Pose: {}, Output Activation: {}".format(p.shape, a.shape))