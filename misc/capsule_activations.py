import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt


class TwinAugmentation(nn.Module):
    """
        double random augmentation of a input 
        value example:
        rotade=30, translate=[0.2,0.2], scale=[0.9, 1.1], shear=[-3,3]
    """
    def __init__(self, rotade=None, translate=None, scale=None, shear=None):
        
        super().__init__()
        self.transforms = T.Compose([
                                T.RandomAffine(degrees=rotade,
                                                translate=translate,
                                                scale=scale,
                                                shear=shear),
                                ])
    
    def forward(self, x):
        x1 = self.transforms(x)
        x2 = self.transforms(x)
        return x1, x2


def cross_correlation(X1, X2):
    """
    normalized cross correlation of 2 Matrices (signal, capsuls)
    """
    X1_norm = (X1 - X1.mean(dim=0)) / X1.std(dim=0)
    X2_norm = (X2 - X2.mean(dim=0)) / X2.std(dim=0)

    c = torch.matmul(X1_norm.T, X2_norm) / X1.shape[0]
    return c


def show_imgrid(x,y=None,nrow=8):
    if y is not None and (y.shape[-1] % nrow) == 0:
        print(y.view(-1,nrow))

    img = torchvision.utils.make_grid(x[:64,:1,:,:], nrow=nrow)
    img = img.permute((1,2,0))
    plt.imshow(img)
    plt.show()


def show_pc_vals(pc, ncol=8, cmap="copper", figsize=(12,50)):
    """
    IN: mat -> torch.tensor(pc_no, pc_dim), Primecaps of one Input 
    """
    
    mat = pc.view(ncol, -1, pc.shape[1])
    mat = mat.cpu().detach().numpy()
    
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots(1, mat.shape[0], sharex=False, sharey=True)

    for i in range(mat.shape[0]):
        ax[i].imshow(mat[i], cmap="copper")

    plt.show()


def calc_pc_delta(pc1, pc2):
    """
    generate norm over capsule dimension
    subtract values of 2 inputs elementwise
    generate norm over dims for scarlar output -> maybe some better way?

    IN: pc1, pc2 -> PrimeCaps of 2 inputs (pc_no, pc_dim)
    OUT: norm_delta -> calculated delta
    """
    pc1 = pc1.norm(dim=-1) 
    pc2 = pc2.norm(dim=-1)
    delta = pc1 - pc2
    norm_delta = delta.norm(dim=-1)
    return norm_delta



def calc_pc_corr(pc_ref, pc_corr):
    """
    calculate correlation between reference tensor of primecaps and any other primecaps
    - generate norm over capsule dimension
    - calculate correlation matrix
    - output correlation to reference

    IN:     pc_ref  -> PrimeCaps reference (bs, pc_no, pc_dim)
            pc_corr -> PrimeCaps of any input (bs, pc_no, pc_dim)
    OUT:    correlation coeffs
    """

    corr = torch.cat((pc_ref, pc_corr),dim=0)
    corr = corr.norm(dim=-1) 
    corr = torch.corrcoef(corr)

    out = corr[0,1:]

    return out


def plt_lin(vals):
    """
    simple x y plot
    """
    norm_lin = np.arange(0,len(vals),1)

    plt.plot(norm_lin, vals)
    plt.show


def affine_xtrans(img, target, range=[-5.,5.,1]):
    """
    generate one transform output for each value in range
    x transpose
    IN:  img      -> input tensor (1,c,h,w)
         target   -> input target tensor (1)
         range    -> space of transforms
    OUT: x_trans  -> input tensor (range,c,h,w)
         l_target -> input target tensor (range)
    """
    arange = np.arange(range[0],(range[1]+range[2]),range[2])
    x_trans = torch.zeros([len(arange),img.shape[1],img.shape[2],img.shape[3]])
    l_target = torch.zeros(len(arange))

    for i, l in enumerate(arange):
        x_trans[i] = T.functional.affine(img=img, angle=0, translate=[l,0], scale=1.,shear=0)
        l_target[i] = target
    
    return x_trans, l_target


def affine_ytrans(img, target, range=[-5.,5.,1]):
    """
    generate one transform output for each value in range
    y transpose
    IN:  img      -> input tensor (1,c,h,w)
         target   -> input target tensor (1)
         range    -> space of transforms
    OUT: x_trans  -> input tensor (range,c,h,w)
         l_target -> input target tensor (range)
    """
    arange = np.arange(range[0],(range[1]+range[2]),range[2])
    y_trans = torch.zeros([len(arange),img.shape[1],img.shape[2],img.shape[3]])
    l_target = torch.zeros(len(arange))

    for i, l in enumerate(arange):
        y_trans[i] = T.functional.affine(img=img, angle=0, translate=[0,l], scale=1.,shear=0)
        l_target[i] = target
    
    return y_trans, l_target


def affine_rot(img, target, range=[-25.,25.,1]):
    """
    generate one transform output for each value in range
    rotation
    IN:  img      -> input tensor (1,c,h,w)
         target   -> input target tensor (1)
         range    -> space of transforms
    OUT: x_trans  -> input tensor (range,c,h,w)
         l_target -> input target tensor (range)
    """
    arange = np.arange(range[0],(range[1]+range[2]),range[2])
    rot = torch.zeros([len(arange),img.shape[1],img.shape[2],img.shape[3]])
    l_target = torch.zeros(len(arange))

    for i, l in enumerate(arange):
        rot[i] = T.functional.affine(img=img, angle=l, translate=[0,0], scale=1.,shear=0)
        l_target[i] = target
    print(img.shape)
    
    return rot, l_target


def affine_scale(img, target, range=[0.75,1.25,0.05]):
    """
    generate one transform output for each value in range
    scale
    IN:  img      -> input tensor (1,c,h,w)
         target   -> input target tensor (1)
         range    -> space of transforms
    OUT: x_trans  -> input tensor (range,c,h,w)
         l_target -> input target tensor (range)
    """
    arange = np.arange(range[0],(range[1]+range[2]),range[2])
    rot = torch.zeros([len(arange),img.shape[1],img.shape[2],img.shape[3]])
    l_target = torch.zeros(len(arange))

    for i, l in enumerate(arange):
        rot[i] = T.functional.affine(img=img, angle=0, translate=[0,0], scale=l,shear=0)
        l_target[i] = target
    
    return rot, l_target


def affine_shear(img, target, range=[-10.,10.,2]):
    """
    generate one transform output for each value in range
    shear
    IN:  img      -> input tensor (1,c,h,w)
         target   -> input target tensor (1)
         range    -> space of transforms
    OUT: x_trans  -> input tensor (range,c,h,w)
         l_target -> input target tensor (range)
    """
    arange = np.arange(range[0],(range[1]+range[2]),range[2])
    rot = torch.zeros([len(arange),img.shape[1],img.shape[2],img.shape[3]])
    l_target = torch.zeros(len(arange))

    for i, l in enumerate(arange):
        rot[i] = T.functional.affine(img=img, angle=0, translate=[0,0], scale=1,shear=l)
        l_target[i] = target
    
    return rot, l_target


