import torch
import torch.nn as nn
from .layers import View, Squash, PrimaryCaps, FCCaps, PrimaryCapsLayer, CapsLayer
from .functions import max_norm_masking, masking

class SmallNorbEcnBackbone(nn.Module):
    """
        Backbone model from Efficient-CapsNet for SmallNorb
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(7, 7),stride=2, padding="valid"),
            nn.LeakyReLU(0.3,inplace=True),
            nn.InstanceNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding="valid"),
            nn.LeakyReLU(0.3,inplace=True),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding="valid"),
            nn.LeakyReLU(0.3,inplace=True),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding="valid"),
            nn.LeakyReLU(0.3,inplace=True),
            nn.InstanceNorm2d(128),
        )
    def forward(self, x):
        """
            IN:
                x (b, 2, 48, 48)
            OUT:
                x (b, 128, 8, 8)
        """
        return self.layers(x)
    

class SmallNorbEcnDecoder(nn.Module):
    """
        Decoder model from Efficient-CapsNet for SmallNorb
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(16*5, 64)
        )
        
        self.layer2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding="valid"),
            nn.LeakyReLU(0.3,inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding="valid"),
            nn.LeakyReLU(0.3,inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding="valid"),
            nn.LeakyReLU(0.3,inplace=True),
            nn.Conv2d(128, 2, kernel_size=(3, 3), padding="valid"),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
            IN:
                x (b, n, d) with n=10 and d=16
            OUT:
                x_rec (b, 1, 36, 36)
            Notes:
                input must be masked!
        """
        x = self.layer1(x)
        x = x.view(-1, 1, 8, 8)
        x = self.layer2(x)
        return x

    
class SmallNorbEffCapsNet(nn.Module):
    """
        EffCaps Implementation for SmallNorb
        all parameters taken from the paper
    """
    def __init__(self):
        super().__init__()
        # values from paper, are fixed!
        self.n_l = 16  # num of primary capsules
        self.d_l = 8   # dim of primary capsules
        self.n_h = 5   # num of output capsules
        self.d_h = 16  # dim of output capsules
        
        self.backbone = SmallNorbEcnBackbone()
        self.primcaps = PrimaryCaps(F=128, K=8, N=self.n_l, D=self.d_l, s=2) # F = n_l * d_l !!!
        self.fcncaps = FCCaps(self.n_l, self.n_h, self.d_l, self.d_h) 
        self.decoder = SmallNorbEcnDecoder()

    def forward(self, x, y_true=None):
        """
            IN:
                x (b, 2, 48, 48)
            OUT:
                u_h    
                    (b, n_h, d_h)
                    output caps
                x_rec  
                    (b, 2, 48, 48)
                    reconstruction of x
        """
        #Encoder
        u_l = self.backbone(x)
        u_l = self.primcaps(u_l)
        u_h = self.fcncaps(u_l)
        
        #Decoder
        u_h_masked = masking(u_h, y_true)
        x_rec = self.decoder(u_h_masked)
        
        return u_h,  x_rec


class CapsNet(nn.Module):
    """
        CapsNet Implementation for SmallNorb
        all parameters taken from the paper
    """
    def __init__(self):
        super().__init__()
        self.n_l = (32 * 12 * 12) # num of primary capsules
        self.d_l = 8            # dim of primary capsules
        self.n_h = 5           # num of output capsules
        self.d_h = 16           # dim of output capsules
        self.n_iter = 3
        
        self.backbone = nn.Sequential(
                        nn.Conv2d(2, 256, kernel_size=9, stride=1),
                        nn.Conv2d(256, 256, kernel_size=9, stride=1)
        )
        self.primcaps = PrimaryCapsLayer(c_in=256,c_out=32,d_l=self.d_l, kernel_size=9, stride=2)
        self.digitcaps = CapsLayer(self.n_l, self.d_l, self.n_h, self. d_h, self.n_iter) 
        self.decoder = SmallNorbEcnDecoder()

    def forward(self, x, y_true=None):
        """
            IN:
                x (b, 2, 48, 48)
            OUT:
                u_h    
                    (b, n_h, d_h)
                    output caps
                x_rec  
                    (b, 2, 48, 48)
                    reconstruction of x
        """
        #Encoder
        u_l = self.backbone(x)
        u_l = self.primcaps(u_l)
        u_h = self.digitcaps(u_l)
        
        #Decoder
        u_h_masked = masking(u_h, y_true)
        x_rec = self.decoder(u_h_masked)
        
        return u_h,  x_rec