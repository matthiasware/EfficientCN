import torch
import torch.nn as nn
from .layers import View, Squash, PrimaryCaps, FCCaps, PrimaryCapsLayer, CapsLayer
from .functions import masking


class MultiMnistBaselineCNN(nn.Module):
    pass


class MultiMnistEcnBackbone(nn.Module):
    """
        Backbone model from Efficient-CapsNet for MultiMNIST
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2, padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
    def forward(self, x):
        """
            IN:
                x (b, 1, 36, 36)
            OUT:
                x (b, 128, 6, 6)
        """
        return self.layers(x)
    
    
class MultiMnistEcnDecoder(nn.Module):
    """
        Decoder model from Efficient-CapsNet for MultiMNIST
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16*10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 36*36),
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
        x = self.layers(x)
        x = x.view(-1, 1, 36, 36)
        return x

    
class MultiMnistEffCapsNet(nn.Module):
    """
        EffCaps Implementation for MultiMNIST
        all parameters taken from the paper
    """
    def __init__(self):
        super().__init__()
        # values from paper, are fixed!
        self.n_l = 16  # num of primary capsules
        self.d_l = 8   # dim of primary capsules
        self.n_h = 10  # num of output capsules
        self.d_h = 16  # dim of output capsules
        
        self.backbone = MultiMnistEcnBackbone()
        self.primcaps = PrimaryCaps(F=128, K=5, N=self.n_l, D=self.d_l, s=2) # F = n_l * d_l !!! # S=stride=2 reduces [1,128,6,6] -> [1,128,5,5]
        self.fcncaps = FCCaps(self.n_l, self.n_h, self.d_l, self.d_h) 
        self.decoder1 = MultiMnistEcnDecoder()
        self.decoder2 = MultiMnistEcnDecoder()

        #initializer
        #self.initialize_weights()

    def forward(self, x, y_true=None, z_true=None):
        """
            IN:
                x (b, 1, 36, 36)
            OUT:
                u_h    
                    (b, n_h, d_h)
                    output caps
                x_rec  
                    (b, 1, 36, 36)
                    reconstruction of x
        """
        #Encoder
        u_l = self.primcaps(self.backbone(x))
        u_h = self.fcncaps(u_l)
        
        #Decoder
        if y_true is None:
            yz_pred = torch.topk(torch.norm(u_h, dim=2), k=2, dim=1).indices
            y_true = yz_pred[:,0]
            z_true = yz_pred[:,1]

        u_h_masked_y = masking(u_h, y_true)
        x_rec_y = self.decoder1(u_h_masked_y)
        
        u_h_masked_z = masking(u_h, z_true)
        x_rec_z = self.decoder2(u_h_masked_z)

        return u_h, x_rec_y, x_rec_z

    def initialize_weights(self):
        #initialize backbone
        nn.init.kaiming_uniform_(self.backbone.layers[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.backbone.layers[3].weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.backbone.layers[6].weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.backbone.layers[9].weight, a=0, mode='fan_in', nonlinearity='relu')

        #initialize decoder1
        nn.init.kaiming_uniform_(self.decoder1.layers[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.decoder1.layers[2].weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.xavier_normal_(self.decoder1.layers[4].weight)

        #initialize decoder1
        nn.init.kaiming_uniform_(self.decoder2.layers[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.decoder2.layers[2].weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.xavier_normal_(self.decoder2.layers[4].weight)


class MultiMnistEffCapsNet2(nn.Module):
    """
        EffCaps Implementation for MultiMNIST
        all parameters taken from the paper
    """
    def __init__(self):
        super().__init__()
        # values from paper, are fixed!
        self.n_l = 16  # num of primary capsules
        self.d_l = 8   # dim of primary capsules
        self.n_h = 10  # num of output capsules
        self.d_h = 16  # dim of output capsules
        
        self.backbone = MultiMnistEcnBackbone()
        self.primcaps = PrimaryCaps(F=128, K=6, N=self.n_l, D=self.d_l, s=1) # F = n_l * d_l !!! # S=stride=2 reduces [1,128,6,6] -> [1,128,5,5]
        self.fcncaps = FCCaps(self.n_l, self.n_h, self.d_l, self.d_h) 
        self.decoder1 = MultiMnistEcnDecoder()
        self.decoder2 = MultiMnistEcnDecoder()

        #initializer
        #self.initialize_weights()

    def forward(self, x, y_true=None, z_true=None):
        """
            IN:
                x (b, 1, 36, 36)
            OUT:
                u_h    
                    (b, n_h, d_h)
                    output caps
                x_rec  
                    (b, 1, 36, 36)
                    reconstruction of x
        """
        #Encoder
        u_l = self.primcaps(self.backbone(x))
        u_h = self.fcncaps(u_l)
        
        #Decoder
        if y_true is None:
            yz_pred = torch.topk(torch.norm(u_h, dim=2), k=2, dim=1).indices
            y_true = yz_pred[:,0]
            z_true = yz_pred[:,1]

        u_h_masked_y = masking(u_h, y_true)
        x_rec_y = self.decoder1(u_h_masked_y)
        
        u_h_masked_z = masking(u_h, z_true)
        x_rec_z = self.decoder2(u_h_masked_z)

        return u_h, x_rec_y, x_rec_z

    def initialize_weights(self):
        #initialize backbone
        nn.init.kaiming_uniform_(self.backbone.layers[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.backbone.layers[3].weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.backbone.layers[6].weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.backbone.layers[9].weight, a=0, mode='fan_in', nonlinearity='relu')

        #initialize decoder1
        nn.init.kaiming_uniform_(self.decoder1.layers[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.decoder1.layers[2].weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.xavier_normal_(self.decoder1.layers[4].weight)

        #initialize decoder1
        nn.init.kaiming_uniform_(self.decoder2.layers[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.decoder2.layers[2].weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.xavier_normal_(self.decoder2.layers[4].weight)

class CapsNet(nn.Module):
    """
        CapsNet Implementation for MultiMNIST
        all parameters taken from the paper
    """
    def __init__(self):
        super().__init__()
        # values from paper, are fixed!
        self.n_l = (32 * 6 * 6) # num of primary capsules
        self.d_l = 8            # dim of primary capsules
        self.n_h = 10           # num of output capsules
        self.d_h = 16           # dim of output capsules
        self.n_iter = 3
        
        self.backbone = nn.Sequential(
                        nn.Conv2d(1, 256, kernel_size=9, stride=1),
                        nn.Conv2d(256, 256, kernel_size=9, stride=1)
        )
        self.primcaps = PrimaryCapsLayer(c_in=256,c_out=32,d_l=self.d_l, kernel_size=9, stride=2)
        self.digitcaps = CapsLayer(self.n_l, self.d_l, self.n_h, self. d_h, self.n_iter) 
        self.decoder1 = MultiMnistEcnDecoder()
        self.decoder2 = MultiMnistEcnDecoder()

        #initializer
        #self.initialize_weights()

    def forward(self, x, y_true=None, z_true=None):
        """
            IN:
                x (b, 1, 36, 36)
            OUT:
                u_h    
                    (b, n_h, d_h)
                    output caps
                x_rec  
                    (b, 1, 36, 36)
                    reconstruction of x
        """
        #Encoder
        u_l = self.primcaps(self.backbone(x))
        u_h = self.digitcaps(u_l)
        
        #Decoder
        if y_true is None:
            yz_pred = torch.topk(torch.norm(u_h, dim=2), k=2, dim=1).indices
            y_true = yz_pred[:,0]
            z_true = yz_pred[:,1]

        u_h_masked_y = masking(u_h, y_true)
        x_rec_y = self.decoder1(u_h_masked_y)
        
        u_h_masked_z = masking(u_h, z_true)
        x_rec_z = self.decoder2(u_h_masked_z)

        return u_h, x_rec_y, x_rec_z