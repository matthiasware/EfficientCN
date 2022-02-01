import torch
import torch.nn as nn
from .layers import View, Squash, PrimaryCaps, FCCaps, FCCapsWOBias
from .layers import View, Squash, PrimaryCaps, FCCaps
from .functions import max_norm_masking, masking


######################################
# CNN BASELINE MODELS
######################################

class MnistBaselineCNN(nn.Module):
    """
        Baseline CNN Model for MNIST
    """

    def __init__(self):
        super(MnistBaselineCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

######################################
# CN MNIST MODELS
######################################


class MnistEcnBackbone(nn.Module):
    """
        Backbone model from Efficient-CapsNet for MNIST
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
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

    def forward(self, x):
        """
            IN:
                x (b, 1, 28, 28)
            OUT:
                x (b, 128, 9, 9)
        """
        return self.layers(x)


class MnistEcnDecoder(nn.Module):
    """
        Decoder model from Efficient-CapsNet for MNIST
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
            IN:
                x (b, n, d) with n=10 and d=16
            OUT:
                x_rec (b, 1, 28, 28)
            Notes:
                input must be masked!
        """
        x = self.layers(x)
        x = x.view(-1, 1, 28, 28)
        return x


class MnistEffCapsNet(nn.Module):
    """
        EffCaps Implementation for MNIST
        all parameters taken from the paper
    """

    def __init__(self):
        super().__init__()
        # values from paper, are fixed!
        self.n_l = 16  # num of primary capsules
        self.d_l = 8   # dim of primary capsules
        self.n_h = 10  # num of output capsules
        self.d_h = 16  # dim of output capsules

        self.backbone = MnistEcnBackbone()
        self.primcaps = PrimaryCaps(
            F=128, K=9, N=self.n_l, D=self.d_l)  # F = n_l * d_l !!!
        self.fcncaps = FCCaps(self.n_l, self.n_h, self.d_l, self.d_h)
        self.decoder = MnistEcnDecoder()

    def forward(self, x, y_true=None):
        """
            IN:
                x (b, 1, 28, 28)
            OUT:
                u_h
                    (b, n_h, d_h)
                    output caps
                x_rec
                    (b, 1, 28, 28)
                    reconstruction of x
        """
        u_l = self.primcaps(self.backbone(x))
        u_h = self.fcncaps(u_l)
        #
        u_h_masked = masking(u_h, y_true)
        x_rec = self.decoder(u_h_masked)
        return u_h, x_rec


######################################
# AffNIST Models
######################################

class AffnistEcnDecoder(nn.Module):
    """
        Decoder model for AffNIST (40x40)
        Except for last layer identical with MNIST Decoder
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 40 * 40),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
            IN:
                x (b, n, d) with n=10 and d=16
            OUT:
                x_rec (b, 1, 40, 40)
            Notes:
                input must be masked!
        """
        x = self.layers(x)
        x = x.view(-1, 1, 40, 40)
        return x


class AffnistEcnBackbone(nn.Module):
    """
        Backbone model for AffNIST (40x40)
        Identical to MNIST Backbone
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
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

    def forward(self, x):
        """
            IN:
                x (b, 1, 40, 40)
            OUT:
                x (b, 128, 15, 15)
        """
        return self.layers(x)


class AffnistEffCapsNet(nn.Module):
    """
        EffCaps Implementation for AffNIST
        almost identical to MnistEffCapsNet
        all parameters taken from the paper
    """

    def __init__(self):
        super().__init__()
        # values from paper, are fixed!
        self.n_l = 16  # num of primary capsules
        self.d_l = 8   # dim of primary capsules
        self.n_h = 10  # num of output capsules
        self.d_h = 16  # dim of output capsules

        self.backbone = AffnistEcnBackbone()

        # changed from K=9 to K=15 to compensate for larger image dims
        self.primcaps = PrimaryCaps(
            F=128, K=15, N=self.n_l, D=self.d_l)  # F = n_l * d_l !!!
        self.fcncaps = FCCaps(self.n_l, self.n_h, self.d_l, self.d_h)

        # changed large layer to output (40x40) instead of (28x28)
        self.decoder = AffnistEcnDecoder()

    def forward(self, x):
        """
            IN:
                x (b, 1, 40, 40)
            OUT:
                u_h
                    (b, n_h, d_h)
                    output caps
                x_rec
                    (b, 1, 40, 40)
                    reconstruction of x
        """
        u_l = self.primcaps(self.backbone(x))
        u_h = self.fcncaps(u_l)
        #
        u_h_masked = max_norm_masking(u_h)
        u_h_masked = torch.flatten(u_h_masked, start_dim=1)
        x_rec = self.decoder(u_h_masked)
        return u_h, x_rec


class AffnistEffCapsNetWOBias(nn.Module):
    """
        EffCaps Implementation for AffNIST
        almost identical to MnistEffCapsNet
        all parameters taken from the paper
    """

    def __init__(self):
        super().__init__()
        # values from paper, are fixed!
        self.n_l = 16  # num of primary capsules
        self.d_l = 8   # dim of primary capsules
        self.n_h = 10  # num of output capsules
        self.d_h = 16  # dim of output capsules

        self.backbone = AffnistEcnBackbone()

        # changed from K=9 to K=15 to compensate for larger image dims
        self.primcaps = PrimaryCaps(
            F=128, K=15, N=self.n_l, D=self.d_l)  # F = n_l * d_l !!!
        self.fcncaps = FCCapsWOBias(self.n_l, self.n_h, self.d_l, self.d_h)

        # changed large layer to output (40x40) instead of (28x28)
        self.decoder = AffnistEcnDecoder()

    def forward(self, x):
        """
            IN:
                x (b, 1, 40, 40)
            OUT:
                u_h
                    (b, n_h, d_h)
                    output caps
                x_rec
                    (b, 1, 40, 40)
                    reconstruction of x
        """
        u_l = self.primcaps(self.backbone(x))
        u_h = self.fcncaps(u_l)
        #
        u_h_masked = max_norm_masking(u_h)
        u_h_masked = torch.flatten(u_h_masked, start_dim=1)
        x_rec = self.decoder(u_h_masked)
        return u_h, x_rec


class AffnistEffCapsNetWORec(nn.Module):
    """
        EffCaps Implementation for AffNIST
        almost identical to MnistEffCapsNet
        all parameters taken from the paper
    """

    def __init__(self):
        super().__init__()
        # values from paper, are fixed!
        self.n_l = 16  # num of primary capsules
        self.d_l = 8   # dim of primary capsules
        self.n_h = 10  # num of output capsules
        self.d_h = 16  # dim of output capsules

        self.backbone = AffnistEcnBackbone()

        # changed from K=9 to K=15 to compensate for larger image dims
        self.primcaps = PrimaryCaps(
            F=128, K=15, N=self.n_l, D=self.d_l)  # F = n_l * d_l !!!
        self.fcncaps = FCCaps(self.n_l, self.n_h, self.d_l, self.d_h)

    def forward(self, x):
        """
            IN:
                x (b, 1, 40, 40)
            OUT:
                u_h
                    (b, n_h, d_h)
                    output caps
                x_rec
                    (b, 1, 40, 40)
                    reconstruction of x
        """
        u_l = self.primcaps(self.backbone(x))
        u_h = self.fcncaps(u_l)
        return u_h

######################################
# CNN-R MNIST MODELS
######################################

class CNN_R_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=256,            
                kernel_size=9,              
                stride=1,                   
                padding=0,                  
            ),                              
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(256, 256, 9, 2),     
            nn.ReLU(),                                 
        )
        self.fc1 = nn.Linear(256 * 6 * 6, 10)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)        
        x = x.view(-1,10)
        return x


class CNN_R_Decoder(nn.Module):
    """
        Decoder model from Efficient-CapsNet for MNIST
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
            IN:
                x (b, n, d) with n=10 and d=16
            OUT:
                x_rec (b, 1, 28, 28)
            Notes:
                input must be masked!
        """
        x = self.layers(x)
        x = x.view(-1, 1, 28, 28)
        return x


class CNN_R(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNN_R_Backbone()
        self.decoder = CNN_R_Decoder()

    def forward(self, x, y_true=None):
        """
            IN:
                x (b, 1, 28, 28)
            OUT:
                u_h
                    (b, n_h, d_h)
                    output caps
                x_rec
                    (b, 1, 28, 28)
                    reconstruction of x
        """

        y_pred = self.backbone(x)
        x_rec = self.decoder(y_pred)

        return y_pred, x_rec        


######################################
# CNN-CR MNIST MODELS
######################################

class CNN_CR_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=256,            
                kernel_size=9,              
                stride=1,                   
                padding=0,                  
            ),                              
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(256, 256, 9, 2),     
            nn.ReLU(),                                 
        )
        self.fc1 = nn.Linear(256 * 6 * 6, 10 * 16)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)        
        x = x.view(-1,10,16)
        return x


class CNN_CR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNN_CR_Backbone()
        self.decoder = MnistEcnDecoder()

    def forward(self, x, y_true=None):
        """
            IN:
                x (b, 1, 28, 28)
            OUT:
                u_h
                    (b, n_h, d_h)
                    output caps
                x_rec
                    (b, 1, 28, 28)
                    reconstruction of x
        """

        u_h = self.backbone(x)

        u_h_masked = masking(u_h, y_true)
        x_rec = self.decoder(u_h_masked)

        return u_h, x_rec


######################################
# CNN-CR-SF MNIST MODELS
######################################

class CNN_CR_SF_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=256,            
                kernel_size=9,              
                stride=1,                   
                padding=0,                  
            ),                              
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(256, 256, 9, 2),     
            nn.ReLU(),                                 
        )
        self.fc1 = nn.Linear(256 * 6 * 6, 16 * 8)
        self.sq = Squash(eps=1e-20)
        self.fc2 =  nn.Linear(16 * 8, 16 * 10)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = x.view(-1,16,8)
        x = self.sq(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc2(x)
        x = x.view(-1,10,16)
        x = self.sq(x)
        return x


class CNN_CR_SF(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNN_CR_SF_Backbone()
        self.decoder = MnistEcnDecoder()

    def forward(self, x, y_true=None):
        """
            IN:
                x (b, 1, 28, 28)
            OUT:
                u_h
                    (b, n_h, d_h)
                    output caps
                x_rec
                    (b, 1, 28, 28)
                    reconstruction of x
        """

        u_h = self.backbone(x)

        u_h_masked = masking(u_h, y_true)
        x_rec = self.decoder(u_h_masked)

        return u_h, x_rec