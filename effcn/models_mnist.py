import torch
import torch.nn as nn
from .layers import Squash, SquashHinton, PrimaryCaps, FCCaps, PrimaryCapsLayer, CapsLayer
from .functions import max_norm_masking, masking


######################################
# CNN BASELINE MODELS
######################################

class BaselineCNN(nn.Module):
    """
        Baseline CNN Model for MNIST
    """

    def __init__(self):
        super(BaselineCNN, self).__init__()
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
# EffCN MNIST MODELS
######################################


class Backbone(nn.Module):
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


class Decoder(nn.Module):
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


class EffCapsNet(nn.Module):
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

        self.backbone = Backbone()
        self.primcaps = PrimaryCaps(
            F=128, K=9, N=self.n_l, D=self.d_l)  # F = n_l * d_l !!!
        self.fcncaps = FCCaps(self.n_l, self.n_h, self.d_l, self.d_h)
        self.decoder = Decoder()

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
# CN MNIST MODELS
######################################


class BackboneHinton(nn.Module):
    """
        Backbone model from Efficient-CapsNet for MNIST
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                        nn.Conv2d(1, 256, kernel_size=9, stride=1),
                        nn.ReLU(inplace=True),
                        #nn.BatchNorm2d(256),
        )

    def forward(self, x):
        """
            IN:
                x (b, 1, 28, 28)
            OUT:
                x (b, 128, 9, 9)
        """
        return self.layers(x)


class BackboneHiNoStride(nn.Module):
    """
        Backbone model from Efficient-CapsNet for MNIST
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size=11, stride=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64, 256, kernel_size=10, stride=1, padding=0),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(256),
        )

    def forward(self, x):
        """
            IN:
                x (b, 1, 28, 28)
            OUT:
                x (b, 256, 9, 9)
        """
        return self.layers(x)


class BackboneHiDeep(nn.Module):
    """
        Backbone model from Efficient-CapsNet for MNIST
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=5, padding="valid"),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32, 64, kernel_size=5, padding="valid"),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64, 128, kernel_size=3, padding="valid"),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(128),
                        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding="valid"),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(256),
        )

    def forward(self, x):
        """
            IN:
                x (b, 1, 28, 28)
            OUT:
                x (b, 128, 9, 9)
        """
        return self.layers(x)


class CapsNet(nn.Module):
    """
        CapsNet Implementation for MNIST
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

        self.backbone = BackboneHinton()
        self.primcaps = PrimaryCapsLayer(c_in=256,c_out=32,d_l=self.d_l, kernel_size=9, stride=2)
        self.digitcaps = CapsLayer(self.n_l, self.d_l, self.n_h, self. d_h, self.n_iter)
        self.decoder = Decoder()

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
        u_h = self.digitcaps(u_l)
        #
        u_h_masked = masking(u_h, y_true)
        x_rec = self.decoder(u_h_masked)
        return u_h, x_rec

    def forward_pc_study(self, x, y_true=None):
        """
        IN as forward

        OUT as forward + u_l, bb
        """
        bb = self.backbone(x)
        u_l = self.primcaps(bb)
        u_h = self.digitcaps(u_l)
        #
        u_h_masked = masking(u_h, y_true)
        x_rec = self.decoder(u_h_masked)
        return u_h, x_rec, u_l, bb

#####################################
# CN EffCN with BB PC  cross implemetations
#####################################

class CapsNetCross(nn.Module):
    """
        CapsNet Implementation for MNIST
        all parameters taken from the paper
    """

    def __init__(self):
        super().__init__()
        # values from paper, are fixed!
        self.n_l = 16#(32 * 6 * 6) # num of primary capsules
        self.d_l = 8            # dim of primary capsules
        self.n_h = 10           # num of output capsules
        self.d_h = 16           # dim of output capsules
        self.n_iter = 3

        #self.backbone = BackboneHinton()
        #self.primcaps = PrimaryCapsLayer(c_in=256,c_out=32,d_l=self.d_l, kernel_size=9, stride=2)

        self.backbone = Backbone()
        self.primcaps = PrimaryCaps(F=128, K=9, N=self.n_l, D=self.d_l, sq=False)  # F = n_l * d_l !!!
        self.squash =  SquashHinton()

        self.digitcaps = CapsLayer(self.n_l, self.d_l, self.n_h, self. d_h, self.n_iter)
        self.decoder = Decoder()

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
        x = self.backbone(x)
        u_l = self.primcaps(x)
        u_l = self.squash(u_l)
        u_h = self.digitcaps(u_l)
        print(u_l.shape, u_h.shape)
        #
        u_h_masked = masking(u_h, y_true)
        x_rec = self.decoder(u_h_masked)
        return u_h, x_rec



class EffCapsNetCross(nn.Module):
    """
        EffCaps Implementation for MNIST
        all parameters taken from the paper
    """

    def __init__(self):
        super().__init__()
        # values from paper, are fixed!
        self.n_l = (32 * 6 * 6) #16  # num of primary capsules
        self.d_l = 8   # dim of primary capsules
        self.n_h = 10  # num of output capsules
        self.d_h = 16  # dim of output capsules

        self.backbone = BackboneHinton()
        self.primcaps = PrimaryCapsLayer(c_in=256,c_out=32,d_l=self.d_l, kernel_size=9, stride=2, sq=False)
        self.squash = Squash(eps=1e-20)
        self.fcncaps = FCCaps(self.n_l, self.n_h, self.d_l, self.d_h)
        self.decoder = Decoder()

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
        x   = self.backbone(x)
        u_l = self.primcaps(x)
        u_l = self.squash(u_l)
        u_h = self.fcncaps(u_l)
        print(u_l.shape, u_h.shape)
        #
        u_h_masked = masking(u_h, y_true)
        x_rec = self.decoder(u_h_masked)
        return u_h, x_rec



#####################################
# CN with alternative BB PC implemetations
#####################################

class CapsNetNoStride(nn.Module):
    """
        CapsNet Implementation for MNIST
        all parameters taken from the paper
    """

    def __init__(self):
        super().__init__()
        # values from paper, are fixed!
        self.n_l = 128 # num of primary capsules
        self.d_l = 8            # dim of primary capsules
        self.n_h = 10           # num of output capsules
        self.d_h = 16           # dim of output capsules
        self.n_iter = 3

        self.backbone = BackboneHiNoStride()
        self.primcaps = PrimaryCapsLayer(c_in=256,c_out=128,d_l=self.d_l, kernel_size=9, stride=1)
        self.digitcaps = CapsLayer(self.n_l, self.d_l, self.n_h, self. d_h, self.n_iter)
        self.decoder = Decoder()

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
        u_h = self.digitcaps(u_l)
        #
        u_h_masked = masking(u_h, y_true)
        x_rec = self.decoder(u_h_masked)
        return u_h, x_rec



class CapsNetDeep(nn.Module):
    """
        CapsNet Implementation for MNIST
        all parameters taken from the paper
    """

    def __init__(self):
        super().__init__()
        # values from paper, are fixed!
        self.n_l = 1024         # num of primary capsules
        self.d_l = 8            # dim of primary capsules
        self.n_h = 10           # num of output capsules
        self.d_h = 16           # dim of output capsules
        self.n_iter = 3

        self.backbone = BackboneHiDeep()
        self.primcaps = PrimaryCapsLayer(c_in=256,c_out=64,d_l=self.d_l, kernel_size=9, stride=2)
        self.digitcaps = CapsLayer(self.n_l, self.d_l, self.n_h, self. d_h, self.n_iter)
        self.decoder = Decoder()

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
        u_h = self.digitcaps(u_l)
        #
        u_h_masked = masking(u_h, y_true)
        x_rec = self.decoder(u_h_masked)
        return u_h, x_rec


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
        self.fc1 = nn.Linear(256 * 6 * 6, 10 * 16)
        self.fc2 = nn.Linear(10 * 16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        out = self.fc2(x)
        out = out.view(-1, 10)
        return out, x


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
        self.decoder = Decoder()

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

        y_pred, x_dec = self.backbone(x)
        x_rec = self.decoder(x_dec)

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
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = x.view(-1, 10, 16)

        return x


class CNN_CR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNN_CR_Backbone()
        self.decoder = Decoder()

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
        self.fc2 = nn.Linear(16 * 8, 16 * 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = x.view(-1, 16, 8)
        x = self.sq(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc2(x)
        x = x.view(-1, 10, 16)
        x = self.sq(x)
        return x


class CNN_CR_SF(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CNN_CR_SF_Backbone()
        self.decoder = Decoder()

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
