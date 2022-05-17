import torch
import torch.nn as nn

###############
# MNIST
###############
class BBFC_mnist_deep(nn.Module):
    """
        Backbone model from Efficient-CapsNet for MNIST
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding="valid"),
            nn.Conv2d(32, 64, kernel_size=5, padding="valid"),
            nn.Conv2d(64, 128, kernel_size=3, padding="valid"),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding="valid"),
            nn.Conv2d(256, 512, kernel_size=9, stride=2, groups=1, padding="valid"),
        )

    def forward(self, x):
        """
            IN:
                x (b, 1, 28, 28)
            OUT:
                x (b, 128, 9, 9)
        """
        return self.layers(x)


class BBFC_mnist_shallow(nn.Module):
    """
        Backbone model from Efficient-CapsNet for MNIST
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                        nn.Conv2d(1, 256, kernel_size=9, stride=1, padding=0),
                        nn.Conv2d(256, 256, kernel_size=9, stride=2, padding=0),
        )

    def forward(self, x):
        """
            IN:
                x (b, 1, 28, 28)
            OUT:
                x (b, 128, 9, 9)
        """
        return self.layers(x)



class BBFC_mnist_nostride(nn.Module):
    """
        Backbone model from Efficient-CapsNet for MNIST
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size=11, stride=1, padding=0),
                        nn.Conv2d(64, 256, kernel_size=10, stride=1, padding=0),
                        nn.Conv2d(256, 1024, kernel_size=9, stride=1, padding=0, groups=1),
        )

    def forward(self, x):
        """
            IN:
                x (b, 1, 28, 28)
            OUT:
                x (b, 128, 9, 9)
        """
        return self.layers(x)


###############
# SMALLNORB
###############

class BBFC_smallnorb_shallow_48(nn.Module):
    """
        Backbone model from Efficient-CapsNet for MNIST
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                        nn.Conv2d(2, 256, kernel_size=32, stride=1),
                        nn.ReLU(inplace=True),
                        #nn.Conv2d(256, 256, kernel_size=9, stride=2),
                        #nn.ReLU(inplace=True),
                        nn.Conv2d(256, 32*8, kernel_size=9, stride=2),
        )

    def forward(self, x):
        """
            IN:
                x (b, 1, 48, 48)
            OUT:
                x (b, 128, 9, 9)
        """
        return self.layers(x)


class BBFC_smallnorb_shallow_96(nn.Module):
    """
        Backbone model from Efficient-CapsNet for MNIST
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                        nn.Conv2d(2, 256, kernel_size=80, stride=1),
                        nn.ReLU(inplace=True),
                        #nn.Conv2d(256, 256, kernel_size=9, stride=2),
                        #nn.ReLU(inplace=True),
                        nn.Conv2d(256, 32*8, kernel_size=9, stride=2),
        )


    def forward(self, x):
        """
            IN:
                x (b, 1, 48, 48)
            OUT:
                x (b, 128, 9, 9)
        """
        return self.layers(x)


###############
# Affnist
###############


class BBFC_affnist_deep(nn.Module):
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
            nn.Conv2d(32, 64, kernel_size=(7), stride=1, padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(7), stride=2, padding="valid")
        )

    def forward(self, x):
        """
            IN:
                x (b, 1, 40, 40)
            OUT:
                x (b, 128, 15, 15)
        """
        return self.layers(x)




class BBFC_affnist_shallow(nn.Module):
    """
        Backbone model for AffNIST (40x40)
        Identical to MNIST Backbone
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(13), padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(15), stride=2, padding="valid")
        )

    def forward(self, x):
        """
            IN:
                x (b, 1, 40, 40)
            OUT:
                x (b, 128, 15, 15)
        """
        return self.layers(x)