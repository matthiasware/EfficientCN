import torch
import torch.nn as nn


class BBFC_mnist_deep(nn.Module):
    """
        Backbone model from Efficient-CapsNet for MNIST
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding="valid"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=9, stride=1, groups=1, padding="valid"),
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
                        nn.Conv2d(1, 256, kernel_size=9, stride=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(256),
                        nn.Conv2d(256, 32*8, kernel_size=9, stride=2, groups=1),
        )

    def forward(self, x):
        """
            IN:
                x (b, 1, 28, 28)
            OUT:
                x (b, 128, 9, 9)
        """
        return self.layers(x)