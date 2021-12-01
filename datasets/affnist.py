from pathlib import Path

import torch
from torch.utils.data import Dataset


class AffNIST(Dataset):
    """
        Requires files in p_root:
        - affnist_train.pt
        - affnist_valid.pt
        - mnist_train.pt
        - mnist_valid.pt

        CHECK NOTEBOOK TO CREATE THE DATASET
    """

    DATA_DIR = "AFFNIST"
    SPLIT_FILES = {
        'mnist_train': 'mnist_train.pt',
        'mnist_valid': 'mnist_valid.pt',
        'affnist_train': 'affnist_train.pt',
        'affnist_valid': 'affnist_valid.pt'
    }

    def __init__(self, p_root: str, split: str = 'mnist_train', transform=None, target_transform=None):
        """
            p_root:
                path to directory where directory AFFNIST resides
            split:
                mnist_train:
                    original mnist train set, randomly padded to 40x40
                mnist_valid:
                    original mnist test set, randomly padded to 40x40
                affnist_train:
                    affine transformed mnist train set
                affnist_valid
                    affine transformed mnist test set
            transform:
                transform function on input imgs, format Tensors
            target_transform:
                transform function on input targets, list of ints
        """
        self.p_root = p_root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if self.split not in self.SPLIT_FILES:
            raise Exception("split: '{}' does not exist!".format(self.split))
        assert self.split in self.SPLIT_FILES

        self.p_file = Path(self.p_root) / self.DATA_DIR / \
            self.SPLIT_FILES[self.split]
        if not self.p_file.exists():
            raise Exception("File: '{}' does not exists!".format(self.p_file))

        self.data, self.targets = torch.load(self.p_file)

        # add channel dimension to be compatible
        self.data.unsqueeze_(dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
            It may be necessary to convert the img Tensor
            to a PIL.Image to be consistent with 
            different implementations, but here we
            try to work without that first
        """
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
