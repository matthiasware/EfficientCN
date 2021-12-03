from pathlib import Path
import hashlib

# external libraries
import torch
from torch.utils.data import Dataset
from PIL import Image

# local
from .download_utils import download_file_from_google_drive


class AffNIST(Dataset):
    """
        CHECK NOTEBOOK TO CREATE THE DATASET FROM SCRATCH
        IF DOWNLOAD DOES NOT WORK
    """

    DATA_DIR = "AFFNIST"
    SPLIT_FILES = {
        'mnist_train': 'mnist_train.pt',
        'mnist_valid': 'mnist_valid.pt',
        'affnist_train': 'affnist_train.pt',
        'affnist_valid': 'affnist_valid.pt'
    }
    FILE_LINKS = {
        "mnist_train": 'https://drive.google.com/file/d/1ugK4SBt-GNjET3fTrozV3UEuZ_tC_yjJ/view?usp=sharing',
        "mnist_valid": 'https://drive.google.com/file/d/1Puc7tQgBJ9Vl_2Zjbdp6EUXs6T6Jz_7I/view?usp=sharing',
        "affnist_train": 'https://drive.google.com/file/d/1EilN9NiDh_rcL-SjDYbM4oEVxIXBCc6t/view?usp=sharing',
        "affnist_valid": 'https://drive.google.com/file/d/1BNIvJBRdwDQdzhznv9A8P8hL0iIKDNUF/view?usp=sharing'
    }
    FILE_LINK_ID = {
        "mnist_train": '1ugK4SBt-GNjET3fTrozV3UEuZ_tC_yjJ',
        "mnist_valid": '1Puc7tQgBJ9Vl_2Zjbdp6EUXs6T6Jz_7I',
        "affnist_train": '1EilN9NiDh_rcL-SjDYbM4oEVxIXBCc6t',
        "affnist_valid": '1BNIvJBRdwDQdzhznv9A8P8hL0iIKDNUF'
    }
    FILE_CHECKSUMS = {
        "mnist_train": '5646df9ccec5ec450919a2e2db453c96',
        "mnist_valid": '05fb76ad01cc2c1fd5cc22926b95c901',
        "affnist_train": '2849b72ca42795ef2ee99094b9255b9a',
        "affnist_valid": '297649ec1df45bea6920ff24818ef607'
    }

    def __init__(self, p_root: str, split: str = 'mnist_train', transform=None, target_transform=None, download=True):
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

        if download is True:
            self.download_files()

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
        img, target = self.data[idx], self.targets[idx]

        # could be doing this so that it is consistent with all other datasets
        # to return a PIL Image

        # img = Image.fromarray(img.squeeze().numpy(), mode="L")


        # this should be much faster than the PIL version
        img = img.to(dtype=torch.float32).div(255)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def download_files(self):
        for split in self.SPLIT_FILES.keys():
            # check if alread exists on drive
            p_file = Path(self.p_root) / self.DATA_DIR / \
                self.SPLIT_FILES[split]
            if p_file.exists():
                continue

            print("Downloading {} from file {}".format(
                split, self.FILE_LINKS[split]))
            # create data directory
            p_file.parents[0].mkdir(exist_ok=True, parents=True)

            # download file
            file_id = self.FILE_LINK_ID[split]
            download_file_from_google_drive(file_id, p_file)

            # checksum
            checksum_exp = self.FILE_CHECKSUMS[split]

            checksum_act = hashlib.md5(open(p_file, 'rb').read()).hexdigest()

            if not checksum_act == checksum_exp:
                raise Exception(
                    "Checksums do not match for split {}!".format(split))
