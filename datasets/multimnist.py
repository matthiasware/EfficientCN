import sys
sys.path.append("./..")
sys.path.append(".")

import os

from pathlib import Path
import pprint
import hashlib
import pickle

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from dotted_dict import DottedDict

#local imports
from misc.utils import mkdir_directories


#def mkdir_directories(dirs, parents, exist_ok):
#    for director in dirs:
#        Path(director).mkdir(parents=parents, exist_ok=exist_ok)


class MultiMNist(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, generate=False, g_samples=[1000,1000]):
        
        #paths
        self.p_root = Path(root)
        if train==True:
            self.p_data  = self.p_root / "Train"
        else:
            self.p_data  = self.p_root / "Valid"

        #transforms
        self.transform = transform
        self.target_transform = target_transform
        
        #generation
        if generate == True:
            self.__generator__(g_samples=g_samples)
        
        #path exist check
        if not self.__check_exists__():
            raise RuntimeError('Dataset on choosen file not found or corrupted.' +
                               ' You can use generate=True to generate from MNist to choosen file')


        #load mnist dataset
        if train==True:
            self.mnist_ds = datasets.MNIST(root=self.p_root, train=True, download=False, transform=T.ToTensor())
        else:
            self.mnist_ds = self.mnist_valid = datasets.MNIST(root=self.p_root, train=False, download=False, transform=T.ToTensor())
        self.labels = self.mnist_ds.targets
        #load target files
        self.mnist_idx_1, self.mnist_idx_2, self.padding_1, self.padding_2 = self.__load_meta__()

    def __getitem__(self, mmn_idx):
        x, xy, xz = self.__mm_clamp__(mmn_idx)
        y = self.labels[self.mnist_idx_1[mmn_idx]]
        z = self.labels[self.mnist_idx_2[mmn_idx]]
        
        if self.transform is not None:
            x = self.transform(x)
            xy = self.transform(xy)
            xz = self.transform(xz)

        if self.target_transform is not None:
            y = self.target_transform(y)
            z = self.target_transform(z)
        
        return x, y, z, xy, xz
    
    def __len__(self):
        return len(self.mnist_idx_1)

    def __check_exists__(self):
        """ Check if processed files exists."""
        files = (
            self.p_root,
            self.p_data
        )
        fpaths = [os.path.exists(f) for f in files]
        return False not in fpaths

    def __load_meta__(self):
        #load meta profil for mmnist
        file1 = open(self.p_data / 'idx_1.pkl', 'rb')
        id_1 = pickle.load(file1)
        file1.close()

        file2 = open(self.p_data / 'idx_2.pkl', 'rb')
        id_2 = pickle.load(file2)
        file2.close()  

        file3 = open(self.p_data / 'padding_1.pkl', 'rb')
        padding_1 = pickle.load(file3)
        file3.close()

        file4 = open(self.p_data / 'padding_2.pkl', 'rb')
        padding_2 = pickle.load(file4)
        file4.close()

        return id_1, id_2, padding_1, padding_2 

    def __mm_clamp__(self, mmn_idx):
        base  = T.Pad(padding=self.padding_1[mmn_idx])(self.mnist_ds[self.mnist_idx_1[mmn_idx]][0])
        top   = T.Pad(padding=self.padding_2[mmn_idx])(self.mnist_ds[self.mnist_idx_2[mmn_idx]][0])
        merge = torch.clamp(base + top, min=0, max=1)
        return merge, base, top

    def __generator__(self,g_samples=[1000,1000]):
        #mnist
        ds_train = datasets.MNIST(root=self.p_root, train=True, download=True, transform=T.ToTensor())
        ds_valid = datasets.MNIST(root=self.p_root, train=False, download=True, transform=T.ToTensor())

        print('generate multimnist training data...')
        self. __multimatch__(p_data=self.p_root / "Train", data = ds_train,n=g_samples[0])
        print('generate multimnist valid data...')
        self. __multimatch__(p_data=self.p_root / "Valid", data = ds_valid,n=g_samples[1])
        print('generation done')

    def __pad_rand_l__(self, m,n):
        #generate random pos for padding
        ref = 8
        left = np.random.randint(1, high=9, size=(m*n), dtype=int)
        rigth = ref - left
        up = np.random.randint(1, high=9, size=(m*n), dtype=int)
        down = ref - up

        l_stack = np.stack((left,up, rigth, down), axis=1).tolist()
        
        return l_stack

    def __multimatch__(self, p_data, data, n=1000):
        
        #labels
        labels=data.targets

        # proof path
        mkdir_directories([self.p_root, p_data], parents=True, exist_ok=True)

        #generate index list for mmnist
        all_ref = []
        all_top = []

        #generate top label idx for each class
        all_label = np.unique(labels)
        top_idx = []
        for k in all_label:
            top_idx.append(np.where(labels != k)[0])


        #generate merging labels
        for j, label in enumerate(labels):
            list_top = np.random.choice(top_idx[label.item()],n,replace=False).tolist()
            list_ref = np.full(n,j).tolist()
            all_ref.extend(list_ref)
            all_top.extend(list_top)

        #create target 1 id's
        file1 = open(p_data /'idx_1.pkl', 'wb')
        pickle.dump(all_ref, file1)
        file1.close()

        #create target 2
        file2 = open(p_data /'idx_2.pkl', 'wb')
        pickle.dump(all_top, file2)
        file2.close()

        #create rand padding pos
        padding1 = self.__pad_rand_l__(len(labels),n)
        padding2 = self.__pad_rand_l__(len(labels),n)

        #create padding 1 
        file3 = open(p_data /'padding_1.pkl', 'wb')
        pickle.dump(padding1, file3)
        file3.close()

        #create padding 2 
        file4 = open(p_data /'padding_2.pkl', 'wb')
        pickle.dump(padding2, file4)
        file4.close()



if __name__ == '__main__':
    print('example')

    A = MultiMNist(root='/mnt/data/datasets/multimnist_test',train=True, generate=True, g_samples=[1,1])

    print(A)
    print(A[0]) 





