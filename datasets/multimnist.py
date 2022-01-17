import sys
sys.path.append("./..")
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
#from misc.utils import mkdir_directories


def mkdir_directories(dirs, parents, exist_ok):
    for director in dirs:
        Path(director).mkdir(parents=parents, exist_ok=exist_ok)


class MultiMNist(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, generate=False, g_samples=[1000,1000]):
        
        #paths
        self.p_root = Path(root)
        if train==True:
            self.p_data  = self.p_root / "Train"
        else:
            self.p_data  = self.p_root / "Valid"
        self.p_img = self.p_data / "Img"

        #transforms
        self.transform = transform
        self.target_transform = target_transform
        
        #generation
        if generate == True:
            self.generator(n_train=g_samples[0], n_valid=g_samples[1])
        
        #path exist check
        if not self.__check_exists__():
            raise RuntimeError('Dataset on choosen file not found or corrupted.' +
                               ' You can use generate=True to generate from MNist to choosen file')

        #load target files
        file = open(self.p_data / 'targets_1.plk', 'rb')
        self.targets_1 = pickle.load(file)
        file.close()

        file = open(self.p_data / 'targets_2.plk', 'rb')
        self.targets_2 = pickle.load(file)
        file.close()
        
        #load list of imgs
        self.data = os.listdir(self.p_img)

    def __getitem__(self, idx):
        data_loc = os.path.join(self.p_img, self.data[idx])
        x = Image.open(data_loc).convert("L")
        y = self.targets_1[idx]
        z = self.targets_2[idx]
        
        if self.transform is not None:
            x = self.transform(x)
            
        if self.target_transform is not None:
            y = self.target_transform(y)
            z = self.target_transform(z)
        
        return x, y, z
    
    def __len__(self):
        return len(self.data)

    def __check_exists__(self):
        """ Check if processed files exists."""
        files = (
            self.p_root,
            self.p_data,
            self.p_img
        )
        fpaths = [os.path.exists(f) for f in files]
        return False not in fpaths

    def generator(self,n_train=1000, n_valid=1000):
        #mnist
        ds_train = datasets.MNIST(root=self.p_root, train=True, download=True, transform=T.ToTensor())
        ds_valid = datasets.MNIST(root=self.p_root, train=False, download=True, transform=T.ToTensor())

        print('generate multimnist...')
        
        self. __multimatch__(p_data=self.p_root / "Train", images=ds_train.data, labels=ds_train.targets,n=n_train)
        self. __multimatch__(p_data=self.p_root / "Valid", images=ds_valid.data, labels=ds_valid.targets,n=n_valid)

        print('generation done')

    def __pad_rand__(self):
        ref = 8
        left = np.random.randint(1, high=9, size=None, dtype=int)
        rigth = ref - left
        up = np.random.randint(1, high=9, size=None, dtype=int)
        down = ref - up
        
        return [left, up, rigth, down]
    
    def __multimatch__(self, p_data, images, labels, n=1000):
        #paths
        p_imgs = p_data / 'Img'
        
        mkdir_directories([p_data, p_imgs], parents=True, exist_ok=True)

        #lists
        all_targets1 = []
        all_targets2 = []
        
        #generator index
        index = 1
        
        #test dataset
        #test = images[0:10]
        #print(test.size())
        #for j, image in enumerate(test):
        
        #generate for whole dataset
        for j, image in enumerate(images):
        
            #reference img
            img_ref = images[j]
            lab_ref = labels[j]

            #choose random top images from different classes
            top_idx = np.where(labels != lab_ref)[0]
            top_idx = np.random.choice(top_idx,n,replace=False)

            #generate images
            for i, idx in enumerate(top_idx):
                
                #randomize position
                base  = T.Pad(padding=self.__pad_rand__())(images[j])
                top   = T.Pad(padding=self.__pad_rand__())(images[top_idx[i]])
                
                #merge images
                merge = torch.clamp(base + top,min=0, max=1)
                merge = merge.unsqueeze(0)
                
                #add labels to list
                label1 = labels[j]
                label2 = labels[top_idx[i]]
                all_targets1.append(label1)
                all_targets2.append(label2)

                #Save Img as png
                torchvision.utils.save_image(merge.float(), p_imgs / "{:08d}.png".format(index))
                index += 1 
        
        #create target 1
        file_targets1 = open(p_data /'targets_1.plk', 'wb')
        pickle.dump(all_targets1, file_targets1)
        file_targets1.close()
        
        #create target 2
        file_targets2 = open(p_data /'targets_2.plk', 'wb')
        pickle.dump(all_targets2, file_targets2)
        file_targets2.close()



def loadtrageds():
    #load target files
    file = open('/mnt/data/datasets/test/Valid/targets_1.plk', 'rb')
    targets_1 = pickle.load(file)
    file.close()

    file = open('/mnt/data/datasets/test/Valid/targets_2.plk', 'rb')
    targets_2 = pickle.load(file)
    file.close()  

    for i, target in enumerate(targets_1):
    	print(i+1, targets_1[i], targets_2[i])

if __name__ == '__main__':
    print('example')

    A = MultiMNist(root='/mnt/data/datasets/multimnist_test',train=True)#, generate=True, g_samples=[20,10])
    print(A[1])
    B = MultiMNist(root='/mnt/data/datasets/multimnist_test',train=False)#, generate=True, g_samples=[20,10])
    print(B[1])
    #MultiMNist(root='/mnt/data/datasets/multimnist1000',train=True, generate=True, g_samples=[1000,1000])
    #loadtrageds()

    #create target 1
    file_targets1 = open('/mnt/data/datasets/multimnist11/done.plk', 'wb')
    pickle.dump('Done', file_targets1)
    file_targets1.close()




