import sys
sys.path.append("./../")
sys.path.append(".")

from pathlib import Path
import math
import pickle
import pprint
import time
import datetime
#
import torch
import torchvision
from torchvision import utils
import torchvision.transforms as T
import torchvision.datasets as datasets
#
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm
from dotted_dict import DottedDict
#
from misc.plot_utils import plot_mat, imshow
from effcn.functions import max_norm_masking
from effcn.models import MnistEcnBackbone, MnistEcnDecoder, MnistEffCapsNet
from misc.utils import mkdir_directories

def affine_xtrans(img, target, range=[-5.,5.,1]):
    arange = np.arange(range[0],(range[1]+range[2]),range[2])
    x_trans = torch.zeros([len(arange),img.shape[-3],img.shape[-2],img.shape[-1]])
    l_target = torch.zeros(len(arange))

    for i, l in enumerate(arange):
        x_trans[i] = T.functional.affine(img=img, angle=0, translate=[l,0], scale=1.,shear=0)
        l_target[i] = target
    
    return x_trans, l_target

def affine_ytrans(img, target, range=[-5.,5.,1]):
    arange = np.arange(range[0],(range[1]+range[2]),range[2])
    y_trans = torch.zeros([len(arange),img.shape[-3],img.shape[-2],img.shape[-1]])
    l_target = torch.zeros(len(arange))

    for i, l in enumerate(arange):
        y_trans[i] = T.functional.affine(img=img, angle=0, translate=[0,l], scale=1.,shear=0)
        l_target[i] = target
    
    return y_trans, l_target

def affine_rot(img, target, range=[-25.,25.,1]):
    arange = np.arange(range[0],(range[1]+range[2]),range[2])
    rot = torch.zeros([len(arange),img.shape[-3],img.shape[-2],img.shape[-1]])
    l_target = torch.zeros(len(arange))

    for i, l in enumerate(arange):
        rot[i] = T.functional.affine(img=img, angle=l, translate=[0,0], scale=1.,shear=0)
        l_target[i] = target
    
    return rot, l_target

def affine_scale(img, target, range=[0.75,1.25,0.05]):
    arange = np.arange(range[0],(range[1]+range[2]),range[2])
    scale = torch.zeros([len(arange),img.shape[-3],img.shape[-2],img.shape[-1]])
    l_target = torch.zeros(len(arange))

    for i, l in enumerate(arange):
        scale[i] = T.functional.affine(img=img, angle=0, translate=[0,0], scale=l,shear=0)
        l_target[i] = target
    
    return scale, l_target

def affine_shear(img, target, range=[-10.,10.,2]):
    arange = np.arange(range[0],(range[1]+range[2]),range[2])
    shear = torch.zeros([len(arange),img.shape[-3],img.shape[-2],img.shape[-1]])
    l_target = torch.zeros(len(arange))

    for i, l in enumerate(arange):
        shear[i] = T.functional.affine(img=img, angle=0, translate=[0,0], scale=1,shear=l)
        l_target[i] = target
    
    return shear, l_target   

def cov_uh_trans(uh):
    """
    uh in [k,n,m]
    k -> number of transformed images
    n -> number of output classes
    m -> number of capsul values
    """

    uh_mean = uh.mean(dim=0)
    z = uh - uh_mean
    c_k = torch.einsum('...ij, ...ik -> ...jk', z,z)
    c = torch.einsum('ijk -> jk', c_k) / c_k.shape[0]
    return c

def sem_comp(conf):
    
    config = DottedDict(conf)

    p_experiment = Path(config.path.p_experiment)
    p_ckpts = p_experiment / config.path.p_ckpts
    p_model = p_ckpts / config.path.p_model
    p_data = Path(config.path.p_data)
    p_semcomp = Path(config.path.p_semcomp)
    p_stats = p_semcomp / config.path.p_stats
    
    #proof Path
    for p in [p_model, p_data, p_semcomp]:
        if not p.exists():
            print("Path does not exist: " + str(p))
            exit()
    
    #create Path for semantic compareness
    mkdir_directories([p_stats], parents=True, exist_ok=False)

    #save config
    file1 = open(p_stats /'config.pkl', 'wb')
    pickle.dump(conf, file1)
    file1.close()

    #Model to device
    device = torch.device("cuda")
    model = MnistEffCapsNet()
    model.load_state_dict(torch.load(p_model))
    model = model.to(device)
    model.eval()

    print(model)

    #load dataset
    ds_train = datasets.MNIST(root=p_data, train=config.ds.train, download=True, transform=T.ToTensor())

    if config.ds.batch_size == None:
        config.ds.batch_size = len(ds_train)

    dl_train = torch.utils.data.DataLoader(ds_train,
                                            batch_size=config.ds.batch_size, 
                                            shuffle=False)

    #prepare affine trans funcs
    affines = []
    if  config.affine.xtrans == True:
        affines.append([affine_xtrans,config.affine.x_range])
    if  config.affine.ytrans == True:
        affines.append([affine_ytrans,config.affine.y_range])
    if  config.affine.rot == True:
        affines.append([affine_rot,config.affine.rot_range])
    if  config.affine.scale == True:
        affines.append([affine_scale,config.affine.sc_range])
    if  config.affine.shear == True:
        affines.append([affine_shear,config.affine.sh_range])

    #run for each affine trans
    for affine in affines:
        pca_eig = []
        kl_div = []
        name = str(affine[0].__name__)
        desc = name + " [{:3}/{:3}]:".format(affines.index(affine)+1, len(affines))
        pbar = tqdm(dl_train, bar_format= desc + '{bar:10}{r_bar}{bar:-10b}')

        #load batchwise
        for x, y in pbar:
            #calculate staistical vals
            for i, img in enumerate(x):
                #generate aff transforms
                x_aff, y_aff = affine[0](x[i],y[i],affine[1])
                
                #Generate Caps from affine Transform
                x_aff = x_aff.to(device)
                uh_aff, _ = model.forward(x_aff)

                #PCA
                #Covariance from Caps
                cov_uh = cov_uh_trans(uh_aff)
                #Eigenvals
                eig, v_eig = torch.linalg.eig(cov_uh)
                sig = eig.float() / eig.float().sum()
                #PCA eigenvalues
                pca_eig.append(sig.tolist())


                #KL-Divergence
                #Caps from valid
                uh_aff_th = uh_aff[:,y[i],:]
                #Variance over each dimension
                var_uh_aff = torch.var(uh_aff_th, dim=0)
                #Variance normalized
                nor_uh_aff = var_uh_aff / var_uh_aff.sum()
                #uniform prior
                uni_p = 1/nor_uh_aff.shape[0]
                #Kullback-Leibler-Divergenz
                kl = (nor_uh_aff * torch.log((nor_uh_aff/uni_p))).sum()
                kl_div.append(kl.tolist())


        pca_mean = torch.tensor(pca_eig).mean(dim=0)
        kld_mean = torch.tensor(kl_div).mean()    

        #saving stats
        stats = {
            "model": str(p_model),
            "dataset": str(p_data),
            "train": config.ds.train,
            "affine": name,
            "pca": {
                'pca_eig': pca_eig,
                'pca_mean': pca_mean,
            },
            "kld": {
                'kld_val': kl_div,
                'kld_mean': kld_mean,
            }
        }    
        #save stats
        file1 = open(p_stats /('stats_' + name + '.pkl'), 'wb')
        pickle.dump(stats, file1)
        file1.close()

        #save overview plot
        plt.plot(pca_mean.detach().numpy(),".")
        plt.tick_params(colors="k")
        plt.title(name + ", kld_mean: " + str(round(kld_mean.item(),4)),color="k")
        plt.savefig(p_stats /('eigenvals_mean_' + name + '.png'))
        plt.close() 



if __name__ == "__main__":
    config = {
        "path" : {
            "p_experiment": "/mnt/data/experiments/EfficientCN/mnist",
            "p_ckpts": "run_2022-01-19_14-50-25",
            "p_model": "ecn_mnist_epoch_150.ckpt",
            "p_data" : "/mnt/data/datasets",
            "p_semcomp" : "/mnt/data/experiments/EfficientCN/sem_comp",
            "p_stats" : 'semcomp_mnist_{da}'.format(da=datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S'))
        },
        "ds" : {
            "train" : True,
            "batch_size" : 64,
        },
        "affine" : {
            "xtrans": True,
            "ytrans": True,
            "rot"   : True,
            "scale"   : True,
            "shear"   : True,
            "x_range"  : [-5.,5.,1],
            "y_range"  : [-5.,5.,1],
            "rot_range": [-25.,25.,1],
            "sc_range"  : [0.75,1.25,0.05],
            "sh_range"  : [-10.,10.,2],
        }
    }
    
    
    sem_comp(config)