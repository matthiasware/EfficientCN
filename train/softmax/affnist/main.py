import sys
sys.path.append("./../../..")

# standard lib
import shutil
from pathlib import Path

# external imports
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import scipy as sp
import pandas as pd

# local imports
from datasets import AffNIST
from effcn.layers import Squash
from effcn.functions import margin_loss, max_norm_masking
from misc.utils import count_parameters
from misc.plot_utils import plot_couplings, plot_capsules, plot_mat, plot_mat2
from misc.metrics import *
#
from dotted_dict import DottedDict
import pickle

from tmpmodels import *


def main(config):
    device = torch.device(config.device)

    """
        DATA
    """

    transform_train = T.Compose([
        T.RandomAffine(degrees=(-8, 8),
                       shear=(-15, 15),
                       scale=(0.9, 1.1)
                       ),
        T.Normalize((0.0641,), (0.2257))
    ])
    transform_valid = T.Normalize((0.0641,), (0.2257))

    ds_mnist_train = AffNIST(p_root=config.p_data, split="mnist_train",
                             download=True, transform=transform_train,
                             target_transform=None)
    ds_mnist_valid = AffNIST(p_root=config.p_data, split="mnist_valid",
                             download=True, transform=transform_valid,
                             target_transform=None)
    ds_affnist_valid = AffNIST(p_root=config.p_data, split="affnist_valid",
                               download=True, transform=transform_valid,
                               target_transform=None)

    dl_mnist_train = torch.utils.data.DataLoader(
        ds_mnist_train,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4)
    dl_mnist_valid = torch.utils.data.DataLoader(
        ds_mnist_valid,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4)
    dl_affnist_valid = torch.utils.data.DataLoader(
        ds_affnist_valid,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4)

    ns = config.ns
    ds = config.ds
    #
    attention_scaling = config.attention_scaling
    #
    model = DeepCapsNet(ns=ns, ds=ds, attention_scaling=attention_scaling)

    print("tot Model ", count_parameters(model))
    print("Backbone  ", count_parameters(model.backbone))
    #
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=0.96)

    RES = {
        "acc": {
            "epoch": [],
            "valid": []
        },
    }

    num_epochs = config.num_epochs
    for epoch_idx in range(num_epochs):
        # ####################
        # TRAIN
        # ####################
        model.train()
        desc = "Train [{:3}/{:3}]:".format(epoch_idx, num_epochs)
        pbar = tqdm(dl_mnist_train, bar_format=desc +
                    '{bar:10}{r_bar}{bar:-10b}')

        for x, y_true in pbar:
            x = x.to(device)
            y_true = y_true.to(device)
            optimizer.zero_grad()

            u_h = model.forward(x)

            # LOSS
            y_one_hot = F.one_hot(y_true, num_classes=10)
            loss = margin_loss(u_h, y_one_hot)

            loss.backward()

            optimizer.step()

            y_pred = torch.argmax(torch.norm(u_h, dim=2), dim=1)
            acc = (y_true == y_pred).sum() / y_true.shape[0]

            pbar.set_postfix(
                {'loss': loss.item(),
                 'acc': acc.item()
                 }
            )
        lr_scheduler.step()
        #
        # ####################
        # VALID
        # ####################
        if epoch_idx % config.eval_freq != 0:
            continue

        model.eval()

        total_correct = 0
        total = 0

        for x, y_true in dl_mnist_valid:
            x = x.to(device)
            y_true = y_true.to(device)

            with torch.no_grad():
                u_h = model.forward(x)

                y_pred = torch.argmax(torch.norm(u_h, dim=2), dim=1)
                total_correct += (y_true == y_pred).sum()
                total += y_true.shape[0]
        print("   mnist acc_valid: {:.3f}".format(total_correct / total))

        model.eval()

        total_correct = 0
        total = 0

        for x, y_true in dl_affnist_valid:
            x = x.to(device)
            y_true = y_true.to(device)

            with torch.no_grad():
                u_h = model.forward(x)

                y_pred = torch.argmax(torch.norm(u_h, dim=2), dim=1)
                total_correct += (y_true == y_pred).sum()
                total += y_true.shape[0]
        print("   affnist acc_valid: {:.3f}".format(total_correct / total))
        RES["acc"]["epoch"].append(epoch_idx)
        RES["acc"]["valid"].append((total_correct / total).item())

    """
        EVAL
    """

    model.eval()

    YY = []
    CC = [[] for _ in range(len(ns) - 1)]
    US = [[] for _ in range(len(ns))]

    # use whole dataset
    for x, y_true in dl_affnist_valid:
        x = x.to(device)
        with torch.no_grad():
            _, cc, us = model.forward_debug(x.to(device))
            for idx in range(len(cc)):
                CC[idx].append(cc[idx].detach().cpu().numpy())
            for idx in range(len(us)):
                US[idx].append(us[idx].detach().cpu().numpy())
            YY.append(y_true.numpy())
    # Dataset Labels
    YY = np.concatenate(YY)

    # Dataset Coupling Coefficient Matrices
    CC = [np.concatenate(c) for c in CC]

    # Dataset Capsules
    US = [np.concatenate(u) for u in US]

    """
        PARSE TREE
    """
    fig, axes = plt.subplots(1, 6, figsize=(24, 6))

    # Mean parse tree
    cc_mean = [np.mean(c, axis=0) for c in CC]
    cc_std = [np.std(c, axis=0) for c in CC]
    cc_max = [np.max(c, axis=0) for c in CC]
    #
    plot_couplings(cc_mean, ax=axes[0], show=False, title="mean couplings")
    plot_couplings(cc_std, ax=axes[1], show=False, title="std couplings")
    plot_couplings(cc_max, ax=axes[2], show=False, title="max couplings")

    # mean and std capsule activation
    us_mean = [np.linalg.norm(u, axis=-1).mean(axis=0) for u in US]
    us_std = [np.linalg.norm(u, axis=-1).std(axis=0) for u in US]
    us_max = [np.linalg.norm(u, axis=-1).max(axis=0) for u in US]
    #
    plot_capsules(us_mean, scale_factor=1,
                  ax=axes[3], show=False, title="mean activation")
    plot_capsules(us_std, scale_factor=1,
                  ax=axes[4], show=False, title="std activation")
    plot_capsules(us_max, scale_factor=1,
                  ax=axes[5], show=False, title="max activation")
    plt.suptitle("dataset")
    plt.savefig(config.p_tree)
    plt.close()

    """
        NORMALIZED TREE
    """
    fig, axes = plt.subplots(1, 3, figsize=(4 * len(CC), 4))

    CNS = [normalize_couplings(C) for C in CC]

    CNS_MAN = [ma_couplings(C, pr) for C, pr in CNS]
    CNS_MAX = [C.max(axis=0) for C, pr in CNS]
    CNS_STD = [stda_couplings(C, pr) for C, pr in CNS]

    plot_couplings(CNS_MAN, ax=axes[0], show=False, title="mean")
    plot_couplings(CNS_STD, ax=axes[1], show=False, title="std")
    plot_couplings(CNS_MAX, ax=axes[2], show=False, title="max")
    plt.savefig(config.p_norm_tree)
    plt.close()


    RES["norm_caps"] = {
        "mu": [],
        "sd": [],
        "max": [],
        "dead": []
    }
    for idx in range(len(US)):
        U = US[idx]
        U_norm = np.linalg.norm(U, axis=2)
        U_norm_mu = U_norm.mean(axis=0)
        U_norm_sd = U_norm.std(axis=0)
        U_norm_max = U_norm.max(axis=0)
        #
        U_dead = (U_norm_sd < 1e-2) * (U_norm_mu < 1e-2)
        #
        RES["norm_caps"]["mu"].append(U_norm_mu)
        RES["norm_caps"]["sd"].append(U_norm_sd)
        RES["norm_caps"]["max"].append(U_norm_max)
        RES["norm_caps"]["dead"].append(U_dead)

    """
        VIBRANCE
    """
    RES["dead"] = []
    for U in US:
        pr = rate_dead_capsules_norm(U)
        RES["dead"].append(pr.mean())
        print("#Permanently Dead: {:.3f}".format(pr.mean()))


    # sanity check
    RES["rnd"] = []
    RES["rac"] = []
    RES["racnd"] = []
    for idx in range(len(CC)):
        C = CC[idx]
        U = US[idx]
        #
        rnd, rac, racnd = get_vibrance(U, C)
        RES["rnd"].append(rnd)
        RES["rac"].append(rac)
        RES["racnd"].append(racnd)
        #
        print("rate alive: {:.3f} rate active {:.3f} rate active of alive {:.3f}".format(
            rnd, rac, racnd))



    """
        BONDING
    """
    RES["bonding"] = []
    for idx in range(len(CC)):
        C = CC[idx]
        b = get_bonding(C)
        RES["bonding"].append(b)
        print_str = "bonding strength: {:.3f}"
        print(print_str.format(b))


    """
        DYNAMICS
    """
    RES["dynamics"] = []
    for idx in range(len(CC)):
        C = CC[idx]
        dyc = get_dynamics(C)
        RES["dynamics"].append(dyc)
        #
        print("dynamics: {:.3f}".format(
              dyc))

    with open(config.p_res_file, "wb") as file:
        pickle.dump(RES, file)

    return RES


if __name__ == "__main__":
    N_REPS = 10
    # AS = [3, 3.5, 2, 2.5 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    AS = np.logspace(-3, 1, 9)
    p_r = "results2.pkl"

    results = {}
    for attention_scaling in AS:
        results[attention_scaling] = {}
        for rep in range(N_REPS):
            print("[{:3d}/{:3d}] - {:.3f}".format(rep + 1, N_REPS, attention_scaling))
            config = {
                "device": "cuda:1",
                "p_data": '/home/matthias/projects/EfficientCN/data',
                "p_results": "./results2/rep_{}".format(rep),
                "batch_size": 512,
                "attention_scaling": attention_scaling,
                "ns": [32, 32, 32, 10],
                "ds": [8, 8, 8, 16],
                "num_epochs": 101,
                "eval_freq": 1,
            }
            config = DottedDict(config)
            #
            Path(config.p_results).mkdir(exist_ok=True, parents=True)
            config["p_tree"] = Path(config.p_results) / \
                "tree_{:.3f}_{}.png".format(config.attention_scaling, rep)
            config["p_norm_tree"] = Path(
                config.p_results) / "norm_tree_{:.3f}.png".format(config.attention_scaling)
            config["p_res_file"] = Path(config.p_results) / "results_{}.pkl".format(config.attention_scaling)

            try:
                R = main(config)
            except Exception as e:
                R = str(e)
                print("\n\n EXCEPTION \n {} \n".format(R))
                print(e)
            results[attention_scaling][rep] = R
            with open(p_r, "wb") as file:
                pickle.dump(results, file)
