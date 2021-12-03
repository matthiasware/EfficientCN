import sys
sys.path.append("./..")

# standard lib
import shutil
from pathlib import Path
import pickle
import math
import time

# external imports
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from dotted_dict import DottedDict
import pprint
from tqdm import tqdm

# local imports
from datasets import AffNIST
from effcn.models import AffnistEffCapsNet
from effcn.layers import PrimaryCaps, FCCaps
from effcn.functions import create_margin_loss, max_norm_masking, margin_loss
from effcn.utils import count_parameters
from misc.optimizer import get_optimizer, get_scheduler
from misc.utils import get_sting_timestamp, mkdir_directories

# will most likely result in a 30% speed up
torch.backends.cudnn.benchmark = True

# These may improve persormance around 5%-10%
#torch.autograd.set_detect_anomaly(False)
#torch.autograd.profiler.profile(False)
#torch.autograd.profiler.emit_nvtx(False)


def eval_model(model, device, data_loader, config, func_margin, func_rec):
    model.eval()
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    for x, y_true in data_loader:
        x = x.to(device)
        y_true = y_true.to(device)

        with torch.no_grad():
            u_h, x_rec = model.forward(x)

            # LOSS
            y_one_hot = F.one_hot(y_true, num_classes=10)
            loss_margin = func_margin(u_h, y_one_hot)
            loss_rec = func_rec(x, x_rec)

            # total loss
            loss = (loss_margin * config.loss.margin.weight) + \
                (loss_rec * config.loss.rec.weight)

            # validate batch
            y_pred = torch.argmax(torch.norm(u_h, dim=2), dim=1)

            epoch_loss += loss.item()
            epoch_correct += (y_true == y_pred).sum().item()
            epoch_total += x.shape[0]
    epoch_acc = epoch_correct / epoch_total
    return epoch_loss, epoch_acc


def create_reconstruction_grid_img(model, device, x, permute=False):
    model.eval()
    with torch.no_grad():
        _, x_rec = model.forward(x.to(device))
        x_rec = x_rec.cpu()
    img = torchvision.utils.make_grid(
        torch.cat([x, x_rec], dim=0), nrow=x.shape[0])
    if permute:
        img = img.permute(1, 2, 0)
    return img


def plot_acc_from_stats(stats, p_file):
    train_max = max(stats["train"]["acc"])
    valid_mnist_max = max(stats["valid"]["mnist"]["acc"])
    valid_affnist_max = max(stats["valid"]["affnist"]["acc"])

    plt.figure(figsize=(10, 10))
    plt.plot(stats["train"]["epoch"], stats["train"]["acc"],
             label="train {:.5f}".format(train_max), color='b')
    plt.plot(stats["valid"]["mnist"]["epoch"], stats["valid"]["mnist"]["acc"],
             label="valid mnist {:.5f}".format(valid_mnist_max), color='red')
    plt.plot(stats["valid"]["affnist"]["epoch"], stats["valid"]["affnist"]["acc"],
             label="valid affnist {:.5f}".format(valid_affnist_max), color='orange')
    #
    plt.axhline(y=train_max, color='b', linestyle='dotted')
    plt.axhline(y=valid_mnist_max, color='red', linestyle='dotted')
    plt.axhline(y=valid_affnist_max, color='orange', linestyle='dotted')
    plt.title("ACC")
    plt.legend()
    plt.tight_layout()

    plt.savefig(p_file)
    plt.close()


def plot_loss_from_stats(stats, p_file):
    plt.figure(figsize=(10, 10))
    plt.plot(stats["train"]["epoch"], stats["train"]
             ["loss"], label="train", color='b')
    plt.plot(stats["valid"]["mnist"]["epoch"], stats["valid"]
             ["mnist"]["loss"], label="valid mnist", color='red')
    plt.plot(stats["valid"]["affnist"]["epoch"], stats["valid"]
             ["affnist"]["loss"], label="valid affnist", color='orange')
    #
    plt.title("LOSS")
    plt.legend()
    plt.tight_layout()

    plt.savefig(p_file)
    plt.close()


def train(config):
    print("#"* 100)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)

    p_data = config.paths.data
    p_experiment = Path(config.paths.experiments) / config.names.model_dir
    p_ckpts = p_experiment / config.names.ckpt_dir
    p_logs = p_experiment / config.names.log_dir
    p_config = p_experiment / config.names.config_file
    p_stats = p_experiment / config.names.stats_file
    p_imgs = p_experiment / config.names.img_dir
    p_acc_plot = p_experiment / config.names.acc_plot
    p_loss_plot = p_experiment / config.names.loss_plot
    #
    device = torch.device(config.device)

    # transform_train = T.Compose([T.ToTensor()])
    # converts [0,255] to [0,1] by dividing through 255
    # transform_valid = T.Compose([T.ToTensor()])

    ds_mnist_train = AffNIST(p_root=p_data, split="mnist_train",
                             download=True, transform=None, target_transform=None)
    ds_mnist_valid = AffNIST(p_root=p_data, split="mnist_valid",
                             download=True, transform=None, target_transform=None)
    ds_affnist_valid = AffNIST(p_root=p_data, split="affnist_valid",
                               download=True, transform=None, target_transform=None)

    dl_mnist_train = torch.utils.data.DataLoader(
        ds_mnist_train,
        batch_size=config.train.batch_size,
        shuffle=True,
        # prefetch_factor=3,
        persistent_workers=True,
        pin_memory=config.train.pin_memory,
        num_workers=config.train.num_workers)
    dl_mnist_valid = torch.utils.data.DataLoader(
        ds_mnist_valid,
        batch_size=config.valid.batch_size,
        shuffle=True,
        pin_memory=config.valid.pin_memory,
        persistent_workers=True,
        num_workers=config.valid.num_workers)
    dl_affnist_valid = torch.utils.data.DataLoader(
        ds_affnist_valid,
        batch_size=config.valid.batch_size,
        shuffle=True,
        persistent_workers=True,
        pin_memory=config.valid.pin_memory,
        num_workers=config.valid.num_workers)

    x, _ = next(iter(dl_mnist_train))
    x_vis_train = x[:config.train.num_vis]

    x, _ = next(iter(dl_mnist_valid))
    x_vis_mnist_valid = x[:config.valid.num_vis]

    x, _ = next(iter(dl_affnist_valid))
    x_vis_affnist_valid = x[:config.valid.num_vis]

    model = AffnistEffCapsNet()
    model = model.to(device)

    optimizer = get_optimizer(
        config.optimizer, model.parameters(), config.optimizer_args)
    if config.scheduler is not None:
        scheduler = get_scheduler(
            config.scheduler, optimizer, config.scheduler_args)
    else:
        scheduler = None

    # create directories
    if config.debug:
        # remove dir and recreate it if in debug mode
        if p_experiment.exists():
            shutil.rmtree(p_experiment)
        mkdir_directories([p_experiment, p_ckpts, p_logs,
                           p_imgs], parents=True, exist_ok=True)
    else:
        mkdir_directories([p_experiment, p_ckpts, p_logs,
                           p_imgs], parents=True, exist_ok=False)

    print("p_experiment:  {}".format(p_experiment))
    # summary writer
    sw = SummaryWriter(p_logs)
    print("tensorboard --logdir={}".format(str(p_logs)))
    print("#" * 100)

    # save configs
    with open(p_config, "wb") as file:
        pickle.dump(config, file)

    # custom training stats
    stats = {
        "train": {
            'acc': [],
            'loss': [],
            'epoch': [],
        },
        "valid": {
            'mnist': {
                'acc': [],
                'loss': [],
                'epoch': [],
            },
            'affnist': {
                'acc': [],
                'loss': [],
                'epoch': [],
            }
        },
        "notes": []
    }

    start = time.time()
    stop_run = False  # set if some event occurs

    # LOSS FUNCTIONS [create in advance for speed]
    func_margin_loss = create_margin_loss(
        lbd=config.loss.margin.lbd,
        m_plus=config.loss.margin.m_plus,
        m_minus=config.loss.margin.m_minus
    )
    func_rec_loss = torch.nn.MSELoss()


    for epoch_idx in range(1, config.train.num_epochs + 1, 1):
        ###################
        # TRAIN
        ###################
        model.train()
        desc = "Train [{:3}/{:3}]:".format(epoch_idx, config.train.num_epochs)
        pbar = tqdm(dl_mnist_train, bar_format=desc +
                    '{bar:10}{r_bar}{bar:-10b}')

        epoch_loss = 0
        epoch_correct = 0

        for x,y_true in pbar:
            x = x.to(device)
            y_true = y_true.to(device)

            # way faster than optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None
            u_h, x_rec = model.forward(x)
            # LOSS
            y_one_hot = F.one_hot(y_true, num_classes=10)
            loss_margin = func_margin_loss(u_h, y_one_hot)
            loss_margin = loss_margin * config.loss.margin.weight
            loss_rec = func_rec_loss(x, x_rec)
            loss_rec = loss_rec * config.loss.rec.weight

            # total loss
            loss = loss_margin + loss_rec
            loss.backward()

            optimizer.step()

            # validate batch
            y_pred = torch.argmax(torch.norm(u_h, dim=2), dim=1)

            correct = (y_true == y_pred).sum()
            acc = correct / x.shape[0]

            epoch_correct += correct.item()
            epoch_loss += loss.item()

            pbar.set_postfix(
                    {'loss': loss.item(),
                     'mar': loss_margin.item(),
                     'rec': loss_rec.item(),
                     'acc': acc.item()
                    }
            )
        # TRAIN STAS
        sw.add_scalar("train/loss", epoch_loss, epoch_idx)
        sw.add_scalar("train/acc", epoch_correct /
                      len(ds_mnist_train), epoch_idx)

        stats["train"]["epoch"].append(epoch_idx)
        stats["train"]["acc"].append(epoch_correct / len(ds_mnist_train))
        stats["train"]["loss"].append(epoch_loss)

        if scheduler is not None and (epoch_idx > config.scheduler_burnin):
            scheduler.step()

        if math.isnan(epoch_loss):
            print_str = "Stopping epoch {}: epoch_loss={}".format(
                epoch_idx, epoch_loss)
            print(print_str)
            stats["notes"].append(print_str)
            stop_run = True

        ###################
        # EVAL
        ###################
        model.eval()
        if (epoch_idx % config.freqs.ckpt == 0) or (config.train.num_epochs == epoch_idx):
            p_ckpt = p_ckpts / config.names.model_file.format(epoch_idx)
            torch.save(model.state_dict(), p_ckpt)

        if (epoch_idx % config.freqs.rec == 0) or (config.train.num_epochs == epoch_idx):

            img_train = create_reconstruction_grid_img(
                model, device, x_vis_train)
            img_mnist_valid = create_reconstruction_grid_img(
                model, device, x_vis_mnist_valid)
            img_affnist_valid = create_reconstruction_grid_img(
                model, device, x_vis_affnist_valid)

            #plt.figure(figsize=(10, 10))
            plt.imshow(img_train.permute(1,2,0))
            plt.tight_layout()
            plt.savefig(p_imgs / "img_train_{:03d}.png".format(epoch_idx))
            plt.tight_layout()
            plt.close()

            #plt.figure(figsize=(10, 10))
            plt.imshow(img_mnist_valid.permute(1,2,0))
            plt.savefig(p_imgs / "img_valid_mnist_{:03d}.png".format(epoch_idx))
            plt.close()

            #plt.figure(figsize=(10, 10))
            plt.imshow(img_affnist_valid.permute(1,2,0))
            plt.savefig(p_imgs / "img_valid_affnist_{:03d}.png".format(epoch_idx))
            plt.close()

            sw.add_image("train/rec", img_train, epoch_idx)
            sw.add_image("valid/mnist", img_mnist_valid, epoch_idx)
            sw.add_image("valid/affnist", img_affnist_valid, epoch_idx)

        if (epoch_idx % config.freqs.valid == 0) or (config.train.num_epochs == epoch_idx):
            loss_mnist_valid, acc_mnist_valid = eval_model(
                model, device, dl_mnist_valid, config, func_margin_loss, func_rec_loss)

            sw.add_scalar("valid/mnist/loss", loss_mnist_valid, epoch_idx)
            sw.add_scalar("valid/mnist/acc", acc_mnist_valid, epoch_idx)

            stats["valid"]["mnist"]["epoch"].append(epoch_idx)
            stats["valid"]["mnist"]["acc"].append(acc_mnist_valid)
            stats["valid"]["mnist"]["loss"].append(loss_mnist_valid)

            loss_affnist_valid, acc_affnist_valid = eval_model(
                model, device, dl_affnist_valid, config, func_margin_loss, func_rec_loss)
            sw.add_scalar("valid/affnist/loss", loss_affnist_valid, epoch_idx)
            sw.add_scalar("valid/affnist/acc", acc_affnist_valid, epoch_idx)

            stats["valid"]["affnist"]["epoch"].append(epoch_idx)
            stats["valid"]["affnist"]["acc"].append(acc_affnist_valid)
            stats["valid"]["affnist"]["loss"].append(loss_affnist_valid)

            print_str = "Valid: mnist_loss: {:.5f}, affnist_loss: {:.5f}, mnist_acc: {:.5f} affnist_acc: {:.5f}"
            print(print_str.format(loss_mnist_valid, loss_affnist_valid,
                                   acc_mnist_valid, acc_affnist_valid))

            if acc_mnist_valid >= config.stop_acc:
                print_str = "Stopping epoch {}: acc_valid {:.5f} > {:.5f}".format(
                    epoch_idx, acc_mnist_valid, config.stop_acc)
                print(print_str)
                stats["notes"].append(print_str)
                stop_run = True

        if stop_run:
            break

    end = time.time()
    stats["train_time"] = "{:.1f}".format(end - start)
    print("Training time: {:.1f}".format(end - start))

    with open(p_stats, "wb") as file:
        pickle.dump(p_stats, file)

    sw.close()

    plot_acc_from_stats(stats, p_acc_plot)
    plot_loss_from_stats(stats, p_loss_plot)
    return stats


if __name__ == "__main__":
    config = {
        'device': 'cuda:0',
        'debug': True,
        'train': {
            'batch_size': 512,
            'num_epochs': 150,
            'num_workers': 2,
            'num_vis': 8,
            'pin_memory': True,
        },
        'valid': {
            'num_workers': 2,       # Either set num_worker high or pin_memory=True
            'batch_size': 512,
            'num_vis': 8,
            'pin_memory': True,
        },
        'optimizer': 'adam',
        'optimizer_args': {
            'lr': 0.001
        },
        'scheduler': 'exponential_decay',
        'scheduler_burnin': 10,  # [epochs]
        'scheduler_args': {
            'gamma': 0.96
        },
        'freqs': {
            'valid': 1,   # [epochs]
            'rec': 1,    # [epochs] show reconstructions
            'ckpt': 10,   # [epochs]
        },
        'paths': {
            'data': '/home/matthias/projects/EfficientCN/data',
            'experiments': '/mnt/experiments/effcn/affnist/tmp',
        },
        'names': {
            'model_dir': 'effcn_affnist_{}'.format(get_sting_timestamp()),
            'ckpt_dir': 'ckpts',
            'img_dir': 'imgs',
            'log_dir': 'logs',
            'model_file': 'model_{}.ckpt',
            'stats_file': 'stats.pkl',
            'config_file': 'config.pkl',
            'acc_plot': 'acc.png',
            'loss_plot': 'loss.png',
        },
        'loss': {
            'margin': {
                'lbd': 0.5,
                'm_plus': 0.9,
                'm_minus': 0.1,
                'weight': 1.0
            },
            'rec': {
                'weight': 0.3
            }
        },
        'stop_acc': 0.9922
    }
    config = DottedDict(config)
    train(config)
