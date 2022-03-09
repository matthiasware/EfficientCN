import time
import datetime
from pathlib import Path
from torchvision import transforms
import numpy as np


def get_sting_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')


def mkdir_directories(dirs, parents, exist_ok):
    for director in dirs:
        Path(director).mkdir(parents=parents, exist_ok=exist_ok)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_transform(means, stds):
    return transforms.Normalize(
        mean=means,
        std=stds)


def inverse_normalize_transform(means, stds):
    return transforms.Normalize(
        mean=-1 * np.array(means) / np.array(stds),
        std=1 / np.array(stds))