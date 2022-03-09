import sys
sys.path.append("./../../..")

from pathlib import Path
import pickle
#
from dotted_dict import DottedDict
import torchvision.transforms as T
import torchvision
import numpy as np
#
from misc.utils import normalize_transform, inverse_normalize_transform
from datasets.csprites import ClassificationDataset
from torch.utils.data import DataLoader
from data import get_dataloaders


def main(config):
    device = torch.device(config.device)
    dl_train, dl_valid, dl_concepts = get_dataloaders(
        config.p_data, config.num_workers, config.batch_size)

    n_vis = 64
    x, y = next(iter(dl_train))
    x = x[:n_vis]
    y = y[:n_vis]
    #
    grid_img = torchvision.utils.make_grid(x, nrow=int(np.sqrt(n_vis)))
    plt.imshow(grid_img.permute(1, 2, 0))


if __name__ == "__main__":
    config = {
        'p_data': '/home/matthias/projects/data/single_csprites_32x32_n7_c24_a12_p6_s2_bg_1_constant_color_145152',
        'p_root': "/mnt/experiments/capsnet/softmax/csprites",
        'num_workers': 4,
        'batch_size': 512,
        'device': 'cuda:0'

    }
    config = DottedDict(config)
    main(config)
