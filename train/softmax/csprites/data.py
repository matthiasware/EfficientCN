import sys
sys.path.append("./../../..")

from pathlib import Path
import pickle
#
import torchvision.transforms as T
#
from misc.utils import normalize_transform, inverse_normalize_transform
from datasets.csprites import ClassificationDataset
from torch.utils.data import DataLoader


def get_dataloaders(p_data, num_workers, batch_size):
    p_ds_config = Path(p_data) / "config.pkl"
    with open(p_ds_config, "rb") as file:
        ds_config = pickle.load(file)
    target_variable = "shape"
    target_idx = [idx for idx, target in enumerate(
        ds_config["classes"]) if target == target_variable][0]
    n_classes = ds_config["n_classes"][target_variable]
    #
    norm_transform = normalize_transform(ds_config["means"],
                                         ds_config["stds"])
    #
    def target_transform(x): return x[target_idx]
    transform = T.Compose(
        [T.ToTensor(),
         norm_transform,
         ])
    inverse_norm_transform = inverse_normalize_transform(
        ds_config["means"],
        ds_config["stds"]
    )
    ds_train = ClassificationDataset(
        p_data=p_data,
        transform=transform,
        target_transform=target_transform,
        split="train"
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    # VALID
    ds_valid = ClassificationDataset(
        p_data=p_data,
        transform=transform,
        target_transform=target_transform,
        split="valid"
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    #
    ds_concepts = ClassificationDataset(
        p_data=p_data,
        transform=transform,
        target_transform=None,
        split="valid"
    )
    dl_concepts = DataLoader(
        ds_concepts,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    return dl_train, dl_valid, dl_concepts
