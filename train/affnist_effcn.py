import time
import numpy as np
from affnist_effcn_train import train
import pickle
from dotted_dict import DottedDict
from misc.utils import get_sting_timestamp
import subprocess

DEVICE = "cuda:0"
P_EXPERIMENT = '/mnt/experiments/effcn/affnist/grid_search'

if DEVICE == "cuda:0":
    bss = [64, 32]
    lrs = [0.0001, 0.001, 0.01]
    rec_weights = [10, 1, 0.1, 0.01, 0.001, 0]
    weight_decays = [1e-5, 1e-4, 1e-3, 0]
elif DEVICE == "cuda:1":
    bss = [1024, 512, 256, 128]
    lrs = [0.0001, 0.001, 0.01]
    rec_weights = [0.01, 0]
    weight_decays = [1e-5, 1e-4, 1e-3, 0]
else:
    raise Exception("Bla")

lrs = [0.0001, 0.001, 0.01]
rec_weights = [10, 1, 0.1, 0.001]
weight_decays = [1e-5, 1e-4, 1e-3, 0]


all_stats = []

n_runs = len(bss) * len(lrs) * len(rec_weights) * len(weight_decays)
run_idx = 1
for weight_decay in weight_decays:
    for bs in bss:
        for lr in lrs:
            for rec_weight in rec_weights:
                print("Run [{}/{}]".format(run_idx, n_runs))
                run_idx += 1
                try:
                    subprocess.call(
                        "python affnist_effcn_train.py --lr {} --bs {} --num_epochs 150 --weight_decay {} --loss_weight_rec {} --device {} --p_experiment {}".format(
                            lr, bs, weight_decay, rec_weight, DEVICE, P_EXPERIMENT
                        ),
                        shell=True)
                    all_stats.append(("success", None))
                except Exception as e:
                    all_stats.append(("failed", e))

for idx, stat in enumerate(all_stats):
    print(idx, stat[0], stat[1])
