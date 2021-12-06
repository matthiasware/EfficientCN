import time
import numpy as np
from affnist_effcn_train import train
import pickle
from dotted_dict import DottedDict
from misc.utils import get_sting_timestamp

DEVICE = "cuda:0"

if DEVICE == "cuda:0":
    bss = [512]
    p_results = "results_cuda0_512.pkl"
elif DEVICE == "cuda:1":
    bss = [32, 64, 128][::-1]
    p_results = "results_cuda1.pkl"
else:
    raise Exception("Bla")

lrs = [0.0001, 0.001, 0.01]
rec_weights = [1, 0.1, 0.01, 0.001]
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
                config = {
                    'device': DEVICE,
                    'debug': True,
                    'train': {
                        'batch_size': bs,
                        'num_epochs': 60,
                        'num_workers': 2,
                        'num_vis': 8,
                        'pin_memory': True,
                    },
                    'valid': {
                        'num_workers': 2,
                        'batch_size': bs,
                        'num_vis': 8,
                        'pin_memory': True,
                    },
                    'optimizer': 'adam',
                    'optimizer_args': {
                        'lr': lr,
                        'weight_decay': weight_decay,
                    },
                    'scheduler': 'exponential_decay',
                    'scheduler_burnin': 10,  # [epochs]
                    'scheduler_args': {
                        'gamma': 0.96
                    },
                    'freqs': {
                        'valid': 1,  # [epochs]
                        'rec': 1,    # [epochs] show reconstructions
                        'ckpt': 10,   # [epochs]
                    },
                    'paths': {
                        'data': '/home/matthias/projects/EfficientCN/data',
                        'experiments': '/mnt/experiments/effcn/affnist',
                    },
                    'names': {
                        'model_dir': 'effcn_affnist_{}_{}'.format(get_sting_timestamp(), hex(np.random.randint(0, 10000))),
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
                            'weight': rec_weight
                        }
                    },
                    'stop_acc': 0.9922
                }
                config = DottedDict(config)
                try:
                    _ = train(config)
                    all_stats.append((config, "success", None))
                except Exception as e:
                    print(e)
                    all_stats.append((config, "failed", str(e)))
                print(config.names.model_dir)
                time.sleep(0.5)


with open(p_results, "wb") as file:
    pickle.dump(all_stats, file)
