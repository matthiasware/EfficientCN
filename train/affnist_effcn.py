from affnist_effcn_train import train
import pickle
from dotted_dict import DottedDict
from misc.utils import get_sting_timestamp

DEVICE = "cuda:1"

if DEVICE == "cuda:0":
    bss = [512, 1024]
    p_results = "results_0.pkl"
elif DEVICE == "cuda:1":
    bss = [64, 128]
    p_results = "results_1.pkl"
else:
    raise Exception("Bla")

lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01]
weights = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]


all_stats = []

n_runs = len(bss) * len(lrs) * len(weights)
run_idx = 1
for bs in bss:
    for lr in lrs:
        for weight in weights:
            print("Run [{}/{}]".format(run_idx, n_runs))
            run_idx += 1
            config = {
                'device': DEVICE,
                'debug': True,
                'train': {
                    'batch_size': bs,
                    'num_epochs': 30,
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
                    'lr': lr
                },
                'scheduler': 'exponential_decay',
                'scheduler_burnin': 10,  # [epochs]
                'scheduler_args': {
                    'gamma': 0.96
                },
                'freqs': {
                    'valid': 1,  # [epochs]
                    'rec': 1,    # [epochs] show reconstructions
                    'ckpt': 30,   # [epochs]
                },
                'paths': {
                    'data': '/home/matthias/projects/EfficientCN/data',
                    'experiments': '/mnt/experiments/effcn/affnist',
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
                        'weight': weight
                    }
                },
                'stop_acc': 0.9922
            }
            config = DottedDict(config)
            # stats = train(config)
            # all_stats.append((bs, lr, weight, config, "succes"))
            try:
                stats = train(config)
                all_stats.append((bs, lr, weight, config, "succes"))
            except Exception as e:
                print(e)
                all_stats.append((bs, lr, weight, config, "failed"))


with open(p_results, "wb") as file:
    pickle.dump(all_stats, file)
