# EfficientCN
Pytorch Implementation for [Efficient-CapsNet](https://arxiv.org/abs/2101.12491).
### Install:
```sh
conda env create -f environment.yml
```
### Todo
- check hyperparameter
- preprocessing!
- check calculations!
- check attention scores
- add SmallNORBS example
- add MultiMNIST example

### Features
- batch vs epoch statistics


### Questions

#### MNIST
- what is my best baseline?
- how do attention scores look like?
- how do attention scores vary over training process?
- training without augementation?
- how to speedup training?
- can i get the same results as in the paper?


### Results MNIST

#### ECN Implementation

ACC=0.xxx, LR=5e-4,     BS=16
ACC=0.995, LR=5e-4 * 8, BS=256  data/ckpts/run_2021-11-27_23-23-16

#### EfficientCapsNet Paper
The paper reports a mean performance of 0.9974 acc for single model predictions.

#### Baseline CNN

The baseline CNN uses 28938 trainable parameters and after training yields an acc of 0.993 on the test with 82 misclassified samples using the following setting:
- bs: 256
- optimizer: Adam
- learning rate: 0.01
- exponential decay scheduler with 0.96