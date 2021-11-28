# EfficientCN
Pytorch Implementation for [Efficient-CapsNet](https://arxiv.org/abs/2101.12491).

## Install:

on linux:
```sh
conda env create -f environment-linux.yml --name=effcn
conda activate effcn
```

on windows:
```sh
conda env create -f environment-windows.yml --name=effcn
conda activate effcn
```


## Usage

For training Efficient-CapsNet on MNIST with the original settings from the paper, run:
```sh
python mnist_effcn_train.py
```

For advanced training options use:
```sh
python mnist_effcn_advanced_train.py
```

## TODO

#### Small Stuff
- controll used hypterparamters 
- use original preprocessing and augmentation
- unittest routing and masking opterations
- check attention scores

#### Feature Request
- batch vs epoch statistics
- tensorboard support
- animated recs
- save training statistics data
- add SmallNORBS example
- add MultiMNIST example

## Experimental and Conceptual Questions

well we should answer those:
- what is my best baseline?
- how do attention scores look like?
- how do attention scores vary over training process?
- training without augementation?
- how to speedup training?
- can i get the same results as in the paper?


## MNIST Results

### EfficientCapsNet Paper
The paper reports a mean performance of 0.9974 acc for single model predictions.

### Our Pytorch Implementation

Runs:
- ACC=0.997, LR=5e-4,     BS=16  epchs=150     (data/ckpts/run_2021-11-27_23-49-25)
- ACC=0.996, LR=53-4 * 8, BS=256 epochs=10000  (data/ckpts/run_2021-11-27_23-56-50)

#### Baseline CNN

The baseline CNN uses 28938 trainable parameters and after training yields an acc of 0.993 on the test with 82 misclassified samples using the following setting:
- bs: 256
- optimizer: Adam
- learning rate: 0.01
- exponential decay scheduler with 0.96


## Notes

Export environments for different platforms via:

```sh
conda env export --no-builds > environment.yml
```

and delete the prefix in the end of the file.