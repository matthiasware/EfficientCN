# EfficientCN
Pytorch Implementation for [Efficient-CapsNet](https://arxiv.org/abs/2101.12491).

Experimental results can be found in RESULTS.md

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
- reread critique paper and check their results
- neural turing machine paper > memory extension
- controll used hypterparamters 
- use original preprocessing and augmentation
- unittest routing and masking opterations
- check attention scores

#### Feature Request
- different scheduler
- batch vs epoch statistics
- tensorboard support
- animated recs
- visualize filters and capsnets
- save training statistics data
- add SmallNORBS example
- add MultiMNIST example
- add affNIST test set

## Experimental and Conceptual Questions
well we should answer those:
- what is my best baseline?
- how do attention scores look like?
- how do attention scores vary over training process?
- training without augementation?
- how to speedup training?
- can i get the same results as in the paper?


## Notes

Export environments for different platforms via:

```sh
conda env export --no-builds > environment.yml
```

and delete the prefix in the end of the file.