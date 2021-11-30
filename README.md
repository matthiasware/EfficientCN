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

#### Todo queue (Matthias)
- refactor ImbaCaps notes and understand the different approaches and methods
- implement advanced training script which is more customizable
- train on MNIST with original preprocessing and test on affNIST
- train on MNIST without preprocessing and test on affNIST
- train on CIFAR10
- check attention scores

### Todo queue (Marcel)
- add code for SmallNORBS with original parameters
- add code for MultiMNIST with orignal parameters 
- (unit)test attention layer
- (unit)test masking

#### Small Stuff
- neural turing machine paper > memory extension
- controll used hypterparamters 
- unittest routing and masking opterations
- check attention scores
- my implementation of the masking operation might be wrong. i masked out everything but the most significant capsule, but i guess in the original apprach they mask out depending on the ground truth labels and not on the CN output.

#### Feature Request
- different schedulers
- batch vs epoch statistics
- add support for accs on mnist and affnist
- create 32 images from test set and create rec set
- tensorboard support
- animated recs
- visualize filters and capsnets
- save training statistics data alongside model

## Experimental and Conceptual Questions
well we should answer these questions:
- what is my best baseline?
- how do attention scores look like?
- how do attention scores vary over training process?
- training without augementation?
- how to speedup training?
- can i get the same results as in the paper?
- can i get rid of the rec loss and to which degree?
- self supervised loss?
- how to scale the model to more clases withot parameter explosion?

## Notes

Export environments for different platforms via:

```sh
conda env export --no-builds > environment.yml
```

and delete the prefix in the end of the file.