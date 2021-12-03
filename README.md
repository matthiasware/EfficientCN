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
cd train
python mnist_effcn_train.py
```

For advanced training options use:
```sh
cd train
python mnist_effcn_advanced_train.py
```

### Todo queue (Marcel)
- add code for SmallNORBS with original parameters
- add code for MultiMNIST with orignal parameters
- label masking function
- try train with max_norm_masking + label_norm_masking
- research: unit testing
- (unit) test attention layer
- (unit) test masking




## Notes

Export environments for different platforms via:

```sh
conda env export --no-builds > environment.yml
```

and delete the prefix in the end of the file.