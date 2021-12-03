# Experimental Results

## MNIST Results

### Original EfficientCapsNet Paper
The paper reports a mean performance of 0.9974 acc for single model predictions.

### Our Pytorch Implementation

Training time of the model for different batch sizes:
- 16:   43 sec/epoch
- 32:   20 sec/epoch
- 64:   10 sec/epoch
- 128:  5  sec/epoch
- 256:  3  sec/epoch
- 512:  3  sec/epoch
- 1025: 3  sec/epoch
- 2048: 3  sec/epoch
- 4096: 4  sec/epoch
- 8192: 4  sec/epoch


Reported: max(acc) of last 10 epochs on test set, all trained for 150 epochs:

#### bs=16
- ACC=0.9954, LR=5e-4 * 2**-1, BS=16
- ACC=0.9958, LR=5e-4 * 2**0,  BS=16
- ACC=0.9963, LR=5e-4 * 2**1,  BS=16

#### bs=32
- ACC=0.9961, LR=5e-4 * 2**-1, BS=32
- ACC=0.9963, LR=5e-4 * 2**0,  BS=32
- ACC=0.9963, LR=5e-4 * 2**1,  BS=32

#### bs=64
- ACC=0.9950, LR=5e-4 * 2**-1, BS=64
- ACC=0.9962, LR=5e-4 * 2**0,  BS=64
- ACC=0.9964, LR=5e-4 * 2**1,  BS=64

#### bs=128
- ACC=0.9952, LR=5e-4 * 2**-1, BS=128
- ACC=0.9961, LR=5e-4 * 2**0,  BS=128
- ACC=0.9961, LR=5e-4 * 2**1,  BS=128
- ACC=, LR=5e-4 * 2**2,  BS=128

#### bs=256
- ACC=0.99550, LR=5e-4 * 2**0, BS=256
- ACC=0.99600, LR=5e-4 * 2**1, BS=256
- ACC=0.99620, LR=5e-4 * 2**2, BS=256
- ACC=0.99600, LR=5e-4 * 2**4, BS=256
- ACC=0.99590, LR=5e-4 * 2**6, BS=256

#### bs=512
- ACC=0.99530, LR=5e-4 * 2**0, BS=512
- ACC=0.99620, LR=5e-4 * 2**1, BS=512
- ACC=0.99540, LR=5e-4 * 2**2, BS=512
- ACC=0.99650, LR=5e-4 * 2**4, BS=512
- ACC=0.99620, LR=5e-4 * 2**6, BS=512

#### bs=1024
- ACC=0.99440, LR=5e-4 * 2**0, BS=1024
- ACC=0.99460, LR=5e-4 * 2**1, BS=1024
- ACC=0.99480, LR=5e-4 * 2**2, BS=1024
- ACC=0.99550, LR=5e-4 * 2**4, BS=1024
- ACC=0.99602, LR=5e-4 * 2**6, BS=1024

#### bs=2048
- ACC=0.9920, LR=5e-4 * 2**0, BS=2048
- ACC=0.9937, LR=5e-4 * 2**1, BS=2048
- ACC=0.9937, LR=5e-4 * 2**2, BS=2048
- ACC=0.9957, LR=5e-4 * 2**4, BS=2048
- ACC=0.9960, LR=5e-4 * 2**6, BS=2048
- ACC=nannan, LR=5e-4 * 2**8, BS=2048, -> NAN

#### bs=4096
- ACC=0.99040, LR=5e-4 * 2**0, BS=4096
- ACC=0.99050, LR=5e-4 * 2**1, BS=4096
- ACC=0.99399, LR=5e-4 * 2**2, BS=4096
- ACC=0.99480, LR=5e-4 * 2**4, BS=4096
- ACC=0.99480, LR=5e-4 * 2**6, BS=4096

#### bs=8192
- ACC=0.99360, LR=5e-4 * 2**4, BS=8192
- ACC=0.99240, LR=5e-4 * 2**6, BS=8192

#### Baseline CNN
The baseline CNN uses 28938 trainable parameters and after training yields an acc of 0.993 on the test with 82 misclassified samples using the following setting:
- bs: 256
- optimizer: Adam
- learning rate: 0.01
- exponential decay scheduler with 0.96