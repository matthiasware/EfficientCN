"""
Splits
- train:         MNIST train images, randomlay placed on 40x40
- valid:         MNIST valid images, randomly placed on 40x40
- train_affnist: AffNIST train images
- valid_affnist: AffNIST test images
"""

import torchvision.datasets as datasets

P_DATA = "./../data"
ds_train = datasets.MNIST(root=P_DATA, train=True,
                          download=True, transform=None)

# dl_train = torch.utils.data.DataLoader(ds_train,
#                                     batch_size=BATCH_SIZE,
#                                     shuffle=True,
#                                     pin_memory=True,
#                                     num_workers=NUM_WORKERS)
# dl_valid = torch.utils.data.DataLoader(ds_valid,
#                                     batch_size=BATCH_SIZE,
#                                     shuffle=True,
#                                     pin_memory=True,
#                                     num_workers=NUM_WORKERS)
