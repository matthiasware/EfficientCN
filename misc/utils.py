import time
import datetime
from pathlib import Path
import torch
from torchvision import transforms
import numpy as np


def get_sting_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')


def mkdir_directories(dirs, parents, exist_ok):
    for director in dirs:
        Path(director).mkdir(parents=parents, exist_ok=exist_ok)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_transform(means, stds):
    return transforms.Normalize(
        mean=means,
        std=stds)


def inverse_normalize_transform(means, stds):
    return transforms.Normalize(
        mean=-1 * np.array(means) / np.array(stds),
        std=1 / np.array(stds))


def calc_receptive_field(Layer):
    """
    input: L -> Layerwise Kernelsize and Stride
    input example: L = [[kernel1, stride1],[kernel2, stride2], ...]
    output: r -> receptive field 1D
    """
    Layer = np.array(Layer)
    r = 1
    for i, la in enumerate(Layer):
        r += (la[0] - 1)*np.prod(Layer[0:i+1,1])
    return r

def layer_conv(bb_fc_cls):
    """
    input: bb_fc_cls -> Class with Backbone and Conv Layers
    input example: bb_fc_cls =         
                    self.layers = nn.Sequential(
                        nn.Conv2d(1, 256, kernel_size=9, stride=1), \n
                        nn.ReLU(inplace=True), \n
                        nn.BatchNorm2d(256), \n
                        nn.Conv2d(256, 32*8, kernel_size=9, stride=2, groups=1),
        )
    output: L -> Layerwise Kernelsize and Stride
    """

    Lh = []
    Lw = []
    for layer in bb_fc_cls.layers:
        if type(layer) == torch.nn.modules.conv.Conv2d:
            Lh.append([layer.kernel_size[0], layer.stride[0]])
            Lw.append([layer.kernel_size[1], layer.stride[1]])
    return Lh, Lw

def calc_field_delta(x_shape, rec_fld):
    """
    input:  x_shape  -> shape of the input tensor
            rec_fld  -> receptive field from primecap neurons
    output: delta_rf -> delta of receptive field and image size 
    """
    delta_h = rec_fld[0] - x_shape[2] 
    delta_w = rec_fld[1] - x_shape[3] 
    delta_rf = (delta_h, delta_w)
    return delta_rf

def calc_caps_nr(img_shape, caps_dims=None):
    """
    input: img_shape -> shape of the PC Conv tensor
           caps_dims -> list of caps dims 

    output: caps_shapes -> Number and Dimenson of PrimeCaps
    """
    if caps_dims == None:
        caps_dims = 1
    
    param_all = img_shape[1] * img_shape[2] * img_shape[3]
    caps_dims = np.array(caps_dims)
    caps_nr = param_all / caps_dims
    caps_shapes = np.array([caps_nr, caps_dims]).transpose().tolist()

    #with batch size
    #bs = np.zeros_like(caps_dims)
    #bs.fill(a[0])
    #caps_shapes = np.array([bs, caps_nr, caps_dims]).transpose().tolist()

    return caps_shapes

def bb_pc_vals(model, x, caps_dims=None, print_vals=False):
    """
    input: model      -> Class with Backbone and Conv Layers
           x        -> Tensor in shape of input Image
           caps_dims  -> list of caps dims
           print_vals -> pretty print of the output
    input example: model =         
                    self.layers = nn.Sequential(
                        nn.Conv2d(1, 256, kernel_size=9, stride=1), \n
                        nn.ReLU(inplace=True), \n
                        nn.BatchNorm2d(256), \n
                        nn.Conv2d(256, 32*8, kernel_size=9, stride=2, groups=1)
                        )
                   x: torch.tensor(1,1,28,28)
    output: params -> Nr. of Model Parameter
            res.shape   -> output shape of img
            rec_field   -> area of the receptive field
            caps_shapes -> Number and Dimenson of PrimeCaps
    """

    res = model(x)
    layer_vals_h, layer_vals_w = layer_conv(model)
    rec_field = (calc_receptive_field(layer_vals_h), calc_receptive_field(layer_vals_w))
    field_delta = calc_field_delta(x_shape=x.shape, rec_fld=rec_field)
    caps_shapes = calc_caps_nr(img_shape=res.shape, caps_dims=caps_dims)
    params = count_parameters(model)

    if print_vals == True:
        print("nr. of parmeters:    {}".format(params))
        print("image output shape:  {}".format(res.shape))
        print("layer values 'h':    {}".format(layer_vals_h))
        print("layer values 'w':    {}".format(layer_vals_w))
        print("receptive field:     {}".format(rec_field))
        print("delta of fields:     {}".format(field_delta))
        print("shape of primecaps:  {}".format(caps_shapes))    

    return params, res.shape, rec_field, field_delta, caps_shapes