import torch
import torch.nn as nn

def weights_init(model):
    classname = model.__class__.__name__
    # checks for the appropriate layers
    #  and applies the criteria explained above
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

UPPER_FAKE = 0.3
def generate_fake_labels(tensor_size):
    return torch.rand(tensor_size) * UPPER_FAKE

UPPER_REAL = 1.2
LOWER_REAL = 0.7
def generate_real_labels(tensor_size):
    return torch.rand(tensor_size) * (UPPER_REAL - LOWER_REAL) + LOWER_REAL