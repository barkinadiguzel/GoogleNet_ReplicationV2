import torch
import torch.nn as nn

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

# 3x3 -> Asymmetric factorization: 3x3 = 3x1 + 1x3
class AsymmetricConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=False),
            nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

# 5x5 -> Factorized: 5x5 = 3x3 + 3x3
class Factorized5x5(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

# Factory functions 

def conv_1x1(in_c, out_c):
    return ConvBNReLU(in_c, out_c, 1)

def conv_3x3(in_c, out_c):
    return ConvBNReLU(in_c, out_c, 3, padding=1)

def conv_asym(in_c, out_c):
    return AsymmetricConv(in_c, out_c)

def conv_5x5_fact(in_c, out_c):
    return Factorized5x5(in_c, out_c)
