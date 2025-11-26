import torch
import torch.nn as nn
from layers.conv_layer import conv_1x1, conv_3x3, conv_5x5_fact

class InceptionModule(nn.Module):
    def __init__(self, in_channels, base_out_channels, coarsest_size=8):
        super().__init__()
        self.in_channels = in_channels
        self.base_out_channels = base_out_channels
        self.coarsest_size = coarsest_size

    def forward(self, x):
        _, _, h, w = x.shape

        if min(h, w) == self.coarsest_size:
            # P branch: pooling
            branch_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
            # C branch: conv stride 2
            branch_conv = conv_1x1(self.in_channels, self.base_out_channels[0], stride=2)(x)
            # concat
            x = torch.cat([branch_pool, branch_conv], dim=1)
            return x
            
        branch1 = conv_1x1(self.in_channels, self.base_out_channels[0])(x)
        branch2 = conv_3x3(self.in_channels, self.base_out_channels[1])(x)
        branch3 = conv_5x5_fact(self.in_channels, self.base_out_channels[2])(x)
        branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_1x1(self.in_channels, self.base_out_channels[3])
        )(x)

        return torch.cat([branch1, branch2, branch3, branch4], dim=1)
