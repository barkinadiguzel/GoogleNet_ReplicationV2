import torch
import torch.nn as nn
from layers.conv_layer import conv_1x1, conv_3x3, conv_asym, conv_5x5_fact

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, out_5x5, out_pool):
        super().__init__()
        # 1x1 branch
        self.branch1 = conv_1x1(in_channels, out_1x1)
        
        # 3x3 branch (optionally factorized in v2)
        self.branch2 = conv_3x3(in_channels, out_3x3)

        # 5x5 branch (factorized as 3x3 + 3x3)
        self.branch3 = conv_5x5_fact(in_channels, out_5x5)

        # Pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_1x1(in_channels, out_pool)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)
