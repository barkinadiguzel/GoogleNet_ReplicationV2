import torch
import torch.nn as nn
from layers.conv_layer import conv_1x1

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.aux = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            conv_1x1(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 768),  # assuming feature map reduces to 4x4
            nn.ReLU(inplace=True),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        return self.aux(x)
