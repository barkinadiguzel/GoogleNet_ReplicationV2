import torch
import torch.nn as nn
from src.layers.conv_layer import ConvLayer
from src.layers.inception_module import InceptionModule
from src.layers.flatten_layer import FlattenLayer
from src.layers.fc_layer import FCLayer
from src.layers.pool_layers.maxpool_layer import MaxPoolLayer
from src.layers.pool_layers.avgpool_layer import AvgPoolLayer
from src.blocks.auxiliary_classifier import AuxiliaryClassifier

class InceptionV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV2, self).__init__()

        # Initial convolution and pooling layers
        self.stem = nn.Sequential(
            ConvLayer(3, 32, 3, stride=2),  # 299x299 -> 149x149
            ConvLayer(32, 32, 3),
            ConvLayer(32, 64, 3, padding=1),
            MaxPoolLayer(kernel_size=3, stride=2),  # 149x149 -> 73x73
            ConvLayer(64, 80, 1),
            ConvLayer(80, 192, 3),
            MaxPoolLayer(kernel_size=3, stride=2)   # 71x71 -> 35x35
        )

        # Inception modules
        self.inception3a = InceptionModule(192, [64, 128, 32, 32])
        self.inception3b = InceptionModule(256, [128, 192, 64, 64])
        self.maxpool3 = MaxPoolLayer(kernel_size=3, stride=2)  # Grid reduction

        self.inception4a = InceptionModule(384, [192, 208, 48, 64])
        self.aux_classifier = AuxiliaryClassifier(512, num_classes)
        self.inception4b = InceptionModule(512, [160, 224, 64, 64])
        self.inception4c = InceptionModule(512, [128, 256, 64, 64])
        self.inception4d = InceptionModule(512, [112, 288, 64, 64])
        self.inception4e = InceptionModule(528, [256, 320, 128, 128])
        self.maxpool4 = MaxPoolLayer(kernel_size=3, stride=2)  # Grid reduction

        self.inception5a = InceptionModule(832, [256, 320, 128, 128])
        self.inception5b = InceptionModule(832, [384, 384, 128, 128])

        # Final layers
        self.avgpool = AvgPoolLayer(kernel_size=8)  
        self.flatten = FlattenLayer()
        self.fc = FCLayer(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        aux = self.aux_classifier(x) 
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x, aux  # The auxiliary output is used only during training.
