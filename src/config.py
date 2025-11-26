# General model parameters
INPUT_SIZE = (3, 299, 299)  # Channels x Height x Width
NUM_CLASSES = 1000           # ILSVRC 2012

# Convolutional layer parameters
CONV_PARAMS = {
    'conv1': {'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 0},
    'conv2': {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
    'conv3': {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
}

# Inception module filter sizes
INCEPTION_FILTERS = {
    '35x35': [64, 64, 64, 64],   # Example filter sizes for 35x35 modules
    '17x17': [128, 128, 128, 128],
    '8x8':   [256, 256, 256, 256],
}

# Auxiliary classifier parameters
AUX_PARAMS = {
    'fc1_out': 1024,
    'dropout': 0.7
}
