from __future__ import absolute_import, division # what does this line mean?

import torch
import torch.nn as nn
import torch.nn.init as init

from torch_deform_conv.layers import ConvOffset2D

class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2)) # why input channel is 1 ???
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
	init.orthogonal(self.conv4.weight)

class NetARCNN(nn.Module):
    def __init__(self):  #deleted upscale?
        super(NetARCNN, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, 9, 1, 4) # does padding 0 means "same"?
        self.conv2 = nn.Conv2d(64, 32, 7, 1, 3)
        self.conv3 = nn.Conv2d(32, 16, 1, 1, 0)
        self.conv4 = nn.Conv2d(16, 1, 5, 1, 2) # why is padding 9 necessary???? 

	self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv4.weight, init.calculate_gain('relu'))

class NetARCNN_deform(nn.Module):
    def __init__(self):  #deleted upscale?
        super(NetARCNN_deform, self).__init__()
         
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 64, 9, 1, 4) # does padding 0 means "same"?
        self.offset1 = ConvOffset2D(64)

        self.conv2 = nn.Conv2d(64, 32, 7, 1, 3)

        self.offset2 = ConvOffset2D(32)          

        self.conv3 = nn.Conv2d(32, 16, 1, 1, 0)

        self.offset3 = ConvOffset2D(16)

        self.conv4 = nn.Conv2d(16, 1, 5, 1, 2) # why is padding 9 necessary???? 

	self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.offset1(x)
        x = self.relu(self.conv2(x))
        x = self.offset2(x)
        x = self.relu(self.conv3(x))
        x = self.offset3(x)
        x = self.relu(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv4.weight, init.calculate_gain('relu'))
