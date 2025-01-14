import torch
from torch.nn import Sequential, Conv2d, LeakyReLU, ConvTranspose2d, GELU, PReLU
from collections import OrderedDict
import torch.nn.functional as F

import torch.nn as nn

class QRCNN(nn.Module):
    def __init__(self, final_activation='tanh', sr=4):
        super(QRCNN, self).__init__()
        G_layers = []
        G_layers.append(nn.Sequential(nn.Conv2d(1,512,3,1,1), nn.LeakyReLU(), nn.PixelShuffle(upscale_factor=2)))
        if sr == 4:
            G_layers.append(nn.Sequential(nn.Conv2d(128,256,3,1,1), nn.LeakyReLU(), nn.PixelShuffle(upscale_factor=2)))
        elif sr==2:
            G_layers.append(nn.Sequential(nn.Conv2d(128,64,3,1,1), nn.LeakyReLU()))
        else:
            print('Super resolution factor not supported')
        if final_activation == 'tanh':
            G_layers.append(nn.Sequential(nn.Conv2d(64,1,9,1,4), nn.Tanh()))
        elif final_activation == 'sigmoid':
            G_layers.append(nn.Sequential(nn.Conv2d(64,1,9,1,4), nn.Sigmoid()))
        else:
            G_layers.append(nn.Sequential(nn.Conv2d(64,1,9,1,4)))
        self.G_model = nn.Sequential(*G_layers)
    def forward(self, x):
        return self.G_model(x)

class QRGAN_Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(QRGAN_Discriminator, self).__init__()
        in_channels, in_height, in_width = input_shape
        self.output_shape = (1, int(in_height/2**4), int(in_width/2**4))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels,128,3,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(128,1,9,8,4))
    def forward(self, y):
        y = self.conv1(y)
        D_out  = self.conv2(y)
        return D_out