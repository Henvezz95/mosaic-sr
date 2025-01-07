import torch
import torch.nn as nn
import torch.nn.functional as F
from ECBSR.ecb import ECB

class ECBSR_ITER(nn.Module):
    def __init__(self, module_nums, channel_nums, with_idt, act_type, scale, colors, num=32):
        super(ECBSR_ITER, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.with_idt = with_idt
        self.act_type = act_type
        self.backbone = None
        self.tail = []
        self.num = num
        self.upsampler = None

        backbone = []
        tail = []
        backbone += [ECB(self.colors, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        for i in range(self.module_nums):
            backbone += [ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        tail += [nn.Conv2d(self.channel_nums, self.num, 3, 1, padding=1)]
        tail += [nn.LeakyReLU(negative_slope=0.1)]
        tail += [nn.Conv2d(self.num, self.num, 1, 1)]
        tail += [nn.LeakyReLU(negative_slope=0.1)]
        #tail += [nn.Conv2d(self.num, self.num, 1, 1)]
        #tail += [nn.LeakyReLU(negative_slope=0.1)]
        #tail += [nn.Conv2d(self.num, self.num, 1, 1)]
        #tail += [nn.LeakyReLU(negative_slope=0.1)]
        tail += [nn.Conv2d(self.num, self.colors*self.scale*self.scale, 1, 1)]

        self.backbone = nn.Sequential(*backbone)
        self.tail = nn.Sequential(*tail)
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        y = self.backbone(x) + x
        y = self.tail(y)
        y = self.upsampler(y)
        return y
