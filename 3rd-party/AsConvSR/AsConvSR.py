import torch
import torch.nn as nn
import torch.nn.functional as F

from AsConvSR.Assembled_conv import AssembledBlock

# Implementation of AsConvSR 
class AsConvSR(nn.Module):
    def __init__(self, in_ch: int=1, out_ch: int=1, scale_factor: int=2, device=torch.device('cpu')):
        super(AsConvSR, self).__init__()
        self.scale_factor = scale_factor
        
        self.pixelUnShuffle = nn.PixelUnshuffle(2)
        self.conv1 = nn.Conv2d(4*in_ch, 32, kernel_size=3, stride=1, padding=1)
        self.assemble = AssembledBlock(32, 32, E=3, temperature=30, kernel_size=3, stride=1, padding=1, device=device)
        self.conv2 = nn.Conv2d(32, 16*out_ch, kernel_size=3, stride=1, padding=1)
        self.pixelShuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.pixelUnShuffle(x)
        out2 = self.conv1(out1)
        out3 = self.assemble(out2)
        out4 = self.conv2(out3)
        out5 = self.pixelShuffle(out4)
        
        x = torch.cat((x, x, x, x), dim=1)
        out6 = torch.add(out5, x)
        out7 = self.pixelShuffle(out6)
        
        return out7