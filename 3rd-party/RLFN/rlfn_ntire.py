# -*- coding: utf-8 -*-
# Copyright 2022 ByteDance
import torch.nn as nn
from RLFN import block


class RLFN_NTIRE(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN in NTIRE 2022 Efficient SR Challenge
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 feature_channels=46,
                 mid_channels=48,
                 upscale=2):
        super(RLFN_NTIRE, self).__init__()

        self.conv_1 = block.conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.block_1 = block.RLFB(feature_channels, mid_channels)
        self.block_2 = block.RLFB(feature_channels, mid_channels)
        self.block_3 = block.RLFB(feature_channels, mid_channels)
        self.block_4 = block.RLFB(feature_channels, mid_channels)

        self.conv_2 = block.conv_layer(feature_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.upsampler = block.pixelshuffle_block(feature_channels,
                                                  out_channels,
                                                  upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)

        out_low_resolution = self.conv_2(out_b4) + out_feature
        output = self.upsampler(out_low_resolution)

        return output
