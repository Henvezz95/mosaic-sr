import torch
import numpy as np
from torch import nn


class edgeSR_MAX(nn.Module):
    def __init__(self, model_id):
        self.model_id = model_id
        super().__init__()

        assert self.model_id.startswith('eSR-MAX_')

        parse = self.model_id.split('_')

        self.channels = int([s for s in parse if s.startswith('C')][0][1:])
        self.kernel_size = (int([s for s in parse if s.startswith('K')][0][1:]), ) * 2
        self.stride = (int([s for s in parse if s.startswith('s')][0][1:]), ) * 2

        self.pixel_shuffle = nn.PixelShuffle(self.stride[0])
        self.filter = nn.Conv2d(
            in_channels=1,
            out_channels=self.stride[0]*self.stride[1]*self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(
                (self.kernel_size[0]-1)//2,
                (self.kernel_size[1]-1)//2
            ),
            groups=1,
            bias=False,
            dilation=1
        )
        nn.init.xavier_normal_(self.filter.weight, gain=1.)
        self.filter.weight.data[:, 0, self.kernel_size[0]//2, self.kernel_size[0]//2] = 1.

    def forward(self, input):
        return self.pixel_shuffle(self.filter(input)).max(dim=1, keepdim=True)[0]


class edgeSR_TM(nn.Module):
    def __init__(self, model_id):
        self.model_id = model_id
        super().__init__()

        assert self.model_id.startswith('eSR-TM_')

        parse = self.model_id.split('_')

        self.channels = int([s for s in parse if s.startswith('C')][0][1:])
        self.kernel_size = (int([s for s in parse if s.startswith('K')][0][1:]), ) * 2
        self.stride = (int([s for s in parse if s.startswith('s')][0][1:]), ) * 2

        self.pixel_shuffle = nn.PixelShuffle(self.stride[0])
        self.softmax = nn.Softmax(dim=1)
        self.filter = nn.Conv2d(
            in_channels=1,
            out_channels=2*self.stride[0]*self.stride[1]*self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(
                (self.kernel_size[0]-1)//2,
                (self.kernel_size[1]-1)//2
            ),
            groups=1,
            bias=False,
            dilation=1
        )
        nn.init.xavier_normal_(self.filter.weight, gain=1.)
        self.filter.weight.data[:, 0, self.kernel_size[0]//2, self.kernel_size[0]//2] = 1.

    def forward(self, input):
        filtered = self.pixel_shuffle(self.filter(input))

        value, key = torch.split(filtered, [self.channels, self.channels], dim=1)
        return torch.sum(
            value * self.softmax(key),
            dim=1, keepdim=True
        )


class edgeSR_TR(nn.Module):
    def __init__(self, model_id):
        self.model_id = model_id
        super().__init__()

        assert self.model_id.startswith('eSR-TR_')

        parse = self.model_id.split('_')

        self.channels = int([s for s in parse if s.startswith('C')][0][1:])
        self.kernel_size = (int([s for s in parse if s.startswith('K')][0][1:]), ) * 2
        self.stride = (int([s for s in parse if s.startswith('s')][0][1:]), ) * 2

        self.pixel_shuffle = nn.PixelShuffle(self.stride[0])
        self.softmax = nn.Softmax(dim=1)
        self.filter = nn.Conv2d(
            in_channels=1,
            out_channels=3*self.stride[0]*self.stride[1]*self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(
                (self.kernel_size[0]-1)//2,
                (self.kernel_size[1]-1)//2
            ),
            groups=1,
            bias=False,
            dilation=1
        )
        nn.init.xavier_normal_(self.filter.weight, gain=1.)
        self.filter.weight.data[:, 0, self.kernel_size[0]//2, self.kernel_size[0]//2] = 1.

    def forward(self, input):
        filtered = self.pixel_shuffle(self.filter(input))

        value, query, key = torch.split(filtered, [self.channels, self.channels, self.channels], dim=1)
        return torch.sum(
            value * self.softmax(query*key),
            dim=1, keepdim=True
        )


class edgeSR_CNN(nn.Module):
    def __init__(self, model_id):
        self.model_id = model_id
        super().__init__()

        assert self.model_id.startswith('eSR-CNN_')

        parse = self.model_id.split('_')

        self.channels = int([s for s in parse if s.startswith('C')][0][1:])
        self.stride = (int([s for s in parse if s.startswith('s')][0][1:]), ) * 2
        D = int([s for s in parse if s.startswith('D')][0][1:])
        S = int([s for s in parse if s.startswith('S')][0][1:])
        assert S>0 and D>=0

        self.softmax = nn.Softmax(dim=1)
        if D == 0:
            self.filter = nn.Sequential(
                nn.Conv2d(D, S, (3, 3), (1, 1), (1, 1)),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=S,
                    out_channels=2*self.stride[0]*self.stride[1]*self.channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    groups=1,
                    bias=False,
                    dilation=1
                ),
                nn.PixelShuffle(self.stride[0]),
            )
        else:
            self.filter = nn.Sequential(
                nn.Conv2d(1, D, (5, 5), (1, 1), (2, 2)),
                nn.Tanh(),
                nn.Conv2d(D, S, (3, 3), (1, 1), (1, 1)),
                nn.Tanh(),
                nn.Conv2d(
                    in_channels=S,
                    out_channels=2*self.stride[0]*self.stride[1]*self.channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    groups=1,
                    bias=False,
                    dilation=1
                ),
                nn.PixelShuffle(self.stride[0]),
            )

        if D == 0:
            nn.init.xavier_normal_(self.filter[0].weight, gain=1.)
            nn.init.xavier_normal_(self.filter[2].weight, gain=1.)
            self.filter[0].weight.data[:, 0, 1, 1] = 1.
            self.filter[2].weight.data[:, 0, 1, 1] = 1.
        else:
            nn.init.xavier_normal_(self.filter[0].weight, gain=1.)
            nn.init.xavier_normal_(self.filter[2].weight, gain=1.)
            nn.init.xavier_normal_(self.filter[4].weight, gain=1.)
            self.filter[0].weight.data[:, 0, 2, 2] = 1.
            self.filter[2].weight.data[:, 0, 1, 1] = 1.
            self.filter[4].weight.data[:, 0, 1, 1] = 1.

    def forward(self, input):
        filtered = self.filter(input)

        value, key = torch.split(filtered, [self.channels, self.channels], dim=1)
        return torch.sum(
            value * self.softmax(key*key),
            dim=1, keepdim=True
        )