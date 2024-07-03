import torch
import torch.nn as nn
from .common import RepBlock, ConvBNReLU, BiFusion, make_divisible
from .model_cfg import neck_cfg as cfg


# https://user-images.githubusercontent.com/26833433/240750557-3e9ec4f0-0598-49a8-83ea-f33c91eb6d68.png
class RepBiPAN(nn.Module):
    def __init__(self):
        super().__init__()
        # [64, 128, 256, 512, 1024]
        # [256, 128, 128, 256, 256, 512]

        # Todo, Make global

        channels = cfg["channels"]
        num_repeats = cfg["num_repeats"]
        width_mul = cfg["width_mul"]
        depth_mul = cfg["depth_mul"]

        channels = [make_divisible(i * width_mul, 8) for i in channels]

        num_repeats = [
            (max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeats)
        ]

        self.reduce_layer0 = ConvBNReLU(channels[4], channels[5], 1)

        self.bifusion0 = BiFusion([channels[3], channels[2]], channels[5])
        self.rep0 = RepBlock(channels[5], num_repeats[0])
        # done upto p4

        self.reduce_layer1 = ConvBNReLU(channels[5], channels[6], 1)
        self.bifusion1 = BiFusion([channels[2], channels[1]], channels[6])
        self.rep1 = RepBlock(channels[6], num_repeats[1])
        # done upto p3, n3 comes out

        # second column starts
        self.downsamp2 = ConvBNReLU(channels[6], channels[7], kernel_size=3, stride=2)
        self.rep2 = RepBlock(channels[6] + channels[7], num_repeats[2])
        # n4 comes out upto here

        self.downsamp1 = ConvBNReLU(channels[8], channels[9], kernel_size=3, stride=2)
        self.rep3 = RepBlock(channels[5] + channels[9], num_repeats[3])
        # N5 out upto here.

    def forward(self, input):
        (x0, x1, x2, x3) = input

        fpn_out0 = self.reduce_layer0(x3)
        f0_concat = self.bifusion0([fpn_out0, x2, x1])
        p4 = self.rep0(f0_concat)

        p4_reduce = self.reduce_layer1(p4)
        f1_concat = self.bifusion1([p4_reduce, x1, x0])
        p3 = self.rep1(f1_concat)
        n3 = p3

        # second stage
        n3_downsamp = self.downsamp2(n3)
        n4 = self.rep2(torch.cat([n3_downsamp, p4_reduce], dim=1))

        n4_downsamp = self.downsamp1(n4)
        print(n4_downsamp.shape)
        n5 = self.rep3(torch.cat([n4_downsamp, fpn_out0], dim=1))

        return (n3, n4, n5)
