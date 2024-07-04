import torch
import torch.nn as nn
from .common import ConvBNReLU, make_divisible
from .model_cfg import head_cfg as cfg


class DetectionHead(nn.Module):
    def __init__(self, num_classes=20, num_heads=3):
        super().__init__()
        self.nc = num_classes
        self.nl = num_heads
        channels = cfg["channels"]
        width_mul = cfg["width_mul"]
        channels = [make_divisible(i * width_mul, 8) for i in channels]

        stride = [32, 16, 8] if num_heads == 3 else None
        self.stride = torch.tensor(stride)

        # stem 0, for (13, 13)
        self.stem0 = ConvBNReLU(channels[0], channels[0], 1)
        self.cls_conv0 = ConvBNReLU(channels[0], channels[0], 3)
        self.reg_conv0 = ConvBNReLU(channels[0], channels[0], 3)
        self.cls_pred0 = ConvBNReLU(channels[0], self.nc, 1)
        self.reg_pred0 = ConvBNReLU(channels[0], 4 + 1, 1)

        # stem1 for (26, 26)
        self.stem1 = ConvBNReLU(channels[1], channels[1], 1)
        self.cls_conv1 = ConvBNReLU(channels[1], channels[1], 3)
        self.reg_conv1 = ConvBNReLU(channels[1], channels[1], 3)
        self.cls_pred1 = ConvBNReLU(channels[1], self.nc, 1)
        self.reg_pred1 = ConvBNReLU(channels[1], 4 + 1, 1)

        # stem1 for (52, 52)
        self.stem2 = ConvBNReLU(channels[2], channels[2], 1)
        self.cls_conv2 = ConvBNReLU(channels[2], channels[2], 3)
        self.reg_conv2 = ConvBNReLU(channels[2], channels[2], 3)
        self.cls_pred2 = ConvBNReLU(channels[2], self.nc, 1)
        self.reg_pred2 = ConvBNReLU(channels[2], 4 + 1, 1)

    def forward(self, x):
        (x2_o, x1_o, x0_o) = x

        x0 = self.stem0(x0_o)
        cls_pred0 = self.cls_pred0(self.cls_conv0(x0))
        reg_pred0 = self.reg_pred0(self.reg_conv0(x0))

        x1 = self.stem1(x1_o)
        cls_pred1 = self.cls_pred1(self.cls_conv1(x1))
        reg_pred1 = self.reg_pred1(self.reg_conv1(x1))

        x2 = self.stem2(x2_o)
        cls_pred2 = self.cls_pred2(self.cls_conv2(x2))
        reg_pred2 = self.reg_pred2(self.reg_conv2(x2))

        out0 = torch.cat([cls_pred0, reg_pred0], dim=1)
        out1 = torch.cat([cls_pred1, reg_pred1], dim=1)
        out2 = torch.cat([cls_pred2, reg_pred2], dim=1)

        return (
            out0.permute(0, 2, 3, 1),
            out1.permute(0, 2, 3, 1),
            out2.permute(0, 2, 3, 1),
        )
