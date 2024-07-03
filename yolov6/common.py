import torch
import torch.nn as nn
import math


class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        if kernel_size == 3:
            self.conv = nn.Conv2d(
                in_channels, out_channels, 3, padding=1, stride=stride
            )

        elif kernel_size == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, 1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=26 / 256)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepVGG(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv3 = RepConv(in_channels, in_channels, 3)
        self.conv1 = RepConv(in_channels, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv3(x)
        x2 = self.conv1(x)
        x3 = self.bn(x)

        y = x1 + x2 + x3
        return y


class RepBlock(nn.Module):
    def __init__(self, in_channels, num_repeats):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_repeats):
            self.layers += [RepVGG(in_channels)]

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x


class ConvBNReLU(nn.Module):
    """Convolutional layer followed by Batch Normalization and LeakyReLU activation"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2  # Default padding to maintain spatial dimensions

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=26 / 256)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)  # Activation is not in-place
        return x


class BiFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = ConvBNReLU(in_channels[0], out_channels, 1, 1)
        self.cv2 = ConvBNReLU(in_channels[1], out_channels, 1, 1)
        self.cv3 = ConvBNReLU(out_channels * 3, out_channels, 1, 1)

        self.upsamp = nn.Upsample(scale_factor=2)
        self.downsamp = ConvBNReLU(in_channels[1], out_channels, 3, stride=2)

    def forward(self, x):
        x0 = self.upsamp(x[0])
        x1 = self.cv1(x[1])
        x2 = self.cv2(x[2])
        x2 = self.downsamp(x2)

        x = torch.cat([x0, x1, x2], dim=1)

        return self.cv3(x)


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor
