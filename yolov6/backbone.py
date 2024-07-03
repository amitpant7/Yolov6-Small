import torch.nn as nn
from .common import RepBlock, RepConv, make_divisible
from .model_cfg import backbone_cfg as config


class EfficientRep(nn.Module):
    """EfficientRep Backbone"""

    def __init__(self, config=config, in_channels=3):
        super().__init__()
        self.config = config
        self.num_blocks = len(self.config["num_repeats"])
        channels_list = [
            make_divisible(i * self.config["width_mul"], 8)
            for i in self.config["out_channels"]
        ]

        self.out_pos = [1, 2, 3, 4]
        num_repeat = [
            (max(round(i * config["depth_mul"]), 1) if i > 1 else i)
            for i in (self.config["num_repeats"])
        ]

        self.repconv1 = RepConv(in_channels, 32, 3, stride=1)
        self.layers = nn.ModuleList()
        in_channels = 32

        for i in range(self.num_blocks):
            self.layers += [
                nn.Sequential(
                    RepConv(in_channels, channels_list[i], 3, stride=2),
                    RepBlock(channels_list[i], num_repeat[i]),
                )
            ]

            in_channels = channels_list[i]

    def forward(self, x):
        x = self.repconv1(x)
        out = []

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i in self.out_pos:
                out.append(x)

        return out
