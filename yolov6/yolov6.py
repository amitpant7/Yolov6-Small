import torch.nn as nn
from .backbone import EfficientRep
from .neck import RepBiPAN
from .head import DetectionHead


class YOLOv6s(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.backbone = EfficientRep()
        self.neck = RepBiPAN()
        self.head = DetectionHead(num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x
