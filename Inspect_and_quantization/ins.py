import torch
import sys
import os
from pytorch_nndct.apis import Inspector
from model.yolov4 import *


target = "DPUCZDX8G_ISA1_B4096"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = torch.load("yolov6m.pth", map_location="cpu")

# Random Input
random_input = torch.randn(1, 3, 416, 416)

# inspection
inspector = Inspector(target)
inspector.inspect(model, random_input, device, output_dir="inspect", image_format="svg")
