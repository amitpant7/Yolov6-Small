import torch
from torchvision import datasets
from torchvision import tv_tensors
from config import *

# lets write custom tranform to transform the targets in appropriate format.
import torch
from torchvision import datasets
from torchvision import tv_tensors

# lets write custom tranform to transform the targets in appropriate format.


class MyCustomTransformatioms(torch.nn.Module):
    # pase the dictionary format of targets in pascal voc and create bboxes and labels , from it from it
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, img, data):
        img = tv_tensors.Image(img)
        labels = []
        bboxes = []

        class_encoding = CLASS_ENCODING

        annotation = data["annotation"]
        objects = annotation["object"]

        for obj in objects:
            label = obj["name"]
            bbox = obj["bndbox"]
            xmin = int(bbox["xmin"])
            ymin = int(bbox["ymin"])
            xmax = int(bbox["xmax"])
            ymax = int(bbox["ymax"])

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            width = xmax - xmin
            height = ymax - ymin

            labels.append(class_encoding[label])
            bboxes += [[x_center, y_center, width, height]]

        bboxes = tv_tensors.BoundingBoxes(
            bboxes, format="CXCYWH", canvas_size=img.shape[-2:]
        )

        sample = {"image": img, "labels": torch.tensor(labels), "bboxes": bboxes}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class FinalTranform(torch.nn.Module):
    # Retruns target in the shape [S, S, N, C+5] for every Scale,
    # So a tesor represtnation of target for all anchor boxes and all scale values .

    def __init__(self):
        super().__init__()

    def forward(self, sample):
        image = sample["image"]
        bboxes = sample["bboxes"]
        labels = sample["labels"]
        bbox_with_labesl = (bboxes, labels)
        targets = []

        SCALES = [13, 26, 52]
        for scale in SCALES:
            targets.append(self.preprocess_targets(bbox_with_labesl, scale=scale))

        return image, targets

    def preprocess_targets(self, targets, nc=20, scale=26, img_size=416):
        stride = img_size / scale
        bboxes, labels = targets
        cls = torch.zeros(nc, scale, scale)  # class asssigmnetnt
        reg = torch.zeros(5, scale, scale)  # includes centerness too

        for boxes, label in zip(bboxes, labels):
            pos = int(boxes[0] / stride), int(boxes[1] / stride)
            box_pos = int(0.5 * boxes[2] / stride), int(0.5 * boxes[3] / stride)
            cls[
                label,
                pos[0] - box_pos[0] : pos[0] + box_pos[0],
                pos[1] - box_pos[1] : pos[1] + box_pos[1],
            ] = 1

            for i in range(pos[0] - box_pos[0], pos[0] + box_pos[0]):
                for j in range(pos[1] - box_pos[1], pos[1] + box_pos[1]):
                    l = (i + 0.5) * stride - (
                        boxes[0] - int(boxes[2] / 2)
                    )  # top calculation
                    r = boxes[0] + int(boxes[2] / 2) - (i + 0.5) * stride
                    t = (j + 0.5) * stride - (boxes[1] - int(boxes[3] / 2))
                    b = boxes[1] + int(boxes[3] / 2) - (j + 0.5) * stride

                    centerness = torch.sqrt(
                        min(l, r) * min(t, b) / (max(l, r) * max(t, b))
                    )

                    # check for level assign
                    if (
                        (scale == 13 and max(l, r, t, b) > 256)
                        or (scale == 26 and 64 < max(l, r, t, b) <= 256)
                        or (scale == 52 and max(l, r, t, b) <= 64)
                    ):

                        if torch.max(reg[:, i, j]) == 0:  # no box assigned yet
                            reg[:, i, j] = torch.tensor(
                                [
                                    l / stride,
                                    t / stride,
                                    r / stride,
                                    b / stride,
                                    centerness,
                                ]
                            ).clamp_(min=0)

                        elif max(l, r, t, b) < torch.max(
                            reg[:, i, j]
                        ):  # box already assigned but current box is smaller than previously assigned
                            reg[:, i, j] = torch.tensor(
                                [
                                    l / stride,
                                    t / stride,
                                    r / stride,
                                    b / stride,
                                    centerness,
                                ]
                            ).clamp_(min=0)

        return torch.cat([cls, reg], dim=0).permute(2,1,0)
