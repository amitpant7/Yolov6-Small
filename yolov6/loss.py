import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import device


def calculate_iou(box1, box2, mode=None):
    assert box1.shape == box2.shape
    assert box1.dtype == box2.dtype
    assert mode is not None
    epsilon = 1e-7

    if mode == "corner":
        assert torch.all(box1[..., 0] < box1[..., 2]), "box1: x1 must be less than x2"
        assert torch.all(box1[..., 1] < box1[..., 3]), "box1: y1 must be less than y2"
        assert torch.all(box2[..., 0] < box2[..., 2]), "box2: x1 must be less than x2"
        assert torch.all(box2[..., 1] < box2[..., 3]), "box2: y1 must be less than y2"

    if mode == "center":
        x1 = torch.max(
            box1[..., 0:1] - box1[..., 2:3] / 2, box2[..., 0:1] - box2[..., 2:3] / 2
        )
        y1 = torch.max(
            box1[..., 1:2] - box1[..., 3:4] / 2, box2[..., 1:2] - box2[..., 3:4] / 2
        )
        x2 = torch.min(
            box1[..., 0:1] + box1[..., 2:3] / 2, box2[..., 0:1] + box2[..., 2:3] / 2
        )
        y2 = torch.min(
            box1[..., 1:2] + box1[..., 3:4] / 2, box2[..., 1:2] + box2[..., 3:4] / 2
        )
    elif mode == "corner":
        x1 = torch.max(box1[..., 0:1], box2[..., 0:1])
        y1 = torch.max(box1[..., 1:2], box2[..., 1:2])
        x2 = torch.min(box1[..., 2:3], box2[..., 2:3])
        y2 = torch.min(box1[..., 3:4], box2[..., 3:4])
    elif mode == "width_height":
        intersection = torch.min(box1[..., 0], box2[..., 0]) * torch.min(
            box1[..., 1], box2[..., 1]
        )
        union = box1[..., 0] * box1[..., 1] + box2[..., 0] * box2[..., 1] - intersection
        iou = intersection / (union + epsilon)
        return iou
    else:
        raise ValueError("mode should be 'center' or 'corner' or 'width_height'")

    if mode == "corner":
        box1_area = (box1[..., 2:3] - box1[..., 0:1]) * (
            box1[..., 3:4] - box1[..., 1:2]
        )
        box2_area = (box2[..., 2:3] - box2[..., 0:1]) * (
            box2[..., 3:4] - box2[..., 1:2]
        )
    else:  # 'center' mode
        box1_area = box1[..., 2:3] * box1[..., 3:4]
        box2_area = box2[..., 2:3] * box2[..., 3:4]
    # intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    intersection = (x2 - x1) * (y2 - y1)
    union = box1_area + box2_area - intersection
    iou = intersection / (union + epsilon)
    # print("intersection: {:.0f}".format(intersection.sum().item()))
    # print("union: {:.0f}".format((union+epsilon).sum().item()))
    # iou2 =intersection.sum().item()/union.sum().item()
    # print("iou: {:.4f}".format(iou2))
    return iou


class IOULoss(nn.Module):
    def __init__(self, mode=None, theta=4):
        """
        SIOU loss : https://arxiv.org/pdf/2205.12740
        CIOU loss : TODO
        GIOU loss : TODO

        Takes input of the format x1, y1, x2, y2
        :param mode:
        """
        super().__init__()
        self.mode = mode
        self.theta = theta

    def forward(self, pred, target):
        """
        pred, target in xyxy format
        :param pred:[bs,grid_size,grid_size,4]
        :param target: [bs,grid_size,grid_size,4]
        :return:
        """
        assert pred.shape == target.shape
        assert pred.dtype == target.dtype
        assert self.mode is not None
        assert not torch.any(pred < 0)
        assert not torch.any(target < 0)

        epsilon = 1e-10

        pred = convert_to_xy(pred)
        target = convert_to_xy(target)

        iou = calculate_iou(pred, target, mode=self.mode)
        b1_x1, b1_y1, b1_x2, b1_y2 = pred.unbind(dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = target.unbind(dim=-1)

        w1, h1 = b1_x2 - b1_x1 + epsilon, b1_y2 - b1_y1 + epsilon
        w2, h2 = b2_x2 - b2_x1 + epsilon, b2_y2 - b2_y1 + epsilon
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1) + epsilon
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1) + epsilon

        Cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
        Ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
        sigma = torch.sqrt(Cw**2 + Ch**2) + epsilon
        # sin_alpha = torch.clamp(ch/sigma, -1+epsilon, 1-epsilon)
        # angle_cost = 1 - 2 * torch.pow( torch.sin(torch.arcsin(sin_alpha) - np.pi/4), 2)

        sin_alpha_1 = torch.abs(Cw) / sigma
        sin_alpha_2 = torch.abs(Ch) / sigma
        threshold = np.pi / 4
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)

        angle_cost = torch.sin(torch.arcsin(sin_alpha) * 2)
        gamma = 2 - angle_cost
        rho_x = (Cw / cw) ** 2
        rho_y = (Ch / ch) ** 2
        distance_cost = (1 - torch.exp(-gamma * rho_x)) + (
            1 - torch.exp(-gamma * rho_y)
        )

        omega_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omega_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow((1 - torch.exp(-omega_w)), self.theta) + (
            torch.pow((1 - torch.exp(-omega_h)), self.theta)
        )
        iou = iou.squeeze(-1) - (distance_cost + shape_cost) * 0.5
        loss = 1 - iou
        # print(f"angle_cost: {angle_cost.sum()}")
        # print(f"shape_cost: {shape_cost.sum()}")
        # print(f"distance_cost: {distance_cost.sum()}")
        # print(f"iou:{iou.sum()}")
        return loss


class VariFocalLoss(nn.Module):
    def __init__(self):
        super(VariFocalLoss, self).__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):

        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            pred_score = F.sigmoid(pred_score)
            loss = (
                F.binary_cross_entropy(
                    pred_score.float(), gt_score.float(), reduction="none"
                )
                * weight
            ).sum()

        return loss


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_class = 1.0
        self.lambda_iou = 2.5

        self.varifocal_loss = VariFocalLoss()
        self.box_loss = IOULoss("corner")

    @torch.no_grad()
    def forward(self, pred, target):
        # class loss
        # One-hot encoded labels
        target_labels = target[..., 5:]

        # class_score, xyxy [bs, 13, 13, 5]
        loss_cls = self.varifocal_loss(pred[..., :5], target[..., :5], target_labels)

        # SIOU loss
        # xyxy
        loss_iou = self.box_loss(pred[..., 1:5], target[..., 1:5])
        loss = self.lambda_class * loss_cls + self.lambda_iou * loss_iou
        return loss


def convert_to_xy(pred, img_size=416):
    """Takes input tensors and converts them into x1y1x2y2 format using center anchors"""
    s = pred.size(1)
    stride = img_size / s
    range_vals = torch.arange(s, device=device)
    x_grid, y_grid = torch.meshgrid(range_vals, range_vals, indexing="ij")

    pred[..., 0:1] = x_grid.unsqueeze(dim=-1) * stride - pred[..., 0:1]  # x1
    pred[..., 1:2] = y_grid.unsqueeze(dim=-1) * stride - pred[..., 1:2]  # Y1
    pred[..., 2:3] = x_grid.unsqueeze(dim=-1) * stride + pred[..., 2:3]  # x2
    pred[..., 3:4] = y_grid.unsqueeze(dim=-1) * stride + pred[..., 3:4]  # y2

    return pred
