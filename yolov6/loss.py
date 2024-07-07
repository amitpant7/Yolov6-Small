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

    #     if mode == "corner":
    #         assert torch.all(box1[..., 0] < box1[..., 2]), "box1: x1 must be less than x2"
    #         assert torch.all(box1[..., 1] < box1[..., 3]), "box1: y1 must be less than y2"
    #         assert torch.all(box2[..., 0] < box2[..., 2]), "box2: x1 must be less than x2"
    #         assert torch.all(box2[..., 1] < box2[..., 3]), "box2: y1 must be less than y2"

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

        epsilon = 1e-10
        mask = target[..., 0] > 0

        pred = torch.exp(pred).clone()
        pred = convert_to_xy(pred)
        target = convert_to_xy(target)

        pred = pred[mask]
        target = target[mask]

        iou = calculate_iou(pred, target, mode=self.mode)
        b1_x1, b1_y1, b1_x2, b1_y2 = (
            pred[..., 0],
            pred[..., 1],
            pred[..., 2],
            pred[..., 3],
        )
        b2_x1, b2_y1, b2_x2, b2_y2 = (
            target[..., 0],
            target[..., 1],
            target[..., 2],
            target[..., 3],
        )

        w1, h1 = b1_x2 - b1_x1 + epsilon, b1_y2 - b1_y1 + epsilon
        w2, h2 = b2_x2 - b2_x1 + epsilon, b2_y2 - b2_y1 + epsilon
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1) + epsilon
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1) + epsilon

        Cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
        Ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
        sigma = torch.sqrt(Cw**2 + Ch**2 + epsilon)

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
        #         print('shape:', loss.shape)
        return loss.nanmean()


class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, preds, targets, alpha=0.75, gamma=2.0):
        """

        :param preds:[bs, grid_size, grid_size,num_classes]
        :param targets:[bs, grid_size, grid_size,num_classes]
        :param alpha:
        :param gamma:
        :return:
        """
        pred_prob = torch.sigmoid(preds)

        weight = alpha * pred_prob.pow(gamma) * (1 - targets) + targets
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy(pred_prob, targets, reduction="none") * weight

            #         loss = F.binary_cross_entropy_with_logits(preds, targets.float(), reduction="none")
        return loss.mean()


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_class = 1.0
        self.lambda_iou = 2.5
        self.lambda_center = 1.5

        self.varifocal_loss = VarifocalLoss()
        self.box_loss = IOULoss("corner")

    def forward(self, preds, targets):
        # class loss
        # One-hot encoded labels
        total_loss = 0

        for pred, target in zip(preds, targets):
            # creating mask to perform cal only where there is object
            loss_iou = self.box_loss(pred[..., 20:24], target[..., 20:24])
            loss_cls = self.varifocal_loss(pred[..., :20], target[..., :20])

            mask = target[..., 24:25] > 0
            center_loss = F.binary_cross_entropy_with_logits(
                pred[..., 24:25][mask], target[..., 24:25][mask]
            )

            loss = (
                self.lambda_class * loss_cls
                + self.lambda_iou * loss_iou
                + self.lambda_center * center_loss
            )
            total_loss += loss
        return total_loss


def convert_to_xy(pred, img_size=416):
    """Takes input tensors and converts them into x1y1x2y2 format using center anchors"""
    s = pred.size(1)
    stride = img_size / s
    range_vals = torch.arange(s, device=device)
    x_grid, y_grid = torch.meshgrid(range_vals, range_vals, indexing="ij")

    pred[..., 0:1] = x_grid.unsqueeze(dim=-1) - pred[..., 0:1]  # x1
    pred[..., 1:2] = y_grid.unsqueeze(dim=-1) - pred[..., 1:2]  # Y1
    pred[..., 2:3] = x_grid.unsqueeze(dim=-1) + pred[..., 2:3]  # x2
    pred[..., 3:4] = y_grid.unsqueeze(dim=-1) + pred[..., 3:4]  # y2
    pred = pred.clamp(min=0)

    return pred + 0.5  # offset
