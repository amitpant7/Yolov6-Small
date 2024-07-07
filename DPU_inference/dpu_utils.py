import numpy as np


def decode_preds(
    targets, nc=20, scales=[13, 26, 52], img_size=416, center_thres=0, conf_thres=0.65
):
    outputs = []

    for tensor, scale in zip(targets, scales):
        # Remove batch dimension
        tensor = np.squeeze(tensor, axis=0)  # Shape [scale, scale, nc + 5]

        stride = img_size / scale
        cls = 1 / (1 + np.exp(-tensor[..., :nc]))  # Sigmoid for class assignments
        reg = tensor[..., nc:]  # Regression info including centerness
        reg[..., :4] = np.exp(reg[..., :4])

        # Find positions where class assignments are made (any class score > conf_thres)
        class_mask = cls > conf_thres
        class_indices = np.argwhere(class_mask)

        # Extract class labels and positions
        labels = class_indices[:, 2]  # Class indices
        i_indices = class_indices[:, 0]  # Grid i positions
        j_indices = class_indices[:, 1]  # Grid j positions

        # Gather regression values using advanced indexing
        cls_conf = cls[class_mask]
        reg_values = reg[i_indices, j_indices, :]

        l, t, r, b, centerness = reg_values.T
        l, t, r, b = l * stride, t * stride, r * stride, b * stride

        center_x = (i_indices.astype(np.float32) + 0.5) * stride
        center_y = (j_indices.astype(np.float32) + 0.5) * stride

        left = center_x - l
        top = center_y - t
        right = center_x + r
        bottom = center_y + b

        centers = (left + right) / 2, (top + bottom) / 2

        widths = right - left
        heights = bottom - top

        centerness = 1 / (1 + np.exp(-centerness))  # Sigmoid

        bboxes = np.stack(
            [
                labels.astype(np.float32),
                cls_conf,
                centers[0],
                centers[1],
                widths,
                heights,
                centerness,
            ],
            axis=1,
        )

        keep = bboxes[:, -1] >= center_thres

        if np.sum(keep) > 0:
            outputs.extend(bboxes[keep].tolist())

    return np.clip(np.array(outputs), a_min=0, a_max=None)


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
    - boxes (numpy.ndarray): Bounding boxes array of shape [N, 4], where each row represents [x1, y1, x2, y2].
    - scores (numpy.ndarray): Confidence scores array of shape [N].
    - iou_threshold (float): IOU threshold for suppression.

    Returns:
    - keep_indices (list): List of indices of the kept bounding boxes.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # Sort by scores in descending order

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]  # Add 1 because we skipped the current box

    return keep
