import numpy as np
from .utils import visualize_bb, decode_preds, non_max_suppression
from config import device


def visualize_outputs(model, train_data, center_thres=0.65, conf_thres=0.6):
    samples = []
    model.eval()
    ind = np.random.choice(len(train_data), size=10, replace=False)
    for indices in ind:
        img = train_data[indices][0]
        image = img.unsqueeze(0)
        image = image.to(device)
        preds = model(image)
        outputs = decode_preds(preds, center_thres=center_thres, conf_thres=conf_thres)

        if outputs.dim() <= 1:
            continue

        (
            labels,
            conf,
            bboxes,
        ) = (
            outputs[:, 0],
            outputs[:, 1],
            outputs[:, 2:6],
        )
        keep = non_max_suppression(bboxes.clone(), conf, 0.2)
        print(keep.shape)

        sample = {"image": img, "bbox": bboxes[keep], "labels": labels[keep].long()}
        samples += [sample]

    visualize_bb(samples)
