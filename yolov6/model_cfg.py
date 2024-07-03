backbone_cfg = {
    "width_mul": 0.5,
    "depth_mul": 0.33,
    "num_repeats": [1, 6, 12, 18, 6],
    "out_channels": [64, 128, 256, 512, 1024],
}

head_cfg = {"channels": [512, 256, 128], "width_mul": 0.5}

neck_cfg = {
    "channels": [
        64,
        128,
        256,
        512,
        1024,
        256,
        128,
        128,
        256,
        256,
        512,
    ],  # incudes backbone cfg upto 4th index
    "num_repeats": [12, 12, 12, 12],
    "width_mul": 0.5,
    "depth_mul": 0.33,
}
