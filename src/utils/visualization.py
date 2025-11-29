# src/utils/visualization.py

import numpy as np
from skimage.feature import blob_log
import matplotlib.pyplot as plt


def create_error_heatmap(
    errors: np.ndarray,
    H: int,
    W: int,
    patch_size: int,
    stride: int
):
    heatmap = np.zeros((H, W))
    idx = 0
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            heatmap[i:i+patch_size, j:j+patch_size] = errors[idx]
            idx += 1
    return heatmap


def detect_blobs(error_map, threshold_rel=0.5):
    blobs = blob_log(
        error_map,
        max_sigma=10,
        threshold=threshold_rel * np.max(error_map)
    )
    points = [(int(b[1]), int(b[0])) for b in blobs]
    return points


def show_heatmap(heatmap, points):
    plt.imshow(heatmap, cmap='hot')
    if points:
        plt.scatter([p[0] for p in points], [p[1] for p in points], c='cyan')
    plt.show()
