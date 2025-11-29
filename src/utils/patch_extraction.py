# src/utils/patch_extraction.py

import torch


def extract_patch_features(
    feat_map: torch.Tensor,
    patch_size: int,
    stride: int
) -> torch.Tensor:
    """
    Convert a feature map into flattened patches.
    """
    C, H, W = feat_map.shape
    patches = []

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = feat_map[:, i:i+patch_size, j:j+patch_size].reshape(-1)
            patches.append(patch)

    return torch.stack(patches)
