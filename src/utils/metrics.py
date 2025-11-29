# src/utils/metrics.py

import torch
import numpy as np


def compute_patch_error(
    patches_ref: torch.Tensor,
    patches_insp: torch.Tensor,
    student
) -> np.ndarray:
    """
    Compute reconstruction errors between student predictions and reference.
    """
    errors = []

    for pr, pi in zip(patches_ref, patches_insp):
        pred = student(pi)
        err = torch.norm(pred - pr).item()
        errors.append(err)

    return np.array(errors)
