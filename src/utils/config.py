# src/utils/config.py

import torch

# --- Global Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = (256, 256)

PATCH_SIZE = 4
PATCH_STRIDE = 4

TEACHER_FEATURE_DIM = 256  # depends on teacher output
LEARNING_RATE = 1e-3
EPOCHS = 5
