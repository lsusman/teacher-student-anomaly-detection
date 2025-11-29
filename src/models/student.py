# src/models/student.py

import torch.nn as nn
from src.utils.config import TEACHER_FEATURE_DIM


class StudentModel(nn.Module):
    """
    Regression network mapping inspected teacher features → reference features.
    """

    def __init__(self, latent_dim: int = TEACHER_FEATURE_DIM):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        return self.regressor(x)
