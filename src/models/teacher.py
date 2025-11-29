# src/models/teacher.py

import torch.nn as nn
from torchvision import models


class TeacherModel(nn.Module):
    """
    Pretrained convolutional encoder model (AlexNet features).
    Outputs a spatial feature map.
    """

    def __init__(self):
        super().__init__()
        backbone = models.alexnet(
            weights=models.AlexNet_Weights.IMAGENET1K_V1
        )
        self.encoder = backbone.features

    def forward(self, x):
        return self.encoder(x)
