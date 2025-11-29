# src/data/preprocessing.py

from torchvision import transforms
from PIL import Image
import torch
from src.utils.config import IMAGE_SIZE


transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def load_image(path: str) -> torch.Tensor:
    """Load an image from disk and apply preprocessing."""
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def random_intensity_modulation(img: torch.Tensor,
                                min_scale: float = 0.8,
                                max_scale: float = 1.2) -> torch.Tensor:
    """Apply random brightness modulation."""
    scale = torch.empty(1).uniform_(min_scale, max_scale).item()
    return img * scale
