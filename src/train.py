# src/train.py

import torch
from torch import nn, optim

from src.models.teacher import TeacherModel
from src.models.student import StudentModel
from src.data.preprocessing import load_image, random_intensity_modulation
from src.utils.patch_extraction import extract_patch_features
from src.utils.metrics import compute_patch_error
from src.utils.config import (
    DEVICE, PATCH_SIZE, PATCH_STRIDE,
    EPOCHS, LEARNING_RATE
)


def train_one_pair(ref_path: str, insp_path: str):

    teacher = TeacherModel().to(DEVICE).eval()
    student = StudentModel().to(DEVICE)

    # --- Load and preprocess ---
    ref = load_image(ref_path).to(DEVICE)
    insp = load_image(insp_path).to(DEVICE)
    insp_mod = random_intensity_modulation(insp)

    # --- Extract teacher features ---
    with torch.no_grad():
        ref_feat = teacher(ref).squeeze(0)
        insp_feat = teacher(insp_mod).squeeze(0)

    # --- Patchify ---
    patches_ref = extract_patch_features(ref_feat, PATCH_SIZE, PATCH_STRIDE)
    patches_insp = extract_patch_features(insp_feat, PATCH_SIZE, PATCH_STRIDE)

    # --- Train student ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for pr, pi in zip(patches_ref, patches_insp):
            pr, pi = pr.to(DEVICE), pi.to(DEVICE)
            optimizer.zero_grad()
            pred = student(pi)
            loss = criterion(pred, pr)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss {epoch_loss:.4f}")

    return student


if __name__ == "__main__":
    student = train_one_pair(
        ref_path="data/raw/case1_aligned_reference.png",
        insp_path="data/raw/case1_aligned_inspected.png",
    )
