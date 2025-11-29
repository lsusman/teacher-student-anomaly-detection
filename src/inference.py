# src/inference.py

import torch
from src.models.teacher import TeacherModel
from src.utils.visualization import (
    create_error_heatmap,
    detect_blobs,
    show_heatmap
)
from src.utils.patch_extraction import extract_patch_features
from src.data.preprocessing import load_image
from src.utils.metrics import compute_patch_error
from src.utils.config import DEVICE, PATCH_SIZE, PATCH_STRIDE


def run_inference(student, ref_path, insp_path):
    teacher = TeacherModel().to(DEVICE).eval()

    ref = load_image(ref_path).to(DEVICE)
    insp = load_image(insp_path).to(DEVICE)

    # Teacher features
    with torch.no_grad():
        ref_feat = teacher(ref).squeeze(0)
        insp_feat = teacher(insp).squeeze(0)

    # Patchify
    patches_ref = extract_patch_features(ref_feat, PATCH_SIZE, PATCH_STRIDE)
    patches_insp = extract_patch_features(insp_feat, PATCH_SIZE, PATCH_STRIDE)

    # Compute error
    errors = compute_patch_error(patches_ref, patches_insp, student)

    # Heatmap
    H, W = ref_feat.shape[1], ref_feat.shape[2]
    heatmap = create_error_heatmap(errors, H, W, PATCH_SIZE, PATCH_STRIDE)

    # Blob detection
    points = detect_blobs(heatmap)

    show_heatmap(heatmap, points)
    return heatmap, points


if __name__ == "__main__":
    # Example usage
    student = torch.load("student_model.pth", map_location=DEVICE)
    run_inference(
        student,
        ref_path="examples/ref_sample.png",
        insp_path="examples/insp_sample.png",
    )
