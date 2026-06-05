"""
Ultrasound inspect pipeline: grayscale → Otsu ROI mask → processed view.
Radiomics feature extraction for ML inference on uploads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "data_processing"))

from extract_custom_features import (  # noqa: E402
    extract_first_order_features,
    extract_gradient_features,
    extract_shape_features,
    extract_texture_features,
)

from paths import PROJECT_ROOT as ROOT  # noqa: E402

_GUI_DATASET = ROOT / "output" / "gui_real_ultrasound_dataset.csv"
DATASET_PATH = (
    _GUI_DATASET
    if _GUI_DATASET.exists()
    else ROOT / "output" / "final_ultrasound_dataset.csv"
)


def load_bgr(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image

        img = np.array(Image.open(image_path).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def build_roi_mask(grayscale: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Otsu threshold + morphology + largest contour (thesis Figure 3.2)."""
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask_filled = np.zeros_like(cleaned)
        cv2.drawContours(mask_filled, [largest], -1, 255, -1)
        cleaned = mask_filled

    return thresholded, cleaned, blurred


def processed_roi_view(original_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Green-tinted ROI overlay with contour."""
    overlay = original_rgb.copy()
    mask_color = np.zeros_like(original_rgb)
    mask_color[:, :, 1] = 120
    mask_color[:, :, 0] = 60
    mask_bool = mask > 0
    if np.any(mask_bool):
        overlay[mask_bool] = cv2.addWeighted(
            original_rgb[mask_bool], 0.7, mask_color[mask_bool], 0.3, 0
        )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 100), 2)
    return overlay


def normalized_masked_grayscale(grayscale: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked region on black background, normalized for display."""
    out = np.zeros_like(grayscale)
    region = grayscale.copy()
    if region.max() > 0:
        region = cv2.normalize(region, None, 0, 255, cv2.NORM_MINMAX)
    out[mask > 0] = region[mask > 0]
    return out


def run_inspect_pipeline(image_path: Path) -> dict:
    """
    Returns dict with RGB arrays for Streamlit:
    original, grayscale, threshold, mask_binary, roi_overlay, processed.
    """
    bgr = load_bgr(image_path)
    original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    grayscale = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    thresholded, mask, _ = build_roi_mask(grayscale)
    roi_overlay = processed_roi_view(original_rgb, mask)
    processed = normalized_masked_grayscale(grayscale, mask)

    return {
        "original": original_rgb,
        "grayscale": grayscale,
        "threshold": thresholded,
        "mask": mask,
        "roi_overlay": roi_overlay,
        "processed": cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB),
        "mask_uint8": (mask > 0).astype(np.uint8) * 255,
        "gray_norm": grayscale,
    }


IMG_SIZE_CNN = (224, 224)


def prepare_ultrasound_cnn_tensor(
    image_path: Path | str,
    preprocess_fn,
    img_size: tuple[int, int] = IMG_SIZE_CNN,
    augment: bool = False,
) -> np.ndarray | None:
    """CLAHE + ROI mask → 3-channel tensor for disease CNN (matches enhanced training)."""
    bgr = load_bgr(Path(image_path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if augment:
        if np.random.rand() > 0.5:
            gray = cv2.flip(gray, 1)
        angle = np.random.uniform(-18, 18)
        m = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), angle, 1.0)
        gray = cv2.warpAffine(gray, m, (gray.shape[1], gray.shape[0]), borderMode=cv2.BORDER_REFLECT)
        if np.random.rand() > 0.5:
            beta = np.random.randint(-25, 25)
            gray = np.clip(gray.astype(np.int16) + beta, 0, 255).astype(np.uint8)

    _, mask, _ = build_roi_mask(gray)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)
    roi = enhanced.copy()
    roi[mask == 0] = 0
    rgb = cv2.cvtColor(cv2.resize(roi, img_size), cv2.COLOR_GRAY2RGB)
    return preprocess_fn(rgb.astype(np.float32))


def extract_radiomics_dict(grayscale: np.ndarray, mask: np.ndarray) -> dict:
    """Same feature families as thesis custom radiomics extraction."""
    if grayscale.max() > 0:
        gray = cv2.normalize(grayscale, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        gray = grayscale.astype(np.uint8)

    m = (mask > 0).astype(np.uint8) * 255
    feats = {}
    feats.update(extract_first_order_features(gray, m))
    feats.update(extract_texture_features(gray, m))
    feats.update(extract_shape_features(m))
    feats.update(extract_gradient_features(gray, m))
    return feats


def extract_feature_vector(image_path: Path, feature_columns: list[str]) -> tuple[np.ndarray | None, str]:
    pipe = run_inspect_pipeline(image_path)
    raw = extract_radiomics_dict(pipe["gray_norm"], pipe["mask_uint8"])

    # Optional severity from master CSV when image is in dataset
    severity_val = 0.0
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH, usecols=["image_path", "severity"], nrows=None)
        name = image_path.name
        match = df[df["image_path"].astype(str).str.endswith(name)]
        if not match.empty and pd.notna(match.iloc[0]["severity"]):
            severity_val = float(match.iloc[0]["severity"])

    raw["severity"] = severity_val

    vec = []
    for col in feature_columns:
        val = raw.get(col, np.nan)
        if pd.isna(val):
            val = 0.0
        vec.append(float(val))
    return np.array(vec, dtype=float), ""
