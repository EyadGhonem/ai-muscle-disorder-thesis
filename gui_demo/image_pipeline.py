"""
image_pipeline.py
-----------------
Core preprocessing and feature-extraction pipeline for the thesis GUI.

This module implements two complementary pipelines:

1. ROI Inspection Pipeline (``run_inspect_pipeline``)
   -------------------------------------------------
   Produces the five intermediate images shown in the GUI's
   "How the AI Sees This Image" panel:
     Original → Grayscale → Otsu Threshold → ROI Mask Overlay → Processed ROI

2. Radiomics Feature Extraction (``extract_feature_vector``)
   ---------------------------------------------------------
   Applies the same preprocessing steps and then computes 28 hand-crafted
   radiomics-inspired features (first-order statistics, GLCM texture,
   shape descriptors, gradient statistics) from the masked ROI.
   The resulting feature vector is passed directly to the ML classifiers.

3. CNN Tensor Preparation (``prepare_ultrasound_cnn_tensor``)
   ----------------------------------------------------------
   Applies CLAHE contrast enhancement and ROI masking before resizing the
   image to 224×224 for input to the disease CNN models.
   This preprocessing matches the enhanced training configuration used during
   model training (see train_gui_on_real_ultrasound.py).

Input  : a single ultrasound image file (PNG / JPEG / TIFF / BMP)
Output : intermediate arrays for display, a feature vector (np.ndarray),
         or a preprocessed CNN input tensor (np.ndarray)
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
    """Load an image from disk in BGR colour format (OpenCV convention).

    Falls back to PIL if OpenCV cannot decode the file (e.g. some TIFF variants).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image
        img = np.array(Image.open(image_path).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def build_roi_mask(grayscale: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a binary ROI mask from a grayscale ultrasound image.

    Pipeline (corresponds to thesis Figure 3.2):
    1. Gaussian blur  : reduces high-frequency noise before thresholding.
    2. Otsu threshold : automatically selects the optimal binarisation threshold.
    3. Morphological closing (3 iterations) : fills small holes in the tissue region.
    4. Morphological opening  (2 iterations) : removes small isolated noise blobs.
    5. Largest contour selection : retains only the biggest connected region,
       assumed to be the muscle of interest.

    Returns
    -------
    thresholded : raw Otsu binary image (before morphology)
    cleaned     : final binary mask (largest connected region only)
    blurred     : Gaussian-blurred grayscale (for display / debugging)
    """
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological refinement to clean the binary mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

    # Keep only the largest contiguous region (the primary muscle area)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask_filled = np.zeros_like(cleaned)
        cv2.drawContours(mask_filled, [largest], -1, 255, -1)
        cleaned = mask_filled

    return thresholded, cleaned, blurred


def processed_roi_view(original_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Render the ROI mask as a semi-transparent green overlay on the original image.

    Used in the GUI's Step 4 inspection panel to visualise which pixels are
    included in feature extraction.
    """
    overlay = original_rgb.copy()
    mask_color = np.zeros_like(original_rgb)
    mask_color[:, :, 1] = 120   # green channel
    mask_color[:, :, 0] = 60    # slight blue tint
    mask_bool = mask > 0
    if np.any(mask_bool):
        # Blend original and colour mask within the ROI region (70 / 30 split)
        overlay[mask_bool] = cv2.addWeighted(
            original_rgb[mask_bool], 0.7, mask_color[mask_bool], 0.3, 0
        )
    # Draw the ROI contour in bright green
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 100), 2)
    return overlay


def normalized_masked_grayscale(grayscale: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the masked ROI region on a black background, min-max normalised.

    Pixels outside the ROI mask are set to zero. Used in the GUI's Step 5 panel
    to show the exact image content passed to the feature extractor.
    """
    out = np.zeros_like(grayscale)
    region = grayscale.copy()
    if region.max() > 0:
        region = cv2.normalize(region, None, 0, 255, cv2.NORM_MINMAX)
    out[mask > 0] = region[mask > 0]
    return out


def run_inspect_pipeline(image_path: Path) -> dict:
    """Execute the full ROI inspection pipeline on a single image.

    Produces the five intermediate representations shown in the GUI:
    1. original      : RGB image as loaded from disk
    2. grayscale     : single-channel luminance image
    3. threshold     : raw Otsu binary mask (before morphological cleaning)
    4. roi_overlay   : original image with semi-transparent ROI overlay
    5. processed     : normalised masked-ROI region on black background (RGB)

    Also returns the raw mask and grayscale arrays needed by the feature extractor.

    Returns
    -------
    dict with keys: original, grayscale, threshold, mask, roi_overlay,
                    processed, mask_uint8, gray_norm
    """
    bgr = load_bgr(image_path)
    original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    grayscale = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    thresholded, mask, _ = build_roi_mask(grayscale)
    roi_overlay = processed_roi_view(original_rgb, mask)
    processed = normalized_masked_grayscale(grayscale, mask)

    return {
        "original":   original_rgb,
        "grayscale":  grayscale,
        "threshold":  thresholded,
        "mask":       mask,
        "roi_overlay": roi_overlay,
        "processed":  cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB),
        "mask_uint8": (mask > 0).astype(np.uint8) * 255,
        "gray_norm":  grayscale,
    }


# Target spatial resolution for all CNN inputs
IMG_SIZE_CNN = (224, 224)


def prepare_ultrasound_cnn_tensor(
    image_path: Path | str,
    preprocess_fn,
    img_size: tuple[int, int] = IMG_SIZE_CNN,
    augment: bool = False,
) -> np.ndarray | None:
    """Prepare an ultrasound image as a CNN input tensor using CLAHE + ROI masking.

    Preprocessing steps (matches the enhanced training configuration):
    1. Convert to grayscale.
    2. Optional augmentation (horizontal flip, random rotation, brightness jitter).
    3. Build ROI mask via Otsu + morphology (``build_roi_mask``).
    4. Min-max normalise the grayscale image.
    5. CLAHE contrast enhancement (clipLimit=2.0, tileGridSize=8×8).
    6. Zero-out pixels outside the ROI mask.
    7. Resize to *img_size* (default 224×224).
    8. Convert to 3-channel RGB and apply architecture-specific preprocessing.

    Parameters
    ----------
    image_path   : path to the input ultrasound image
    preprocess_fn: Keras application preprocessing function (from get_preprocess_fn)
    img_size     : target (width, height) for CNN input
    augment      : if True, apply random augmentations (used during training only)

    Returns
    -------
    np.ndarray of shape (224, 224, 3) ready for model.predict(), or None on error.
    """
    bgr = load_bgr(Path(image_path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if augment:
        # Horizontal flip with 50% probability
        if np.random.rand() > 0.5:
            gray = cv2.flip(gray, 1)
        # Random rotation within ±18 degrees
        angle = np.random.uniform(-18, 18)
        m = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), angle, 1.0)
        gray = cv2.warpAffine(gray, m, (gray.shape[1], gray.shape[0]), borderMode=cv2.BORDER_REFLECT)
        # Random brightness offset within ±25 intensity units
        if np.random.rand() > 0.5:
            beta = np.random.randint(-25, 25)
            gray = np.clip(gray.astype(np.int16) + beta, 0, 255).astype(np.uint8)

    # Build ROI mask and apply CLAHE contrast enhancement to the masked region
    _, mask, _ = build_roi_mask(gray)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)

    # Zero out pixels outside the ROI before passing to the CNN
    roi = enhanced.copy()
    roi[mask == 0] = 0

    # Convert to RGB and apply architecture-specific pixel normalisation
    rgb = cv2.cvtColor(cv2.resize(roi, img_size), cv2.COLOR_GRAY2RGB)
    return preprocess_fn(rgb.astype(np.float32))


def extract_radiomics_dict(grayscale: np.ndarray, mask: np.ndarray) -> dict:
    """Compute all 28 radiomics-inspired features from the masked ROI.

    Feature families (same as thesis custom extraction pipeline):
    - First-order statistics : mean, std, min, max, median, quartiles,
                               skewness, kurtosis, entropy
    - GLCM texture           : energy, contrast, homogeneity, correlation,
                               dissimilarity, ASM
    - Shape descriptors      : area, perimeter, circularity, aspect ratio,
                               extent, solidity, equivalent diameter
    - Gradient statistics    : mean, max, std, energy of the Sobel gradient

    Parameters
    ----------
    grayscale : single-channel (H×W) uint8 grayscale image
    mask      : binary (H×W) uint8 mask; non-zero pixels define the ROI

    Returns
    -------
    dict mapping feature name → float value
    """
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
    """Run the full preprocessing + feature extraction pipeline for a single upload.

    Steps:
    1. Run ``run_inspect_pipeline`` to obtain the normalised grayscale and mask.
    2. Compute all radiomics features via ``extract_radiomics_dict``.
    3. Look up the numeric severity label from the dataset CSV (defaults to 0.0
       if the image is not found, preserving the 28-feature dimensionality that
       the trained scaler expects).
    4. Assemble the ordered feature vector matching ``feature_columns``.

    Returns
    -------
    (feature_vector, error_message)
    ``feature_vector`` is None if extraction fails; ``error_message`` is empty
    on success.
    """
    pipe = run_inspect_pipeline(image_path)
    raw = extract_radiomics_dict(pipe["gray_norm"], pipe["mask_uint8"])

    # Attempt to retrieve the numeric severity label from the master CSV.
    # Severity is included as a feature to match the dimensionality of the
    # trained StandardScaler (28 features total).
    severity_val = 0.0
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH, usecols=["image_path", "severity"], nrows=None)
        name = image_path.name
        match = df[df["image_path"].astype(str).str.endswith(name)]
        if not match.empty and pd.notna(match.iloc[0]["severity"]):
            severity_val = float(match.iloc[0]["severity"])

    raw["severity"] = severity_val

    # Build the ordered feature vector; missing features default to 0.0
    vec = []
    for col in feature_columns:
        val = raw.get(col, np.nan)
        if pd.isna(val):
            val = 0.0
        vec.append(float(val))
    return np.array(vec, dtype=float), ""
