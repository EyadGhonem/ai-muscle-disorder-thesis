#!/usr/bin/env python3
"""
run_general_ai_comparison.py
-----------------------------
Step 1 & 2 of the General AI Comparison Study.

Purpose
-------
Select 15 representative ultrasound test images (3 per disease class),
copy them into general_ai_test_cases/, run MyoScan AI (ML pipeline)
on each image, and write the predictions to:

    results/general_ai_comparison/test_cases.csv

This CSV will later be used to compare MyoScan AI results against
responses from general-purpose multimodal AI assistants (e.g. ChatGPT,
Gemini) when those assistants are shown the same images.

Columns in output CSV
---------------------
case_id               : integer 1–15
image_path            : path to the copied image in general_ai_test_cases/
ground_truth          : disease class label
myoscan_prediction    : predicted class from best available ML model
myoscan_confidence    : softmax confidence % (or "Not available" if absent)

Usage
-----
    python scripts/run_general_ai_comparison.py

Requires the trained ML bundle:
    output/baseline_and_advanced_models/trained_models.pkl
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ── project root and sys.path setup ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GUI_DIR      = PROJECT_ROOT / "gui_demo"
sys.path.insert(0, str(GUI_DIR))          # makes gui_demo importable as a package

# ── output paths ──────────────────────────────────────────────────────────────
DEST_ROOT   = PROJECT_ROOT / "general_ai_test_cases"
RESULTS_DIR = PROJECT_ROOT / "results" / "general_ai_comparison"
OUTPUT_CSV  = RESULTS_DIR / "test_cases.csv"

# ── source demo images (3 per class, chosen for visual clarity) ───────────────
# All sourced from demo_data/ — files are only COPIED, never moved/deleted.
DEMO_DIR = PROJECT_ROOT / "demo_data"

# Image selection: 3 clear, representative images per class
# FSHD: 1 Mild (_00_) + 2 Severe (_01_) for variety
SELECTED_IMAGES: list[tuple[str, str, str]] = [
    # (class_label, source_subfolder, filename)
    # Normal — 3 images
    ("Normal",                  "Normal",           "N008_20151013153636_idx0229.png"),
    ("Normal",                  "Normal",           "N021_20160107163403_idx0740.png"),
    ("Normal",                  "Normal",           "N031_2016021614341705_idx1137.png"),
    # IBM — 3 images
    ("Inclusion Body Myositis", "IBM",              "M017_20160107165757_idx0572.png"),
    ("Inclusion Body Myositis", "IBM",              "M034_2016022418290401_idx1256.png"),
    ("Inclusion Body Myositis", "IBM",              "M071_2016081610135108_idx2744.png"),
    # Dermatomyositis — 3 images
    ("Dermatomyositis",         "Dermatomyositis",  "M001_20150922134651_idx0031.png"),
    ("Dermatomyositis",         "Dermatomyositis",  "M049_2016032810561319_idx1893.png"),
    ("Dermatomyositis",         "Dermatomyositis",  "M067_2016072712234211_idx2573.png"),
    # Polymyositis — 3 images
    ("Polymyositis",            "Polymyositis",     "M002_20151016175732_idx0067.png"),
    ("Polymyositis",            "Polymyositis",     "M010_20151103112254_idx0313.png"),
    ("Polymyositis",            "Polymyositis",     "M018_2016010717002322_idx0607.png"),
    # FSHD — 1 Mild + 2 Severe
    ("FSHD",                    "FSHD",             "00319_011_00_2.png"),   # Mild (severity 0)
    ("FSHD",                    "FSHD",             "00076_008_01_3.png"),   # Severe (severity 1)
    ("FSHD",                    "FSHD",             "02244_008_01_1.png"),   # Severe (severity 1)
]

# ── ML model preference order (try each until one loads) ─────────────────────
# These are the best-performing models from the thesis evaluation.
PREFERRED_MODELS = [
    "Gradient Boosting",
    "XGBoost",
    "Extra Trees",
    "Random Forest",
    "SVM",
    "LightGBM",
    "CatBoost",
    "Logistic Regression",
    "Stacking",
]


def copy_images() -> list[tuple[str, str, Path]]:
    """Copy the 15 selected demo images into general_ai_test_cases/.

    For each image:
    - Source: demo_data/{subfolder}/{filename}
    - Destination: general_ai_test_cases/{dest_folder}/{filename}

    The destination subfolder for IBM images is named "IBM" on disk
    (matching the original folder name) but the ground-truth label
    stored in the CSV is the full canonical name.

    Returns
    -------
    list of (class_label, filename, dest_path) tuples
    """
    dest_folder_map = {
        "Normal":                  "Normal",
        "Inclusion Body Myositis": "IBM",
        "Dermatomyositis":         "Dermatomyositis",
        "Polymyositis":            "Polymyositis",
        "FSHD":                    "FSHD",
    }

    copied = []
    missing = []

    for class_label, src_subfolder, filename in SELECTED_IMAGES:
        src_path  = DEMO_DIR / src_subfolder / filename
        dest_dir  = DEST_ROOT / dest_folder_map[class_label]
        dest_path = dest_dir / filename

        if not src_path.exists():
            print(f"  [MISSING] {src_path}")
            missing.append(str(src_path))
            continue

        # Copy only if destination does not already exist (safe — no overwrite of originals)
        if not dest_path.exists():
            shutil.copy2(str(src_path), str(dest_path))
            print(f"  [COPIED ] {filename}  ->  {dest_dir.name}/")
        else:
            print(f"  [EXISTS ] {filename}  (skipped, already present)")

        copied.append((class_label, filename, dest_path))

    if missing:
        print(f"\n  WARNING: {len(missing)} source file(s) not found — check demo_data/")

    return copied


def load_ml_pipeline():
    """Load the ML bundle and return (bundle, model_name, warnings).

    Tries PREFERRED_MODELS in order and returns the first that is available.
    Returns (None, None, [error_message]) if bundle cannot be loaded.
    """
    from model_registry import load_ml_bundle
    from cohort import MAT

    bundle, warnings = load_ml_bundle()
    if bundle is None:
        return None, None, warnings + ["ML bundle could not be loaded — check trained_models.pkl path."]

    # Pick the best available model
    selected_model = None
    for name in PREFERRED_MODELS:
        if name in bundle.models:
            selected_model = name
            break

    if selected_model is None:
        return bundle, None, warnings + [f"None of the preferred models found. Available: {list(bundle.models.keys())}"]

    print(f"  [MODEL  ] Using: {selected_model}  (bundle has {len(bundle.models)} models)")
    return bundle, selected_model, warnings


def run_prediction(image_path: Path, class_label: str, bundle, model_name: str) -> tuple[str, str]:
    """Run MyoScan AI ML prediction on a single image.

    Parameters
    ----------
    image_path   : path to the copied image
    class_label  : ground-truth disease label (used for feature lookup)
    bundle       : loaded MLBundle
    model_name   : name of the ML model to use

    Returns
    -------
    (predicted_class, confidence_str)
    confidence_str is "Not available" if model has no predict_proba.
    """
    from inference import extract_features_for_image, predict_ml
    from cohort import FSHD, MAT

    # Determine cohort from ground truth
    cohort = FSHD if class_label == "FSHD" else MAT

    # Extract features (fast CSV lookup or live ROI radiomics)
    feats, err, source = extract_features_for_image(
        image_path,
        bundle.feature_columns,
        cohort=cohort,
        true_label=class_label,
    )

    if feats is None:
        return "ERROR", f"Feature extraction failed: {err}"

    # Run model prediction
    pred = predict_ml(bundle, model_name, feats)

    if "error" in pred:
        return "ERROR", pred["error"]

    predicted_class = pred.get("predicted_class", "Unknown")
    confidence      = pred.get("confidence", float("nan"))

    # Format confidence — use raw softmax value (no presentation calibration)
    if confidence != confidence or np.isnan(float(confidence)):   # NaN check
        conf_str = "Not available"
    else:
        conf_str = f"{float(confidence):.1f}%"

    return predicted_class, conf_str


def build_and_run_csv(copied_images, bundle, model_name: str) -> pd.DataFrame:
    """Build the test_cases.csv with MyoScan AI predictions filled in.

    Parameters
    ----------
    copied_images : list of (class_label, filename, dest_path) from copy_images()
    bundle        : loaded MLBundle (or None)
    model_name    : ML model to use for prediction (or None)

    Returns
    -------
    DataFrame written to OUTPUT_CSV
    """
    rows = []
    prediction_errors = []

    for case_id, (class_label, filename, dest_path) in enumerate(copied_images, start=1):
        print(f"\n  [{case_id:02d}/15] {class_label:<28} {filename}")

        if bundle is not None and model_name is not None:
            predicted, confidence = run_prediction(dest_path, class_label, bundle, model_name)
        else:
            # ML bundle failed to load — record unavailability
            predicted, confidence = "Not available", "Not available"
            prediction_errors.append(f"Case {case_id}: ML bundle not available")

        # Flag prediction errors for summary
        if predicted.startswith("ERROR"):
            prediction_errors.append(f"Case {case_id} ({filename}): {confidence}")
            predicted, confidence = "ERROR", "Not available"

        correct_flag = "OK" if predicted == class_label else (
            "OK(IBM)" if (class_label == "Inclusion Body Myositis" and predicted == "Inclusion Body Myositis")
            else "WRONG" if predicted not in ("ERROR", "Not available") else "?"
        )
        print(f"         Predicted: {predicted:<28}  Conf: {confidence:>10}  {correct_flag}")

        rows.append({
            "case_id":              case_id,
            "image_path":           str(dest_path.relative_to(PROJECT_ROOT)),
            "ground_truth":         class_label,
            "myoscan_model":        model_name or "Not available",
            "myoscan_prediction":   predicted,
            "myoscan_confidence":   confidence,
        })

    df = pd.DataFrame(rows)

    # Write CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  [SAVED  ] {OUTPUT_CSV}")

    return df, prediction_errors


def print_summary(df: pd.DataFrame, prediction_errors: list[str], model_name: str | None):
    """Print a clean results summary table to the console."""
    sep  = "=" * 72
    thin = "-" * 72

    print(f"\n{sep}")
    print("  MyoScan AI Comparison — Run Summary")
    print(sep)
    print(f"  Model used        : {model_name or 'Not available'}")
    print(f"  Total test cases  : {len(df)}")

    # Count correct predictions (exclude errors)
    valid = df[~df["myoscan_prediction"].isin(["ERROR", "Not available"])]
    if not valid.empty:
        correct = (valid["myoscan_prediction"] == valid["ground_truth"]).sum()
        print(f"  Correct (exact)   : {correct}/{len(valid)} ({correct/len(valid)*100:.0f}%)")

    print(f"\n  {'#':<5} {'Ground Truth':<28} {'Predicted':<28} {'Confidence':>10}")
    print(thin)
    for _, row in df.iterrows():
        marker = "OK" if row["myoscan_prediction"] == row["ground_truth"] else "WRONG"
        print(
            f"  {int(row['case_id']):<5} "
            f"{row['ground_truth']:<28} "
            f"{row['myoscan_prediction']:<28} "
            f"{str(row['myoscan_confidence']):>10}  {marker}"
        )

    if prediction_errors:
        print(f"\n{thin}")
        print("  ERRORS / WARNINGS:")
        for e in prediction_errors:
            print(f"    • {e}")

    print(f"\n{sep}")
    print(f"  Output CSV : {OUTPUT_CSV}")
    print(sep)


def main():
    """Main entry point — orchestrates all steps."""
    print("\n" + "=" * 72)
    print("  MyoScan AI — General AI Comparison Setup")
    print("  Steps: select images -> copy -> run ML predictions -> save CSV")
    print("=" * 72)

    # ── Step 1: Copy 15 images ────────────────────────────────────────────────
    print("\n[Step 1] Copying 15 representative demo images...")
    copied_images = copy_images()
    print(f"         {len(copied_images)}/15 images ready in general_ai_test_cases/")

    if not copied_images:
        print("  ERROR: No images could be copied. Check demo_data/ folder.")
        sys.exit(1)

    # ── Step 2: Load ML model ─────────────────────────────────────────────────
    print("\n[Step 2] Loading MyoScan AI ML model...")
    bundle, model_name, ml_warnings = load_ml_pipeline()
    for w in ml_warnings:
        print(f"  [WARN] {w}")

    # ── Step 3: Run predictions and build CSV ─────────────────────────────────
    print("\n[Step 3] Running MyoScan AI predictions...")
    df, prediction_errors = build_and_run_csv(copied_images, bundle, model_name)

    # ── Step 4: Print summary ─────────────────────────────────────────────────
    print_summary(df, prediction_errors, model_name)


if __name__ == "__main__":
    main()
