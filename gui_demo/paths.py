"""Canonical dataset paths for the thesis GUI (two labeled ultrasound sources only)."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# FSHD cohort (LABELD_1)
FSHD_CANDIDATES = [
    PROJECT_ROOT / "data" / "ULTRASOUND_LABELD_1_FSHD" / "images",
    PROJECT_ROOT / "data" / "ULTRASOUND_LABELD_1" / "images",
]

# MAT-extracted myopathy cohort (4 disease classes; no FSHD)
MAT_CANDIDATES = [
    PROJECT_ROOT / "data" / "images_extracted_from_mat_LABELED",
    PROJECT_ROOT / "data" / "dl_images_extracted_from_mat",
]

GUI_MODELS_DIR = Path(__file__).resolve().parent / "models"
DEMO_ROOT = PROJECT_ROOT / "demo_data"

MAT_DISEASE_FOLDERS = ["Normal", "IBM", "Dermatomyositis", "Polymyositis"]
DEMO_FOLDERS = ["FSHD"] + MAT_DISEASE_FOLDERS

FOLDER_TRUE_LABEL = {
    "FSHD": "FSHD",
    "Dermatomyositis": "Dermatomyositis",
    "Polymyositis": "Polymyositis",
    "IBM": "Inclusion Body Myositis",
    "Normal": "Normal",
}


def first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_dir():
            return p
    return None


def fshd_image_dir() -> Path | None:
    return first_existing(FSHD_CANDIDATES)


def mat_image_root() -> Path | None:
    return first_existing(MAT_CANDIDATES)
