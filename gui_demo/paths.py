"""
paths.py
--------
Centralised path configuration for the thesis GUI.

Defines all dataset, model, and demo directory locations used across the
gui_demo package. Keeping paths in one place avoids hard-coded strings
scattered throughout the codebase.

Two labeled ultrasound sources are used in this thesis:
- ULTRASOUND_LABELD_1  : ~25,005 FSHD images with Heckmatt severity labels
- images_extracted_from_mat_LABELED : ~3,194 MAT-derived images (4 disease classes)
"""

from __future__ import annotations

from pathlib import Path

# Root of the entire thesis project (one level above gui_demo/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Candidate directories for the FSHD ultrasound dataset (LABELD_1).
# The first existing directory is used at runtime.
FSHD_CANDIDATES = [
    PROJECT_ROOT / "data" / "ULTRASOUND_LABELD_1_FSHD" / "images",
    PROJECT_ROOT / "data" / "ULTRASOUND_LABELD_1" / "images",
]

# Candidate directories for MAT-extracted myopathy images (4 disease classes; no FSHD).
MAT_CANDIDATES = [
    PROJECT_ROOT / "data" / "images_extracted_from_mat_LABELED",
    PROJECT_ROOT / "data" / "dl_images_extracted_from_mat",
]

# Directory where trained .keras CNN weights and metrics JSON are stored
GUI_MODELS_DIR = Path(__file__).resolve().parent / "models"

# Optional demo_data/ folder used when running the GUI from pre-copied samples
DEMO_ROOT = PROJECT_ROOT / "demo_data"

# Known sub-folder names inside the MAT cohort and demo_data/
MAT_DISEASE_FOLDERS = ["Normal", "IBM", "Dermatomyositis", "Polymyositis"]
DEMO_FOLDERS = ["FSHD"] + MAT_DISEASE_FOLDERS

# Maps demo folder names to canonical disease label strings used in the dataset CSV
FOLDER_TRUE_LABEL = {
    "FSHD": "FSHD",
    "Dermatomyositis": "Dermatomyositis",
    "Polymyositis": "Polymyositis",
    "IBM": "Inclusion Body Myositis",   # IBM folder maps to full name
    "Normal": "Normal",
}


def first_existing(candidates: list[Path]) -> Path | None:
    """Return the first directory in *candidates* that exists on disk, or None."""
    for p in candidates:
        if p.is_dir():
            return p
    return None


def fshd_image_dir() -> Path | None:
    """Resolve the FSHD image directory from the candidate list."""
    return first_existing(FSHD_CANDIDATES)


def mat_image_root() -> Path | None:
    """Resolve the MAT myopathy image directory from the candidate list."""
    return first_existing(MAT_CANDIDATES)
