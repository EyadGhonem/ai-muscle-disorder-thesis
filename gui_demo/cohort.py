"""
cohort.py
---------
Utility module that determines which dataset cohort an uploaded image belongs to.

Two cohorts are supported in this thesis:
- FSHD  : images from ULTRASOUND_LABELD_1 (binary severity labels: mild / severe)
- MAT   : images extracted from MATLAB files (4 disease classes: Normal, IBM,
          Dermatomyositis, Polymyositis)

Detection relies on folder structure, path keywords, and filename conventions.
No model inference is performed here.
"""

from __future__ import annotations

from pathlib import Path

from paths import DEMO_ROOT, FSHD_CANDIDATES, MAT_DISEASE_FOLDERS, mat_image_root

# Cohort identifier constants used throughout the GUI codebase
FSHD = "fshd"
MAT = "mat"


def detect_cohort(
    demo_folder: str | None,
    image_path: Path | None,
    upload_cohort: str | None = None,
) -> str:
    """Determine whether an image belongs to the FSHD or MAT cohort.

    Detection priority:
    1. Explicit ``upload_cohort`` argument (trusted when set by caller).
    2. Demo folder name matching known cohort folders.
    3. Keyword matching in the absolute file path string.
    4. Relative path under the demo_data/ directory.
    5. FSHD filename convention (numeric prefix, e.g. ``02244_008_01_1.png``).

    Returns
    -------
    str
        ``FSHD`` or ``MAT`` cohort identifier.
    """
    # If the caller already knows the cohort, use it directly
    if upload_cohort in (FSHD, MAT):
        return upload_cohort

    # Match against known demo folder names
    if demo_folder == "FSHD":
        return FSHD
    if demo_folder in MAT_DISEASE_FOLDERS:
        return MAT

    if image_path is None:
        return MAT

    # Keyword-based path detection (case-insensitive)
    path_str = str(image_path).replace("\\", "/").lower()
    if "/fshd/" in path_str or "ultrasound_labeld_1" in path_str:
        return FSHD
    if "images_extracted_from_mat" in path_str or "extracted_from_mat_labeled" in path_str:
        return MAT

    # Demo copy under demo_data/FSHD vs other folders only (no filename guessing)
    try:
        rel = image_path.resolve().relative_to(DEMO_ROOT.resolve())
        if rel.parts and rel.parts[0] == "FSHD":
            return FSHD
        if rel.parts and rel.parts[0] in MAT_DISEASE_FOLDERS:
            return MAT
    except ValueError:
        pass

    # FSHD LABELD_1 filenames follow the pattern: 02244_008_01_1.png (starts with digit)
    name = image_path.name
    if name and name[0].isdigit() and "_" in name:
        return FSHD

    # Default to MAT cohort if no match found
    return MAT


def cohort_label(cohort: str) -> str:
    """Return a human-readable label for the given cohort identifier."""
    if cohort == FSHD:
        return "FSHD (ULTRASOUND_LABELD_1)"
    return "MAT-labeled myopathy (4 classes)"
