"""Detect whether an image belongs to FSHD or MAT-labeled demo cohort."""

from __future__ import annotations

from pathlib import Path

from paths import DEMO_ROOT, FSHD_CANDIDATES, MAT_DISEASE_FOLDERS, mat_image_root

FSHD = "fshd"
MAT = "mat"


def detect_cohort(
    demo_folder: str | None,
    image_path: Path | None,
    upload_cohort: str | None = None,
) -> str:
    if upload_cohort in (FSHD, MAT):
        return upload_cohort
    if demo_folder == "FSHD":
        return FSHD
    if demo_folder in MAT_DISEASE_FOLDERS:
        return MAT

    if image_path is None:
        return MAT

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

    # FSHD LABELD_1 filenames: 02244_008_01_1.png
    name = image_path.name
    if name and name[0].isdigit() and "_" in name:
        return FSHD

    return MAT


def cohort_label(cohort: str) -> str:
    if cohort == FSHD:
        return "FSHD (ULTRASOUND_LABELD_1)"
    return "MAT-labeled myopathy (4 classes)"
