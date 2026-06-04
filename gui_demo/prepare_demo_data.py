#!/usr/bin/env python3
"""
Copy demo images from the two labeled ultrasound sources only:
  - FSHD: ULTRASOUND_LABELD_1_FSHD (or LABELD_1)
  - Other diseases: MAT-extracted images (images_extracted_from_mat_LABELED)

Does NOT use ULTRASOUND_LABELD_2 tabular rows or feature_samples.csv.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GUI_DIR))

from paths import (  # noqa: E402
    DEMO_ROOT,
    FOLDER_TRUE_LABEL,
    MAT_DISEASE_FOLDERS,
    fshd_image_dir,
    mat_image_root,
)

PER_CLASS = 6
RANDOM_STATE = 42


def copy_samples(src_dir: Path, dest_dir: Path, n: int) -> list[str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for old in dest_dir.glob("*.png"):
        old.unlink()
    for old in dest_dir.glob("*.jpg"):
        old.unlink()
    for old in dest_dir.glob("feature_samples.csv"):
        old.unlink()

    files = sorted(src_dir.glob("*.png")) + sorted(src_dir.glob("*.jpg"))
    if not files:
        return []
    rng = random.Random(RANDOM_STATE)
    pick = rng.sample(files, min(n, len(files)))
    lines = []
    for src in pick:
        dst = dest_dir / src.name
        shutil.copy2(src, dst)
        lines.append(f"  IMAGE: {dst.name} <- {src}")
    return lines


def main():
    lines = [
        "Demo dataset summary (two labeled ultrasound sources only)",
        "==========================================================",
        f"Target per class: {PER_CLASS}",
        "",
    ]

    fshd_dir = fshd_image_dir()
    mat_root = mat_image_root()

    if fshd_dir:
        dest = DEMO_ROOT / "FSHD"
        lines.append(f"\n[FSHD] source={fshd_dir}")
        lines.extend(copy_samples(fshd_dir, dest, PER_CLASS) or ["  (no images)"])
    else:
        lines.append("\n[FSHD] SKIP — FSHD image folder not found")

    if mat_root:
        for folder in MAT_DISEASE_FOLDERS:
            src = mat_root / folder
            if not src.is_dir():
                lines.append(f"\n[{folder}] missing folder under MAT root")
                continue
            dest = DEMO_ROOT / folder
            disease = FOLDER_TRUE_LABEL[folder]
            lines.append(f"\n[{folder}] disease={disease} | source={src}")
            lines.extend(copy_samples(src, dest, PER_CLASS) or ["  (no images)"])
    else:
        lines.append("\nMAT cohort SKIP — images_extracted_from_mat_LABELED not found")

    for folder in FOLDER_TRUE_LABEL:
        meta_path = DEMO_ROOT / folder / "folder_meta.json"
        if (DEMO_ROOT / folder).exists():
            n_img = len(list((DEMO_ROOT / folder).glob("*.png")))
            meta_path.write_text(
                json.dumps(
                    {
                        "disease": FOLDER_TRUE_LABEL[folder],
                        "folder": folder,
                        "image_count": n_img,
                        "source": "FSHD_LABELD_1" if folder == "FSHD" else "MAT_LABELED",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    summary = DEMO_ROOT / "demo_summary.txt"
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
