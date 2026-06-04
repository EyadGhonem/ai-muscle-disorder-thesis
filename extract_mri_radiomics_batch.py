"""
Batch MRI radiomics extraction for BMD/DMD classification pipeline.

Expected layout:
- data/mri/raw/images/*.nii.gz
- data/mri/raw/masks/*.nii.gz

Output:
- output/mri_radiomics_features.csv

⚠️ NOTE: This extracts features for BMD/DMD proxy classification.
Real clinical classification requires confirmed BMD/DMD patient data.
"""

from pathlib import Path
import logging
import pandas as pd
from radiomics import featureextractor


logging.getLogger("radiomics").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DATA_DIR = Path("data") / "mri" / "raw"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"
OUTPUT_DIR = Path("output")


def normalize_image_key(path: Path):
    """
    Example:
    01_calf_in_phase.nii.gz -> 01_calf
    """
    name = path.name.lower()
    for suffix in [".nii.gz", ".nii"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    parts = name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else name


def normalize_mask_key(path: Path):
    """
    Example:
    01_calf_mask_muscles.nii.gz -> 01_calf
    """
    name = path.name.lower()
    for suffix in [".nii.gz", ".nii"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    marker = "_mask_"
    if marker in name:
        return name.split(marker)[0]
    return name


def mask_priority(path: Path):
    n = path.name.lower()
    if "mask_muscles" in n:
        return 0
    if "mask_whole_muscle_sat" in n:
        return 1
    return 2


def build_pairs():
    image_files = sorted(
        list(IMAGES_DIR.glob("*.nii.gz")) + list(IMAGES_DIR.glob("*.nii"))
    )
    mask_files = sorted(
        list(MASKS_DIR.glob("*.nii.gz")) + list(MASKS_DIR.glob("*.nii"))
    )

    mask_map = {}
    for mask_path in mask_files:
        key = normalize_mask_key(mask_path)
        mask_map.setdefault(key, []).append(mask_path)

    pairs = []
    skipped = []
    for image_path in image_files:
        key = normalize_image_key(image_path)
        candidates = mask_map.get(key, [])
        if not candidates:
            skipped.append(image_path.name)
            continue
        best_mask = sorted(candidates, key=mask_priority)[0]
        pairs.append((image_path, best_mask))

    return pairs, skipped


def extract_features_for_pair(extractor, image_path: Path, mask_path: Path):
    features = extractor.execute(str(image_path), str(mask_path))
    feature_row = {
        key: float(value)
        for key, value in features.items()
        if key.startswith("original_")
    }
    feature_row["image_name"] = image_path.name
    feature_row["mask_name"] = mask_path.name
    feature_row["case_key"] = normalize_image_key(image_path)
    return feature_row


def main():
    if not IMAGES_DIR.exists() or not MASKS_DIR.exists():
        print("Required MRI folders not found:")
        print(f"- {IMAGES_DIR}")
        print(f"- {MASKS_DIR}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    extractor = featureextractor.RadiomicsFeatureExtractor()

    pairs, skipped = build_pairs()
    print(f"Found {len(pairs)} image-mask pairs.")
    print(f"Skipped {len(skipped)} images without compatible mask.")

    all_rows = []
    for idx, (image_path, mask_path) in enumerate(pairs, start=1):
        if idx % 10 == 0 or idx == 1:
            print(f"Processing {idx}/{len(pairs)}: {image_path.name}")
        try:
            all_rows.append(extract_features_for_pair(extractor, image_path, mask_path))
        except Exception as exc:
            logger.warning("Failed %s: %s", image_path.name, exc)

    if not all_rows:
        print("No features extracted.")
        return

    df = pd.DataFrame(all_rows)
    front_cols = ["case_key", "image_name", "mask_name"]
    front_cols = [c for c in front_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    output_csv = OUTPUT_DIR / "mri_radiomics_features.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved MRI features to {output_csv}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    if skipped:
        skipped_path = OUTPUT_DIR / "mri_unpaired_images.txt"
        skipped_path.write_text("\n".join(skipped), encoding="utf-8")
        print(f"Saved unpaired MRI image list to {skipped_path}")


if __name__ == "__main__":
    main()
