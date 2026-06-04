"""
Extract REAL PyRadiomics features from ULTRASOUND_LABELD_1 images
that exist in final_ultrasound_dataset.csv (FSHD cohort with real PNG files).

Output:
  output/pyradiomics_labeled1_features.csv  (features + image_path + disease labels)
"""

from pathlib import Path
import logging

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("radiomics").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_CSV = PROJECT_ROOT / "output" / "final_ultrasound_dataset.csv"
OUTPUT_CSV = PROJECT_ROOT / "output" / "pyradiomics_labeled1_features.csv"
MAX_IMAGES = None  # set e.g. 500 for quick test


def setup_extractor():
    settings = {"force2D": True, "force2Ddimension": 0, "label": 1}
    return featureextractor.RadiomicsFeatureExtractor(**settings)


def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        from PIL import Image
        img = np.array(Image.open(path).convert("L"))
    return img.astype(np.float32)


def make_mask(img):
    _, mask = cv2.threshold(
        img.astype(np.uint8) if img.max() > 1 else (img * 255).astype(np.uint8),
        max(img.mean() * 0.5, 1),
        1,
        cv2.THRESH_BINARY,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(np.uint8)


def extract_one(extractor, image_path: Path):
    img = load_image(image_path)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    mask = make_mask(img)
    if int(mask.sum()) == 0:
        return None
    image_sitk = sitk.GetImageFromArray(img)
    mask_sitk = sitk.GetImageFromArray(mask)
    features = extractor.execute(image_sitk, mask_sitk)
    return {
        k: float(v)
        for k, v in features.items()
        if k.startswith("original_")
    }


def main():
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing {DATASET_CSV}")

    df = pd.read_csv(DATASET_CSV)
    subset = df[df["dataset_source"] == "ULTRASOUND_LABELD_1"].copy()
    subset = subset.drop_duplicates(subset=["image_path"])

    rows = []
    for _, row in subset.iterrows():
        p = PROJECT_ROOT / row["image_path"]
        if p.exists():
            rows.append(row)

    if not rows:
        print("No existing LABELD_1 image files found.")
        return

    meta = pd.DataFrame(rows)
    if MAX_IMAGES:
        meta = meta.head(MAX_IMAGES)

    print(f"Extracting PyRadiomics from {len(meta)} real ultrasound images...")
    extractor = setup_extractor()
    all_features = []
    failed = 0

    for _, row in tqdm(meta.iterrows(), total=len(meta)):
        path = PROJECT_ROOT / row["image_path"]
        try:
            feats = extract_one(extractor, path)
            if feats is None:
                failed += 1
                continue
            feats["image_path"] = row["image_path"]
            feats["patient_id"] = row["patient_id"]
            feats["disease"] = row["disease"]
            feats["severity_label"] = row.get("severity_label", "")
            feats["dataset_source"] = row["dataset_source"]
            all_features.append(feats)
        except Exception as exc:
            failed += 1
            logger.warning("Failed %s: %s", path.name, exc)

    if not all_features:
        print("No features extracted.")
        return

    out = pd.DataFrame(all_features)
    front = ["image_path", "patient_id", "disease", "severity_label", "dataset_source"]
    out = out[front + [c for c in out.columns if c not in front]]
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(out)} rows to {OUTPUT_CSV}")
    print(f"Failed: {failed}")
    print(f"Feature columns: {len(out.columns) - len(front)}")


if __name__ == "__main__":
    main()
