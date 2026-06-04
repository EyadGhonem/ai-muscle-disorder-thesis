#!/usr/bin/env python3
"""
Extract ultrasound images from PatientData.mat (MATLAB v7.3 / HDF5)
and organize by disease label using PatientImages_PLOS2017.xlsx.

Does NOT modify or delete existing files. Skips PNGs that already exist
unless --force is passed.

Output:
  data/dl_images_extracted_from_mat/{DiseaseFolder}/*.png
  data/processed/mat_extracted_image_labels.csv  (or timestamped if exists)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "myopathy_US"
MAT_PATH = DATA_DIR / "PatientData.mat"
XLSX_PATH = DATA_DIR / "PatientImages_PLOS2017.xlsx"
OUT_IMAGE_ROOT = PROJECT_ROOT / "data" / "dl_images_extracted_from_mat"
OUT_CSV_DEFAULT = PROJECT_ROOT / "data" / "processed" / "mat_extracted_image_labels.csv"
INSPECTION_JSON = PROJECT_ROOT / "data" / "processed" / "mat_dataset_inspection.json"

# Excel diagnosis code -> folder name (no FSHD in this cohort)
DISEASE_FOLDER = {
    "D": "Dermatomyositis",
    "P": "Polymyositis",
    "I": "IBM",
    "N": "Normal",
}

DRB_CODE_TO_LETTER = {1: "D", 2: "I", 3: "N", 4: "P"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def detect_mat_format(path: Path) -> str:
    with open(path, "rb") as f:
        header = f.read(128)
    if b"MATLAB 7.3" in header or header.startswith(b"\x89HDF"):
        return "MAT v7.3 (HDF5) — use h5py"
    return "Legacy MAT — scipy.io.loadmat may work"


def decode_h5_ascii_ref(f: h5py.File, ref) -> str:
    """Decode MATLAB char array stored as uint8 columns in #refs#."""
    obj = f[ref]
    data = np.asarray(obj[()])
    if data.ndim == 2:
        chars = [chr(int(x)) for x in data.flatten() if 32 <= int(x) < 127]
        return "".join(chars).strip()
    if data.size == 1:
        v = int(data.flat[0])
        return chr(v) if 32 <= v < 127 else str(v)
    return str(data)


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Scale image to 0–255 uint8."""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim > 2:
        # Take first slice / channel for display
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Cannot convert shape {img.shape} to 2D")
    if arr.dtype == np.uint8 and arr.max() <= 255 and arr.min() >= 0:
        return arr.astype(np.uint8)
    amin, amax = np.nanmin(arr), np.nanmax(arr)
    if amax <= amin:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - amin) / (amax - amin) * 255.0
    return scaled.clip(0, 255).astype(np.uint8)


def save_png_safe(path: Path, img2d: np.ndarray, force: bool) -> bool:
    """Save PNG; return True if written, False if skipped."""
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img2d).save(path)
    return True


def inspect_mat(path: Path) -> dict:
    """Print and return MAT structure summary."""
    info = {"path": str(path), "format": detect_mat_format(path), "variables": {}}
    log(f"\n=== PatientData.mat ===")
    log(f"Format: {info['format']}")

    with h5py.File(path, "r") as f:
        top_keys = [k for k in f.keys() if not k.startswith("#")]
        log(f"Top-level keys: {top_keys}")
        for key in top_keys:
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                info["variables"][key] = {
                    "type": "Dataset",
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }
                log(f"  {key}: Dataset shape={obj.shape} dtype={obj.dtype}")
            else:
                info["variables"][key] = {"type": "Group"}

        # Key metadata arrays in #refs#
        for ref_key in ("GRb", "DRb", "GHc"):
            if ref_key in f["#refs#"]:
                ds = f["#refs#"][ref_key]
                info["variables"][f"#refs#/{ref_key}"] = {
                    "shape": list(ds.shape),
                    "dtype": str(ds.dtype),
                }
                log(f"  #refs#/{ref_key}: shape={ds.shape} dtype={ds.dtype}")

    return info


def inspect_excel(path: Path) -> dict:
    log(f"\n=== PatientImages_PLOS2017.xlsx ===")
    df = pd.read_excel(path)
    info = {
        "path": str(path),
        "shape": list(df.shape),
        "columns": [str(c) for c in df.columns],
        "head": df.head(10).astype(str).to_dict(orient="records"),
    }
    log(f"Shape: {df.shape}")
    log("Columns:")
    for c in df.columns:
        log(f"  - {c}")
    log("\nFirst 10 rows (abbreviated):")
    print(df.head(10).to_string(max_cols=8))
    return info


def build_excel_lookup(xlsx_path: Path) -> pd.DataFrame:
    """
    One row per unique 2D Image timestamp.
    Diagnosis is consistent per timestamp (verified).
    """
    df = pd.read_excel(xlsx_path)
    pid_col = "Patient Identifier"
    img_col = "2D Image"
    diag_col = next(c for c in df.columns if "Diagnosis" in str(c))
    muscle_col = next(
        (c for c in df.columns if "Muscle" in str(c) and "Strength" not in str(c)),
        None,
    )

    df = df.copy()
    df[pid_col] = df[pid_col].ffill()
    df["image_id"] = df[img_col].apply(
        lambda x: str(int(float(x))) if pd.notna(x) else ""
    )

    agg = {diag_col: "first", pid_col: "first"}
    if muscle_col:
        agg[muscle_col] = "first"

    lk = df.groupby("image_id", as_index=False).agg(agg)
    lk = lk.rename(
        columns={
            diag_col: "diagnosis_code",
            pid_col: "patient_id",
            muscle_col: "muscle" if muscle_col else "muscle",
        }
    )
    if muscle_col is None:
        lk["muscle"] = ""
    lk = lk[lk["image_id"] != ""]
    return lk.set_index("image_id")


def extract_images_and_labels(
    mat_path: Path,
    lookup: pd.DataFrame,
    out_root: Path,
    force: bool,
) -> pd.DataFrame:
    rows = []
    stats = {"written": 0, "skipped_existing": 0, "unknown": 0}

    with h5py.File(mat_path, "r") as f:
        im_ds = f["im"]
        n_images = im_ds.shape[1]
        GRb = f["#refs#"]["GRb"]
        DRb = f["#refs#"]["DRb"][()].flatten()

        log(f"\nExtracting {n_images} images from variable 'im'...")

        for idx in range(n_images):
            ref = im_ds[0, idx]
            if not isinstance(ref, h5py.Reference):
                continue

            img_raw = np.asarray(f[ref][()])
            img_u8 = normalize_to_uint8(img_raw)

            # Timestamp ID from GRb
            try:
                image_id = decode_h5_ascii_ref(f, GRb[0, idx])
            except Exception:
                image_id = ""

            confidence = "unknown"
            diagnosis_code = ""
            patient_id = ""
            disease_folder = "Unknown"
            match_method = ""

            if image_id and image_id in lookup.index:
                row = lookup.loc[image_id]
                diagnosis_code = str(row["diagnosis_code"]).strip()
                patient_id = str(row["patient_id"]).strip()
                match_method = "excel_2D_Image_timestamp"
                confidence = "high"

                if diagnosis_code in DISEASE_FOLDER:
                    disease_folder = DISEASE_FOLDER[diagnosis_code]
                else:
                    disease_folder = "Unknown"
                    confidence = "medium_unmapped_code"

                # Cross-check with DRb
                drb_code = DRB_CODE_TO_LETTER.get(int(DRb[idx]))
                if drb_code and drb_code != diagnosis_code:
                    confidence = "medium_excel_drb_mismatch"
            elif image_id:
                match_method = "timestamp_not_in_excel"
                stats["unknown"] += 1
            else:
                match_method = "no_timestamp"
                stats["unknown"] += 1

            # Filename: patient_timestamp_idx.png
            safe_pid = patient_id.replace("/", "_") or "nopatient"
            fname = f"{safe_pid}_{image_id or 'noid'}_idx{idx:04d}.png"
            out_path = out_root / disease_folder / fname

            written = save_png_safe(out_path, img_u8, force=force)
            if written:
                stats["written"] += 1
            else:
                stats["skipped_existing"] += 1

            rows.append(
                {
                    "image_path": str(out_path.relative_to(PROJECT_ROOT)),
                    "patient_id": patient_id,
                    "image_id": image_id,
                    "disease_label": disease_folder,
                    "diagnosis_code": diagnosis_code,
                    "drb_code": DRB_CODE_TO_LETTER.get(int(DRb[idx]), ""),
                    "source_file": str(mat_path.name),
                    "original_mat_variable": "im",
                    "original_index": idx,
                    "confidence_of_label_matching": confidence,
                    "match_method": match_method,
                    "png_written": written,
                }
            )

            if (idx + 1) % 500 == 0:
                log(f"  ... {idx + 1}/{n_images}")

    log(f"\nPNG written: {stats['written']}, skipped (exist): {stats['skipped_existing']}")
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, out_root: Path) -> None:
    log("\n=== EXTRACTION SUMMARY ===")
    log(f"Total records: {len(df)}")
    log("\nImages per disease folder:")
    for folder in sorted(df["disease_label"].unique()):
        n = (df["disease_label"] == folder).sum()
        on_disk = len(list((out_root / folder).glob("*.png"))) if (out_root / folder).exists() else 0
        log(f"  {folder}: {n} labeled ({on_disk} PNG files on disk)")

    unk = (df["disease_label"] == "Unknown").sum()
    log(f"\nUnknown labels: {unk}")

    log("\nConfidence breakdown:")
    print(df["confidence_of_label_matching"].value_counts().to_string())

    log("\nExample matched rows (high confidence):")
    hi = df[df["confidence_of_label_matching"] == "high"].head(5)
    print(hi[["image_path", "patient_id", "image_id", "disease_label"]].to_string(index=False))

    if (df["confidence_of_label_matching"].str.contains("mismatch")).any():
        log("\nWARNING: Some rows have Excel/DRb mismatch — review CSV.")

    log("\n=== DEEP LEARNING SUITABILITY ===")
    log("Classes in this MAT cohort: Normal, Dermatomyositis, Polymyositis, IBM (NO FSHD).")
    log("Labels: REAL clinical codes from Excel (D/P/I/N), matched by 2D Image timestamp.")
    counts = df["disease_label"].value_counts()
    for cls, n in counts.items():
        log(f"  {cls}: {n} images")
    min_n = counts.min() if len(counts) else 0
    if min_n >= 400:
        log("Verdict: Suitable for CNN training per class (reasonable counts).")
    elif min_n >= 100:
        log("Verdict: Usable for CNN with augmentation / class weights (moderate counts).")
    else:
        log("Verdict: Small classes may need augmentation; still better than synthetic features only.")
    log("Also suitable for real radiomics + ML (recommended baseline).")


def resolve_csv_path(force: bool) -> Path:
    if not OUT_CSV_DEFAULT.exists() or force:
        return OUT_CSV_DEFAULT
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    alt = OUT_CSV_DEFAULT.with_name(f"mat_extracted_image_labels_{ts}.csv")
    log(f"CSV already exists — writing to {alt.name} (use --force to overwrite default)")
    return alt


def main():
    parser = argparse.ArgumentParser(description="Extract myopathy US images from MAT")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing PNG/CSV files",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect MAT/Excel, do not extract",
    )
    args = parser.parse_args()

    if not MAT_PATH.exists():
        log(f"ERROR: Missing {MAT_PATH}")
        sys.exit(1)
    if not XLSX_PATH.exists():
        log(f"ERROR: Missing {XLSX_PATH}")
        sys.exit(1)

    OUT_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_CSV_DEFAULT.parent.mkdir(parents=True, exist_ok=True)

    # Create disease folders (no FSHD — not in this dataset)
    for folder in list(DISEASE_FOLDER.values()) + ["Unknown"]:
        (OUT_IMAGE_ROOT / folder).mkdir(parents=True, exist_ok=True)

    mat_info = inspect_mat(MAT_PATH)
    xlsx_info = inspect_excel(XLSX_PATH)

    INSPECTION_JSON.parent.mkdir(parents=True, exist_ok=True)
    INSPECTION_JSON.write_text(
        json.dumps({"mat": mat_info, "excel": xlsx_info}, indent=2, default=str),
        encoding="utf-8",
    )
    log(f"\nInspection saved: {INSPECTION_JSON}")

    if args.inspect_only:
        return

    lookup = build_excel_lookup(XLSX_PATH)
    log(f"\nExcel lookup: {len(lookup)} unique image timestamps")

    df = extract_images_and_labels(MAT_PATH, lookup, OUT_IMAGE_ROOT, force=args.force)

    csv_path = resolve_csv_path(args.force)
    df.to_csv(csv_path, index=False)
    log(f"\nLabels CSV: {csv_path}")

    print_summary(df, OUT_IMAGE_ROOT)


if __name__ == "__main__":
    main()
