"""
Import and normalize MRI segmentation label-map JSON files.

This script:
1) Copies label dictionaries into the project under data/mri/label_maps
2) Normalizes label IDs to integer keys and enforces a stable structure
3) Exports a combined CSV lookup for downstream preprocessing/training
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_labels(payload: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    if "labels" not in payload or not isinstance(payload["labels"], dict):
        raise ValueError(f"'labels' dictionary missing in {source_path}")

    region = payload.get("anatomical_region", source_path.stem.replace("_segmentation_labels", ""))
    description = payload.get("description", "")

    normalized_labels: Dict[str, Dict[str, Any]] = {}
    seen_names = set()

    for raw_id, label_info in sorted(payload["labels"].items(), key=lambda kv: int(kv[0])):
        try:
            label_id = int(raw_id)
        except ValueError as exc:
            raise ValueError(f"Non-integer label id '{raw_id}' in {source_path}") from exc

        if not isinstance(label_info, dict):
            raise ValueError(f"Invalid label entry for id {raw_id} in {source_path}")

        name = label_info.get("name", f"label_{label_id}")
        if name in seen_names:
            raise ValueError(f"Duplicate label name '{name}' in {source_path}")
        seen_names.add(name)

        color = label_info.get("color", [0, 0, 0])
        if not (isinstance(color, list) and len(color) == 3):
            raise ValueError(f"Invalid RGB color for label {label_id} in {source_path}")

        normalized_labels[str(label_id)] = {
            "name": name,
            "description": label_info.get("description", ""),
            "abbreviation": label_info.get("abbreviation", ""),
            "color": [int(c) for c in color],
        }

    return {
        "anatomical_region": region,
        "description": description,
        "source_file": source_path.name,
        "labels": normalized_labels,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _write_lookup_csv(path: Path, datasets: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "anatomical_region",
        "label_id",
        "name",
        "abbreviation",
        "description",
        "color_r",
        "color_g",
        "color_b",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ds in datasets:
            dataset_name = ds["source_file"].replace(".json", "")
            for label_id, info in ds["labels"].items():
                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "anatomical_region": ds["anatomical_region"],
                        "label_id": int(label_id),
                        "name": info["name"],
                        "abbreviation": info.get("abbreviation", ""),
                        "description": info.get("description", ""),
                        "color_r": info["color"][0],
                        "color_g": info["color"][1],
                        "color_b": info["color"][2],
                    }
                )


def integrate_label_maps(input_files: List[Path], output_dir: Path, lookup_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = []

    for path in input_files:
        payload = _load_json(path)
        normalized = _normalize_labels(payload, path)

        out_name = path.name
        out_path = output_dir / out_name
        _write_json(out_path, normalized)
        datasets.append(normalized)
        print(f"Imported: {path} -> {out_path}")

    _write_lookup_csv(lookup_path, datasets)
    print(f"Created lookup table: {lookup_path}")
    print("\nNext:")
    print("- Put MRI images in data/mri/raw/images and segmentation masks in data/mri/raw/masks")
    print("- Ensure mask voxel values match label_id values from these JSON files")
    print("- For binary radiomics (muscle vs background), remap selected labels to 1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate MRI segmentation label JSONs")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSON label-map files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/mri/label_maps",
        help="Destination folder inside project",
    )
    parser.add_argument(
        "--lookup-path",
        default="data/mri/metadata/combined_label_lookup.csv",
        help="Path to write merged CSV label lookup",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]
    out_dir = Path(args.output_dir)
    integrate_label_maps(input_paths, out_dir, Path(args.lookup_path))
