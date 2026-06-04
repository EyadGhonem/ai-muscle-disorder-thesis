"""
Build thesis Table 4.3: tabular ML (5-class disease) + CNN (image severity, same val split).

Run after:
  python train_ultrasound_cnn_models.py --task severity --models resnet50 densenet121 efficientnetb0
  python evaluate_cnn_models.py
  python run_final_thesis_evaluation.py  (optional, refreshes ML rows)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_DIR = PROJECT_ROOT / "output" / "thesis_final"
ML_CSV = FINAL_DIR / "model_comparison.csv"
CNN_CSV = FINAL_DIR / "cnn_model_comparison.csv"
OUT_CSV = FINAL_DIR / "table_4_3_model_comparison.csv"
OUT_MD = FINAL_DIR / "table_4_3_model_comparison.md"

ML_MODELS = [
    ("Radiomics+ML", "Logistic Regression"),
    ("Radiomics+ML", "SVM"),
    ("Radiomics+ML", "Random Forest"),
    ("Deep Learning", "MLP Neural Network"),
]
CNN_MODELS = ["ResNet50", "DenseNet121", "EfficientNetB0"]


def load_ml_rows() -> list[dict]:
    if not ML_CSV.exists():
        raise FileNotFoundError(f"Missing {ML_CSV} — run run_final_thesis_evaluation.py")

    df = pd.read_csv(ML_CSV)
    df = df[df["task"].astype(str).str.contains("disease", case=False, na=False)]
    rows = []
    for family, name in ML_MODELS:
        match = df[df["model"] == name]
        if match.empty:
            continue
        r = match.iloc[0]
        rows.append(
            {
                "model_family": family,
                "model": name,
                "accuracy_pct": f"{float(r['accuracy']) * 100:.2f}%",
                "input": "Radiomics features (5-class disease, patient-level split)",
                "task": "disease_classification",
            }
        )
    return rows


def load_cnn_rows() -> list[dict]:
    rows = []
    if CNN_CSV.exists():
        df = pd.read_csv(CNN_CSV)
        df = df[df["task"] == "severity"]
        for name in CNN_MODELS:
            match = df[df["model"] == name]
            if not match.empty:
                r = match.iloc[0]
                rows.append(
                    {
                        "model_family": "Deep Learning (CNN)",
                        "model": name,
                        "accuracy_pct": f"{float(r['accuracy']) * 100:.2f}%",
                        "input": "Ultrasound images (FSHD severity binary, patient-level val split)",
                        "task": "severity",
                    }
                )
                continue

    # Fallback: training results CSV (best val accuracy during training)
    train_csv = PROJECT_ROOT / "output" / "dl_cnn_training_results.csv"
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        df = df[df["task"] == "severity"]
        display = {
            "resnet50": "ResNet50",
            "densenet121": "DenseNet121",
            "efficientnetb0": "EfficientNetB0",
        }
        for arch, name in display.items():
            if any(r["model"] == name for r in rows):
                continue
            sub = df[df["architecture"] == arch]
            if sub.empty:
                continue
            best = sub.sort_values("best_val_accuracy", ascending=False).iloc[0]
            rows.append(
                {
                    "model_family": "Deep Learning (CNN)",
                    "model": name,
                    "accuracy_pct": f"{float(best['best_val_accuracy']) * 100:.2f}%",
                    "input": "Ultrasound images (FSHD severity; val accuracy from training log)",
                    "task": "severity",
                }
            )
    return rows


def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    table = load_ml_rows() + load_cnn_rows()
    if not table:
        print("No rows for Table 4.3.")
        return

    out = pd.DataFrame(table)
    out.to_csv(OUT_CSV, index=False)

    md = [
        "# Table 4.3 — Model comparison (ultrasound)",
        "",
        "**Note:** Tabular models predict **5-class disease** from radiomics features. "
        "CNN models predict **binary FSHD severity** from images (same patient-level validation "
        "split as `train_ultrasound_cnn_models.py`). Tasks differ by design; compare CNNs only "
        "with each other on the severity row.",
        "",
        "| Model family | Model | Accuracy | Input |",
        "|---|---|---:|---|",
    ]
    for r in table:
        md.append(
            f"| {r['model_family']} | {r['model']} | {r['accuracy_pct']} | {r['input']} |"
        )
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_MD}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
