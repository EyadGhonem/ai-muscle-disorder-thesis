"""
Final thesis evaluation (ultrasound-only, multi-class muscle disease).

Uses: output/final_ultrasound_dataset.csv
- Patient-level train/test split (reduces leakage)
- Radiomics+ML: SVM, Random Forest, Logistic Regression
- Deep Learning baseline: MLP neural network on same features
- Per-class metrics, confusion matrices, feature importance
- Dataset-source bias analysis (ULTRASOUND_LABELD_1 vs ULTRASOUND_LABELD_2)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from thesis_metrics import compute_multiclass_metrics
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FINAL_DIR = OUTPUT_DIR / "thesis_final"
DATASET_PATH = OUTPUT_DIR / "final_ultrasound_dataset.csv"
PYRADIOMICS_PATH = OUTPUT_DIR / "pyradiomics_labeled1_features.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

METADATA_COLS = {
    "image_path",
    "patient_id",
    "disease",
    "severity",
    "severity_label",
    "dataset_source",
}


def load_dataset():
    using_pyradiomics = False
    if PYRADIOMICS_PATH.exists():
        df = pd.read_csv(PYRADIOMICS_PATH)
        using_pyradiomics = True
        print(f"Using REAL PyRadiomics features: {PYRADIOMICS_PATH.name}")
    elif DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
        print(
            "WARNING: Using final_ultrasound_dataset.csv (mixed custom + synthetic features). "
            "Run: python extract_pyradiomics_labeled1.py for real PyRadiomics."
        )
    else:
        raise FileNotFoundError(
            f"Missing {DATASET_PATH} and {PYRADIOMICS_PATH}"
        )

    df = df.dropna(subset=["disease", "patient_id"])
    df["disease"] = df["disease"].astype(str).str.strip()

    # Remove unknown / empty disease labels
    invalid = {"", "Unknown", "unknown", "nan", "NAN", "NaN"}
    df = df[~df["disease"].isin(invalid)]
    df = df[df["disease"].notna()]

    meta = set(METADATA_COLS) | {"severity_label"}
    feature_cols = [c for c in df.columns if c not in meta and not c.startswith("original_diagnostic")]
    if using_pyradiomics:
        feature_cols = [c for c in feature_cols if c.startswith("original_") or c not in meta]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    valid = ~X.isna().all(axis=1)
    df = df.loc[valid].reset_index(drop=True)
    X = X.loc[valid].reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(df["disease"])

    return df, X, y, le, feature_cols


def patient_split(df, X, y):
    groups = df["patient_id"].astype(str)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return train_idx, test_idx


def build_models():
    return {
        "SVM": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=1.0, class_weight="balanced")),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "MLP Neural Network": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        max_iter=80,
                        random_state=RANDOM_STATE,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }


def evaluate_by_source(test_df, y_true, y_pred, le):
    from sklearn.metrics import accuracy_score, f1_score
    rows = []
    for source in sorted(test_df["dataset_source"].unique()):
        mask = test_df["dataset_source"] == source
        if mask.sum() < 5:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        labels = list(range(len(le.classes_)))
        rows.append(
            {
                "dataset_source": source,
                "n_samples": int(mask.sum()),
                "accuracy": float(accuracy_score(yt, yp)),
                "f1_macro": float(
                    f1_score(yt, yp, average="macro", zero_division=0, labels=labels)
                ),
            }
        )
    return pd.DataFrame(rows)


def save_feature_importance(model, feature_cols, class_names, out_path):
    rf = model.named_steps["model"]
    if not hasattr(rf, "feature_importances_"):
        return

    imp = pd.DataFrame(
        {"feature": feature_cols, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)
    imp.to_csv(out_path, index=False)

    md = ["# Top radiomics / imaging features (Random Forest)", ""]
    for _, row in imp.head(15).iterrows():
        md.append(f"- `{row['feature']}`: {row['importance']:.4f}")
    out_path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")


def main():
    print("=" * 70)
    print("FINAL THESIS EVALUATION — ULTRASOUND MULTI-CLASS")
    print("=" * 70)

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    df, X, y, le, feature_cols = load_dataset()
    train_idx, test_idx = patient_split(df, X, y)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Samples: {len(df)} | Patients: {df['patient_id'].nunique()}")
    print(f"Classes: {list(le.classes_)}")
    print(f"Train: {len(train_idx)} | Test: {len(test_idx)} (patient-level split)")

    all_rows = []
    per_class_reports = {}
    labels = list(range(len(le.classes_)))

    models = build_models()
    for model_name, pipeline in models.items():
        family = (
            "Deep Learning"
            if "MLP" in model_name or "Neural" in model_name
            else "Radiomics+ML"
        )
        print(f"\nTraining: {model_name} ({family})...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = compute_multiclass_metrics(y_test, y_pred, len(labels))
        row = {
            "task": "disease_classification",
            "model_family": family,
            "model": model_name,
            **metrics,
        }
        all_rows.append(row)

        report = classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=le.classes_,
            zero_division=0,
            output_dict=True,
        )
        per_class_reports[model_name] = report

        print(
            f"  acc={metrics['accuracy']:.4f} | "
            f"f1_macro={metrics['f1_macro']:.4f} | f1_weighted={metrics['f1_weighted']:.4f}"
        )

        if model_name == "Random Forest":
            save_feature_importance(
                pipeline,
                feature_cols,
                le.classes_,
                FINAL_DIR / "feature_importance.csv",
            )

        # Confusion matrix CSV
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        cm_df.to_csv(FINAL_DIR / f"confusion_matrix_{model_name.replace(' ', '_')}.csv")

        # Dataset-source bias on test set
        bias_df = evaluate_by_source(test_df, y_test, y_pred, le)
        if not bias_df.empty:
            bias_df.to_csv(
                FINAL_DIR / f"dataset_source_metrics_{model_name.replace(' ', '_')}.csv",
                index=False,
            )

    # CNN models (ResNet50, DenseNet121, EfficientNetB0) — same metrics format
    print("\n" + "-" * 70)
    print("Evaluating CNN models (if trained)...")
    print("-" * 70)
    try:
        from evaluate_cnn_models import evaluate_all_cnn_rows

        cnn_rows = evaluate_all_cnn_rows()
        if cnn_rows:
            all_rows.extend(cnn_rows)
            print(f"Added {len(cnn_rows)} CNN result rows.")
    except Exception as exc:
        print(f"CNN evaluation skipped: {exc}")

    # Save combined results
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(FINAL_DIR / "model_comparison.csv", index=False)

    cols = ["task", "model_family", "model", "accuracy", "f1_macro"]
    if "f1_weighted" in results_df.columns:
        cols.append("f1_weighted")
    comparison = results_df[cols].copy()
    comparison["accuracy_pct"] = (comparison["accuracy"] * 100).round(2).astype(str) + "%"
    comparison["f1_macro"] = comparison["f1_macro"].round(4)
    comparison.to_csv(FINAL_DIR / "thesis_final_comparison_table.csv", index=False)

    # Markdown table for thesis Chapter 4
    md = [
        "# Final comparison: Radiomics+ML vs Deep Learning (Ultrasound)",
        "",
        "Diseases (tabular ML/MLP): " + ", ".join(le.classes_),
        "",
        "Split: patient-level (20% patients) for disease task; CNN tasks use held-out validation from image training.",
        "",
        "| Task | Model family | Model | Accuracy | F1 (macro) | F1 (weighted) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for _, r in results_df.iterrows():
        task = r.get("task", "disease_classification")
        f1w = r.get("f1_weighted", r.get("f1_macro", 0))
        md.append(
            f"| {task} | {r['model_family']} | {r['model']} | {r['accuracy']*100:.2f}% | "
            f"{r['f1_macro']:.4f} | {f1w:.4f} |"
        )
    (FINAL_DIR / "thesis_final_comparison_table.md").write_text(
        "\n".join(md), encoding="utf-8"
    )

    (FINAL_DIR / "per_class_metrics.json").write_text(
        json.dumps(per_class_reports, indent=2), encoding="utf-8"
    )

    (FINAL_DIR / "class_labels.json").write_text(
        json.dumps({"classes": list(le.classes_)}, indent=2), encoding="utf-8"
    )

    # Scope / limitations document
    scope = f"""# Thesis scope and limitations (ultrasound-only)

## Scope
- Modality: **ultrasound only** (MRI excluded from final study).
- Task: **multi-class muscle disease classification**.
- Classes in this run: {', '.join(le.classes_)}.
- DMD/BMD comparison not performed (data unavailable); study uses available labeled diseases.

## Features
- Tabular radiomics/texture features from `final_ultrasound_dataset.csv`
  (shape, intensity, GLCM, gradient, morphology descriptors).
- Ultrasound ROI: automatic foreground mask approximation during feature extraction pipeline.

## Validation
- **Patient-level** train/test split ({int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%).
- Metrics: accuracy, macro/weighted precision, recall, F1, confusion matrix.
- Dataset-source bias report saved per model under `thesis_final/`.

## Model families compared
1. **Radiomics + ML**: SVM, Random Forest, Logistic Regression
2. **Deep Learning (features)**: MLP neural network on radiomics feature vectors
3. **Deep Learning (CNN)**: ResNet50, DenseNet121, EfficientNetB0 on ultrasound images (binary + severity tasks)

## Notes for examiner
- High performance may partly reflect correlation between dataset source and disease class;
  see `dataset_source_metrics_*.csv` and discuss in thesis limitations.
- Progression over time not modeled (single time-point classification).
"""
    (FINAL_DIR / "THESIS_SCOPE_AND_LIMITATIONS.md").write_text(scope, encoding="utf-8")

    # Sync to legacy paths used by evaluate_thesis_models.py readers
    results_df.to_csv(OUTPUT_DIR / "thesis_metrics_detailed.csv", index=False)
    comparison_export = results_df[["model_family", "model", "accuracy", "f1_macro"]].copy()
    comparison_export.rename(columns={"f1_macro": "f1_score"}, inplace=True)
    comparison_export["accuracy"] = comparison_export["accuracy"].apply(
        lambda x: f"{x * 100:.2f}%"
    )
    comparison_export["f1_score"] = comparison_export["f1_score"].map(lambda x: f"{x:.4f}")
    comparison_export.to_csv(OUTPUT_DIR / "thesis_final_comparison_table.csv", index=False)

    print("\n" + "=" * 70)
    print("DONE — outputs in:", FINAL_DIR)
    print("=" * 70)
    print(comparison_export.to_string(index=False))


if __name__ == "__main__":
    main()
