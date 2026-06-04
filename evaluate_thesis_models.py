"""
Legacy binary evaluation (small labeled ultrasound set).

For the main thesis (multi-class, patient-level split, FSHD/IBM/DM/PM/Normal),
run instead:

    python run_final_thesis_evaluation.py

Outputs go to output/thesis_final/
"""

from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

OUTPUT_DIR = Path("output")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_binary_labels(labels_series, name):
    labels = pd.to_numeric(labels_series, errors="coerce")
    if labels.isna().any():
        raise ValueError(f"{name}: label column contains non-numeric values.")
    unique_values = set(labels.astype(int).unique().tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"{name}: expected binary labels in {{0,1}}, got {sorted(unique_values)}")
    return labels.astype(int)


def evaluate_predictions(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def load_radiomics_dataset(features_path, labels_path, dataset_name):
    if not features_path.exists():
        print(f"⚠️  {dataset_name}: missing features file {features_path}")
        return None
    if not labels_path.exists():
        print(f"⚠️  {dataset_name}: missing labels file {labels_path}")
        return None

    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    if "image_name" not in features_df.columns:
        print(f"⚠️  {dataset_name}: features file must contain image_name column")
        return None
    if "image_name" not in labels_df.columns or "label" not in labels_df.columns:
        print(f"⚠️  {dataset_name}: labels file must contain image_name and label columns")
        return None

    merged = features_df.merge(labels_df[["image_name", "label"]], on="image_name", how="inner")
    if merged.empty:
        print(f"⚠️  {dataset_name}: no overlap between features and labels")
        return None

    y = ensure_binary_labels(merged["label"], dataset_name)
    drop_cols = {"image_name", "label", "mask_name", "case_key"}
    feature_cols = [c for c in merged.columns if c not in drop_cols]
    if not feature_cols:
        print(f"⚠️  {dataset_name}: no usable feature columns found")
        return None

    X = merged[feature_cols].apply(pd.to_numeric, errors="coerce")
    valid_rows = ~X.isna().all(axis=1)
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if len(X) < 10:
        print(f"⚠️  {dataset_name}: too few samples after cleaning ({len(X)})")
        return None
    if y.nunique() < 2:
        print(f"⚠️  {dataset_name}: labels must contain both classes")
        return None

    print(f"✓ {dataset_name}: {len(X)} samples, {X.shape[1]} features")
    return X, y


def train_radiomics_models(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "SVM": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=1.0)),
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
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }

    rows = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        rows.append(
            {
                "dataset": dataset_name,
                "model_family": "Radiomics+ML",
                "model": model_name,
                **metrics,
            }
        )
        print(
            f"  {dataset_name} | {model_name}: "
            f"acc={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}"
        )
    return rows


def evaluate_dl_predictions(predictions_path, labels_path, dataset_name, model_name):
    if not predictions_path.exists() or not labels_path.exists():
        print(f"⚠️  {dataset_name}: missing DL predictions or labels")
        return None

    pred_df = pd.read_csv(predictions_path)
    labels_df = pd.read_csv(labels_path)

    if "image_name" not in pred_df.columns:
        print(f"⚠️  {dataset_name}: predictions must include image_name")
        return None
    if "predicted_class" not in pred_df.columns and "model_score" not in pred_df.columns:
        print(f"⚠️  {dataset_name}: predictions need predicted_class or model_score")
        return None
    if "image_name" not in labels_df.columns or "label" not in labels_df.columns:
        print(f"⚠️  {dataset_name}: labels must include image_name and label")
        return None

    merged = pred_df.merge(labels_df[["image_name", "label"]], on="image_name", how="inner")
    if merged.empty:
        print(f"⚠️  {dataset_name}: no overlap between DL predictions and labels")
        return None

    y_true = ensure_binary_labels(merged["label"], f"{dataset_name} DL")
    if "predicted_class" in merged.columns:
        y_pred = ensure_binary_labels(merged["predicted_class"], f"{dataset_name} DL predicted_class")
    else:
        y_pred = (pd.to_numeric(merged["model_score"], errors="coerce") >= 0.5).astype(int)

    metrics = evaluate_predictions(y_true, y_pred)
    print(
        f"  {dataset_name} | {model_name}: "
        f"acc={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}"
    )
    return {
        "dataset": dataset_name,
        "model_family": "Deep Learning",
        "model": model_name,
        **metrics,
    }


def format_percentage(x):
    return f"{x * 100:.2f}%"


def save_results(rows):
    if not rows:
        print("❌ No evaluation results to save.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    results_df = pd.DataFrame(rows)
    metrics_csv = OUTPUT_DIR / "thesis_metrics_detailed.csv"
    results_df.to_csv(metrics_csv, index=False)

    cm_json = OUTPUT_DIR / "thesis_confusion_matrices.json"
    cm_payload = []
    for _, row in results_df.iterrows():
        cm_payload.append(
            {
                "dataset": row["dataset"],
                "model_family": row["model_family"],
                "model": row["model"],
                "confusion_matrix": row["confusion_matrix"],
            }
        )
    cm_json.write_text(json.dumps(cm_payload, indent=2), encoding="utf-8")

    comparison_df = results_df[["dataset", "model_family", "model", "accuracy", "f1_score"]].copy()
    comparison_df["accuracy"] = comparison_df["accuracy"].apply(format_percentage)
    comparison_df["f1_score"] = comparison_df["f1_score"].map(lambda x: f"{x:.4f}")

    comparison_csv = OUTPUT_DIR / "thesis_final_comparison_table.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    md_lines = [
        "# Thesis Final Comparison Table",
        "",
        "| Dataset | Model Family | Model | Accuracy | F1 |",
        "|---|---|---|---:|---:|",
    ]
    for _, row in comparison_df.iterrows():
        md_lines.append(
            f"| {row['dataset']} | {row['model_family']} | {row['model']} | "
            f"{row['accuracy']} | {row['f1_score']} |"
        )
    comparison_md = OUTPUT_DIR / "thesis_final_comparison_table.md"
    comparison_md.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print("THESIS EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Detailed metrics: {metrics_csv}")
    print(f"Confusion matrices: {cm_json}")
    print(f"Final comparison CSV: {comparison_csv}")
    print(f"Final comparison Markdown: {comparison_md}")
    print("=" * 70)


def main():
    print("=" * 70)
    print("LEGACY BINARY EVALUATION (small ultrasound / MRI sets)")
    print("For multi-class thesis results, run: python run_final_thesis_evaluation.py")
    print("=" * 70)

    all_rows = []

    print("\n[1/3] Radiomics + ML (SVM, Random Forest, Logistic Regression)")
    radiomics_tasks = [
        (
            OUTPUT_DIR / "ultrasound_radiomics_features.csv",
            OUTPUT_DIR / "labels.csv",
            "Ultrasound",
        ),
        (
            OUTPUT_DIR / "mri_radiomics_features.csv",
            OUTPUT_DIR / "mri_labels.csv",
            "MRI",
        ),
    ]
    for features_path, labels_path, dataset_name in radiomics_tasks:
        loaded = load_radiomics_dataset(features_path, labels_path, dataset_name)
        if loaded is None:
            continue
        X, y = loaded
        all_rows.extend(train_radiomics_models(X, y, dataset_name))

    print("\n[2/3] Deep Learning prediction evaluation")
    dl_rows = [
        evaluate_dl_predictions(
            OUTPUT_DIR / "ultrasound_predictions.csv",
            OUTPUT_DIR / "labels.csv",
            "Ultrasound",
            "EfficientNetB0 CNN",
        ),
        evaluate_dl_predictions(
            OUTPUT_DIR / "mri_predictions.csv",
            OUTPUT_DIR / "mri_labels.csv",
            "MRI",
            "3D CNN",
        ),
    ]
    for row in dl_rows:
        if row is not None:
            all_rows.append(row)

    print("\n[3/3] Saving thesis outputs")
    save_results(all_rows)


if __name__ == "__main__":
    main()
