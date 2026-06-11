"""
run_roc_analysis.py
-------------------
One-vs-rest ROC (Receiver Operating Characteristic) curve analysis for all ML models.

Input:
  output/baseline_and_advanced_models/trained_models.pkl  — fitted models + scaler
  output/final_ultrasound_dataset.csv                     — radiomics features

Processing:
  1. Recreates the patient-level test split (GroupShuffleSplit, test_size=0.2,
     random_state=42) — identical split to training.
  2. Binarises the multi-class labels using a one-vs-rest scheme.
  3. For each ML model that supports ``predict_proba``, computes per-class ROC
     curves and their AUC scores.
  4. Computes the macro-average AUC (unweighted mean across all disease classes).

Output (saved to output/aplus/run_roc_analysis/):
  roc_<ModelName>.png    — per-class ROC curves with AUC legend
  roc_summary_bar.png    — bar chart comparing macro-AUC across all models

Run: python scripts/run_roc_analysis.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# ── paths ────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_PATH = os.path.join(BASE, "output", "baseline_and_advanced_models", "trained_models.pkl")
CSV_PATH = os.path.join(BASE, "output", "final_ultrasound_dataset.csv")
OUT_ROOT = os.path.join(BASE, "output", "aplus", "run_roc_analysis")
os.makedirs(OUT_ROOT, exist_ok=True)

PALETTE  = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]
EXCLUDE  = {"image_path", "patient_id", "disease", "severity_label",
            "dataset_source"}

PROBA_MODELS = {"SVM", "Random Forest", "Gradient Boosting", "XGBoost",
                "LightGBM", "CatBoost", "Extra Trees", "Logistic Regression"}

plt.style.use("seaborn-v0_8-whitegrid")

# ── load ─────────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(CSV_PATH)
df = df[df["disease"].notna() & (df["disease"] != "NAN")].copy()

feature_cols = [c for c in df.columns if c not in EXCLUDE]
X      = df[feature_cols].fillna(df[feature_cols].median())
groups = df["patient_id"].astype(str)

with open(PKL_PATH, "rb") as f:
    bundle = pickle.load(f)
models = bundle["models"]
scaler = bundle["scaler"]
le     = bundle["label_encoder"]
classes = le.classes_
n_cls   = len(classes)

mask   = np.isin(df["disease"].values, classes)
X      = X[mask]
y_raw  = df["disease"].values[mask]
groups = groups[mask]
y      = le.transform(y_raw)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, test_idx = next(gss.split(X, y, groups))
X_test = scaler.transform(X.iloc[test_idx].values)
y_test = y[test_idx]

y_bin = label_binarize(y_test, classes=list(range(n_cls)))

print(f"Test set: {len(y_test)} samples | Classes: {list(classes)}")

# ── ROC per model ─────────────────────────────────────────────────────────────
macro_aucs = {}

for model_name, model in models.items():
    if model_name not in PROBA_MODELS:
        continue
    if not hasattr(model, "predict_proba"):
        print(f"  [{model_name}] no predict_proba — skipping")
        continue

    print(f"  ROC for {model_name} …")
    try:
        proba = model.predict_proba(X_test)          # (n, n_cls)
    except Exception as e:
        print(f"    Error: {e}")
        continue

    fig, ax = plt.subplots(figsize=(8, 6))
    aucs_per_class = []

    for i, cls_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        roc_auc      = auc(fpr, tpr)
        aucs_per_class.append(roc_auc)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2,
                label=f"{cls_name} (AUC={roc_auc:.3f})")

    macro_auc = float(np.mean(aucs_per_class))
    macro_aucs[model_name] = macro_auc

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {model_name}\n(macro-avg AUC = {macro_auc:.3f})")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    safe = model_name.replace(" ", "_")
    fig.savefig(os.path.join(OUT_ROOT, f"roc_{safe}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    macro AUC={macro_auc:.3f}")

# ── summary bar chart ─────────────────────────────────────────────────────────
if macro_aucs:
    names  = list(macro_aucs.keys())
    values = [macro_aucs[n] for n in names]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Macro-average AUC")
    ax.set_title("Macro-Average AUC per ML Model (One-vs-Rest)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "roc_summary_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\nMacro AUC summary:")
    for n, v in sorted(macro_aucs.items(), key=lambda x: -x[1]):
        print(f"  {n:<25} {v:.4f}")

print(f"\nAll ROC outputs saved to: {OUT_ROOT}")
