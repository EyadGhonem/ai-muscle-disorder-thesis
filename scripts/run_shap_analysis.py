"""
SHAP analysis for tree-based ML models.
Outputs: beeswarm, bar, and waterfall plots per model.
Run: python scripts/run_shap_analysis.py
"""
import sys
import subprocess
import importlib

def ensure_shap():
    try:
        import shap
        return shap
    except ImportError:
        print("Installing shap...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "-q"])
        return importlib.import_module("shap")

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

shap = ensure_shap()

# ── paths ────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_PATH  = os.path.join(BASE, "output", "baseline_and_advanced_models", "trained_models.pkl")
CSV_PATH  = os.path.join(BASE, "output", "final_ultrasound_dataset.csv")
OUT_ROOT  = os.path.join(BASE, "output", "aplus", "run_shap_analysis")
os.makedirs(OUT_ROOT, exist_ok=True)

PALETTE   = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]
EXCLUDE   = {"image_path", "patient_id", "disease", "severity_label",
             "dataset_source"}
TREE_MODELS = ["Random Forest", "Gradient Boosting", "Extra Trees"]

plt.style.use("seaborn-v0_8-whitegrid")

# ── load data ────────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(CSV_PATH)
df = df[df["disease"].notna() & (df["disease"] != "NAN")].copy()

feature_cols = [c for c in df.columns if c not in EXCLUDE]
X = df[feature_cols].fillna(df[feature_cols].median())
groups = df["patient_id"].astype(str)

with open(PKL_PATH, "rb") as f:
    bundle = pickle.load(f)

models  = bundle["models"]
scaler  = bundle["scaler"]
le      = bundle["label_encoder"]

# encode labels
y_raw = df["disease"].values
mask  = np.isin(y_raw, le.classes_)
X, y_raw, groups = X[mask], y_raw[mask], groups[mask]
y = le.transform(y_raw)

# patient-level split (same seed as training)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, test_idx = next(gss.split(X, y, groups))
X_test_raw = X.iloc[test_idx]
y_test     = y[test_idx]

X_test_sc = scaler.transform(X_test_raw.values)
X_test_df = pd.DataFrame(X_test_sc, columns=feature_cols)

print(f"Test set: {len(X_test_df)} samples")

# ── SHAP per model ───────────────────────────────────────────────────────────
for model_name in TREE_MODELS:
    if model_name not in models:
        print(f"  [{model_name}] not in pkl — skipping")
        continue

    model = models[model_name]
    print(f"\nRunning SHAP for {model_name} …")
    out_dir = os.path.join(OUT_ROOT, model_name.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)

    # Normalise to a single numpy array regardless of shap version output format:
    # list-of-arrays  → stack to (n, f, c)
    # already 3D array (n, f, c) → keep as-is
    # 2D array (n, f) → keep as-is
    if isinstance(shap_values, list):
        shap_vals = np.stack(shap_values, axis=-1)   # (n, f, c)
    else:
        shap_vals = np.array(shap_values)            # (n, f) or (n, f, c)

    # collapse multi-class SHAP to mean absolute value per feature
    if shap_vals.ndim == 3:
        mean_abs_shap = np.abs(shap_vals).mean(axis=(0, 2))  # shape: (n_features,)
    else:
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)       # shape: (n_features,)

    # 1. Beeswarm (summary plot) — top 20 features
    # pass a 2D slice to shap.summary_plot; use mean across classes for 3D
    if shap_vals.ndim == 3:
        sv_plot = shap_vals.mean(axis=2)   # (n, f) — averaged over classes
    else:
        sv_plot = shap_vals
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv_plot, X_test_df, max_display=20, show=False,
                      color_bar=True)
    plt.title(f"{model_name} — SHAP Beeswarm (top 20 features)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved beeswarm")

    # 2. Bar plot (mean |SHAP|)
    top20_idx        = np.argsort(mean_abs_shap)[-20:]
    feature_cols_arr = np.array(feature_cols)
    top20_feat       = feature_cols_arr[top20_idx].tolist()
    top20_vals       = mean_abs_shap[top20_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top20_feat[::-1], top20_vals[::-1], color=PALETTE[0])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"{model_name} — Mean |SHAP| Importance (top 20)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved bar plot")

    # 3. Waterfall for 3 individual samples (one per dominant class)
    present_classes = [c for c in range(len(le.classes_)) if np.any(y_test == c)]
    sample_classes  = present_classes[:3]

    for cls_idx in sample_classes:
        cls_name    = le.classes_[cls_idx]
        sample_rows = np.where(y_test == cls_idx)[0]
        if len(sample_rows) == 0:
            continue
        row = sample_rows[0]

        if shap_vals.ndim == 3:
            sv_row = shap_vals[row, :, cls_idx]   # (n_features,) for this class
            ev     = explainer.expected_value
            base   = ev[cls_idx] if isinstance(ev, (list, np.ndarray)) else float(ev)
        else:
            sv_row = shap_vals[row]               # (n_features,)
            ev     = explainer.expected_value
            base   = float(ev[0]) if isinstance(ev, (list, np.ndarray)) else float(ev)

        exp = shap.Explanation(
            values        = sv_row,
            base_values   = base,
            data          = X_test_df.iloc[row].values,
            feature_names = feature_cols,
        )
        plt.figure(figsize=(10, 5))
        shap.plots.waterfall(exp, max_display=15, show=False)
        plt.title(f"{model_name} — Waterfall sample (class: {cls_name})", fontsize=11)
        plt.tight_layout()
        safe_cls = cls_name.replace(" ", "_")
        plt.savefig(os.path.join(out_dir, f"shap_waterfall_{safe_cls}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved waterfall for {cls_name}")

print(f"\nAll SHAP outputs saved to: {OUT_ROOT}")
