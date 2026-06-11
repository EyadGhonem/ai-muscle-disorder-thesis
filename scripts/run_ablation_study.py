"""
run_ablation_study.py
---------------------
Feature ablation study for the radiomics feature set.

Systematically removes one feature group at a time to quantify each group's
contribution to classification accuracy, providing evidence for feature
selection decisions in the thesis.

Feature groups:
  - first_order : mean, std, min, max, median, Q25, Q75, skewness, kurtosis, entropy
  - texture     : GLCM (glcm_* columns)
  - shape       : area, perimeter, circularity, aspect_ratio, extent, solidity,
                  equivalent_diameter
  - gradient    : gradient_* columns

Five configurations evaluated (all using Random Forest, balanced class weight):
  1. All features (baseline)
  2. Without gradient features
  3. Without texture (GLCM) features
  4. Without shape features
  5. Without first-order statistics

For each configuration the same patient-level GroupShuffleSplit (test_size=0.2,
random_state=42) is applied, and accuracy + macro F1 are reported.

Input:
  output/final_ultrasound_dataset.csv  — radiomics features with disease labels

Output (saved to output/aplus/run_ablation_study/):
  ablation_results.csv    — per-configuration accuracy and macro F1
  ablation_bar.png        — bar chart comparing accuracy across configurations

Run: python scripts/run_ablation_study.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ── paths ────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE, "output", "final_ultrasound_dataset.csv")
OUT_ROOT = os.path.join(BASE, "output", "aplus", "run_ablation_study")
os.makedirs(OUT_ROOT, exist_ok=True)

EXCLUDE_COLS = {"image_path", "patient_id", "disease", "severity_label", "dataset_source"}
PALETTE      = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]
plt.style.use("seaborn-v0_8-whitegrid")

# ── load ─────────────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(CSV_PATH)
df = df[df["disease"].notna() & (df["disease"] != "NAN")].copy()

all_feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

# ── define feature groups ─────────────────────────────────────────────────────
first_order_prefixes = ("mean_", "std_", "min_", "max_", "median_",
                        "q25_", "q75_")
first_order_exact    = {"skewness", "kurtosis", "entropy",
                        "mean_intensity", "std_intensity", "min_intensity",
                        "max_intensity", "median_intensity",
                        "q25_intensity", "q75_intensity"}

texture_cols  = [c for c in all_feature_cols if c.startswith("glcm_")]
shape_cols    = [c for c in all_feature_cols if c in {
                    "area", "perimeter", "circularity", "aspect_ratio",
                    "extent", "solidity", "equivalent_diameter"}]
gradient_cols = [c for c in all_feature_cols if c.startswith("gradient_")]
first_order_cols = [c for c in all_feature_cols
                    if c.startswith(first_order_prefixes) or c in first_order_exact]

print(f"Feature groups:")
print(f"  All features  : {len(all_feature_cols)}")
print(f"  First-order   : {len(first_order_cols)} — {first_order_cols}")
print(f"  Texture (GLCM): {len(texture_cols)} — {texture_cols}")
print(f"  Shape         : {len(shape_cols)} — {shape_cols}")
print(f"  Gradient      : {len(gradient_cols)} — {gradient_cols}")

# ── 5 ablation configurations ─────────────────────────────────────────────────
configs = {
    "All features (baseline)":           all_feature_cols,
    "Without gradient features":         [c for c in all_feature_cols if c not in gradient_cols],
    "Without texture (GLCM) features":   [c for c in all_feature_cols if c not in texture_cols],
    "Without shape features":            [c for c in all_feature_cols if c not in shape_cols],
    "First-order only":                  first_order_cols,
}

# ── train / eval ──────────────────────────────────────────────────────────────
groups = df["patient_id"].astype(str).values
le_enc = LabelEncoder()
y      = le_enc.fit_transform(df["disease"].values)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

results = []
for config_name, feat_cols in configs.items():
    if not feat_cols:
        print(f"  [{config_name}] no features — skipping")
        continue

    X = df[feat_cols].fillna(df[feat_cols].median()).values

    _, test_idx = next(gss.split(X, y, groups))
    train_idx   = [i for i in range(len(X)) if i not in set(test_idx)]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = RandomForestClassifier(n_estimators=100, random_state=42,
                                 class_weight="balanced", n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc      = accuracy_score(y_test, y_pred) * 100
    f1_macro = f1_score(y_test, y_pred, average="macro")
    n_feat   = len(feat_cols)

    results.append((config_name, n_feat, acc, f1_macro))
    print(f"  [{config_name}]  features={n_feat}  acc={acc:.2f}%  macro_F1={f1_macro:.4f}")

# ── plot ──────────────────────────────────────────────────────────────────────
names    = [r[0] for r in results]
accs     = [r[2] for r in results]
f1s      = [r[3] for r in results]
x        = np.arange(len(names))
bar_w    = 0.38

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2      = ax1.twinx()

b1 = ax1.bar(x - bar_w/2, accs, bar_w, color=PALETTE[0], label="Accuracy (%)", alpha=0.85)
b2 = ax2.bar(x + bar_w/2, f1s,  bar_w, color=PALETTE[2], label="Macro F1",     alpha=0.85)

ax1.bar_label(b1, fmt="%.1f%%", fontsize=8, padding=2)
ax2.bar_label(b2, fmt="%.3f",   fontsize=8, padding=2)

ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
ax1.set_ylabel("Accuracy (%)", color=PALETTE[0])
ax2.set_ylabel("Macro F1",     color=PALETTE[2])
ax1.set_ylim(0, 110); ax2.set_ylim(0, 1.1)
ax1.set_title("Ablation Study — Random Forest across Feature Configurations", fontsize=12)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(OUT_ROOT, "ablation_study.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved ablation_study.png")

# ── summary table ─────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print(f"  {'Configuration':<40} {'Features':>8}  {'Accuracy':>9}  {'Macro F1':>9}")
print(f"  {'-'*40} {'-'*8}  {'-'*9}  {'-'*9}")
for name, n_feat, acc, f1_macro in results:
    print(f"  {name:<40} {n_feat:>8}  {acc:>8.2f}%  {f1_macro:>9.4f}")
print(f"{'═'*70}")
print(f"\nAll outputs saved to: {OUT_ROOT}")
