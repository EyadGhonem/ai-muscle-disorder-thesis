"""
run_bias_and_learning_curves.py
--------------------------------
Dataset bias assessment and model learning curve analysis.

Three analysis stages:

Part A — Cramér's V bias test
  Tests whether the disease label distribution is statistically associated
  with the dataset source (ULTRASOUND_LABELD_1 vs MAT_LABELED).
  A high Cramér's V value (≈ 1.0) would indicate that models may be learning
  source-specific artefacts rather than true disease patterns.
  Output: cramer_v_heatmap.png

Part B — ML learning curves
  Plots validation accuracy as a function of training set size for Random
  Forest and Gradient Boosting, using 5-fold cross-validation.
  Helps diagnose whether the models suffer from high bias (underfitting) or
  high variance (overfitting).
  Output: learning_curve_<ModelName>.png  (one per model)

Part C — CNN training history
  Reads validation accuracy history from gui_training_metrics.json
  (written by train_gui_on_real_ultrasound.py) and plots per-epoch learning
  curves for all trained CNN architectures.
  Output: cnn_training_history.png

Input:
  output/final_ultrasound_dataset.csv        — radiomics dataset with source labels
  gui_demo/models/gui_training_metrics.json  — CNN training history (Part C only)

Output directory: output/aplus/run_bias_and_learning_curves/

Run: python scripts/run_bias_and_learning_curves.py
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve, GroupShuffleSplit

# ── paths ────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH  = os.path.join(BASE, "output", "final_ultrasound_dataset.csv")
METRICS   = os.path.join(BASE, "gui_demo", "models", "gui_training_metrics.json")
OUT_ROOT  = os.path.join(BASE, "output", "aplus", "run_bias_and_learning_curves")
os.makedirs(OUT_ROOT, exist_ok=True)

PALETTE  = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]
EXCLUDE  = {"image_path", "patient_id", "disease", "severity_label",
            "dataset_source", "severity"}

plt.style.use("seaborn-v0_8-whitegrid")

# ─────────────────────────────────────────────────────────────────────────────
# PART A — Cramér's V
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PART A — Cramér's V bias test")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
df = df[df["disease"].notna() & (df["disease"] != "NAN")].copy()

ct = pd.crosstab(df["dataset_source"], df["disease"])
print("\nContingency table (dataset_source × disease):")
print(ct.to_string())

chi2, p_val, dof, _ = chi2_contingency(ct)
n         = ct.values.sum()
r, c      = ct.shape
cramers_v = float(np.sqrt(chi2 / (n * (min(r, c) - 1))))

print(f"\nchi2  = {chi2:.4f}")
print(f"p-val = {p_val:.2e}")
print(f"dof   = {dof}")
print(f"n     = {n}")
print(f"Cramér's V = {cramers_v:.4f}  "
      f"({'strong' if cramers_v > 0.5 else 'moderate' if cramers_v > 0.3 else 'weak'} association)")

# heatmap figure
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
            ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title(
    f"Dataset Source × Disease — Contingency Table\n"
    f"chi²={chi2:.1f}  p={p_val:.2e}  Cramér's V={cramers_v:.3f}",
    fontsize=11,
)
ax.set_xlabel("Disease"); ax.set_ylabel("Dataset Source")
plt.tight_layout()
fig.savefig(os.path.join(OUT_ROOT, "bias_contingency_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved bias_contingency_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# PART B — ML learning curves
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART B — ML learning curves")
print("=" * 60)

feature_cols = [c for c in df.columns if c not in EXCLUDE]
X = df[feature_cols].fillna(df[feature_cols].median()).values
le_enc = LabelEncoder()
y = le_enc.fit_transform(df["disease"].values)
groups = df["patient_id"].astype(str).values

lc_models = {
    "Random Forest":      RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    "Gradient Boosting":  GradientBoostingClassifier(n_estimators=50, random_state=42),
}
train_sizes = np.linspace(0.1, 1.0, 8)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, (name, clf) in zip(axes, lc_models.items()):
    print(f"  Learning curve for {name} …")
    cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    ts, tr_scores, cv_scores = learning_curve(
        clf, X, y,
        groups       = groups,
        cv           = cv,
        train_sizes  = train_sizes,
        scoring      = "accuracy",
        n_jobs       = -1,
        error_score  = 0,
    )
    tr_mean = tr_scores.mean(axis=1)
    tr_std  = tr_scores.std(axis=1)
    cv_mean = cv_scores.mean(axis=1)
    cv_std  = cv_scores.std(axis=1)

    ax.fill_between(ts, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color=PALETTE[0])
    ax.fill_between(ts, cv_mean - cv_std, cv_mean + cv_std, alpha=0.15, color=PALETTE[2])
    ax.plot(ts, tr_mean, "o-", color=PALETTE[0], label="Train accuracy")
    ax.plot(ts, cv_mean, "s-", color=PALETTE[2], label="CV accuracy")
    ax.set_title(f"Learning Curve — {name}")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_ylim(0, 1.05)

plt.suptitle("ML Learning Curves (patient-level GroupShuffleSplit, 5 folds)", y=1.02, fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUT_ROOT, "learning_curves_ml.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved learning_curves_ml.png")

# ─────────────────────────────────────────────────────────────────────────────
# PART C — CNN training history
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART C — CNN training history")
print("=" * 60)

if not os.path.exists(METRICS):
    print(f"Metrics file not found: {METRICS} — skipping Part C")
else:
    with open(METRICS) as f:
        metrics = json.load(f)

    # check for per-epoch history key
    history_found = False
    for section_key in ("dl_disease_history", "dl_severity_history", "history"):
        if section_key in metrics:
            history = metrics[section_key]
            history_found = True
            print(f"  Found history under key '{section_key}'")

            fig, axes = plt.subplots(1, len(history), figsize=(5 * len(history), 4), sharey=True)
            if len(history) == 1:
                axes = [axes]
            for ax, (model_name, h) in zip(axes, history.items()):
                ta = h.get("accuracy") or h.get("train_accuracy", [])
                va = h.get("val_accuracy", [])
                if ta:
                    ax.plot(ta, "o-", color=PALETTE[0], label="Train accuracy")
                if va:
                    ax.plot(va, "s-", color=PALETTE[2], label="Val accuracy")
                ax.set_title(model_name)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Accuracy")
                ax.legend(fontsize=8)
            plt.suptitle("CNN Training History", fontsize=12)
            plt.tight_layout()
            fig.savefig(os.path.join(OUT_ROOT, f"cnn_history_{section_key}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved cnn_history_{section_key}.png")
            break

    if not history_found:
        # fall back: bar chart from scalar val_accuracy_pct values
        print("  No per-epoch history found; plotting final val accuracy from scalars.")
        dl_sections = {k: v for k, v in metrics.items()
                       if k.startswith("dl_") and isinstance(v, dict)}
        if dl_sections:
            fig, axes = plt.subplots(1, len(dl_sections), figsize=(6 * len(dl_sections), 4))
            if len(dl_sections) == 1:
                axes = [axes]
            for ax, (section, models_dict) in zip(axes, dl_sections.items()):
                names, vals = [], []
                for mname, mval in models_dict.items():
                    names.append(mname)
                    if isinstance(mval, dict):
                        vals.append(mval.get("val_accuracy_pct", mval.get("val_f1_macro", 0)) / 100
                                    if mval.get("val_accuracy_pct", 0) > 1 else mval.get("val_accuracy_pct", 0))
                    else:
                        vals.append(float(mval) / 100 if float(mval) > 1 else float(mval))
                colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
                bars   = ax.bar(names, vals, color=colors)
                ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
                ax.set_title(section.replace("_", " ").title())
                ax.set_ylabel("Val Accuracy"); ax.set_ylim(0, 1.1)
                ax.set_xticklabels(names, rotation=20, ha="right")
            plt.suptitle("CNN Validation Accuracy (final epoch)", fontsize=12)
            plt.tight_layout()
            fig.savefig(os.path.join(OUT_ROOT, "cnn_val_accuracy.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
            print("  Saved cnn_val_accuracy.png")

print(f"\nAll outputs saved to: {OUT_ROOT}")
