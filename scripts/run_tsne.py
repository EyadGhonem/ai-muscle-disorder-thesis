"""
run_tsne.py
-----------
Dimensionality reduction visualisations of the radiomics feature space.

Produces four scatter plots to illustrate how well the 28 radiomics features
separate the disease classes and data sources in a 2D embedding:

  1. t-SNE coloured by disease class
  2. t-SNE coloured by dataset source (ULTRASOUND_LABELD_1 vs MAT_LABELED)
  3. PCA coloured by disease class
  4. PCA coloured by dataset source

Input:
  output/final_ultrasound_dataset.csv  — radiomics feature table with disease labels

Output (saved to output/aplus/run_tsne/):
  tsne_by_disease.png
  tsne_by_source.png
  pca_by_disease.png
  pca_by_source.png

Note: t-SNE is run with perplexity=40, max_iter=1000, random_state=42 to ensure
reproducibility. Features are standardised with StandardScaler before reduction.

Run: python scripts/run_tsne.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── paths ────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE, "output", "final_ultrasound_dataset.csv")
OUT_ROOT = os.path.join(BASE, "output", "aplus", "run_tsne")
os.makedirs(OUT_ROOT, exist_ok=True)

PALETTE  = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]
EXCLUDE  = {"image_path", "patient_id", "disease", "severity_label",
            "dataset_source", "severity"}

plt.style.use("seaborn-v0_8-whitegrid")

# ── load ─────────────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(CSV_PATH)
df = df[df["disease"].notna() & (df["disease"] != "NAN")].copy()

feature_cols = [c for c in df.columns if c not in EXCLUDE]
X_raw = df[feature_cols].fillna(df[feature_cols].median()).values
labels_disease = df["disease"].values
labels_source  = df["dataset_source"].fillna("Unknown").values

print(f"Samples: {len(df)}  |  Features: {len(feature_cols)}")

# scale
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ── subsample for speed (≤ 3000 points for t-SNE) ────────────────────────────
MAX_TSNE = 3000
rng      = np.random.default_rng(42)
if len(X_scaled) > MAX_TSNE:
    idx_sub = rng.choice(len(X_scaled), MAX_TSNE, replace=False)
    X_sub   = X_scaled[idx_sub]
    d_sub   = labels_disease[idx_sub]
    s_sub   = labels_source[idx_sub]
    print(f"Subsampled to {MAX_TSNE} points for t-SNE")
else:
    X_sub, d_sub, s_sub = X_scaled, labels_disease, labels_source

# ── t-SNE ─────────────────────────────────────────────────────────────────────
print("Running t-SNE (this may take a few minutes) …")
tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000,
            init="pca", learning_rate="auto")
Z_tsne = tsne.fit_transform(X_sub)
print("  Done.")

# ── PCA ───────────────────────────────────────────────────────────────────────
print("Running PCA …")
pca   = PCA(n_components=2, random_state=42)
Z_pca = pca.fit_transform(X_sub)
ev    = pca.explained_variance_ratio_ * 100
print(f"  PC1 {ev[0]:.1f}%  PC2 {ev[1]:.1f}%")


def scatter_by_label(ax, Z, labels, title, xlabel="Dim 1", ylabel="Dim 2"):
    uniq = sorted(set(labels))
    for i, lbl in enumerate(uniq):
        mask = labels == lbl
        ax.scatter(Z[mask, 0], Z[mask, 1],
                   c=PALETTE[i % len(PALETTE)], label=lbl,
                   s=8, alpha=0.6, linewidths=0)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, markerscale=2, loc="best")


# figure 1 – t-SNE by disease
fig, ax = plt.subplots(figsize=(8, 6))
scatter_by_label(ax, Z_tsne, d_sub, "t-SNE — coloured by disease class")
plt.tight_layout()
fig.savefig(os.path.join(OUT_ROOT, "tsne_disease.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved tsne_disease.png")

# figure 2 – t-SNE by source
fig, ax = plt.subplots(figsize=(8, 6))
scatter_by_label(ax, Z_tsne, s_sub, "t-SNE — coloured by dataset source")
plt.tight_layout()
fig.savefig(os.path.join(OUT_ROOT, "tsne_source.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved tsne_source.png")

# figure 3 – PCA by disease
fig, ax = plt.subplots(figsize=(8, 6))
scatter_by_label(ax, Z_pca, d_sub, "PCA — coloured by disease class",
                 xlabel=f"PC1 ({ev[0]:.1f}%)", ylabel=f"PC2 ({ev[1]:.1f}%)")
plt.tight_layout()
fig.savefig(os.path.join(OUT_ROOT, "pca_disease.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved pca_disease.png")

# figure 4 – PCA by source
fig, ax = plt.subplots(figsize=(8, 6))
scatter_by_label(ax, Z_pca, s_sub, "PCA — coloured by dataset source",
                 xlabel=f"PC1 ({ev[0]:.1f}%)", ylabel=f"PC2 ({ev[1]:.1f}%)")
plt.tight_layout()
fig.savefig(os.path.join(OUT_ROOT, "pca_source.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved pca_source.png")

print(f"\nAll t-SNE/PCA outputs saved to: {OUT_ROOT}")
