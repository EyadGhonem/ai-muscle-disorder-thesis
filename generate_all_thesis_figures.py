"""
Generate ALL thesis figures (3.4, 3.5, 4.1–4.8) in one script.
Run: python generate_all_thesis_figures.py
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Shared style ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color palette
COLORS = {
    "blue": "#3B82F6",
    "green": "#10B981",
    "amber": "#F59E0B",
    "red": "#EF4444",
    "purple": "#8B5CF6",
    "teal": "#14B8A6",
    "pink": "#EC4899",
    "indigo": "#6366F1",
    "slate": "#64748B",
    "cyan": "#06B6D4",
}
MODEL_COLORS = {
    "SVM": "#3B82F6",
    "Random Forest": "#10B981",
    "Logistic Regression": "#F59E0B",
    "MLP Neural Network": "#8B5CF6",
    "EfficientNetB0": "#EF4444",
    "ResNet50": "#EC4899",
    "DenseNet121": "#14B8A6",
}
DISEASE_COLORS = {
    "FSHD": "#3B82F6",
    "Normal": "#10B981",
    "IBM": "#F59E0B",
    "Dermatomyositis": "#EF4444",
    "Polymyositis": "#8B5CF6",
}


def draw_pipeline_box(ax, x, y, w, h, text, color, fontsize=9):
    """Draw a rounded box with text."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
        alpha=0.95,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, fontsize=fontsize,
            fontweight="bold", color="white", ha="center", va="center",
            wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color="#64748B"):
    """Draw an arrow between two points."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.15",
                        color=color, lw=2),
    )


# ============================================================
# FIGURE 3.4: Machine Learning Pipeline
# ============================================================
def figure_3_4():
    print("Generating Figure 3.4...")
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")

    steps = [
        ("Feature\nTable", "#64748B"),
        ("Cleaning\n& Scaling", "#3B82F6"),
        ("Train / Test\nSplit", "#06B6D4"),
        ("ML Model\nTraining", "#10B981"),
        ("Prediction", "#F59E0B"),
        ("Evaluation\nMetrics", "#EF4444"),
    ]

    n = len(steps)
    box_w = 0.12
    box_h = 0.45
    gap = (1.0 - n * box_w) / (n + 1)
    y_center = 0.35

    for i, (label, color) in enumerate(steps):
        x = gap + i * (box_w + gap)
        draw_pipeline_box(ax, x, y_center, box_w, box_h, label, color, fontsize=10)

        if i < n - 1:
            x_end = gap + (i + 1) * (box_w + gap)
            draw_arrow(ax, x + box_w + 0.005, y_center + box_h / 2,
                       x_end - 0.005, y_center + box_h / 2)

    # Sub-labels below boxes
    sub_labels = [
        "CSV with\nradiomics features",
        "Imputation,\nStandardScaler",
        "80/20 split,\npatient-level",
        "SVM, RF, LR,\nMLP",
        "Class labels\n& probabilities",
        "Accuracy, F1,\nAUC, CM",
    ]
    for i, sub in enumerate(sub_labels):
        x = gap + i * (box_w + gap)
        ax.text(x + box_w / 2, y_center - 0.08, sub, fontsize=7,
                color="#64748B", ha="center", va="top", fontstyle="italic")

    ax.text(0.5, -0.05,
            "Figure 3.4: Machine learning pipeline using extracted ultrasound features.",
            fontsize=11, fontstyle="italic", color="#444444", ha="center",
            transform=ax.transAxes)

    plt.savefig(OUTPUT_DIR / "figure_3_4_ml_pipeline.png", edgecolor="none")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'figure_3_4_ml_pipeline.png'}")


# ============================================================
# FIGURE 3.5: Deep Learning Pipeline
# ============================================================
def figure_3_5():
    print("Generating Figure 3.5...")
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")

    steps = [
        ("Ultrasound\nImages", "#64748B"),
        ("Resize &\nNormalize", "#3B82F6"),
        ("CNN Model\n(Transfer Learning)", "#8B5CF6"),
        ("Prediction", "#F59E0B"),
        ("Evaluation", "#EF4444"),
    ]

    n = len(steps)
    box_w = 0.14
    box_h = 0.45
    gap = (1.0 - n * box_w) / (n + 1)
    y_center = 0.35

    for i, (label, color) in enumerate(steps):
        x = gap + i * (box_w + gap)
        draw_pipeline_box(ax, x, y_center, box_w, box_h, label, color, fontsize=10)

        if i < n - 1:
            x_end = gap + (i + 1) * (box_w + gap)
            draw_arrow(ax, x + box_w + 0.005, y_center + box_h / 2,
                       x_end - 0.005, y_center + box_h / 2)

    sub_labels = [
        "224×224 RGB\nfrom dataset",
        "ImageNet\npreprocessing",
        "ResNet50, DenseNet121,\nEfficientNetB0",
        "Sigmoid output\n(binary)",
        "Accuracy, AUC,\nF1-score",
    ]
    for i, sub in enumerate(sub_labels):
        x = gap + i * (box_w + gap)
        ax.text(x + box_w / 2, y_center - 0.08, sub, fontsize=7,
                color="#64748B", ha="center", va="top", fontstyle="italic")

    ax.text(0.5, -0.05,
            "Figure 3.5: Deep learning pipeline for ultrasound image classification.",
            fontsize=11, fontstyle="italic", color="#444444", ha="center",
            transform=ax.transAxes)

    plt.savefig(OUTPUT_DIR / "figure_3_5_dl_pipeline.png", edgecolor="none")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'figure_3_5_dl_pipeline.png'}")


# ============================================================
# FIGURE 4.1: Class Distribution
# ============================================================
def figure_4_1():
    print("Generating Figure 4.1...")
    diseases = ["FSHD", "Normal", "IBM", "Dermatomyositis", "Polymyositis"]
    counts = [4775, 1337, 796, 555, 554]
    colors = [DISEASE_COLORS[d] for d in diseases]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars = ax.bar(diseases, counts, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.65, zorder=3)

    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60,
                f"{count:,}", ha="center", va="bottom", fontweight="bold",
                fontsize=11, color="#1E293B")
        pct = count / sum(counts) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{pct:.1f}%", ha="center", va="center", fontweight="bold",
                fontsize=10, color="white")

    ax.set_ylabel("Number of Samples", fontweight="bold")
    ax.set_xlabel("Disease Category", fontweight="bold")
    ax.set_title("Distribution of Ultrasound Samples Across Disease Categories",
                 fontweight="bold", fontsize=13, pad=15)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Total annotation
    ax.text(0.98, 0.95, f"Total: {sum(counts):,} samples",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontstyle="italic", color="#64748B",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F1F5F9",
                      edgecolor="#CBD5E1"))

    fig.text(0.5, -0.02,
             "Figure 4.1: Distribution of ultrasound samples across the five disease categories.",
             fontsize=11, fontstyle="italic", color="#444444", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_4_1_class_distribution.png", edgecolor="none")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'figure_4_1_class_distribution.png'}")


# ============================================================
# FIGURE 4.2: Model Comparison (Accuracy)
# ============================================================
def figure_4_2():
    print("Generating Figure 4.2...")
    models = [
        "SVM", "Random Forest", "MLP Neural\nNetwork",
        "Logistic\nRegression", "EfficientNetB0",
        "DenseNet121", "ResNet50",
    ]
    accuracies = [98.35, 98.23, 98.23, 97.88, 87.49, 79.53, 73.10]
    families = ["Radiomics+ML"] * 4 + ["Deep Learning (CNN)"] * 3
    colors_list = [
        MODEL_COLORS["SVM"], MODEL_COLORS["Random Forest"],
        MODEL_COLORS["MLP Neural Network"], MODEL_COLORS["Logistic Regression"],
        MODEL_COLORS["EfficientNetB0"], MODEL_COLORS["DenseNet121"],
        MODEL_COLORS["ResNet50"],
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(models)), accuracies, color=colors_list,
                  edgecolor="white", linewidth=1.5, width=0.7, zorder=3)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold",
                fontsize=10, color="#1E293B")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Comparison of Classification Accuracy Across Evaluated Models",
                 fontweight="bold", fontsize=13, pad=15)
    ax.set_ylim(65, 103)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Separator line between ML and DL
    ax.axvline(x=3.5, color="#CBD5E1", linestyle="--", linewidth=1.5, zorder=1)
    ax.text(1.5, 67, "Radiomics + ML", ha="center", fontsize=10,
            fontstyle="italic", color="#64748B")
    ax.text(5, 67, "Deep Learning (CNN)", ha="center", fontsize=10,
            fontstyle="italic", color="#64748B")

    fig.text(0.5, -0.02,
             "Figure 4.2: Comparison of classification accuracy across the evaluated models.",
             fontsize=11, fontstyle="italic", color="#444444", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_4_2_model_comparison.png", edgecolor="none")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'figure_4_2_model_comparison.png'}")


# ============================================================
# FIGURE 4.3: Patient-Level Comparison
# ============================================================
def figure_4_3():
    print("Generating Figure 4.3...")
    models = ["SVM", "Random Forest", "Logistic Regression", "MLP Neural Network"]
    accuracy = [98.35, 98.23, 97.88, 98.23]
    f1_macro = [0.4091, 0.2800, 0.2641, 0.2800]
    f1_weighted = [0.9840, 0.9788, 0.9789, 0.9788]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, accuracy, width, label="Accuracy (%)",
                   color="#3B82F6", edgecolor="white", linewidth=1, zorder=3)
    bars2 = ax.bar(x, [f * 100 for f in f1_macro], width, label="Macro F1 (×100)",
                   color="#EF4444", edgecolor="white", linewidth=1, zorder=3)
    bars3 = ax.bar(x + width, [f * 100 for f in f1_weighted], width,
                   label="Weighted F1 (×100)",
                   color="#10B981", edgecolor="white", linewidth=1, zorder=3)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                    f"{height:.1f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color="#1E293B")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("Patient-Level Evaluation: Accuracy vs Macro F1 vs Weighted F1",
                 fontweight="bold", fontsize=13, pad=15)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation about gap
    ax.annotate("Gap indicates\nminority class weakness",
                xy=(0, 40.91), xytext=(0.5, 55),
                arrowprops=dict(arrowstyle="->", color="#EF4444", lw=1.5),
                fontsize=9, fontstyle="italic", color="#EF4444",
                ha="center")

    fig.text(0.5, -0.02,
             "Figure 4.3: Patient-level comparison between radiomics-based machine learning and deep learning models.",
             fontsize=10, fontstyle="italic", color="#444444", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_4_3_patient_level.png", edgecolor="none")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'figure_4_3_patient_level.png'}")


# ============================================================
# FIGURE 4.4: Confusion Matrix (SVM)
# ============================================================
def figure_4_4():
    print("Generating Figure 4.4...")
    classes = ["Dermatomyositis", "FSHD", "IBM", "Normal", "Polymyositis"]
    cm = np.array([
        [3, 0, 0, 2, 2],
        [0, 829, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 1, 2, 1],
        [1, 0, 3, 1, 1],
    ])

    fig, ax = plt.subplots(figsize=(8, 7))

    # Normalized for color mapping
    cm_norm = cm.astype(float)
    for i in range(len(cm)):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_norm[i] = cm[i] / row_sum

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", aspect="auto")

    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm_norm[i, j] > 0.5 else "#1E293B"
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color=color)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    short_labels = ["Derm.", "FSHD", "IBM", "Normal", "Poly."]
    ax.set_xticklabels(short_labels, fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels(short_labels, fontsize=10)
    ax.set_xlabel("Predicted Label", fontweight="bold", fontsize=11)
    ax.set_ylabel("True Label", fontweight="bold", fontsize=11)
    ax.set_title("Confusion Matrix — SVM (Best Radiomics-Based Model)",
                 fontweight="bold", fontsize=13, pad=15)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Proportion", fontsize=10)

    fig.text(0.5, -0.02,
             "Figure 4.4: Confusion matrix for the best radiomics-based machine learning model (SVM).",
             fontsize=10, fontstyle="italic", color="#444444", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_4_4_confusion_matrix.png", edgecolor="none")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'figure_4_4_confusion_matrix.png'}")


# ============================================================
# FIGURE 4.5: Severity Classification
# ============================================================
def figure_4_5():
    print("Generating Figure 4.5...")
    models = ["Random Forest\n(Real Features)", "Earlier Severity\nExperiment"]
    accuracies = [92.01, 93.57]
    colors = ["#3B82F6", "#10B981"]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bars = ax.bar(models, accuracies, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.5, zorder=3)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold",
                fontsize=13, color="#1E293B")

    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Severity Classification Performance on FSHD Subset",
                 fontweight="bold", fontsize=13, pad=15)
    ax.set_ylim(85, 97)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Context box
    ax.text(0.98, 0.15, "Task: Binary severity\nclassification (FSHD subset)\nUsing radiomics features",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontstyle="italic", color="#64748B",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F1F5F9",
                      edgecolor="#CBD5E1"))

    fig.text(0.5, -0.02,
             "Figure 4.5: Severity classification performance on the FSHD subset.",
             fontsize=11, fontstyle="italic", color="#444444", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_4_5_severity.png", edgecolor="none")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'figure_4_5_severity.png'}")


# ============================================================
# FIGURE 4.6: Deep Learning Model Comparison
# ============================================================
def figure_4_6():
    print("Generating Figure 4.6...")
    models = ["MLP Neural\nNetwork", "EfficientNetB0", "DenseNet121", "ResNet50"]
    accuracies = [98.23, 87.49, 79.53, 73.10]
    input_types = ["Radiomics\nFeatures", "Ultrasound\nImages", "Ultrasound\nImages", "Ultrasound\nImages"]
    colors = [
        MODEL_COLORS["MLP Neural Network"],
        MODEL_COLORS["EfficientNetB0"],
        MODEL_COLORS["DenseNet121"],
        MODEL_COLORS["ResNet50"],
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(range(len(models)), accuracies, color=colors,
                   edgecolor="white", linewidth=1.5, height=0.6, zorder=3)

    for i, (bar, acc, inp) in enumerate(zip(bars, accuracies, input_types)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{acc:.2f}%", va="center", fontweight="bold",
                fontsize=11, color="#1E293B")
        ax.text(2, bar.get_y() + bar.get_height() / 2,
                f"Input: {inp}", va="center", fontsize=8,
                color="white", fontstyle="italic")

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Deep Learning Model Comparison",
                 fontweight="bold", fontsize=13, pad=15)
    ax.set_xlim(0, 108)
    ax.grid(axis="x", alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    # Separator
    ax.axhline(y=0.5, color="#CBD5E1", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(105, 0, "Feature-based", ha="right", fontsize=9,
            fontstyle="italic", color="#64748B", va="center")
    ax.text(105, 2, "Image-based", ha="right", fontsize=9,
            fontstyle="italic", color="#64748B", va="center")

    fig.text(0.5, -0.02,
             "Figure 4.6: Comparison between deep learning models evaluated in this thesis.",
             fontsize=10, fontstyle="italic", color="#444444", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_4_6_dl_comparison.png", edgecolor="none")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'figure_4_6_dl_comparison.png'}")


# ============================================================
# FIGURE 4.7: Feature Importance
# ============================================================
def figure_4_7():
    print("Generating Figure 4.7...")
    # Load actual feature importance data
    fi_path = PROJECT_ROOT / "output" / "thesis_final" / "feature_importance.csv"
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)
    else:
        # Fallback data
        fi_df = pd.DataFrame({
            "feature": ["gradient_mean", "perimeter", "area", "glcm_homogeneity",
                         "gradient_max", "equivalent_diameter", "glcm_energy",
                         "glcm_asm", "gradient_std", "entropy",
                         "glcm_dissimilarity", "glcm_contrast", "std_intensity",
                         "glcm_correlation", "solidity"],
            "importance": [0.078, 0.075, 0.061, 0.055, 0.054, 0.044, 0.044,
                           0.043, 0.036, 0.032, 0.032, 0.031, 0.029, 0.029, 0.028],
        })

    top_n = min(15, len(fi_df))
    fi_top = fi_df.head(top_n).iloc[::-1]  # Reverse for horizontal bar

    # Color by feature category
    def get_category_color(feat):
        feat_lower = feat.lower()
        if "glcm" in feat_lower or "glszm" in feat_lower or "glrlm" in feat_lower or "gldm" in feat_lower or "ngtdm" in feat_lower:
            return "#10B981", "Texture"
        elif "gradient" in feat_lower:
            return "#EF4444", "Gradient"
        elif any(k in feat_lower for k in ["area", "perimeter", "circularity",
                                            "aspect_ratio", "solidity", "extent",
                                            "equivalent_diameter"]):
            return "#F59E0B", "Morphology"
        else:
            return "#3B82F6", "Intensity"

    bar_colors = [get_category_color(f)[0] for f in fi_top["feature"]]

    fig, ax = plt.subplots(figsize=(11, 7))

    bars = ax.barh(range(top_n), fi_top["importance"].values, color=bar_colors,
                   edgecolor="white", linewidth=1, height=0.7, zorder=3)

    # Clean feature names for display
    clean_names = []
    for f in fi_top["feature"]:
        name = f.replace("_", " ").replace("original ", "").title()
        clean_names.append(name)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(clean_names, fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)", fontweight="bold")
    ax.set_title("Feature Importance Analysis — Radiomics-Based ML Model",
                 fontweight="bold", fontsize=13, pad=15)
    ax.grid(axis="x", alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}", va="center", fontsize=8, color="#64748B")

    # Legend
    legend_patches = [
        mpatches.Patch(color="#3B82F6", label="Intensity"),
        mpatches.Patch(color="#10B981", label="Texture"),
        mpatches.Patch(color="#F59E0B", label="Morphology"),
        mpatches.Patch(color="#EF4444", label="Gradient"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10,
              framealpha=0.9, title="Feature Category", title_fontsize=10)

    fig.text(0.5, -0.02,
             "Figure 4.7: Feature importance analysis for the radiomics-based machine learning model.",
             fontsize=10, fontstyle="italic", color="#444444", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_4_7_feature_importance.png", edgecolor="none")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'figure_4_7_feature_importance.png'}")


# ============================================================
# FIGURE 4.8: Dataset Source Bias
# ============================================================
def figure_4_8():
    print("Generating Figure 4.8...")

    # Try to load actual data
    master_path = OUTPUT_DIR / "final_ultrasound_dataset.csv"
    if master_path.exists():
        try:
            df = pd.read_csv(master_path, usecols=["disease", "dataset_source"])
            ct = pd.crosstab(df["disease"], df["dataset_source"])
            diseases = ct.index.tolist()
            sources = ct.columns.tolist()
            data = ct.values
        except Exception:
            # Fallback
            diseases = ["FSHD", "Normal", "IBM", "Dermatomyositis", "Polymyositis"]
            sources = ["ULTRASOUND_LABELD_1", "ULTRASOUND_LABELD_2"]
            data = np.array([
                [4775, 0],
                [0, 1337],
                [0, 796],
                [0, 555],
                [0, 554],
            ])
    else:
        diseases = ["FSHD", "Normal", "IBM", "Dermatomyositis", "Polymyositis"]
        sources = ["ULTRASOUND_LABELD_1", "ULTRASOUND_LABELD_2"]
        data = np.array([
            [4775, 0],
            [0, 1337],
            [0, 796],
            [0, 555],
            [0, 554],
        ])

    source_colors = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6"]

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(diseases))
    n_sources = len(sources)
    width = 0.7 / n_sources

    for i, source in enumerate(sources):
        offset = (i - n_sources / 2 + 0.5) * width
        vals = data[:, i]
        color = source_colors[i % len(source_colors)]
        short_name = source.replace("ULTRASOUND_LABELD_", "Source ")
        bars = ax.bar(x + offset, vals, width, label=short_name,
                      color=color, edgecolor="white", linewidth=1, zorder=3)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                        str(int(val)), ha="center", va="bottom", fontsize=8,
                        fontweight="bold", color="#1E293B")

    short_diseases = []
    for d in diseases:
        if d == "Dermatomyositis":
            short_diseases.append("Derm.")
        elif d == "Polymyositis":
            short_diseases.append("Poly.")
        else:
            short_diseases.append(d)

    ax.set_xticks(x)
    ax.set_xticklabels(short_diseases, fontsize=10)
    ax.set_ylabel("Number of Samples", fontweight="bold")
    ax.set_xlabel("Disease Category", fontweight="bold")
    ax.set_title("Dataset Source Distribution Across Disease Classes",
                 fontweight="bold", fontsize=13, pad=15)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Warning annotation
    ax.text(0.98, 0.85,
            "⚠ Disease labels are\ncorrelated with dataset source\n→ potential source bias",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#B91C1C",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF2F2",
                      edgecolor="#FECACA"))

    fig.text(0.5, -0.02,
             "Figure 4.8: Dataset source distribution across disease classes.",
             fontsize=10, fontstyle="italic", color="#444444", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_4_8_dataset_bias.png", edgecolor="none")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'figure_4_8_dataset_bias.png'}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("GENERATING ALL THESIS FIGURES")
    print("=" * 60)

    figure_3_4()
    figure_3_5()
    figure_4_1()
    figure_4_2()
    figure_4_3()
    figure_4_4()
    figure_4_5()
    figure_4_6()
    figure_4_7()
    figure_4_8()

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("\nFiles:")
    for f in sorted(OUTPUT_DIR.glob("figure_*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
