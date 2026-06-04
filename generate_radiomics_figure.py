"""
Generate Figure 3.3: Radiomics-inspired feature extraction pipeline.
Shows: Ultrasound Image + Mask → 4 feature groups → Feature Vector.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_ROOT / "data" / "ULTRASOUND_LABELD_1" / "images"
OUTPUT_PATH = PROJECT_ROOT / "output" / "figure_3_3_radiomics_pipeline.png"


def generate_mask(image_bgr):
    """Generate a foreground mask from the image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(cleaned)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        return mask
    return cleaned


def create_figure(image_path: Path):
    """Create the radiomics pipeline figure."""
    # Load image
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        from PIL import Image
        original = np.array(Image.open(image_path).convert("RGB"))
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    mask = generate_mask(original)

    # Create masked overlay for input panel
    masked_overlay = original_rgb.copy()
    mask_color = np.zeros_like(original_rgb)
    mask_color[:, :, 1] = 100
    mask_bool = mask > 0
    masked_overlay[mask_bool] = cv2.addWeighted(
        original_rgb[mask_bool], 0.75, mask_color[mask_bool], 0.25, 0
    )
    contours_vis, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(masked_overlay, contours_vis, -1, (0, 220, 80), 2)

    # --- Figure layout ---
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("white")

    # Color palette
    colors = {
        "intensity": "#3B82F6",    # Blue
        "texture": "#10B981",      # Green
        "morphology": "#F59E0B",   # Amber
        "gradient": "#EF4444",     # Red
        "input_bg": "#F1F5F9",     # Light gray
        "output_bg": "#EDE9FE",    # Light purple
        "arrow": "#64748B",        # Slate
    }

    # ======== LEFT: Input panel (Image + Mask) ========
    ax_input = fig.add_axes([0.02, 0.15, 0.18, 0.70])
    ax_input.imshow(masked_overlay)
    ax_input.set_title("Ultrasound Image\n+ Foreground Mask", fontsize=11,
                       fontweight="bold", fontfamily="serif", pad=10)
    ax_input.axis("off")
    # Border
    for spine in ax_input.spines.values():
        spine.set_visible(True)
        spine.set_color("#94A3B8")
        spine.set_linewidth(1.5)

    # ======== MIDDLE: Feature group boxes ========
    feature_groups = [
        {
            "name": "First-Order\nIntensity Features",
            "features": [
                "• Mean, Median, Std Dev",
                "• Percentiles (10th, 90th)",
                "• Entropy",
                "• Skewness, Kurtosis",
            ],
            "color": colors["intensity"],
            "icon": "📊",
        },
        {
            "name": "Texture Features\n(GLCM)",
            "features": [
                "• Contrast",
                "• Dissimilarity",
                "• Homogeneity",
                "• Energy, ASM",
                "• Correlation",
            ],
            "color": colors["texture"],
            "icon": "🔲",
        },
        {
            "name": "Morphological\nFeatures",
            "features": [
                "• Area, Perimeter",
                "• Circularity",
                "• Aspect Ratio",
                "• Solidity",
                "• Equivalent Diameter",
            ],
            "color": colors["morphology"],
            "icon": "📐",
        },
        {
            "name": "Gradient\nFeatures",
            "features": [
                "• Gradient Mean",
                "• Gradient Std Dev",
                "• Gradient Maximum",
                "• Gradient Energy",
            ],
            "color": colors["gradient"],
            "icon": "📈",
        },
    ]

    box_x = 0.30
    box_w = 0.16
    box_h = 0.165
    box_gap = 0.015
    total_h = 4 * box_h + 3 * box_gap
    start_y = 0.5 - total_h / 2 + 0.03

    feature_box_axes = []
    for i, group in enumerate(feature_groups):
        y = start_y + (3 - i) * (box_h + box_gap)

        # Draw rounded box using axes
        ax_box = fig.add_axes([box_x, y, box_w, box_h])
        ax_box.set_xlim(0, 1)
        ax_box.set_ylim(0, 1)
        ax_box.axis("off")

        # Background rectangle
        rect = FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02",
            facecolor=group["color"] + "15",  # Very light tint
            edgecolor=group["color"],
            linewidth=2,
        )
        ax_box.add_patch(rect)

        # Title bar
        title_rect = FancyBboxPatch(
            (0.02, 0.70), 0.96, 0.28,
            boxstyle="round,pad=0.02",
            facecolor=group["color"],
            edgecolor=group["color"],
            linewidth=0,
        )
        ax_box.add_patch(title_rect)
        ax_box.text(0.50, 0.84, group["name"], fontsize=8, fontweight="bold",
                    fontfamily="serif", color="white", ha="center", va="center",
                    linespacing=1.1)

        # Feature list
        feat_text = "\n".join(group["features"])
        ax_box.text(0.08, 0.58, feat_text, fontsize=6.5, fontfamily="serif",
                    color="#374151", va="top", linespacing=1.5)

        feature_box_axes.append((ax_box, y, box_h))

    # ======== RIGHT: Feature Vector output ========
    ax_output = fig.add_axes([0.72, 0.18, 0.25, 0.64])
    ax_output.set_xlim(0, 1)
    ax_output.set_ylim(0, 1)
    ax_output.axis("off")

    # Output box
    output_rect = FancyBboxPatch(
        (0.05, 0.05), 0.90, 0.90,
        boxstyle="round,pad=0.03",
        facecolor="#F5F3FF",
        edgecolor="#7C3AED",
        linewidth=2,
    )
    ax_output.add_patch(output_rect)

    ax_output.text(0.50, 0.92, "Feature Vector", fontsize=13, fontweight="bold",
                   fontfamily="serif", color="#7C3AED", ha="center", va="center")

    # Simulated feature vector visualization
    np.random.seed(42)
    n_features = 20
    feature_values = np.random.rand(n_features)
    feature_names_display = [
        "mean", "median", "std", "entropy", "skew",
        "kurt", "p10", "p90", "contrast", "dissim",
        "homog", "energy", "corr", "area", "perim",
        "circ", "ar", "grad_μ", "grad_σ", "grad_E",
    ]
    bar_colors = (
        [colors["intensity"]] * 5 +
        [colors["intensity"]] * 3 +
        [colors["texture"]] * 5 +
        [colors["morphology"]] * 4 +
        [colors["gradient"]] * 3
    )

    bar_y_start = 0.12
    bar_y_end = 0.82
    bar_height = (bar_y_end - bar_y_start) / n_features * 0.75
    bar_gap = (bar_y_end - bar_y_start) / n_features

    for j in range(n_features):
        y_pos = bar_y_end - (j + 1) * bar_gap + bar_gap * 0.125
        bar_w = feature_values[j] * 0.40
        bar = FancyBboxPatch(
            (0.35, y_pos), bar_w, bar_height,
            boxstyle="round,pad=0.002",
            facecolor=bar_colors[j] + "AA",
            edgecolor="none",
        )
        ax_output.add_patch(bar)
        ax_output.text(0.33, y_pos + bar_height / 2, feature_names_display[j],
                       fontsize=5, fontfamily="monospace", color="#4B5563",
                       ha="right", va="center")

    # Legend for feature vector
    legend_items = [
        ("Intensity", colors["intensity"]),
        ("Texture", colors["texture"]),
        ("Morphology", colors["morphology"]),
        ("Gradient", colors["gradient"]),
    ]
    for k, (name, color) in enumerate(legend_items):
        lx = 0.12 + k * 0.20
        ly = 0.03
        legend_rect = FancyBboxPatch(
            (lx, ly), 0.03, 0.03,
            boxstyle="round,pad=0.001",
            facecolor=color,
            edgecolor="none",
        )
        ax_output.add_patch(legend_rect)
        ax_output.text(lx + 0.045, ly + 0.015, name, fontsize=5.5,
                       fontfamily="serif", color="#4B5563", va="center")

    # ======== ARROWS ========
    # Arrow from input image to feature boxes
    for i, (_, y, h) in enumerate(feature_box_axes):
        arrow = FancyArrowPatch(
            (0.21, 0.50),
            (box_x - 0.01, y + h / 2),
            transform=fig.transFigure,
            arrowstyle="->,head_width=4,head_length=3",
            color=feature_groups[i]["color"],
            linewidth=1.5,
            connectionstyle="arc3,rad=0.0",
        )
        fig.patches.append(arrow)

    # Arrows from feature boxes to output
    for i, (_, y, h) in enumerate(feature_box_axes):
        arrow = FancyArrowPatch(
            (box_x + box_w + 0.01, y + h / 2),
            (0.72, 0.50),
            transform=fig.transFigure,
            arrowstyle="->,head_width=4,head_length=3",
            color=feature_groups[i]["color"],
            linewidth=1.5,
            connectionstyle="arc3,rad=0.0",
        )
        fig.patches.append(arrow)

    # Caption
    plt.figtext(0.50, 0.02,
                "Figure 3.3: Radiomics-inspired feature extraction pipeline.",
                fontsize=11, fontfamily="serif", fontstyle="italic",
                color="#444444", ha="center")

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    plt.close()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    images = sorted(IMAGE_DIR.glob("*.png"))
    if not images:
        print("No PNG images found in", IMAGE_DIR)
    else:
        chosen = images[0]
        print(f"Using: {chosen.name}")
        create_figure(chosen)
