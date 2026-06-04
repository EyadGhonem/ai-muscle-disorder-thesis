"""
Generate Figure 3.2: Automatic foreground mask generation pipeline.
Uses a real ultrasound image from the dataset to show each step.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_ROOT / "data" / "ULTRASOUND_LABELD_1" / "images"
OUTPUT_PATH = PROJECT_ROOT / "output" / "figure_3_2_mask_generation.png"


def generate_mask_pipeline(image_path: Path):
    """Generate each step of the mask generation pipeline."""
    # Step 1: Original image
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        from PIL import Image
        original = np.array(Image.open(image_path).convert("RGB"))
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Step 2: Grayscale conversion
    grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Step 3: Thresholding (Otsu's method)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

    # Fill small holes
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask_filled = np.zeros_like(cleaned)
        cv2.drawContours(mask_filled, [largest], -1, 255, -1)
        cleaned = mask_filled

    # Step 5: Final mask overlay
    mask_overlay = original_rgb.copy()
    mask_color = np.zeros_like(original_rgb)
    mask_color[:, :, 1] = 120  # Green tint
    mask_color[:, :, 0] = 60   # Slight red
    mask_bool = cleaned > 0
    mask_overlay[mask_bool] = cv2.addWeighted(
        original_rgb[mask_bool], 0.7,
        mask_color[mask_bool], 0.3, 0
    )
    # Draw contour outline
    contours_final, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_overlay, contours_final, -1, (0, 255, 100), 2)

    return original_rgb, grayscale, thresholded, cleaned, mask_overlay


def create_figure(image_path: Path):
    """Create the thesis figure."""
    steps = generate_mask_pipeline(image_path)
    step_labels = [
        "(a) Original\nUltrasound Image",
        "(b) Grayscale\nConversion",
        "(c) Thresholding",
        "(d) Morphological\nCleaning",
        "(e) Foreground\nMask",
    ]
    cmaps = [None, "gray", "gray", "gray", None]

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    fig.patch.set_facecolor("white")

    for i, (ax, img, label, cmap) in enumerate(zip(axes, steps, step_labels, cmaps)):
        ax.imshow(img, cmap=cmap)
        ax.set_title(label, fontsize=10, fontweight="bold", pad=8,
                     fontfamily="serif")
        ax.axis("off")

    # Draw arrows between subplots
    for i in range(4):
        # Get positions in figure coordinates
        bbox_left = axes[i].get_position()
        bbox_right = axes[i + 1].get_position()

        arrow_y = bbox_left.y0 + bbox_left.height / 2
        arrow_x_start = bbox_left.x1 + 0.005
        arrow_x_end = bbox_right.x0 - 0.005

        arrow = FancyArrowPatch(
            (arrow_x_start, arrow_y),
            (arrow_x_end, arrow_y),
            transform=fig.transFigure,
            arrowstyle="->,head_width=6,head_length=4",
            color="#333333",
            linewidth=1.8,
            mutation_scale=1,
        )
        fig.patches.append(arrow)

    plt.suptitle(
        "Figure 3.2: Automatic foreground mask generation used for ultrasound feature extraction",
        fontsize=11, fontfamily="serif", y=0.02, fontstyle="italic",
        color="#444444",
    )

    plt.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.12, wspace=0.15)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    plt.close()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    # Pick a representative image
    images = sorted(IMAGE_DIR.glob("*.png"))
    if not images:
        print("No PNG images found in", IMAGE_DIR)
    else:
        # Use the first image (or pick one that looks good)
        chosen = images[0]
        print(f"Using: {chosen.name}")
        create_figure(chosen)
