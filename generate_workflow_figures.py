"""
Generate Figure 1.1 and Figure 3.1 - Workflow diagrams for thesis.
Run: python generate_workflow_figures.py
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "blue": "#3B82F6",
    "green": "#10B981",
    "amber": "#F59E0B",
    "red": "#EF4444",
    "purple": "#8B5CF6",
}


def generate_figure_1_1():
    """
    Figure 1.1: General workflow of the proposed ultrasound-based radiomics framework.
    Ultrasound Images → Preprocessing → Feature Extraction → Machine Learning / Deep Learning 
    → Disease Classification and Severity Assessment
    """
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    # Define boxes
    boxes = [
        {"x": 0.5, "label": "Ultrasound\nImages", "color": COLORS["blue"]},
        {"x": 2.8, "label": "Preprocessing", "color": COLORS["green"]},
        {"x": 5.1, "label": "Feature\nExtraction", "color": COLORS["amber"]},
        {"x": 7.4, "label": "Machine\nLearning /\nDeep Learning", "color": COLORS["purple"]},
        {"x": 10.2, "label": "Disease\nClassification\nand Severity\nAssessment", "color": COLORS["red"]},
    ]

    # Draw boxes
    for box in boxes:
        if box["label"] == "Machine\nLearning /\nDeep Learning":
            width, height = 2.0, 2.5
        else:
            width, height = 1.8, 2.2
        
        fancy_box = FancyBboxPatch(
            (box["x"] - width/2, 0.65), width, height,
            boxstyle="round,pad=0.1", 
            edgecolor="black", 
            facecolor=box["color"],
            alpha=0.7,
            linewidth=2
        )
        ax.add_patch(fancy_box)
        
        # Add text
        ax.text(
            box["x"], 1.7,
            box["label"],
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color="white"
        )

    # Draw arrows
    arrow_props = dict(
        arrowstyle="-|>",
        lw=2.5,
        mutation_scale=30,
        color="black"
    )
    
    arrow_x_positions = [1.4, 3.95, 6.5, 9.3]
    for x in arrow_x_positions:
        arrow = FancyArrowPatch((x, 1.7), (x + 0.9, 1.7), **arrow_props)
        ax.add_patch(arrow)

    ax.text(7, 3.2, "Figure 1.1: General workflow of the proposed ultrasound-based radiomics framework.",
            ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_1_1_general_workflow.png", edgecolor="none")
    print(f"  Saved: {OUTPUT_DIR / 'figure_1_1_general_workflow.png'}")
    plt.close()


def generate_figure_3_1():
    """
    Figure 3.1: Overview of the proposed ultrasound-based radiomics and AI framework.
    Ultrasound Datasets → Data Cleaning and Label Mapping → Preprocessing 
    → Feature Extraction → ML/DL Model Training → Evaluation and Interpretation
    """
    fig, ax = plt.subplots(figsize=(15, 3.8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    # Define boxes
    boxes = [
        {"x": 0.8, "label": "Ultrasound\nDatasets", "color": COLORS["blue"]},
        {"x": 2.6, "label": "Data Cleaning\nand Label\nMapping", "color": COLORS["green"]},
        {"x": 4.8, "label": "Preprocessing", "color": COLORS["amber"]},
        {"x": 7.0, "label": "Feature\nExtraction", "color": COLORS["purple"]},
        {"x": 9.5, "label": "ML/DL Model\nTraining", "color": "#EC4899"},
        {"x": 12.2, "label": "Evaluation and\nInterpretation", "color": COLORS["red"]},
    ]

    # Draw boxes
    for box in boxes:
        width, height = 1.6, 2.3
        
        fancy_box = FancyBboxPatch(
            (box["x"] - width/2, 0.75), width, height,
            boxstyle="round,pad=0.1", 
            edgecolor="black", 
            facecolor=box["color"],
            alpha=0.7,
            linewidth=2
        )
        ax.add_patch(fancy_box)
        
        # Add text
        ax.text(
            box["x"], 1.85,
            box["label"],
            ha="center", va="center",
            fontsize=10, fontweight="bold",
            color="white"
        )

    # Draw arrows
    arrow_props = dict(
        arrowstyle="-|>",
        lw=2.5,
        mutation_scale=30,
        color="black"
    )
    
    arrow_x_positions = [1.6, 3.7, 5.9, 8.25, 10.85]
    for x in arrow_x_positions:
        arrow = FancyArrowPatch((x, 1.85), (x + 0.8, 1.85), **arrow_props)
        ax.add_patch(arrow)

    ax.text(7.5, 3.5, "Figure 3.1: Overview of the proposed ultrasound-based radiomics and AI framework.",
            ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_3_1_methodology_overview.png", edgecolor="none")
    print(f"  Saved: {OUTPUT_DIR / 'figure_3_1_methodology_overview.png'}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING WORKFLOW FIGURES (1.1 and 3.1)")
    print("=" * 60)
    
    print("Generating Figure 1.1...")
    generate_figure_1_1()
    
    print("Generating Figure 3.1...")
    generate_figure_3_1()
    
    print("=" * 60)
    print("WORKFLOW FIGURES GENERATED SUCCESSFULLY")
    print("=" * 60)
