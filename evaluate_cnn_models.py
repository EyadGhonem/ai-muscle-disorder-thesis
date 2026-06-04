"""
Evaluate trained CNN models (ResNet50, DenseNet121, EfficientNetB0)
with the same metrics format as run_final_thesis_evaluation.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tensorflow import keras

from thesis_metrics import compute_binary_metrics

PROJECT_ROOT = Path(__file__).resolve().parent
DL_DIR = PROJECT_ROOT / "output" / "dl_models"
FINAL_DIR = PROJECT_ROOT / "output" / "thesis_final"
IMG_SIZE = (224, 224)

ARCH_DISPLAY = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "efficientnetb0": "EfficientNetB0",
}


def load_preprocess(architecture: str):
    if architecture == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif architecture == "densenet121":
        from tensorflow.keras.applications.densenet import preprocess_input
    elif architecture == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    else:
        raise ValueError(architecture)
    return preprocess_input


def load_image_rgb(path: Path, preprocess_fn):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image
        img = np.array(Image.open(path).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = preprocess_fn(img.astype(np.float32))
    return img


def predict_binary_model(model, df: pd.DataFrame, preprocess_fn):
    preds = []
    y_true = df["label"].astype(int).values
    for path in df["filepath"]:
        batch = np.expand_dims(load_image_rgb(Path(path), preprocess_fn), axis=0)
        score = float(model.predict(batch, verbose=0)[0][0])
        preds.append(1 if score >= 0.5 else 0)
    return y_true, np.array(preds)


def evaluate_task(task: str, architectures: list[str] | None = None):
    if architectures is None:
        architectures = ["resnet50", "densenet121", "efficientnetb0"]

    split_path = DL_DIR / f"val_split_{task}.csv"
    if not split_path.exists():
        print(f"Skip CNN eval ({task}): missing {split_path} — run train_ultrasound_cnn_models.py first")
        return []

    val_df = pd.read_csv(split_path)
    rows = []

    for arch in architectures:
        model_path = DL_DIR / f"{arch}_{task}.keras"
        if not model_path.exists():
            print(f"Skip {arch} ({task}): model not found")
            continue

        print(f"Evaluating {ARCH_DISPLAY[arch]} on {task} validation set ({len(val_df)} images)...")
        model = keras.models.load_model(model_path)
        preprocess_fn = load_preprocess(arch)
        y_true, y_pred = predict_binary_model(model, val_df, preprocess_fn)
        metrics = compute_binary_metrics(y_true, y_pred)

        display = ARCH_DISPLAY[arch]
        rows.append(
            {
                "task": task,
                "model_family": "Deep Learning (CNN)",
                "model": display,
                **metrics,
            }
        )

        cm_path = FINAL_DIR / f"confusion_matrix_{display.replace(' ', '_')}_{task}.csv"
        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            metrics["confusion_matrix"],
            index=["true_0", "true_1"],
            columns=["pred_0", "pred_1"],
        ).to_csv(cm_path)

        print(
            f"  {display}: acc={metrics['accuracy']:.4f} | f1_macro={metrics['f1_macro']:.4f}"
        )

    return rows


def evaluate_all_cnn_rows():
    rows = []
    rows.extend(evaluate_task("binary"))
    rows.extend(evaluate_task("severity"))
    return rows


def main():
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = evaluate_all_cnn_rows()
    if not rows:
        print("No CNN models evaluated.")
        return

    df = pd.DataFrame(rows)
    out = FINAL_DIR / "cnn_model_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(df[["task", "model", "accuracy", "f1_macro"]].to_string(index=False))


if __name__ == "__main__":
    main()
