#!/usr/bin/env python3
"""
Train 4-class disease CNNs on MAT-labeled ultrasound images for the GUI.
Saves to gui_demo/models/ (does not overwrite output/dl_models severity weights).

  python scripts/train_gui_disease_cnns.py
  python scripts/train_gui_disease_cnns.py --epochs 3 --models resnet50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gui_demo.paths import mat_image_root  # noqa: E402

GUI_MODELS = PROJECT_ROOT / "gui_demo" / "models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_STATE = 42

ARCHITECTURES = {
    "resnet50": ("ResNet50", "tensorflow.keras.applications.resnet50", "ResNet50", "resnet_preprocess"),
    "densenet121": ("DenseNet121", "tensorflow.keras.applications.densenet", "DenseNet121", "densenet_preprocess"),
    "efficientnetb0": ("EfficientNetB0", "tensorflow.keras.applications.efficientnet", "EfficientNetB0", "efficientnet_preprocess"),
    "mobilenetv2": ("MobileNetV2", "tensorflow.keras.applications.mobilenet_v2", "MobileNetV2", "mobilenet_preprocess"),
}


def load_mat_dataframe(root: Path) -> pd.DataFrame:
    rows = []
    skip = {"Unknown", "UNKNOWN", "unknown"}
    for folder in root.iterdir():
        if not folder.is_dir() or folder.name in skip:
            continue
        for img in folder.glob("*.png"):
            rows.append(
                {
                    "image_name": img.name,
                    "filepath": str(img),
                    "disease": folder.name,
                    "patient_id": img.name.split("_")[0] if "_" in img.name else img.stem,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No PNG files under {root}")
    return pd.DataFrame(rows)


def build_model(architecture: str, num_classes: int):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models

    if architecture == "resnet50":
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input

        base = ResNet50(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    elif architecture == "densenet121":
        from tensorflow.keras.applications import DenseNet121
        from tensorflow.keras.applications.densenet import preprocess_input

        base = DenseNet121(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    elif architecture == "efficientnetb0":
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.applications.efficientnet import preprocess_input

        base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    elif architecture == "mobilenetv2":
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    else:
        raise ValueError(architecture)

    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(base.input, out, name=f"{architecture}_disease")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, preprocess_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(ARCHITECTURES.keys()),
        choices=list(ARCHITECTURES.keys()),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-per-class", type=int, default=400)
    args = parser.parse_args()

    root = mat_image_root()
    if root is None:
        raise FileNotFoundError(
            "MAT image folder not found. Expected data/images_extracted_from_mat_LABELED "
            "or data/dl_images_extracted_from_mat"
        )

    df = load_mat_dataframe(root)
    df["disease"] = df["disease"].replace({"IBM": "Inclusion Body Myositis"})

    if args.max_per_class:
        parts = []
        for _, g in df.groupby("disease"):
            parts.append(
                g.sample(min(len(g), args.max_per_class), random_state=RANDOM_STATE)
            )
        df = pd.concat(parts, ignore_index=True)
    classes = sorted(df["disease"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label"] = df["disease"].map(class_to_idx).astype(int)
    df["image_dir"] = df["filepath"].apply(lambda p: str(Path(p).parent))

    GUI_MODELS.mkdir(parents=True, exist_ok=True)
    meta = {"classes": classes, "class_to_idx": class_to_idx, "task": "disease_multiclass_mat"}
    (GUI_MODELS / "disease_label_classes.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )
    print(f"MAT disease CNN | train={len(train_df)} val={len(val_df)} | classes={classes}")

    results = []
    for arch in args.models:
        print(f"\n=== Training {arch} ===")
        model, preprocess_fn = build_model(arch, len(classes))

        import cv2

        def batch_generator(sub_df, shuffle):
            paths = sub_df["filepath"].values
            labels = sub_df["label"].values
            while True:
                order = np.random.permutation(len(paths)) if shuffle else np.arange(len(paths))
                for start in range(0, len(order), BATCH_SIZE):
                    batch_idx = order[start : start + BATCH_SIZE]
                    xs, ys = [], []
                    for i in batch_idx:
                        import cv2

                        bgr = cv2.imread(str(paths[i]))
                        if bgr is None:
                            continue
                        rgb = cv2.cvtColor(cv2.resize(bgr, IMG_SIZE), cv2.COLOR_BGR2RGB)
                        xs.append(preprocess_fn(rgb.astype(np.float32)))
                        ys.append(labels[i])
                    if xs:
                        yield np.stack(xs), np.array(ys)

        steps_train = max(1, len(train_df) // BATCH_SIZE)
        steps_val = max(1, len(val_df) // BATCH_SIZE)

        history = model.fit(
            batch_generator(train_df, True),
            steps_per_epoch=steps_train,
            validation_data=batch_generator(val_df, False),
            validation_steps=steps_val,
            epochs=args.epochs,
            verbose=1,
        )

        out_path = GUI_MODELS / f"{arch}_disease.keras"
        model.save(out_path)
        best_acc = float(max(history.history["val_accuracy"]))
        results.append({"architecture": arch, "val_accuracy": best_acc, "path": str(out_path)})
        print(f"Saved {out_path} (best val acc {best_acc:.4f})")

    pd.DataFrame(results).to_csv(GUI_MODELS / "training_summary.csv", index=False)
    print("Done. GUI models in", GUI_MODELS)


if __name__ == "__main__":
    main()
