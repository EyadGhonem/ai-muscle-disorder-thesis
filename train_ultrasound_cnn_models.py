"""
Train ultrasound CNN classifiers: ResNet50, DenseNet121, EfficientNetB0.

Usage examples:
  python train_ultrasound_cnn_models.py
  python train_ultrasound_cnn_models.py --models resnet50 densenet121
  python train_ultrasound_cnn_models.py --task severity --models resnet50 densenet121
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121, EfficientNetB0, ResNet50
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "dl_models"
RESULTS_CSV = PROJECT_ROOT / "output" / "dl_cnn_training_results.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # Reduced from 20 for faster training
RANDOM_STATE = 42


def get_preprocess_fn(architecture: str):
    if architecture == "resnet50":
        return resnet_preprocess
    if architecture == "densenet121":
        return densenet_preprocess
    if architecture == "efficientnetb0":
        return efficientnet_preprocess
    raise ValueError(f"Unknown architecture: {architecture}")


def build_model(architecture: str, num_classes: int, learning_rate: float = 1e-3):
    """Transfer-learning head on ImageNet backbones."""
    if architecture == "resnet50":
        base = ResNet50(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    elif architecture == "densenet121":
        base = DenseNet121(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    elif architecture == "efficientnetb0":
        base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy", keras.metrics.AUC(name="auc")]
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model = models.Model(base.input, outputs, name=architecture)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model


def load_binary_small_dataset():
    """~309 images in data/ultrasound_images with output/labels.csv.
    Falls back to FSHD severity if binary dataset doesn't exist."""
    labels_path = PROJECT_ROOT / "output" / "labels.csv"
    image_dir = PROJECT_ROOT / "data" / "ultrasound_images"
    if not labels_path.exists() or not image_dir.exists():
        # Fall back to severity dataset
        print("Binary dataset not found, falling back to severity dataset...")
        return load_fshd_severity_dataset()

    df = pd.read_csv(labels_path)
    df = df[["image_name", "label"]].copy()
    df["label"] = df["label"].astype(str)
    df["filepath"] = df["image_name"].apply(lambda n: str(image_dir / n))
    df = df[df["filepath"].apply(lambda p: Path(p).exists())]
    return df, image_dir, 2, "binary"


def load_fshd_severity_dataset():
    """FSHD images from ULTRASOUND_LABELD_1 with binary severity."""
    master = PROJECT_ROOT / "output" / "final_ultrasound_dataset.csv"
    if not master.exists():
        raise FileNotFoundError(f"Missing {master}")

    df = pd.read_csv(master)
    df = df[df["dataset_source"] == "ULTRASOUND_LABELD_1"].copy()
    df["filepath"] = df["image_path"].apply(lambda p: str(PROJECT_ROOT / p))
    df = df[df["filepath"].apply(lambda p: Path(p).exists())]
    df = df.dropna(subset=["severity"])
    df["label"] = df["severity"].astype(int).astype(str)
    df["image_name"] = df["image_path"].apply(lambda p: Path(p).name)
    image_dir = PROJECT_ROOT / "data" / "ULTRASOUND_LABELD_1" / "images"
    return df[["image_name", "label", "filepath", "patient_id"]], image_dir, 2, "severity"


def load_multiclass_labeled1(max_per_class: int | None = 800):
    """Multi-class disease labels where PNG files exist (mostly FSHD only)."""
    master = PROJECT_ROOT / "output" / "final_ultrasound_dataset.csv"
    df = pd.read_csv(master)
    df = df[df["dataset_source"] == "ULTRASOUND_LABELD_1"].copy()
    df["filepath"] = df["image_path"].apply(lambda p: str(PROJECT_ROOT / p))
    df = df[df["filepath"].apply(lambda p: Path(p).exists())]
    invalid = {"", "NAN", "Unknown", "nan"}
    df = df[~df["disease"].isin(invalid)]

    if max_per_class:
        df = (
            df.groupby("disease", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=RANDOM_STATE))
            .reset_index(drop=True)
        )

    classes = sorted(df["disease"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label"] = df["disease"].map(class_to_idx).astype(int)
    df["image_name"] = df["image_path"].apply(lambda p: Path(p).name)
    image_dir = PROJECT_ROOT / "data" / "ULTRASOUND_LABELD_1" / "images"
    return df, image_dir, len(classes), "multiclass", classes


def make_generators(df, image_dir, num_classes, preprocess_fn, augment_train=True):
    if num_classes == 2:
        class_mode = "binary"
        strat = df["label"]
    else:
        class_mode = "sparse"
        strat = df["label"]

    if "patient_id" in df.columns and df["patient_id"].nunique() > 10:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        train_idx, val_idx = next(
            splitter.split(df, groups=df["patient_id"])
        )
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        print(f"Split: patient-level | train={len(train_df)} val={len(val_df)}")
    else:
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=RANDOM_STATE, stratify=strat
        )
        print(f"Split: stratified random | train={len(train_df)} val={len(val_df)}")

    if augment_train:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest",
        )
    else:
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    # flow_from_dataframe needs directory + filename
    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        directory=str(image_dir),
        x_col="image_name",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=class_mode,
        color_mode="rgb",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        directory=str(image_dir),
        x_col="image_name",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=class_mode,
        color_mode="rgb",
        shuffle=False,
    )
    return train_gen, val_gen, train_df, val_df


def train_one(architecture: str, train_gen, val_gen, num_classes: int):
    model = build_model(architecture, num_classes)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


def plot_history(history, path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title(f"{title} — accuracy")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title(f"{title} — loss")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 / DenseNet121 / EfficientNetB0")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "densenet121"],
        choices=["resnet50", "densenet121", "efficientnetb0"],
    )
    parser.add_argument(
        "--task",
        default="all",
        choices=["all", "binary", "severity", "multiclass"],
        help="all=binary+severity (recommended, like other thesis models)",
    )
    parser.add_argument("--max-per-class", type=int, default=800)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    task_list = ["binary", "severity"] if args.task == "all" else [args.task]
    all_rows = []

    for task_name in task_list:
        if task_name == "binary":
            df, image_dir, num_classes, task_name = load_binary_small_dataset()
            class_names = None
        elif task_name == "severity":
            df, image_dir, num_classes, task_name = load_fshd_severity_dataset()
            class_names = ["severity_0", "severity_1"]
        else:
            df, image_dir, num_classes, task_name, class_names = load_multiclass_labeled1(
                args.max_per_class
            )
            print(f"Classes: {class_names}")
            if num_classes < 2:
                print(f"Skip multiclass: only {num_classes} class with images.")
                continue

        print(f"\nTask={task_name} | samples={len(df)} | classes={num_classes}")
        task_rows = []

        for arch in args.models:
            print("\n" + "=" * 60)
            print(f"Training {arch.upper()} ({task_name})")
            print("=" * 60)

            preprocess_fn = get_preprocess_fn(arch)
            train_gen, val_gen, train_df, val_df = make_generators(
                df, image_dir, num_classes, preprocess_fn
            )

            # Save validation split for same metrics pipeline as other models
            val_export = val_df[["image_name", "label", "filepath"]].copy()
            if "patient_id" in val_df.columns:
                val_export["patient_id"] = val_df["patient_id"]
            val_export.to_csv(OUTPUT_DIR / f"val_split_{task_name}.csv", index=False)

            model, history = train_one(arch, train_gen, val_gen, num_classes)

            model_path = OUTPUT_DIR / f"{arch}_{task_name}.keras"
            model.save(model_path)
            hist_path = OUTPUT_DIR / f"{arch}_{task_name}_history.png"
            plot_history(history, hist_path, f"{arch} ({task_name})")

            best_val_acc = float(max(history.history["val_accuracy"]))
            best_val_loss = float(min(history.history["val_loss"]))
            task_rows.append(
                {
                    "architecture": arch,
                    "task": task_name,
                    "num_classes": num_classes,
                    "train_samples": len(train_df),
                    "val_samples": len(val_df),
                    "best_val_accuracy": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "model_path": str(model_path),
                }
            )
            print(f"Saved: {model_path}")

        all_rows.extend(task_rows)
        meta = {
            "task": task_name,
            "class_names": class_names,
            "models": args.models,
        }
        (OUTPUT_DIR / f"training_meta_{task_name}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    results = pd.DataFrame(all_rows)
    results.to_csv(RESULTS_CSV, index=False)

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE")
    print("=" * 60)
    if not results.empty:
        print(results.to_string(index=False))
    print(f"\nResults CSV: {RESULTS_CSV}")
    print("Next: python evaluate_cnn_models.py")
    print("      python run_final_thesis_evaluation.py")


if __name__ == "__main__":
    main()
