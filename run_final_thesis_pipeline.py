#!/usr/bin/env python3
"""
Final thesis polish pipeline — outputs only to results/final_a_plus_polish/.
Does not modify results/a_plus_full_improvements/ or output/thesis_final/.
"""

from __future__ import annotations

import json
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
OUT = PROJECT_ROOT / "results" / "final_a_plus_polish"
MODELS_DIR = OUT / "models"
GRADCAM_DIR = OUT / "gradcam"
A_PLUS_DIR = PROJECT_ROOT / "results" / "a_plus_full_improvements"
DATASET_PATH = PROJECT_ROOT / "output" / "final_ultrasound_dataset.csv"
OLD_DL_DIR = PROJECT_ROOT / "output" / "dl_models"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5
CNN_EPOCHS = 5
IMG_SIZE = (224, 224)

METADATA = {
    "image_path",
    "patient_id",
    "disease",
    "severity",
    "severity_label",
    "dataset_source",
}

STATUS: dict[str, str] = {}
GENERATED: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)


def out_path(rel: str) -> Path:
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    GENERATED.append(str(p.relative_to(PROJECT_ROOT)))
    return p


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_ml_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["disease", "patient_id"])
    invalid = {"", "Unknown", "unknown", "nan", "NAN", "NaN"}
    df["disease"] = df["disease"].astype(str).str.strip()
    df = df[~df["disease"].isin(invalid)]
    feats = [c for c in df.columns if c not in METADATA]
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    valid = ~X.isna().all(axis=1)
    df, X = df.loc[valid].reset_index(drop=True), X.loc[valid].reset_index(drop=True)
    le = LabelEncoder()
    y = le.fit_transform(df["disease"])
    groups = df["patient_id"].astype(str).values
    return df, X, y, groups, le, feats


def patient_holdout_split(df, X, y, groups):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups=groups))
    overlap = set(groups[tr]) & set(groups[te])
    return tr, te, len(overlap)


def manual_random_oversample(X: pd.DataFrame, y: np.ndarray):
    """Oversample minority classes to majority count (train only)."""
    rng = np.random.default_rng(RANDOM_STATE)
    classes, counts = np.unique(y, return_counts=True)
    max_n = counts.max()
    idx_all = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        idx_all.extend(rng.choice(idx, max_n, replace=True))
    idx_all = np.array(idx_all)
    rng.shuffle(idx_all)
    return X.iloc[idx_all].reset_index(drop=True), y[idx_all]


def apply_smote(X: pd.DataFrame, y: np.ndarray):
    try:
        from imblearn.over_sampling import SMOTE

        k = min(5, int(min(np.bincount(y))) - 1)
        if k < 1:
            return None, None, "too few samples for SMOTE k_neighbors"
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
        Xr, yr = smote.fit_resample(
            SimpleImputer(strategy="median").fit_transform(X), y
        )
        return pd.DataFrame(Xr, columns=X.columns), yr, None
    except Exception as exc:
        return None, None, str(exc)


def build_models():
    models = {
        "SVM": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced")),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=RANDOM_STATE,
                        eval_metric="mlogloss",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        log("  XGBoost not installed — skipped.")

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        log("  LightGBM not installed — skipped.")

    return models


def evaluate_split(model, X_train, y_train, X_test, y_test, le, model_name, strategy):
    """Fit and return summary + per-class rows."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    labels = list(range(len(le.classes_)))

    summary = {
        "model": model_name,
        "strategy": strategy,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    prec = precision_score(y_test, y_pred, average=None, zero_division=0, labels=labels)
    rec = recall_score(y_test, y_pred, average=None, zero_division=0, labels=labels)
    f1c = f1_score(y_test, y_pred, average=None, zero_division=0, labels=labels)

    per_class = []
    for i, cls in enumerate(le.classes_):
        per_class.append(
            {
                **summary,
                "class": cls,
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1c[i]),
            }
        )
    return summary, per_class


def load_original_baseline():
    p = A_PLUS_DIR / "patient_level_results.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "model": r["model"],
                "strategy": "original_a_plus_baseline",
                "accuracy": r["accuracy"],
                "balanced_accuracy": np.nan,
                "f1_macro": r["f1_macro"],
                "f1_weighted": r["f1_weighted"],
                "class": "_overall",
                "precision": r.get("precision_macro", np.nan),
                "recall": r.get("recall_macro", np.nan),
                "f1": r["f1_macro"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# TASK 1 — Imbalance handling
# ---------------------------------------------------------------------------


def task1_imbalance(df, X, y, groups, le):
    log("\n=== TASK 1: Class imbalance handling ===")
    tr, te, overlap = patient_holdout_split(df, X, y, groups)
    log(f"  Patient overlap: {overlap} (expect 0)")
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = y[tr], y[te]

    models = build_models()
    summaries, per_class_rows = [], []

    strategies = [
        ("balanced_class_weight", lambda Xt, yt: (Xt, yt)),
    ]

    # Random oversampling (manual — no imblearn required)
    X_ro, y_ro = manual_random_oversample(X_train.copy(), y_train.copy())
    strategies.append(("random_oversample", lambda Xt, yt: (X_ro, y_ro)))

    # SMOTE if imblearn available
    X_sm, y_sm, smote_err = apply_smote(X_train, y_train)
    if smote_err:
        log(f"  SMOTE skipped: {smote_err}")
        out_path("smote_limitation.txt").write_text(
            f"SMOTE not applied: {smote_err}\nInstall: pip install imbalanced-learn\n",
            encoding="utf-8",
        )
    else:
        strategies.append(("smote", lambda Xt, yt, _x=X_sm, _y=y_sm: (_x, _y)))
        log("  SMOTE applied on training set.")

    for model_name, model_template in models.items():
        for strategy_name, train_fn in strategies:
            from sklearn.base import clone

            model = clone(model_template)
            Xt, yt = train_fn(X_train, y_train)
            log(f"  {model_name} | {strategy_name} | train n={len(yt)}")
            summary, per_rows = evaluate_split(
                model, Xt, yt, X_test, y_test, le, model_name, strategy_name
            )
            summaries.append(summary)
            per_class_rows.extend(per_rows)

    per_class_rows.extend(load_original_baseline())

    detail = pd.DataFrame(per_class_rows)
    detail.to_csv(out_path("imbalance_handling_results.csv"), index=False)

    # Comparison plot: macro F1 by model x strategy (overall rows only)
    overall = detail[detail["class"] == "_overall"].copy()
    if overall.empty:
        overall = detail.groupby(["model", "strategy"], as_index=False).agg(
            {"f1_macro": "mean", "accuracy": "mean", "balanced_accuracy": "mean"}
        )
    else:
        overall = overall.rename(columns={"f1": "f1_macro"})

    plot_df = detail[detail["class"] != "_overall"].groupby(
        ["model", "strategy"], as_index=False
    ).agg({"f1_macro": "first", "balanced_accuracy": "first"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=plot_df, x="model", y="f1_macro", hue="strategy", ax=axes[0])
    axes[0].set_title("Macro F1 — imbalance strategies")
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(data=plot_df, x="model", y="balanced_accuracy", hue="strategy", ax=axes[1])
    axes[1].set_title("Balanced accuracy")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(out_path("imbalance_comparison.png"), dpi=150)
    plt.close()

    best = plot_df.sort_values("f1_macro", ascending=False).iloc[0]
    STATUS["task1_best"] = f"{best['model']} + {best['strategy']} (F1_macro={best['f1_macro']:.4f})"
    STATUS["task1"] = "completed"
    log(f"  Saved imbalance results. Best: {STATUS['task1_best']}")
    return detail, plot_df


# ---------------------------------------------------------------------------
# TASK 2 — GroupKFold CV
# ---------------------------------------------------------------------------


def task2_groupkfold(df, X, y, groups, le):
    log("\n=== TASK 2: Patient-level GroupKFold CV ===")
    models = build_models()
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    fold_rows = []

    for model_name, model in models.items():
        log(f"  CV: {model_name}")
        accs, baccs, f1m, f1w = [], [], [], []
        for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
            from sklearn.base import clone

            m = clone(model)
            m.fit(X.iloc[tr], y[tr])
            pred = m.predict(X.iloc[va])
            accs.append(accuracy_score(y[va], pred))
            baccs.append(balanced_accuracy_score(y[va], pred))
            f1m.append(f1_score(y[va], pred, average="macro", zero_division=0))
            f1w.append(f1_score(y[va], pred, average="weighted", zero_division=0))
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "accuracy": accs[-1],
                    "balanced_accuracy": baccs[-1],
                    "f1_macro": f1m[-1],
                    "f1_weighted": f1w[-1],
                }
            )

        log(
            f"    macro F1: {np.mean(f1m):.4f} +/- {np.std(f1m):.4f} | "
            f"balanced acc: {np.mean(baccs):.4f} +/- {np.std(baccs):.4f}"
        )

    fold_df = pd.DataFrame(fold_rows)
    summary = (
        fold_df.groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            f1_weighted_mean=("f1_weighted", "mean"),
            f1_weighted_std=("f1_weighted", "std"),
        )
        .reset_index()
    )
    summary.to_csv(out_path("groupkfold_cv_results.csv"), index=False)
    fold_df.to_csv(out_path("groupkfold_cv_folds_detail.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=summary,
        x="model",
        y="f1_macro_mean",
        ax=ax,
        color="steelblue",
        errorbar=None,
    )
    ax.errorbar(
        x=range(len(summary)),
        y=summary["f1_macro_mean"],
        yerr=summary["f1_macro_std"],
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.set_ylabel("Macro F1 (mean +/- std)")
    ax.set_title(f"{N_CV_FOLDS}-fold GroupKFold (patient-level)")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(out_path("groupkfold_macro_f1.png"), dpi=150)
    plt.close()

    best_cv = summary.sort_values("f1_macro_mean", ascending=False).iloc[0]
    STATUS["task2"] = "completed"
    STATUS["task2_best"] = (
        f"{best_cv['model']} F1_macro={best_cv['f1_macro_mean']:.4f}"
        f"+/-{best_cv['f1_macro_std']:.4f}"
    )
    log(f"  Saved GroupKFold results. Best CV: {STATUS['task2_best']}")
    return summary


# ---------------------------------------------------------------------------
# TASK 3 — DenseNet121 (+ DL comparison table)
# ---------------------------------------------------------------------------


def _train_cnn_arch(arch: str, out_dir: Path) -> dict | None:
    """Train one CNN on FSHD severity; save to out_dir (not output/dl_models)."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import DenseNet121, EfficientNetB0, ResNet50
        from tensorflow.keras.applications.densenet import preprocess_input as dense_pre
        from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
        from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError as e:
        return {"error": str(e)}

    master = DATASET_PATH
    df = pd.read_csv(master)
    df = df[df["dataset_source"] == "ULTRASOUND_LABELD_1"].copy()
    df["filepath"] = df["image_path"].apply(lambda p: str(PROJECT_ROOT / p))
    df = df[df["filepath"].apply(lambda p: Path(p).exists())]
    df = df.dropna(subset=["severity"])
    df["label"] = df["severity"].astype(int).astype(str)
    df["image_name"] = df["image_path"].apply(lambda p: Path(p).name)
    image_dir = PROJECT_ROOT / "data" / "ULTRASOUND_LABELD_1" / "images"

    pre_map = {
        "resnet50": (ResNet50, res_pre),
        "densenet121": (DenseNet121, dense_pre),
        "efficientnetb0": (EfficientNetB0, eff_pre),
    }
    if arch not in pre_map:
        return {"error": f"unknown arch {arch}"}

    BaseCls, pre_fn = pre_map[arch]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, va = next(gss.split(df, groups=df["patient_id"]))
    train_df, val_df = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)

    val_export = val_df[["image_name", "label", "filepath", "patient_id"]]
    val_export.to_csv(out_dir / "val_split_severity.csv", index=False)

    train_gen = ImageDataGenerator(
        preprocessing_function=pre_fn,
        rotation_range=15,
        horizontal_flip=True,
    ).flow_from_dataframe(
        train_df,
        directory=str(image_dir),
        x_col="image_name",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode="binary",
        shuffle=True,
    )
    val_gen = ImageDataGenerator(preprocessing_function=pre_fn).flow_from_dataframe(
        val_df,
        directory=str(image_dir),
        x_col="image_name",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode="binary",
        shuffle=False,
    )

    base = BaseCls(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(base.input, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    hist = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CNN_EPOCHS,
        verbose=1,
    )
    model_path = out_dir / f"{arch}_severity.keras"
    model.save(model_path)
    best_acc = float(max(hist.history["val_accuracy"]))
    return {
        "architecture": arch,
        "model_path": str(model_path),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "best_val_accuracy": best_acc,
        "epochs": CNN_EPOCHS,
    }


def _eval_cnn(model_path: Path, arch: str, val_csv: Path) -> dict | None:
    try:
        import cv2
        from tensorflow import keras
    except ImportError:
        return None

    pre_map = {
        "resnet50": "tensorflow.keras.applications.resnet50",
        "densenet121": "tensorflow.keras.applications.densenet",
        "efficientnetb0": "tensorflow.keras.applications.efficientnet",
    }
    import importlib

    mod = importlib.import_module(pre_map[arch])
    pre_fn = mod.preprocess_input

    if not model_path.exists() or not val_csv.exists():
        return None

    model = keras.models.load_model(model_path)
    val_df = pd.read_csv(val_csv)
    y_true, y_pred = [], []
    for _, row in val_df.iterrows():
        path = Path(row["filepath"])
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
        x = np.expand_dims(pre_fn(img.astype(np.float32)), 0)
        score = float(model.predict(x, verbose=0)[0][0])
        y_pred.append(1 if score >= 0.5 else 0)
        y_true.append(int(row["label"]))

    if not y_true:
        return None
    from thesis_metrics import compute_binary_metrics

    m = compute_binary_metrics(np.array(y_true), np.array(y_pred))
    return {
        "architecture": arch,
        "accuracy": m["accuracy"],
        "f1_macro": m["f1_macro"],
        "n_val": len(y_true),
        "model_path": str(model_path),
    }


def task3_densenet_and_dl_comparison():
    log("\n=== TASK 3: DenseNet121 + DL comparison ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    candidates = {
        "densenet121": MODELS_DIR / "densenet121_severity.keras",
        "resnet50": MODELS_DIR / "resnet50_severity.keras",
        "efficientnetb0": MODELS_DIR / "efficientnetb0_severity.keras",
    }
    # Also check legacy paths without overwriting
    legacy = {
        "densenet121": OLD_DL_DIR / "densenet121_severity.keras",
        "resnet50": OLD_DL_DIR / "resnet50_severity.keras",
        "efficientnetb0": OLD_DL_DIR / "efficientnetb0_severity.keras",
    }

    val_csv = MODELS_DIR / "val_split_severity.csv"
    if not val_csv.exists() and (OLD_DL_DIR / "val_split_severity.csv").exists():
        val_csv = OLD_DL_DIR / "val_split_severity.csv"

    rows = []
    densenet_done = False

    for arch, polish_path in candidates.items():
        path = polish_path if polish_path.exists() else legacy[arch]
        if not path.exists() and arch == "densenet121":
            log(f"  Training {arch} -> {polish_path} (this may take a while on CPU)...")
            try:
                result = _train_cnn_arch(arch, MODELS_DIR)
                if result and "error" not in result:
                    rows.append(result)
                    densenet_done = True
                    path = polish_path
                else:
                    err = result.get("error", "unknown") if result else "train failed"
                    out_path("densenet121_limitation.txt").write_text(
                        f"DenseNet121 training failed: {err}\n", encoding="utf-8"
                    )
                    STATUS["task3"] = f"failed: {err}"
                    return rows
            except Exception as e:
                out_path("densenet121_limitation.txt").write_text(
                    f"DenseNet121 training failed:\n{e}\n{traceback.format_exc()}\n",
                    encoding="utf-8",
                )
                STATUS["task3"] = f"failed: {e}"
                return rows
        elif not path.exists() and arch in ("resnet50", "efficientnetb0"):
            log(f"  {arch} missing — training to {MODELS_DIR}...")
            try:
                result = _train_cnn_arch(arch, MODELS_DIR)
                if result and "error" not in result:
                    path = MODELS_DIR / f"{arch}_severity.keras"
            except Exception as e:
                log(f"  {arch} training failed: {e}")

        ev = _eval_cnn(path, arch, val_csv)
        if ev:
            rows.append(ev)
            if arch == "densenet121":
                densenet_done = True

    if densenet_done:
        pd.DataFrame([r for r in rows if r.get("architecture") == "densenet121"]).to_csv(
            out_path("densenet121_results.csv"), index=False
        )
        STATUS["task3"] = "completed"
    else:
        if not (OUT / "densenet121_limitation.txt").exists():
            out_path("densenet121_limitation.txt").write_text(
                "DenseNet121 weights not found and training did not complete.\n",
                encoding="utf-8",
            )
        STATUS["task3"] = "not completed"

    if rows:
        pd.DataFrame(rows).to_csv(out_path("cnn_dl_comparison.csv"), index=False)
        log(f"  DL comparison saved ({len(rows)} models).")
    return rows


# ---------------------------------------------------------------------------
# TASK 4 — Grad-CAM (fixed nested backbone)
# ---------------------------------------------------------------------------


def find_last_conv_in_model(model):
    from tensorflow.keras.layers import Conv2D

    def search(m):
        for layer in reversed(m.layers):
            if isinstance(layer, Conv2D):
                return layer
            if hasattr(layer, "layers"):
                found = search(layer)
                if found is not None:
                    return found
        return None

    return search(model)


def make_gradcam_heatmap(grad_model, img_array, pred_index: int):
    import tensorflow as tf

    img_tensor = tf.expand_dims(img_array, axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv = conv_outputs[0]
    heatmap = tf.reduce_sum(conv * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def task4_gradcam():
    log("\n=== TASK 4: Grad-CAM ===")
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as e:
        out_path("gradcam_limitation.txt").write_text(f"Grad-CAM skipped: {e}\n", encoding="utf-8")
        STATUS["task4"] = f"skipped: {e}"
        return

    val_csv = MODELS_DIR / "val_split_severity.csv"
    if not val_csv.exists():
        val_csv = OLD_DL_DIR / "val_split_severity.csv"
    if not val_csv.exists():
        out_path("gradcam_limitation.txt").write_text(
            "No val_split_severity.csv found.\n", encoding="utf-8"
        )
        STATUS["task4"] = "skipped: no val split"
        return

    val_df = pd.read_csv(val_csv)
    archs = ["resnet50", "efficientnetb0", "densenet121"]
    pre_import = {
        "resnet50": "tensorflow.keras.applications.resnet50",
        "densenet121": "tensorflow.keras.applications.densenet",
        "efficientnetb0": "tensorflow.keras.applications.efficientnet",
    }
    saved_total = 0
    errors = []

    for arch in archs:
        paths = [
            MODELS_DIR / f"{arch}_severity.keras",
            OLD_DL_DIR / f"{arch}_severity.keras",
        ]
        model_path = next((p for p in paths if p.exists()), None)
        if model_path is None:
            errors.append(f"{arch}: no .keras file")
            continue

        log(f"  Grad-CAM: {arch} ({model_path.name})")
        import importlib

        pre_fn = importlib.import_module(pre_import[arch]).preprocess_input
        model = keras.models.load_model(model_path)
        last_conv = find_last_conv_in_model(model)
        if last_conv is None:
            errors.append(f"{arch}: no Conv2D layer")
            continue

        grad_model = keras.models.Model(
            inputs=model.input,
            outputs=[last_conv.output, model.output],
        )

        correct_n, wrong_n = 0, 0
        for _, row in val_df.iterrows():
            if correct_n >= 3 and wrong_n >= 3:
                break
            path = Path(row["filepath"])
            if not path.exists():
                path = PROJECT_ROOT / str(row.get("filepath", ""))
            if not path.exists():
                continue

            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                continue
            rgb = cv2.cvtColor(cv2.resize(img_bgr, IMG_SIZE), cv2.COLOR_BGR2RGB)
            x = pre_fn(rgb.astype(np.float32))
            pred = int(model.predict(np.expand_dims(x, 0), verbose=0)[0][0] >= 0.5)
            true = int(row["label"])
            ok = pred == true
            if ok and correct_n >= 3:
                continue
            if not ok and wrong_n >= 3:
                continue

            try:
                heatmap = make_gradcam_heatmap(grad_model, x, pred)
                heatmap = cv2.resize(heatmap, IMG_SIZE)
                hm_uint8 = np.uint8(255 * heatmap)
                hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(rgb, 0.55, hm_color, 0.45, 0)
                tag = "correct" if ok else "wrong"
                fname = f"{arch}_{tag}_t{true}_p{pred}_{path.stem}.png"
                cv2.imwrite(str(GRADCAM_DIR / fname), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                if ok:
                    correct_n += 1
                else:
                    wrong_n += 1
                saved_total += 1
            except Exception as ex:
                errors.append(f"{arch} sample: {ex}")

        log(f"    {arch}: {correct_n} correct, {wrong_n} wrong heatmaps")

    if saved_total > 0:
        STATUS["task4"] = f"completed ({saved_total} images)"
    else:
        msg = "No Grad-CAM images saved.\n" + "\n".join(errors)
        out_path("gradcam_limitation.txt").write_text(msg, encoding="utf-8")
        STATUS["task4"] = "not completed"


# ---------------------------------------------------------------------------
# TASK 5 — Summary + requirements
# ---------------------------------------------------------------------------


def write_requirements():
    reqs = """# Final thesis pipeline dependencies
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
xgboost>=1.7
lightgbm>=3.3
shap>=0.42
tensorflow>=2.12
opencv-python>=4.7
imbalanced-learn>=0.11
"""
    out_path("requirements_final.txt").write_text(reqs, encoding="utf-8")


def write_summary(imbalance_df, cv_summary):
    # Compare macro F1 improvement
    imp = ""
    if imbalance_df is not None and len(imbalance_df):
        sub = imbalance_df[imbalance_df["class"] != "_overall"]
        if len(sub):
            base = sub[sub["strategy"] == "balanced_class_weight"]["f1_macro"].mean()
            best_row = sub.sort_values("f1_macro", ascending=False).iloc[0]
            imp = (
                f"Baseline balanced-weight mean macro F1: {base:.4f}\n"
                f"Best strategy: {best_row['model']} + {best_row['strategy']} "
                f"macro F1={best_row['f1_macro']:.4f}\n"
                f"Improvement: {best_row['f1_macro'] - base:+.4f}\n"
            )

    lines = [
        "FINAL IMPLEMENTATION SUMMARY",
        "============================",
        f"Generated: {datetime.now().isoformat()}",
        f"Output folder: results/final_a_plus_polish/",
        "",
        "TASK STATUS",
        "-----------",
        f"Task 1 (imbalance): {STATUS.get('task1', 'unknown')} — {STATUS.get('task1_best', '')}",
        f"Task 2 (GroupKFold): {STATUS.get('task2', 'unknown')} — {STATUS.get('task2_best', '')}",
        f"Task 3 (DenseNet121): {STATUS.get('task3', 'unknown')}",
        f"Task 4 (Grad-CAM): {STATUS.get('task4', 'unknown')}",
        "",
        "MACRO F1 / IMBALANCE",
        "--------------------",
        imp or "(see imbalance_handling_results.csv)",
        "",
        "HONEST LIMITATIONS",
        "------------------",
        "- Tabular features are custom radiomics from final_ultrasound_dataset.csv.",
        "- FSHD class dominates; high accuracy does not imply balanced per-class performance.",
        "- CNN models use FSHD severity (binary), not 5-class disease (image availability).",
        "- SMOTE requires imbalanced-learn; random oversample used as fallback.",
        "",
        "RECOMMENDED CHAPTER 4 ADDITIONS",
        "-------------------------------",
        "- imbalance_comparison.png + imbalance_handling_results.csv",
        "- groupkfold_macro_f1.png + groupkfold_cv_results.csv",
        "- cnn_dl_comparison.csv (if CNN training completed)",
        "- gradcam/*.png (if generated)",
        "",
        "FILES GENERATED",
        "---------------",
    ]
    for f in sorted(set(GENERATED)):
        lines.append(f"  - {f}")

    out_path("FINAL_IMPLEMENTATION_SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")
    log("\n" + "\n".join(lines[:25]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("=" * 70)
    log("FINAL A+ POLISH PIPELINE")
    log(f"Output: {OUT}")
    log("=" * 70)

    imbalance_df, plot_df = None, None
    cv_summary = None

    try:
        df, X, y, groups, le, _ = load_ml_data()
        imbalance_df, plot_df = task1_imbalance(df, X, y, groups, le)
        cv_summary = task2_groupkfold(df, X, y, groups, le)
    except Exception as e:
        log(f"ML tasks error: {e}")
        traceback.print_exc()
        STATUS["task1"] = STATUS["task2"] = f"error: {e}"

    try:
        task3_densenet_and_dl_comparison()
    except Exception as e:
        log(f"Task 3 error: {e}")
        traceback.print_exc()

    try:
        task4_gradcam()
    except Exception as e:
        log(f"Task 4 error: {e}")
        traceback.print_exc()

    write_requirements()
    write_summary(imbalance_df, cv_summary)

    log("\n" + "=" * 70)
    log(f"FINISHED — see {OUT / 'FINAL_IMPLEMENTATION_SUMMARY.txt'}")
    log("=" * 70)


if __name__ == "__main__":
    main()
