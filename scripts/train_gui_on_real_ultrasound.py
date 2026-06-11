#!/usr/bin/env python3
"""
train_gui_on_real_ultrasound.py
--------------------------------
Full training pipeline for all ML and DL models used in the thesis GUI.

Data sources (real labeled images only):
  1. FSHD  — data/ULTRASOUND_LABELD_1_FSHD/images  (or ULTRASOUND_LABELD_1/images)
             ~25,005 PNG frames with Heckmatt severity labels (0 = Mild, 1 = Severe)
  2. MAT   — data/images_extracted_from_mat_LABELED/
             ~3,194 PNGs in four disease sub-folders:
             Normal, IBM, Dermatomyositis, Polymyositis

DOES NOT use ULTRASOUND_LABELD_2 (tabular-only rows, no real images).

Pipeline stages:
  1. build_manifest()     : scan both image directories → manifest CSV
  2. extract_features()   : Otsu ROI mask + 28 radiomics features per image
  3. train_ml_models()    : 9 sklearn models (5-class disease, patient-level split)
  4. train_dl_*()         : 4 CNNs for FSHD severity (binary) and MAT disease (4-class)

Outputs:
  output/gui_real_ultrasound_dataset.csv          — radiomics feature table
  output/baseline_and_advanced_models/
      trained_models.pkl                          — fitted models, scaler, label encoder
      gui_ml_training_summary.csv                 — per-model accuracy / F1
  output/dl_models/*_severity.keras               — FSHD binary severity CNN weights
  gui_demo/models/*_disease.keras                 — MAT 4-class disease CNN weights
  gui_demo/models/disease_label_classes.json      — class-index mapping for disease CNNs
  gui_demo/models/gui_training_metrics.json       — validation metrics read by the GUI

Usage:
  python scripts/train_gui_on_real_ultrasound.py --epochs 15
  python scripts/train_gui_on_real_ultrasound.py --quick              # 80 imgs/class, 2 epochs
  python scripts/train_gui_on_real_ultrasound.py --ml-only
  python scripts/train_gui_on_real_ultrasound.py --dl-only --epochs 10
  python scripts/train_gui_on_real_ultrasound.py --dl-only --dl-task disease --disease-enhanced --epochs 25
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gui_demo"))
sys.path.insert(0, str(PROJECT_ROOT / "data_processing"))

from extract_custom_features import (  # noqa: E402
    extract_first_order_features,
    extract_gradient_features,
    extract_shape_features,
    extract_texture_features,
)
from gui_demo.image_pipeline import build_roi_mask  # noqa: E402
from gui_demo.paths import FSHD_CANDIDATES, mat_image_root  # noqa: E402

MANIFEST_PATH = PROJECT_ROOT / "output" / "gui_real_ultrasound_manifest.csv"
DATASET_PATH = PROJECT_ROOT / "output" / "gui_real_ultrasound_dataset.csv"
MASTER_PATH = PROJECT_ROOT / "output" / "final_ultrasound_dataset.csv"
ML_BUNDLE_PATH = PROJECT_ROOT / "output" / "baseline_and_advanced_models" / "trained_models.pkl"
DL_SEVERITY_DIR = PROJECT_ROOT / "output" / "dl_models"
GUI_MODELS_DIR = PROJECT_ROOT / "gui_demo" / "models"
METRICS_PATH = GUI_MODELS_DIR / "gui_training_metrics.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_STATE = 42

DL_ARCHS = ["resnet50", "densenet121", "efficientnetb0", "mobilenetv2"]
IBM_FOLDER = "IBM"
IBM_LABEL = "Inclusion Body Myositis"

# Match gui_demo/model_registry.py (severity is a model feature, not metadata)
BASE_COLS = [
    "image_path",
    "patient_id",
    "disease",
    "severity_label",
    "dataset_source",
]


def _fshd_dir() -> Path | None:
    for p in FSHD_CANDIDATES:
        if p.is_dir():
            return p
    return None


def _load_master_lookups() -> tuple[dict, dict, dict]:
    """Build lookup dicts from the master CSV for FSHD rows only.

    Returns three filename-keyed dicts:
    - sev : filename → numeric severity (0.0 = Mild, 1.0 = Severe)
    - pid : filename → patient_id string
    - dis : filename → disease label string
    """
    sev, pid, dis = {}, {}, {}
    if not MASTER_PATH.exists():
        return sev, pid, dis
    df = pd.read_csv(
        MASTER_PATH,
        usecols=["image_path", "patient_id", "disease", "severity", "dataset_source"],
    )
    df = df[df["dataset_source"] == "ULTRASOUND_LABELD_1"].copy()
    for _, row in df.iterrows():
        name = Path(str(row["image_path"])).name
        if pd.notna(row.get("severity")):
            sev[name] = float(row["severity"])
        if pd.notna(row.get("patient_id")):
            pid[name] = row["patient_id"]
        if pd.notna(row.get("disease")):
            dis[name] = str(row["disease"]).strip()
    return sev, pid, dis


def _patient_from_fshd_name(name: str) -> str:
    parts = name.replace(".png", "").split("_")
    if parts and parts[0].isdigit():
        return parts[0].lstrip("0") or parts[0]
    return Path(name).stem


def build_manifest() -> pd.DataFrame:
    """Scan FSHD and MAT image directories and return a manifest DataFrame.

    For each image the manifest records: image_path (relative), filepath (absolute),
    image_name, patient_id, disease label, numeric severity, severity_label string,
    dataset_source, and cohort identifier ('fshd' or 'mat').

    The manifest is also saved to output/gui_real_ultrasound_manifest.csv.
    """
    rows = []
    sev_lookup, pid_lookup, _ = _load_master_lookups()

    fshd_dir = _fshd_dir()
    if fshd_dir:
        for img in sorted(fshd_dir.glob("*.png")):
            name = img.name
            rel = f"data/{fshd_dir.parent.name}/images/{name}"
            rows.append(
                {
                    "image_path": rel.replace("\\", "/"),
                    "filepath": str(img.resolve()),
                    "image_name": name,
                    "patient_id": pid_lookup.get(name, _patient_from_fshd_name(name)),
                    "disease": "FSHD",
                    "severity": sev_lookup.get(name, np.nan),
                    "severity_label": (
                        "Mild" if sev_lookup.get(name) == 0.0 else "Severe"
                        if sev_lookup.get(name) == 1.0
                        else ""
                    ),
                    "dataset_source": "ULTRASOUND_LABELD_1",
                    "cohort": "fshd",
                }
            )
        print(f"FSHD images: {len(list(fshd_dir.glob('*.png')))}")
    else:
        print("WARNING: FSHD image folder not found.")

    mat_root = mat_image_root()
    if mat_root:
        skip = {"Unknown", "UNKNOWN", "unknown"}
        for folder in sorted(mat_root.iterdir()):
            if not folder.is_dir() or folder.name in skip:
                continue
            disease = IBM_LABEL if folder.name == IBM_FOLDER else folder.name
            for img in folder.glob("*.png"):
                stem = img.stem
                patient = stem.split("_")[0] if "_" in stem else stem
                rel = f"data/{mat_root.name}/{folder.name}/{img.name}"
                rows.append(
                    {
                        "image_path": rel.replace("\\", "/"),
                        "filepath": str(img.resolve()),
                        "image_name": img.name,
                        "patient_id": patient,
                        "disease": disease,
                        "severity": np.nan,
                        "severity_label": "",
                        "dataset_source": "MAT_LABELED",
                        "cohort": "mat",
                    }
                )
        print(f"MAT images: {sum(1 for r in rows if r['cohort'] == 'mat')}")
    else:
        print("WARNING: MAT image folder not found.")

    if not rows:
        raise FileNotFoundError("No images found under FSHD or MAT folders.")

    df = pd.DataFrame(rows)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Manifest: {len(df)} rows -> {MANIFEST_PATH}")
    print(df["disease"].value_counts().to_string())
    return df


def _extract_row_features(filepath: str) -> dict:
    bgr = cv2.imread(filepath)
    if bgr is None:
        return {}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask, _ = build_roi_mask(gray)
    if gray.max() > 0:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    m = (mask > 0).astype(np.uint8) * 255
    feats = {}
    feats.update(extract_first_order_features(gray, m))
    feats.update(extract_texture_features(gray, m))
    feats.update(extract_shape_features(m))
    feats.update(extract_gradient_features(gray, m))
    return feats


def _feature_columns_from_master() -> list[str]:
    if MASTER_PATH.exists():
        df = pd.read_csv(MASTER_PATH, nrows=1)
        return [c for c in df.columns if c not in BASE_COLS]
    # fallback
    sample = _extract_row_features(str(next(_fshd_dir().glob("*.png"))))
    cols = list(sample.keys())
    cols.append("severity")
    return cols


def extract_features(manifest: pd.DataFrame, max_per_class: int | None) -> pd.DataFrame:
    """Compute 28 radiomics features for every image in the manifest.

    For each image:
    1. Load BGR → convert to grayscale.
    2. Generate Otsu + morphological ROI mask via ``build_roi_mask``.
    3. Compute first-order, GLCM texture, shape, and gradient features.
    4. Attach the numeric severity value (0 or NaN) from the manifest.

    Supports resuming: if a partial CSV already exists at DATASET_PATH, only
    images not yet processed are computed and the new records are appended.
    The CSV is checkpointed every 500 rows to prevent data loss on long runs.

    Parameters
    ----------
    manifest      : DataFrame produced by ``build_manifest``
    max_per_class : if set, randomly subsample to this many images per disease class
                    before extraction (useful for quick tests)
    """
    df = manifest.copy()
    if max_per_class and max_per_class > 0:
        parts = []
        for _, g in df.groupby("disease"):
            parts.append(g.sample(min(len(g), max_per_class), random_state=RANDOM_STATE))
        df = pd.concat(parts, ignore_index=True)
        print(f"Feature extraction subset: {len(df)} images (max {max_per_class}/class)")
    else:
        print(f"Feature extraction: ALL {len(df)} labeled images (no cap)")

    feature_cols = _feature_columns_from_master()
    done_paths: set[str] = set()
    records: list[dict] = []
    if DATASET_PATH.exists():
        prev = pd.read_csv(DATASET_PATH)
        if len(prev) > 0 and "image_path" in prev.columns:
            done_paths = set(prev["image_path"].astype(str))
            records = prev.to_dict("records")
            print(f"Resuming radiomics: {len(done_paths)} already extracted")

    todo = df[~df["image_path"].astype(str).isin(done_paths)]
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc="Radiomics"):
        feats = _extract_row_features(row["filepath"])
        if not feats:
            continue
        out = {
            "image_path": row["image_path"],
            "patient_id": row["patient_id"],
            "disease": row["disease"],
            "severity_label": row.get("severity_label", ""),
            "dataset_source": row["dataset_source"],
            "severity": float(row["severity"]) if pd.notna(row["severity"]) else 0.0,
        }
        for col in feature_cols:
            if col == "severity":
                continue
            out[col] = feats.get(col, 0.0)
        records.append(out)
        if len(records) % 500 == 0:
            pd.DataFrame(records).to_csv(DATASET_PATH, index=False)

    full = pd.DataFrame(records)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(DATASET_PATH, index=False)
    print(f"Dataset saved: {full.shape} -> {DATASET_PATH}")
    return full


def _dataset_covers_manifest(manifest: pd.DataFrame) -> bool:
    if not DATASET_PATH.exists():
        return False
    n = len(pd.read_csv(DATASET_PATH))
    return n >= int(len(manifest) * 0.98)


def train_ml_models(df: pd.DataFrame) -> dict:
    """Train all 9 ML classifiers on the radiomics feature table.

    Patient-level splitting:
    - Uses ``GroupShuffleSplit`` (test_size=0.2) to ensure that images from the
      same patient appear only in train OR test, preventing data leakage.
    - Features are standardised with ``StandardScaler`` fitted on the train split.
    - A ``LabelEncoder`` maps disease strings to integer indices.

    Models trained: Random Forest, Gradient Boosting, SVM, Logistic Regression,
    XGBoost, LightGBM, CatBoost, Extra Trees, Stacking Ensemble (RF+GB+XGB).

    Outputs saved:
    - ``trained_models.pkl`` : dict with 'models', 'scaler', 'results', 'label_encoder'
    - ``gui_ml_training_summary.csv`` : per-model accuracy and macro F1

    Returns
    -------
    dict with key 'ml' mapping model names to {'val_accuracy_pct', 'f1_macro'}
    """
    print("\n" + "=" * 60)
    print("TRAINING ML MODELS (5-class disease, patient split)")
    print("=" * 60)

    feature_cols = [c for c in df.columns if c not in BASE_COLS]
    X = df[feature_cols].fillna(0).values
    y = df["disease"].astype(str)
    patients = df["patient_id"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    patient_df = pd.DataFrame({"patient_id": patients, "disease": y_enc}).drop_duplicates("patient_id")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_p, test_p = next(gss.split(patient_df, groups=patient_df["patient_id"]))
    train_patients = set(patient_df.iloc[train_p]["patient_id"])
    test_patients = set(patient_df.iloc[test_p]["patient_id"])

    train_mask = patients.isin(train_patients).values
    test_mask = patients.isin(test_patients).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_test = scaler.transform(X[test_mask])
    y_train = y_enc[train_mask]
    y_test = y_enc[test_mask]

    print(f"Train {X_train.shape[0]} | Test {X_test.shape[0]} | classes {list(le.classes_)}")

    try:
        import xgboost as xgb
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
        import xgboost as xgb

    try:
        import lightgbm as lgb
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
        import lightgbm as lgb

    try:
        import catboost as cb
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])
        import catboost as cb

    model_defs = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(iterations=100, random_state=RANDOM_STATE, verbose=0),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
    }

    fitted = {}
    results = {}
    metrics = {"ml": {}, "label_classes": list(le.classes_)}

    for name, model in model_defs.items():
        print(f"  {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1m = f1_score(y_test, pred, average="macro", zero_division=0)
        f1w = f1_score(y_test, pred, average="weighted", zero_division=0)
        results[name] = {
            "accuracy": acc,
            "f1_macro": f1m,
            "f1_weighted": f1w,
            "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        }
        fitted[name] = model
        metrics["ml"][name] = {
            "val_accuracy_pct": round(acc * 100, 2),
            "f1_macro": round(f1m, 4),
        }
        print(f"acc={acc:.4f} macro_f1={f1m:.4f}")

    print("  Stacking...", end=" ", flush=True)
    stacking = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, class_weight="balanced")),
            ("gb", GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE)),
            ("xgb", xgb.XGBClassifier(n_estimators=50, random_state=RANDOM_STATE, verbosity=0)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )
    stacking.fit(X_train, y_train)
    pred = stacking.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro", zero_division=0)
    fitted["Stacking"] = stacking
    results["Stacking"] = {"accuracy": acc, "f1_macro": f1m, "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0)}
    metrics["ml"]["Stacking"] = {"val_accuracy_pct": round(acc * 100, 2), "f1_macro": round(f1m, 4)}
    print(f"acc={acc:.4f} macro_f1={f1m:.4f}")

    ML_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ML_BUNDLE_PATH, "wb") as f:
        pickle.dump({"models": fitted, "scaler": scaler, "results": results, "label_encoder": le}, f)
    print(f"Saved ML bundle: {ML_BUNDLE_PATH}")

    summary = pd.DataFrame(
        [{"model": k, **v} for k, v in results.items()]
    )
    summary.to_csv(ML_BUNDLE_PATH.parent / "gui_ml_training_summary.csv", index=False)
    return metrics


def _build_cnn(architecture: str, num_classes: int):
    """Build a transfer-learning CNN with a frozen ImageNet backbone.

    Architecture: pre-trained base (ImageNet) → GlobalAveragePooling2D →
                  Dense(256, ReLU) → Dropout(0.3) → output layer.

    For binary tasks (num_classes=2): sigmoid output, binary crossentropy.
    For multi-class tasks: softmax output, sparse categorical crossentropy.

    Returns
    -------
    (compiled_keras_model, preprocess_input_fn, base_model)
    """
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
    if num_classes == 2:
        out = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        out = layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
    model = models.Model(base.input, out, name=f"{architecture}_gui")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=loss,
        metrics=["accuracy"],
    )
    return model, preprocess_input, base


def _arch_display(arch: str) -> str:
    return (
        arch.replace("efficientnetb0", "EfficientNetB0")
        .replace("mobilenetv2", "MobileNetV2")
        .replace("resnet50", "ResNet50")
        .replace("densenet121", "DenseNet121")
    )


def _unfreeze_top_layers(base, architecture: str) -> None:
    n_unfreeze = {"resnet50": 40, "densenet121": 40, "efficientnetb0": 30, "mobilenetv2": 35}.get(
        architecture, 40
    )
    base.trainable = True
    for layer in base.layers[:-n_unfreeze]:
        layer.trainable = False


def _load_ultrasound_tensor(filepath: str, preprocess_fn, augment: bool = False) -> np.ndarray | None:
    from image_pipeline import prepare_ultrasound_cnn_tensor  # noqa: E402

    return prepare_ultrasound_cnn_tensor(filepath, preprocess_fn, IMG_SIZE, augment=augment)


def _eval_val_metrics(model, val_df: pd.DataFrame, preprocess_fn, augment: bool = False) -> tuple[float, float]:
    ys, preds = [], []
    for path, label in zip(val_df["filepath"].values, val_df["label"].values):
        x = _load_ultrasound_tensor(path, preprocess_fn, augment=augment)
        if x is None:
            continue
        p = model.predict(np.expand_dims(x, 0), verbose=0)
        ys.append(int(label))
        preds.append(int(np.argmax(p, axis=-1)[0]))
    if not ys:
        return 0.0, 0.0
    acc = accuracy_score(ys, preds)
    f1m = f1_score(ys, preds, average="macro", zero_division=0)
    return float(acc), float(f1m)


def _fit_cnn(df: pd.DataFrame, architectures: list[str], num_classes: int, epochs: int, out_dir: Path, suffix: str):
    from tensorflow import keras

    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["label"] = df["label"].astype(int)

    if df["patient_id"].nunique() > 5:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        tr_idx, va_idx = next(gss.split(df, groups=df["patient_id"]))
        train_df, val_df = df.iloc[tr_idx], df.iloc[va_idx]
    else:
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
        )

    print(f"  CNN split train={len(train_df)} val={len(val_df)}")
    rows = []
    dl_metrics = {}

    for arch in architectures:
        print(f"\n  DL {arch} ({suffix})...")
        model, preprocess_fn, _base = _build_cnn(arch, num_classes)

        def batch_generator(sub_df, shuffle, augment=False):
            paths = sub_df["filepath"].values
            labels = sub_df["label"].values
            while True:
                order = np.random.permutation(len(paths)) if shuffle else np.arange(len(paths))
                for start in range(0, len(order), BATCH_SIZE):
                    batch_idx = order[start : start + BATCH_SIZE]
                    xs, ys = [], []
                    for i in batch_idx:
                        x = _load_ultrasound_tensor(paths[i], preprocess_fn, augment=augment and shuffle)
                        if x is None:
                            continue
                        xs.append(x)
                        ys.append(labels[i])
                    if xs:
                        yield np.stack(xs), np.array(ys)

        steps_train = max(1, len(train_df) // BATCH_SIZE)
        steps_val = max(1, len(val_df) // BATCH_SIZE)
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        ]
        history = model.fit(
            batch_generator(train_df, True),
            steps_per_epoch=steps_train,
            validation_data=batch_generator(val_df, False),
            validation_steps=steps_val,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        out_path = out_dir / f"{arch}_{suffix}.keras"
        model.save(out_path)
        best = float(max(history.history["val_accuracy"]))
        rows.append({"architecture": arch, "task": suffix, "val_accuracy": best, "path": str(out_path)})
        dl_metrics[_arch_display(arch)] = round(best * 100, 2)
        print(f"  Saved {out_path} (val_acc={best:.4f})")

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / ("training_summary.csv" if suffix == "disease" else f"training_summary_{suffix}.csv")
    summary_df.to_csv(summary_path, index=False)
    return dl_metrics


def _fit_cnn_enhanced_disease(
    df: pd.DataFrame,
    architectures: list[str],
    num_classes: int,
    epochs: int,
    out_dir: Path,
) -> dict:
    """Train MAT disease CNNs with an enhanced two-phase fine-tuning strategy.

    Phase 1 (frozen backbone, lr=1e-3, epochs = max(6, total//3)):
    - Only the classification head is trained.
    - CLAHE + ROI augmented batches with per-class sample weights.

    Phase 2 (top-N layers unfrozen, lr=1e-4, remaining epochs):
    - Architecture-specific number of layers unfrozen (30–40 layers).
    - EarlyStopping (patience=6) and ReduceLROnPlateau applied.

    Class weights are computed via scikit-learn's ``compute_class_weight``
    to compensate for the class imbalance across the four MAT disease classes.

    This mode is activated with ``--disease-enhanced`` and is recommended for
    reproducing the best thesis DL results. Use ``--epochs 25``.
    """
    from tensorflow import keras

    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["label"] = df["label"].astype(int)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(gss.split(df, groups=df["patient_id"]))
    train_df, val_df = df.iloc[tr_idx], df.iloc[va_idx]
    print(f"  Enhanced CNN | train={len(train_df)} val={len(val_df)} | epochs={epochs}")

    y_train = train_df["label"].values
    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print(f"  Class weights: {class_weight}")

    phase1_epochs = max(6, epochs // 3)
    phase2_epochs = max(1, epochs - phase1_epochs)
    rows = []
    dl_metrics: dict = {}

    for arch in architectures:
        print(f"\n  === Enhanced DL {arch} (disease) ===")
        model, preprocess_fn, base = _build_cnn(arch, num_classes)

        def batch_generator(sub_df, augment, weighted: bool = False):
            paths = sub_df["filepath"].values
            labels = sub_df["label"].values
            while True:
                order = np.random.permutation(len(paths))
                for start in range(0, len(order), BATCH_SIZE):
                    batch_idx = order[start : start + BATCH_SIZE]
                    xs, ys = [], []
                    for i in batch_idx:
                        x = _load_ultrasound_tensor(paths[i], preprocess_fn, augment=augment)
                        if x is None:
                            continue
                        xs.append(x)
                        ys.append(labels[i])
                    if xs:
                        y_arr = np.array(ys)
                        if weighted:
                            sw = np.array([class_weight[int(y)] for y in y_arr], dtype=np.float32)
                            yield np.stack(xs), y_arr, sw
                        else:
                            yield np.stack(xs), y_arr

        steps_train = max(1, len(train_df) // BATCH_SIZE)
        steps_val = max(1, len(val_df) // BATCH_SIZE)
        val_gen = batch_generator(val_df, augment=False)

        print(f"  Phase 1: frozen backbone, {phase1_epochs} epochs @ lr=1e-3")
        base.trainable = False
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            batch_generator(train_df, augment=True, weighted=True),
            steps_per_epoch=steps_train,
            validation_data=val_gen,
            validation_steps=steps_val,
            epochs=phase1_epochs,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
            ],
            verbose=1,
        )

        print(f"  Phase 2: fine-tune top layers, {phase2_epochs} epochs @ lr=1e-4")
        _unfreeze_top_layers(base, arch)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = model.fit(
            batch_generator(train_df, augment=True, weighted=True),
            steps_per_epoch=steps_train,
            validation_data=val_gen,
            validation_steps=steps_val,
            epochs=phase2_epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
            ],
            verbose=1,
        )

        val_acc, val_f1 = _eval_val_metrics(model, val_df, preprocess_fn)
        hist_acc = float(max(history.history.get("val_accuracy", [0])))
        best_acc = max(val_acc, hist_acc)

        out_path = out_dir / f"{arch}_disease.keras"
        model.save(out_path)
        rows.append(
            {
                "architecture": arch,
                "task": "disease_enhanced",
                "val_accuracy": best_acc,
                "val_f1_macro": val_f1,
                "path": str(out_path),
            }
        )
        display = _arch_display(arch)
        dl_metrics[display] = {
            "val_accuracy_pct": round(best_acc * 100, 2),
            "val_f1_macro": round(val_f1, 4),
        }
        print(f"  Saved {out_path}")
        print(f"  val_acc={best_acc:.4f} | val_macro_f1={val_f1:.4f}")
        pred_vals = []
        true_vals = []
        for path, label in zip(val_df["filepath"].values, val_df["label"].values):
            x = _load_ultrasound_tensor(path, preprocess_fn, augment=False)
            if x is None:
                continue
            pred_vals.append(int(np.argmax(model.predict(np.expand_dims(x, 0), verbose=0), axis=-1)[0]))
            true_vals.append(int(label))
        if true_vals:
            print(classification_report(true_vals, pred_vals, zero_division=0))

    pd.DataFrame(rows).to_csv(out_dir / "training_summary_enhanced.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "training_summary.csv", index=False)
    return dl_metrics


def train_dl_fshd_severity(manifest: pd.DataFrame, architectures: list[str], epochs: int, max_per_class: int | None):
    """Train 4 CNNs for FSHD severity classification (binary: Mild=0 / Severe=1).

    Only rows with ``cohort == 'fshd'`` and a non-null severity label are used.
    Models are saved to output/dl_models/ as ``<arch>_severity.keras``.
    """
    print("\n" + "=" * 60)
    print("TRAINING DL — FSHD severity (binary)")
    print("=" * 60)
    df = manifest[(manifest["cohort"] == "fshd") & manifest["severity"].notna()].copy()
    if df.empty:
        print("No FSHD rows with severity labels; skip severity CNNs.")
        return {}
    df["label"] = df["severity"].astype(int)
    if max_per_class and max_per_class > 0:
        parts = []
        for _, g in df.groupby("label"):
            parts.append(g.sample(min(len(g), max_per_class), random_state=RANDOM_STATE))
        df = pd.concat(parts, ignore_index=True)
    print(f"Severity samples: {len(df)} (0={sum(df['label']==0)}, 1={sum(df['label']==1)})")
    return _fit_cnn(df, architectures, 2, epochs, DL_SEVERITY_DIR, "severity")


def train_dl_mat_disease(
    manifest: pd.DataFrame,
    architectures: list[str],
    epochs: int,
    max_per_class: int | None,
    enhanced: bool = False,
):
    """Train 4 CNNs for MAT 4-class disease classification.

    Classes: Dermatomyositis, Inclusion Body Myositis, Normal, Polymyositis.

    Also writes disease_label_classes.json (used by inference.py and Grad-CAM
    scripts to map class indices back to disease name strings).

    The ``cnn_preprocess`` field in the JSON is set to ``"clahe_roi"`` when
    enhanced training is used, signalling the GUI to apply matching preprocessing
    at inference time.

    Parameters
    ----------
    enhanced : if True, use ``_fit_cnn_enhanced_disease`` (two-phase fine-tuning)
               otherwise use the simpler ``_fit_cnn`` (frozen backbone)
    """
    print("\n" + "=" * 60)
    print("TRAINING DL — MAT 4-class disease")
    print("=" * 60)
    df = manifest[manifest["cohort"] == "mat"].copy()
    classes = sorted(df["disease"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label"] = df["disease"].map(class_to_idx).astype(int)
    if max_per_class and max_per_class > 0:
        parts = []
        for _, g in df.groupby("disease"):
            parts.append(g.sample(min(len(g), max_per_class), random_state=RANDOM_STATE))
        df = pd.concat(parts, ignore_index=True)
    GUI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "task": "disease_multiclass_mat",
        "cnn_preprocess": "clahe_roi" if enhanced else "rgb_resize",
    }
    (GUI_MODELS_DIR / "disease_label_classes.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Disease samples: {len(df)} | classes={classes}")
    if enhanced:
        print("  Mode: ENHANCED (CLAHE ROI, aug, class weights, fine-tune top layers, macro F1)")
        return _fit_cnn_enhanced_disease(df, architectures, len(classes), epochs, GUI_MODELS_DIR)
    return _fit_cnn(df, architectures, len(classes), epochs, GUI_MODELS_DIR, "disease")


def main():
    """Parse CLI arguments and orchestrate the full training pipeline.

    Execution order (flags permitting):
    1. build_manifest()                  : scan image directories
    2. extract_features() / load CSV     : radiomics feature table
    3. train_ml_models()                 : 9 sklearn classifiers
    4. train_dl_fshd_severity()          : 4 CNNs for FSHD binary severity
    5. train_dl_mat_disease()            : 4 CNNs for MAT 4-class disease

    Validation metrics from all stages are merged and written to
    gui_demo/models/gui_training_metrics.json for use by the GUI.
    """
    parser = argparse.ArgumentParser(description="Train all GUI models on real ultrasound images")
    parser.add_argument("--quick", action="store_true", help="Small subset, 2 epochs (smoke test)")
    parser.add_argument("--ml-only", action="store_true")
    parser.add_argument("--dl-only", action="store_true")
    parser.add_argument(
        "--dl-task",
        choices=["all", "severity", "disease"],
        default="all",
        help="With --dl-only: train only FSHD severity or MAT disease CNNs",
    )
    parser.add_argument("--skip-features", action="store_true", help="Reuse existing gui_real_ultrasound_dataset.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--disease-enhanced",
        action="store_true",
        help="MAT disease CNNs: fine-tune, class weights, CLAHE ROI aug, report macro F1 (use --epochs 25)",
    )
    parser.add_argument("--max-per-class", type=int, default=0, help="0 = no cap")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DL_ARCHS,
        choices=DL_ARCHS,
    )
    args = parser.parse_args()

    epochs = 2 if args.quick else args.epochs
    if args.disease_enhanced and not args.quick and args.epochs == 15:
        epochs = 25
    max_pc = 80 if args.quick else (args.max_per_class if args.max_per_class > 0 else None)

    manifest = build_manifest()
    all_metrics = {"data_sources": ["ULTRASOUND_LABELD_1", "MAT_LABELED"], "manifest_rows": len(manifest)}
    if METRICS_PATH.exists():
        try:
            prev = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            if args.dl_only and prev.get("ml"):
                all_metrics["ml"] = prev["ml"]
                all_metrics["label_classes"] = prev.get("label_classes", [])
        except Exception:
            pass

    run_ml = not args.dl_only
    run_dl = not args.ml_only

    if run_ml:
        if args.skip_features and _dataset_covers_manifest(manifest):
            dataset = pd.read_csv(DATASET_PATH)
            print(f"Using existing full feature set: {len(dataset)} rows")
        else:
            if args.skip_features and DATASET_PATH.exists():
                partial = len(pd.read_csv(DATASET_PATH))
                print(
                    f"WARNING: --skip-features ignored — only {partial}/{len(manifest)} "
                    "images extracted. Running full radiomics on all labeled images."
                )
            dataset = extract_features(manifest, max_pc)
        if len(dataset) < len(manifest) * 0.95:
            raise RuntimeError(
                f"Feature CSV has {len(dataset)} rows but manifest has {len(manifest)}. "
                "Re-run without --skip-features."
            )
        all_metrics.update(train_ml_models(dataset))

    if run_dl:
        if not _dataset_covers_manifest(manifest) and not args.skip_features:
            raise RuntimeError(
                "Run full radiomics first (omit --dl-only) or ensure "
                "output/gui_real_ultrasound_dataset.csv has ~28199 rows."
            )
        print(f"\nDL training: ALL labeled images | {epochs} epochs | task={args.dl_task}")
        sev_m, dis_m = {}, {}
        if args.dl_task in ("all", "severity"):
            sev_m = train_dl_fshd_severity(manifest, args.models, epochs, max_pc)
            all_metrics["dl_severity"] = sev_m
        elif METRICS_PATH.exists():
            try:
                prev = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                all_metrics["dl_severity"] = prev.get("dl_severity", {})
            except Exception:
                pass
        if args.dl_task in ("all", "disease"):
            dis_m = train_dl_mat_disease(
                manifest, args.models, epochs, max_pc, enhanced=args.disease_enhanced
            )
            all_metrics["dl_disease"] = dis_m
        elif METRICS_PATH.exists():
            try:
                prev = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                all_metrics["dl_disease"] = prev.get("dl_disease", {})
            except Exception:
                pass

    GUI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"\nMetrics for GUI: {METRICS_PATH}")
    print("\nDone. Restart Streamlit: streamlit run gui_demo/app.py")


if __name__ == "__main__":
    main()
