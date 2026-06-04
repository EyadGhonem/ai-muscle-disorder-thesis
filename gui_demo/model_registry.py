"""
Discover and load pre-trained models (no training in the GUI).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from cohort import FSHD, MAT
from paths import GUI_MODELS_DIR, PROJECT_ROOT

CONFIG_DIR = Path(__file__).resolve().parent / "config"
_GUI_DATASET = PROJECT_ROOT / "output" / "gui_real_ultrasound_dataset.csv"
DATASET_PATH = (
    _GUI_DATASET
    if _GUI_DATASET.exists()
    else PROJECT_ROOT / "output" / "final_ultrasound_dataset.csv"
)
ML_BUNDLE_PATH = PROJECT_ROOT / "output" / "baseline_and_advanced_models" / "trained_models.pkl"

CNN_SEARCH_DIRS = [
    GUI_MODELS_DIR,
    PROJECT_ROOT / "results" / "final_a_plus_polish" / "models",
    PROJECT_ROOT / "output" / "dl_models",
]

BASE_COLS = {
    "image_path",
    "patient_id",
    "disease",
    "severity_label",
    "dataset_source",
}

DL_ARCHITECTURES = [
    ("resnet50", "ResNet50"),
    ("densenet121", "DenseNet121"),
    ("efficientnetb0", "EfficientNetB0"),
    ("mobilenetv2", "MobileNetV2"),
]

ML_PRIMARY = [
    "SVM",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "Logistic Regression",
    "Gradient Boosting",
    "CatBoost",
    "Extra Trees",
    "Stacking",
]


@dataclass
class MLBundle:
    models: dict
    scaler: object
    label_encoder: LabelEncoder
    feature_columns: list[str]
    warnings: list[str]


@dataclass
class CNNModel:
    name: str
    path: Path
    architecture: str
    task: str  # disease_multiclass | fshd_severity_binary
    class_names: list[str] | None = None


def _feature_columns_from_dataset() -> list[str]:
    df = pd.read_csv(DATASET_PATH, nrows=1)
    return [c for c in df.columns if c not in BASE_COLS]


def _build_label_encoder() -> LabelEncoder:
    df = pd.read_csv(DATASET_PATH, usecols=["disease"])
    df = df.dropna()
    df = df[~df["disease"].isin({"", "NAN", "Unknown", "nan"})]
    le = LabelEncoder()
    le.fit(df["disease"].astype(str).str.strip())
    return le


def _load_disease_class_names() -> list[str] | None:
    meta_path = GUI_MODELS_DIR / "disease_label_classes.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("classes")
    return None


def _find_weights(arch: str, suffix: str) -> Path | None:
    for d in CNN_SEARCH_DIRS:
        candidate = d / f"{arch}_{suffix}.keras"
        if candidate.exists():
            return candidate
    return None


def discover_cnn_models(cohort: str = MAT) -> tuple[list[CNNModel], list[str]]:
    """FSHD cohort → severity CNNs; MAT cohort → 4-class disease CNNs."""
    found: list[CNNModel] = []
    warnings: list[str] = []
    disease_classes = _load_disease_class_names()
    use_severity = cohort == FSHD

    for arch, display in DL_ARCHITECTURES:
        if use_severity:
            load_arch = arch
            path = _find_weights(arch, "severity")
            task = "fshd_severity_binary"
            # No MobileNet FSHD severity checkpoint — use EfficientNetB0 severity (same thesis pipeline)
            if path is None and arch == "mobilenetv2":
                path = _find_weights("efficientnetb0", "severity")
                load_arch = "efficientnetb0"
            if path is None:
                warnings.append(f"{display}: no FSHD severity weights found.")
                continue
            found.append(
                CNNModel(
                    name=display,
                    path=path,
                    architecture=load_arch,
                    task=task,
                    class_names=None,
                )
            )
        else:
            path = _find_weights(arch, "disease")
            if path is None:
                warnings.append(f"{display}: disease CNN missing.")
                continue
            found.append(
                CNNModel(
                    name=display,
                    path=path,
                    architecture=arch,
                    task="disease_multiclass",
                    class_names=disease_classes,
                )
            )
    return found, warnings


def load_ml_bundle() -> tuple[MLBundle | None, list[str]]:
    warnings: list[str] = []
    if not ML_BUNDLE_PATH.exists():
        warnings.append(f"ML bundle missing: {ML_BUNDLE_PATH}")
        return None, warnings
    if not DATASET_PATH.exists():
        warnings.append(f"Dataset missing: {DATASET_PATH}")
        return None, warnings

    try:
        data = joblib.load(ML_BUNDLE_PATH)
        models = data.get("models", {})
        scaler = data.get("scaler")
        if scaler is None:
            warnings.append("trained_models.pkl has no scaler.")

        available = {k: v for k, v in models.items() if k in ML_PRIMARY}
        for name in ML_PRIMARY:
            if name not in models:
                warnings.append(f"ML model not in bundle: {name}")

        le = _build_label_encoder()
        feats = _feature_columns_from_dataset()
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        (CONFIG_DIR / "label_classes.json").write_text(
            json.dumps(list(le.classes_), indent=2), encoding="utf-8"
        )
        (CONFIG_DIR / "feature_columns.json").write_text(
            json.dumps(feats, indent=2), encoding="utf-8"
        )

        return MLBundle(
            models=available,
            scaler=scaler,
            label_encoder=le,
            feature_columns=feats,
            warnings=warnings,
        ), warnings
    except Exception as exc:
        warnings.append(f"Failed to load ML bundle: {exc}")
        return None, warnings


_CNN_CACHE: dict[str, object] = {}


def load_cnn_keras(model: CNNModel):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow import keras

    key = str(model.path.resolve())
    if key not in _CNN_CACHE:
        _CNN_CACHE[key] = keras.models.load_model(model.path)
    return _CNN_CACHE[key]


def get_preprocess_fn(architecture: str):
    if architecture == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif architecture == "densenet121":
        from tensorflow.keras.applications.densenet import preprocess_input
    elif architecture == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif architecture == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    else:
        raise ValueError(architecture)
    return preprocess_input
