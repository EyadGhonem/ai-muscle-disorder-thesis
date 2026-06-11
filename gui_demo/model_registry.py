"""
model_registry.py
-----------------
Discovers and loads pre-trained models for the thesis GUI.

No training is performed here. This module provides two main entry points:
- ``load_ml_bundle()``   : loads the scikit-learn ML models, scaler, and label
                           encoder from ``trained_models.pkl``.
- ``discover_cnn_models()``: locates .keras CNN weight files on disk and
                             wraps them in ``CNNModel`` dataclasses.

Models are cached in memory to avoid repeated disk reads during a GUI session.
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

# Directory for cached label/feature config JSON files written at GUI startup
CONFIG_DIR = Path(__file__).resolve().parent / "config"

# Prefer the full real-image dataset CSV; fall back to the master CSV
_GUI_DATASET = PROJECT_ROOT / "output" / "gui_real_ultrasound_dataset.csv"
DATASET_PATH = (
    _GUI_DATASET
    if _GUI_DATASET.exists()
    else PROJECT_ROOT / "output" / "final_ultrasound_dataset.csv"
)

# Pickle file produced by train_gui_on_real_ultrasound.py containing all ML models
ML_BUNDLE_PATH = PROJECT_ROOT / "output" / "baseline_and_advanced_models" / "trained_models.pkl"

# Search order for .keras CNN weight files
CNN_SEARCH_DIRS = [
    GUI_MODELS_DIR,
    PROJECT_ROOT / "results" / "final_a_plus_polish" / "models",
    PROJECT_ROOT / "output" / "dl_models",
]

# Columns in the dataset CSV that are metadata, not radiomics features
BASE_COLS = {
    "image_path",
    "patient_id",
    "disease",
    "severity_label",
    "dataset_source",
}

# CNN architectures included in this thesis (file stem, display name)
DL_ARCHITECTURES = [
    ("resnet50", "ResNet50"),
    ("densenet121", "DenseNet121"),
    ("efficientnetb0", "EfficientNetB0"),
    ("mobilenetv2", "MobileNetV2"),
]

# Ordered list of ML model names expected in trained_models.pkl
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
    """Container for all ML assets loaded from trained_models.pkl.

    Attributes
    ----------
    models         : dict of model_name → fitted sklearn estimator
    scaler         : StandardScaler fitted on the training feature matrix
    label_encoder  : LabelEncoder mapping disease strings to integer indices
    feature_columns: ordered list of the 28 radiomics feature column names
    warnings       : non-fatal loading messages shown in the GUI sidebar
    """
    models: dict
    scaler: object
    label_encoder: LabelEncoder
    feature_columns: list[str]
    warnings: list[str]


@dataclass
class CNNModel:
    """Metadata wrapper for a single CNN weight file.

    Attributes
    ----------
    name        : display name shown in the GUI (e.g. "EfficientNetB0")
    path        : absolute path to the .keras weight file
    architecture: lowercase architecture key used to select the correct
                  Keras preprocessing function (e.g. "efficientnetb0")
    task        : ``"disease_multiclass"`` or ``"fshd_severity_binary"``
    class_names : ordered list of class label strings for disease CNNs; None
                  for severity CNNs (binary output)
    """
    name: str
    path: Path
    architecture: str
    task: str  # disease_multiclass | fshd_severity_binary
    class_names: list[str] | None = None


def _feature_columns_from_dataset() -> list[str]:
    """Read column names from the dataset CSV and return only feature columns.

    Excludes metadata columns defined in BASE_COLS (image path, patient ID, etc.).
    Reads only the header row for efficiency.
    """
    df = pd.read_csv(DATASET_PATH, nrows=1)
    return [c for c in df.columns if c not in BASE_COLS]


def _build_label_encoder() -> LabelEncoder:
    """Fit and return a LabelEncoder on all valid disease labels in the dataset CSV.

    Drops rows with missing or placeholder disease values before fitting.
    """
    df = pd.read_csv(DATASET_PATH, usecols=["disease"])
    df = df.dropna()
    df = df[~df["disease"].isin({"", "NAN", "Unknown", "nan"})]
    le = LabelEncoder()
    le.fit(df["disease"].astype(str).str.strip())
    return le


def _load_disease_class_names() -> list[str] | None:
    """Load the ordered list of MAT disease class names from disease_label_classes.json.

    This JSON is written by the CNN training script and defines the class-index
    mapping used during inference.
    """
    meta_path = GUI_MODELS_DIR / "disease_label_classes.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("classes")
    return None


def _find_weights(arch: str, suffix: str) -> Path | None:
    """Search CNN_SEARCH_DIRS for a weight file named ``<arch>_<suffix>.keras``.

    Returns the first match found, or None if the file does not exist in any
    of the search directories.
    """
    for d in CNN_SEARCH_DIRS:
        candidate = d / f"{arch}_{suffix}.keras"
        if candidate.exists():
            return candidate
    return None


def discover_cnn_models(cohort: str = MAT) -> tuple[list[CNNModel], list[str]]:
    """Locate and register CNN weight files for the given cohort.

    - FSHD cohort  → severity CNNs  (binary: Mild / Severe)
    - MAT cohort   → 4-class disease CNNs (Normal, IBM, Dermatomyositis, Polymyositis)

    Returns
    -------
    found    : list of CNNModel instances with resolved weight paths
    warnings : list of warning strings for missing weight files
    """
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
    """Load the ML model bundle from trained_models.pkl.

    Reads the pickle produced by ``train_gui_on_real_ultrasound.py``, extracts
    the fitted models, StandardScaler, and builds a fresh LabelEncoder and
    feature column list from the dataset CSV.

    Also caches the label classes and feature columns as JSON files under
    gui_demo/config/ for inspection without loading the full pickle.

    Returns
    -------
    bundle   : MLBundle instance, or None if loading failed
    warnings : list of non-fatal messages (e.g. missing models)
    """
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

        # Keep only the nine primary ML models defined in ML_PRIMARY
        available = {k: v for k, v in models.items() if k in ML_PRIMARY}
        for name in ML_PRIMARY:
            if name not in models:
                warnings.append(f"ML model not in bundle: {name}")

        le = _build_label_encoder()
        feats = _feature_columns_from_dataset()

        # Persist label/feature metadata for debugging and reproducibility
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


# Module-level cache: avoids re-loading the same .keras file multiple times
_CNN_CACHE: dict[str, object] = {}


def load_cnn_keras(model: CNNModel):
    """Load a Keras CNN model from disk, with in-memory caching.

    Uses the resolved absolute path as the cache key so the same weights
    file is never loaded twice during a GUI session.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow import keras

    key = str(model.path.resolve())
    if key not in _CNN_CACHE:
        _CNN_CACHE[key] = keras.models.load_model(model.path)
    return _CNN_CACHE[key]


def get_preprocess_fn(architecture: str):
    """Return the Keras application-specific preprocessing function for *architecture*.

    Each Keras application (ResNet50, DenseNet121, etc.) requires a different
    pixel normalisation function. This utility centralises that selection.

    Raises
    ------
    ValueError
        If *architecture* is not one of the four supported strings.
    """
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
