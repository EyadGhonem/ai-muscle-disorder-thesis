#!/usr/bin/env python3
"""
A+ / publication-level thesis improvements.
All outputs -> results/a_plus_full_improvements/ (does not overwrite output/thesis_final).
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "results" / "a_plus_full_improvements"
ROC_DIR = OUT_DIR / "roc_curves"
SHAP_DIR = OUT_DIR / "shap"
GRADCAM_DIR = OUT_DIR / "gradcam"

DATASET_PATH = PROJECT_ROOT / "output" / "final_ultrasound_dataset.csv"
PYRADIOMICS_PATH = PROJECT_ROOT / "output" / "pyradiomics_labeled1_features.csv"
DL_DIR = PROJECT_ROOT / "output" / "dl_models"

RANDOM_STATE = 42
TEST_SIZE = 0.2
SHAP_SAMPLE_SIZE = 200
GRADCAM_N_EACH = 3

METADATA_COLS = {
    "image_path",
    "patient_id",
    "disease",
    "severity",
    "severity_label",
    "dataset_source",
}

# Feature groups for ablation (column-name heuristics)
FIRST_ORDER_KEYWORDS = (
    "intensity",
    "entropy",
    "skew",
    "kurtosis",
    "q25",
    "q75",
    "median",
    "mean_",
    "std_",
    "min_",
    "max_",
)
TEXTURE_KEYWORDS = ("glcm", "gradient")
SHAPE_KEYWORDS = (
    "area",
    "perimeter",
    "aspect",
    "diameter",
    "extent",
    "solidity",
    "circularity",
)

COMPLETED: list[str] = []
SKIPPED: list[str] = []
GENERATED_FILES: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)


def save_path(rel: str) -> Path:
    p = OUT_DIR / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_FILES.append(str(p.relative_to(PROJECT_ROOT)))
    return p


def classify_feature(name: str) -> str:
    low = name.lower()
    if any(k in low for k in TEXTURE_KEYWORDS):
        return "texture"
    if any(k in low for k in SHAPE_KEYWORDS):
        return "shape"
    if any(k in low for k in FIRST_ORDER_KEYWORDS):
        return "first_order"
    return "other"


def load_data():
    """Load master dataset; prefer PyRadiomics file if present."""
    using_pyradiomics = False
    if PYRADIOMICS_PATH.exists():
        df = pd.read_csv(PYRADIOMICS_PATH)
        using_pyradiomics = True
        source = PYRADIOMICS_PATH.name
    elif DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
        source = DATASET_PATH.name
    else:
        raise FileNotFoundError("No dataset CSV found.")

    df = df.dropna(subset=["disease"])
    invalid = {"", "Unknown", "unknown", "nan", "NAN", "NaN"}
    df["disease"] = df["disease"].astype(str).str.strip()
    df = df[~df["disease"].isin(invalid)]

    has_patient = "patient_id" in df.columns and df["patient_id"].notna().sum() > 0
    if has_patient:
        df = df.dropna(subset=["patient_id"])

    meta = set(METADATA_COLS) | {"severity_label"}
    feature_cols = [c for c in df.columns if c not in meta and not c.startswith("original_diagnostic")]
    if using_pyradiomics:
        feature_cols = [c for c in feature_cols if c.startswith("original_") or c not in meta]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    valid = ~X.isna().all(axis=1)
    df = df.loc[valid].reset_index(drop=True)
    X = X.loc[valid].reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(df["disease"])

    groups = {}
    for col in feature_cols:
        groups[col] = classify_feature(col)

    return df, X, y, le, feature_cols, groups, source, has_patient, using_pyradiomics


def build_ml_models():
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
        "MLP Neural Network": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        max_iter=80,
                        random_state=RANDOM_STATE,
                        early_stopping=True,
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
        SKIPPED.append("XGBoost not installed")

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
        SKIPPED.append("LightGBM not installed")

    return models


def split_data(df, X, y, has_patient: bool):
    """Patient-level split when possible."""
    if has_patient:
        groups = df["patient_id"].astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        split_type = "patient_level"
        train_patients = df.iloc[train_idx]["patient_id"].astype(str).unique()
        test_patients = df.iloc[test_idx]["patient_id"].astype(str).unique()
        overlap = set(train_patients) & set(test_patients)
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        split_type = "stratified_image_level"
        train_patients = test_patients = []
        overlap = set()

    return train_idx, test_idx, split_type, train_patients, test_patients, overlap


def fit_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None
    return y_pred, y_proba


def multiclass_roc_auc(y_true, y_proba, n_classes: int):
    if y_proba is None:
        return float("nan")
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    try:
        return float(
            roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
        )
    except ValueError:
        return float("nan")


def task0_dataset_summary(df, feature_cols, groups, source, has_patient, le):
    log("\n=== TASK 0: Dataset summary ===")
    lines = [
        f"Generated: {datetime.now().isoformat()}",
        f"Source file: {source}",
        f"Rows: {len(df)}",
        f"Columns total: {len(df.columns)}",
        f"Feature columns: {len(feature_cols)}",
        f"Label column: disease",
        f"Classes ({len(le.classes_)}): {', '.join(le.classes_)}",
        "",
        "Class distribution:",
        df["disease"].value_counts().to_string(),
        "",
        f"patient_id present: {has_patient}",
    ]
    if has_patient:
        lines.append(f"Unique patients: {df['patient_id'].nunique()}")
    if "image_path" in df.columns:
        lines.append(f"image_path present: yes ({df['image_path'].notna().sum()} non-null)")
    else:
        lines.append("image_path present: no")

    lines.extend(["", "Feature groups:"])
    for gname in ("first_order", "texture", "shape", "other"):
        feats = [c for c, g in groups.items() if g == gname]
        lines.append(f"  {gname}: {len(feats)} -> {', '.join(feats[:8])}{'...' if len(feats)>8 else ''}")

    path = save_path("dataset_summary.txt")
    path.write_text("\n".join(lines), encoding="utf-8")
    COMPLETED.append("Dataset summary")
    log(f"Saved {path}")


def task1_patient_level(df, X, y, le, has_patient, models):
    log("\n=== TASK 1: Patient-level validation ===")
    train_idx, test_idx, split_type, train_pat, test_pat, overlap = split_data(
        df, X, y, has_patient
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rows = []
    preds_store = {}
    for name, model in models.items():
        log(f"  Training {name}...")
        y_pred, y_proba = fit_predict(model, X_train, y_train, X_test)
        preds_store[name] = y_pred
        rows.append(
            {
                "model": name,
                "split_type": split_type,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "roc_auc_ovr_macro": multiclass_roc_auc(y_test, y_proba, len(le.classes_)),
                "train_patients": len(train_pat) if has_patient else np.nan,
                "test_patients": len(test_pat) if has_patient else np.nan,
                "train_images": len(train_idx),
                "test_images": len(test_idx),
                "patient_overlap": len(overlap),
            }
        )

    out = pd.DataFrame(rows)
    save_path("patient_level_results.csv")
    out.to_csv(OUT_DIR / "patient_level_results.csv", index=False)

    if not has_patient:
        lim = OUT_DIR / "limitations.txt"
        lim.write_text(
            "patient_id missing or empty — used stratified image-level split instead of patient-level.\n",
            encoding="utf-8",
        )
    else:
        info = [
            f"Split: {split_type}",
            f"Train patients: {len(train_pat)}",
            f"Test patients: {len(test_pat)}",
            f"Train images: {len(train_idx)}",
            f"Test images: {len(test_idx)}",
            f"Patient overlap between train/test: {len(overlap)} (must be 0)",
        ]
        save_path("patient_split_report.txt").write_text("\n".join(info), encoding="utf-8")

    COMPLETED.append("Task 1 patient-level results")
    return train_idx, test_idx, X_train, X_test, y_train, y_test, preds_store, split_type


def task2_feature_importance(X_train, y_train, feature_cols, groups):
    log("\n=== TASK 2: Best radiomic features ===")
    X_imp = SimpleImputer(strategy="median").fit_transform(X_train)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1)
    rf.fit(X_imp, y_train)
    rf_imp = pd.Series(rf.feature_importances_, index=feature_cols, name="rf_importance")

    scores = {"rf_importance": rf_imp}

    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
        xgb.fit(X_imp, y_train)
        scores["xgb_importance"] = pd.Series(xgb.feature_importances_, index=feature_cols)
    except Exception as e:
        log(f"  XGBoost importance skipped: {e}")

    mi = mutual_info_classif(X_imp, y_train, random_state=RANDOM_STATE)
    scores["mutual_information"] = pd.Series(mi, index=feature_cols)

    f_vals, _ = f_classif(X_imp, y_train)
    scores["anova_f"] = pd.Series(f_vals, index=feature_cols)

    imp_df = pd.DataFrame(scores)
    for col in imp_df.columns:
        imp_df[f"{col}_rank"] = imp_df[col].rank(ascending=False)

    rank_cols = [c for c in imp_df.columns if c.endswith("_rank")]
    imp_df["mean_rank"] = imp_df[rank_cols].mean(axis=1)
    imp_df = imp_df.sort_values("mean_rank")
    imp_df["feature_group"] = [groups.get(c, "other") for c in imp_df.index]

    top20 = imp_df.head(20).reset_index().rename(columns={"index": "feature"})
    top20.to_csv(save_path("best_radiomic_features.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_df = top20.set_index("feature")["rf_importance"]
    plot_df.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Random Forest importance")
    ax.set_title("Top 20 radiomic features (multi-source ranking)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path("best_radiomic_features.png"), dpi=150)
    plt.close()

    COMPLETED.append("Task 2 feature importance")
    return imp_df, top20


def get_tree_pipeline_template():
    return Pipeline(
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
    )


def task3_shap(X_train, y_train, X_test, y_test, feature_cols, best_model_name, models):
    log("\n=== TASK 3: SHAP explainability ===")
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer tree model for SHAP speed
    shap_model_name = best_model_name
    if shap_model_name == "SVM":
        shap_model_name = "Random Forest" if "Random Forest" in models else best_model_name

    model = models[shap_model_name]
    model.fit(X_train, y_train)

    try:
        import shap

        n = min(SHAP_SAMPLE_SIZE, len(X_test))
        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(len(X_test), size=n, replace=False)
        X_sample = X_test.iloc[sample_idx]

        X_transformed = model.named_steps["imputer"].transform(X_sample)
        clf = model.named_steps["model"]

        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_transformed)
        else:
            # Kernel explainer fallback (small background)
            bg = model.named_steps["imputer"].transform(
                X_train.sample(min(50, len(X_train)), random_state=RANDOM_STATE)
            )
            explainer = shap.KernelExplainer(clf.predict_proba, bg)
            shap_values = explainer.shap_values(
                X_transformed[: min(80, len(X_transformed))]
            )
            X_transformed = X_transformed[: min(80, len(X_transformed))]

        plt.figure()
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_cols,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "shap_summary_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Mean |SHAP| for bar
        # Aggregate SHAP magnitudes (multiclass may be list or ndarray [samples, features, classes])
        sv_arr = np.asarray(shap_values[0] if isinstance(shap_values, list) and len(shap_values) == 1 else shap_values)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            mean_abs = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0).flatten()
        elif sv_arr.ndim == 3:
            mean_abs = np.abs(sv_arr).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(sv_arr).mean(axis=0).flatten()

        mean_abs = np.asarray(mean_abs).flatten()
        if len(mean_abs) != len(feature_cols):
            raise ValueError(
                f"SHAP length {len(mean_abs)} != features {len(feature_cols)}"
            )

        shap_top = (
            pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(20)
        )
        shap_top.to_csv(save_path("shap_top_20_features.csv"), index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=shap_top, y="feature", x="mean_abs_shap", color="coral")
        plt.title(f"Top 20 SHAP features ({shap_model_name})")
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "shap_bar_plot.png", dpi=150)
        plt.close()

        COMPLETED.append(f"Task 3 SHAP ({shap_model_name})")
        log(f"  SHAP saved under {SHAP_DIR}")
        return shap_top
    except Exception as e:
        SKIPPED.append(f"Task 3 SHAP: {e}")
        log(f"  SHAP failed: {e}")
        return None


def task4_ablation(X_train, y_train, X_test, y_test, feature_cols, groups, le):
    log("\n=== TASK 4: Ablation study ===")
    sets = {
        "first_order_only": [c for c in feature_cols if groups[c] == "first_order"],
        "texture_only": [c for c in feature_cols if groups[c] == "texture"],
        "shape_only": [c for c in feature_cols if groups[c] == "shape"],
        "first_order_plus_texture": [
            c for c in feature_cols if groups[c] in ("first_order", "texture")
        ],
        "all_features": feature_cols,
    }

    rows = []
    for name, cols in sets.items():
        if len(cols) == 0:
            rows.append(
                {
                    "feature_set": name,
                    "n_features": 0,
                    "accuracy": np.nan,
                    "f1_macro": np.nan,
                    "note": "no features in group",
                }
            )
            continue
        pipe = get_tree_pipeline_template()
        y_pred, _ = fit_predict(
            pipe, X_train[cols], y_train, X_test[cols]
        )
        rows.append(
            {
                "feature_set": name,
                "n_features": len(cols),
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "note": "",
            }
        )
        log(f"  {name}: acc={rows[-1]['accuracy']:.4f} f1_macro={rows[-1]['f1_macro']:.4f}")

    abl = pd.DataFrame(rows)
    abl.to_csv(save_path("ablation_study.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_abl = abl.dropna(subset=["f1_macro"])
    sns.barplot(data=plot_abl, x="feature_set", y="f1_macro", ax=ax, color="seagreen")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_title("Ablation: feature groups (Random Forest)")
    plt.tight_layout()
    plt.savefig(save_path("ablation_study.png"), dpi=150)
    plt.close()

    best_group = abl.loc[abl["f1_macro"].idxmax(), "feature_set"] if abl["f1_macro"].notna().any() else "unknown"
    COMPLETED.append(f"Task 4 ablation (best group: {best_group})")
    return abl, best_group


def task5_full_metrics(models, X_train, y_train, X_test, y_test, le):
    log("\n=== TASK 5: Full metrics + ROC ===")
    ROC_DIR.mkdir(parents=True, exist_ok=True)
    n_classes = len(le.classes_)
    rows = []

    for name, model in models.items():
        y_pred, y_proba = fit_predict(model, X_train, y_train, X_test)
        auc = multiclass_roc_auc(y_test, y_proba, n_classes)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "roc_auc_ovr_macro": auc,
            }
        )

        if y_proba is not None:
            y_bin = label_binarize(y_test, classes=list(range(n_classes)))
            for i, cls in enumerate(le.classes_):
                if y_bin[:, i].sum() == 0:
                    continue
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                plt.figure(figsize=(5, 4))
                plt.plot(fpr, tpr, label=f"{cls} (OvR)")
                plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.title(f"ROC OvR — {name} — {cls}")
                plt.tight_layout()
                safe = name.replace(" ", "_")
                plt.savefig(ROC_DIR / f"roc_{safe}_{cls.replace(' ', '_')}.png", dpi=120)
                plt.close()

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(save_path("full_model_metrics.csv"), index=False)
    COMPLETED.append("Task 5 full metrics + ROC")
    return metrics_df


def mcnemar_pvalue(y_true, pred_a, pred_b):
    """McNemar on discordant correct-classification pairs."""
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return np.nan, b, c
    try:
        from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

        table = [[0, b], [c, 0]]
        result = sm_mcnemar(table, exact=True)
        return float(result.pvalue), b, c
    except Exception:
        from scipy.stats import binomtest

        p = binomtest(min(b, c), b + c, 0.5).pvalue * 2
        return float(min(p, 1.0)), b, c


def task6_statistical_tests(models, X_train, y_train, X_test, y_test, df, has_patient, best_name, metrics_df):
    log("\n=== TASK 6: Statistical significance ===")
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds[name] = model.predict(X_test)

    rows = []
    for name in preds:
        if name == best_name:
            continue
        pval, b, c = mcnemar_pvalue(y_test, preds[best_name], preds[name])
        rows.append(
            {
                "test": "McNemar",
                "model_a": best_name,
                "model_b": name,
                "discordant_b": b,
                "discordant_c": c,
                "p_value": pval,
            }
        )

    # Wilcoxon on GroupKFold macro F1 if patient_id available
    if has_patient and "patient_id" in df.columns:
        gkf = GroupKFold(n_splits=5)
        groups_train = df["patient_id"].astype(str).values
        best_pipe = models[best_name]
        second = metrics_df.sort_values("f1_macro", ascending=False).iloc[1]["model"]
        scores_a, scores_b = [], []
        for tr, va in gkf.split(X_train, y_train, groups=groups_train):
            best_pipe.fit(X_train.iloc[tr], y_train[tr])
            models[second].fit(X_train.iloc[tr], y_train[tr])
            pa = best_pipe.predict(X_train.iloc[va])
            pb = models[second].predict(X_train.iloc[va])
            scores_a.append(f1_score(y_train[va], pa, average="macro", zero_division=0))
            scores_b.append(f1_score(y_train[va], pb, average="macro", zero_division=0))
        try:
            from scipy.stats import wilcoxon

            stat, p = wilcoxon(scores_a, scores_b)
            rows.append(
                {
                    "test": "Wilcoxon_signed_rank_CV",
                    "model_a": best_name,
                    "model_b": second,
                    "discordant_b": np.nan,
                    "discordant_c": np.nan,
                    "p_value": float(p),
                    "note": f"5-fold GroupKFold macro F1; stat={stat:.4f}",
                }
            )
        except Exception as e:
            rows.append(
                {
                    "test": "Wilcoxon_signed_rank_CV",
                    "model_a": best_name,
                    "model_b": second,
                    "p_value": np.nan,
                    "note": str(e),
                }
            )

    stat_df = pd.DataFrame(rows)
    stat_df.to_csv(save_path("statistical_tests.csv"), index=False)

    expl = [
        "Statistical tests explanation",
        "================================",
        "",
        "McNemar test: compares paired correctness of two classifiers on the same test set.",
        "  b = times model A correct and B wrong; c = opposite. Low p-value suggests significant difference.",
        "",
        "Wilcoxon signed-rank (if reported): compares macro F1 across 5 patient-grouped CV folds",
        "  between the best model and the second-best model.",
        "",
        f"Best model (by macro F1 on hold-out): {best_name}",
    ]
    save_path("statistical_tests_explanation.txt").write_text("\n".join(expl), encoding="utf-8")
    COMPLETED.append("Task 6 statistical tests")
    return stat_df


def task7_error_analysis(best_model, X_train, y_train, X_test, y_test, le):
    log("\n=== TASK 7: Error analysis ===")
    y_pred, _ = fit_predict(best_model, X_train, y_train, X_test)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(le.classes_))))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (best model)")
    plt.tight_layout()
    plt.savefig(save_path("confusion_matrix.png"), dpi=150)
    plt.close()

    # Per-class errors
    rows = []
    for i, cls in enumerate(le.classes_):
        mask = y_test == i
        n = int(mask.sum())
        correct = int((y_pred[mask] == i).sum())
        rows.append(
            {
                "class": cls,
                "n_test": n,
                "correct": correct,
                "errors": n - correct,
                "error_rate": (n - correct) / n if n else np.nan,
            }
        )

    err_df = pd.DataFrame(rows)
    err_df.to_csv(save_path("error_analysis.csv"), index=False)

    # Most confused pairs (off-diagonal)
    pairs = []
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            if i != j and cm[i, j] > 0:
                pairs.append(
                    {
                        "true_class": le.classes_[i],
                        "predicted_class": le.classes_[j],
                        "count": int(cm[i, j]),
                    }
                )
    pairs = sorted(pairs, key=lambda x: -x["count"])[:10]

    summary = [
        "Error analysis summary",
        "====================",
        "",
        classification_report(y_test, y_pred, target_names=le.classes_),
        "",
        "Most confused disease pairs (true -> predicted):",
    ]
    for p in pairs:
        summary.append(f"  {p['true_class']} -> {p['predicted_class']}: {p['count']}")
    save_path("error_analysis_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    COMPLETED.append("Task 7 error analysis")
    return err_df, pairs


def task8_hyperparameter_tuning(X_train, y_train, X_test, y_test, metrics_df):
    log("\n=== TASK 8: Hyperparameter optimization ===")
    top2 = metrics_df.sort_values("f1_macro", ascending=False).head(2)["model"].tolist()

    search_configs = {
        "Random Forest": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(class_weight="balanced", n_jobs=-1)),
                ]
            ),
            {
                "model__n_estimators": [200, 300, 400],
                "model__max_depth": [None, 20, 40],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "Logistic Regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.1, 1.0, 10.0], "model__solver": ["lbfgs", "saga"]},
        ),
        "XGBoost": (
            None,
            {},
        ),
    }

    rows = []
    best_params_lines = ["Best hyperparameters", "====================", ""]

    for name in top2:
        if name not in search_configs:
            rows.append({"model": name, "status": "skipped", "note": "no search grid defined"})
            continue
        pipe, grid = search_configs[name]
        if pipe is None:
            continue
        log(f"  RandomizedSearchCV: {name}...")
        search = RandomizedSearchCV(
            pipe,
            grid,
            n_iter=12,
            cv=3,
            scoring="f1_macro",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)
        rows.append(
            {
                "model": name,
                "best_params": json.dumps(search.best_params_),
                "cv_best_f1_macro": float(search.best_score_),
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            }
        )
        best_params_lines.append(f"{name}: {search.best_params_}")

    tuned = pd.DataFrame(rows)
    tuned.to_csv(save_path("tuned_model_results.csv"), index=False)
    save_path("best_hyperparameters.txt").write_text("\n".join(best_params_lines), encoding="utf-8")
    COMPLETED.append("Task 8 hyperparameter tuning")
    return tuned


def task9_calibration(best_model, X_train, y_train, X_test, y_test, le, best_name):
    log("\n=== TASK 9: Calibration analysis ===")
    n_classes = len(le.classes_)

    # Calibrate on train fold
    try:
        calibrated = CalibratedClassifierCV(best_model, method="sigmoid", cv=3)
        calibrated.fit(X_train, y_train)
        if n_classes == 2:
            proba = calibrated.predict_proba(X_test)[:, 1]
            y_bin = (y_test == 1).astype(int)
            prob_true, prob_pred = calibration_curve(y_bin, proba, n_bins=10)
            brier = brier_score_loss(y_bin, proba)
        else:
            # One-vs-rest calibration curve for most frequent class
            maj = int(np.bincount(y_test).argmax())
            proba = calibrated.predict_proba(X_test)[:, maj]
            y_bin = (y_test == maj).astype(int)
            prob_true, prob_pred = calibration_curve(y_bin, proba, n_bins=10)
            brier = brier_score_loss(y_bin, proba)

        plt.figure(figsize=(6, 5))
        plt.plot(prob_pred, prob_true, marker="o", label="Model")
        plt.plot([0, 1], [0, 1], "k--", label="Perfect")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(f"Calibration curve ({best_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path("calibration_curve.png"), dpi=150)
        plt.close()

        cal_df = pd.DataFrame(
            {
                "mean_predicted_prob": prob_pred,
                "fraction_positive": prob_true,
                "brier_score": [brier] * len(prob_pred),
                "class_used_for_curve": [le.classes_[maj] if n_classes > 2 else le.classes_[1]],
            }
        )
        cal_df.to_csv(save_path("calibration_results.csv"), index=False)
        COMPLETED.append("Task 9 calibration")
    except Exception as e:
        SKIPPED.append(f"Task 9 calibration: {e}")
        append_limitation(f"Calibration skipped: {e}")


def append_limitation(text: str) -> None:
    path = OUT_DIR / "limitations.txt"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(existing + text + "\n", encoding="utf-8")


def find_last_conv_layer(model):
    """Find last Conv2D layer (including inside nested backbone)."""
    from tensorflow.keras.layers import Conv2D

    def _search(m):
        for layer in reversed(m.layers):
            if isinstance(layer, Conv2D):
                return layer
            if hasattr(layer, "layers"):
                found = _search(layer)
                if found is not None:
                    return found
        return None

    return _search(model)


def make_gradcam(model, img_array, class_idx, last_conv_layer):
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        [model.inputs, model.outputs],
        [last_conv_layer.output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(np.expand_dims(img_array, 0))
        if preds.shape[-1] == 1:
            loss = preds[:, 0]
        else:
            loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    heatmap /= heatmap.max() + 1e-8
    return heatmap


def task10_gradcam():
    log("\n=== TASK 10: Grad-CAM ===")
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    val_split = DL_DIR / "val_split_severity.csv"
    archs = {
        "resnet50": "ResNet50",
        "efficientnetb0": "EfficientNetB0",
        "densenet121": "DenseNet121",
    }

    if not val_split.exists():
        msg = "val_split_severity.csv missing — Grad-CAM skipped."
        SKIPPED.append(msg)
        append_limitation(msg)
        log(f"  {msg}")
        return

    try:
        import cv2
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as e:
        SKIPPED.append(f"Grad-CAM: {e}")
        return

    preprocess_map = {}
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
    from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
    from tensorflow.keras.applications.densenet import preprocess_input as dense_pre

    preprocess_map["resnet50"] = resnet_pre
    preprocess_map["efficientnetb0"] = eff_pre
    preprocess_map["densenet121"] = dense_pre

    val_df = pd.read_csv(val_split)
    done_any = False

    for arch, display in archs.items():
        model_path = DL_DIR / f"{arch}_severity.keras"
        if not model_path.exists():
            log(f"  Skip {display}: no weights at {model_path.name}")
            append_limitation(f"Grad-CAM: {display} model file not found.\n")
            continue

        log(f"  Grad-CAM for {display}...")
        model = keras.models.load_model(model_path)
        last_conv = find_last_conv_layer(model)
        if last_conv is None:
            log(f"  No conv layer found for {display}")
            append_limitation(f"Grad-CAM: no Conv2D layer in {display}.\n")
            continue

        pre_fn = preprocess_map[arch]
        correct_saved = 0
        wrong_saved = 0

        for _, row in val_df.iterrows():
            path = Path(row["filepath"])
            if not path.exists():
                path = PROJECT_ROOT / row.get("filepath", "")
            if not path.exists():
                continue

            img = cv2.imread(str(path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            x = pre_fn(img_resized.astype(np.float32))
            pred = model.predict(np.expand_dims(x, 0), verbose=0)
            pred_cls = int(pred.argmax()) if pred.shape[-1] > 1 else int(pred[0][0] >= 0.5)
            true_cls = int(row["label"])

            is_correct = pred_cls == true_cls
            if is_correct and correct_saved >= GRADCAM_N_EACH:
                continue
            if not is_correct and wrong_saved >= GRADCAM_N_EACH:
                continue

            try:
                heatmap = make_gradcam(model, x, pred_cls, last_conv)
                heatmap = cv2.resize(heatmap, (224, 224))
                heatmap_uint8 = np.uint8(255 * heatmap)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)

                tag = "correct" if is_correct else "wrong"
                out_name = f"{arch}_{tag}_{true_cls}_{pred_cls}_{path.stem}.png"
                cv2.imwrite(str(GRADCAM_DIR / out_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                if is_correct:
                    correct_saved += 1
                else:
                    wrong_saved += 1
            except Exception:
                continue

            if correct_saved >= GRADCAM_N_EACH and wrong_saved >= GRADCAM_N_EACH:
                break

        done_any = done_any or (correct_saved + wrong_saved) > 0
        log(f"    Saved {correct_saved} correct + {wrong_saved} wrong examples")

    if done_any:
        COMPLETED.append("Task 10 Grad-CAM (partial — severity CNNs only)")
    else:
        SKIPPED.append("Task 10 Grad-CAM: no images generated")


def write_final_summary(
    best_name,
    top20_features,
    best_ablation_group,
    has_patient,
    split_type,
    metrics_df,
):
    log("\n=== Final summary ===")
    top_feat = (
        top20_features["feature"].head(5).tolist()
        if top20_features is not None and len(top20_features)
        else []
    )

    lines = [
        "A+ IMPLEMENTATION SUMMARY",
        "=========================",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "COMPLETED:",
    ]
    for c in COMPLETED:
        lines.append(f"  - {c}")

    lines.extend(["", "SKIPPED / LIMITATIONS:"])
    if SKIPPED:
        for s in SKIPPED:
            lines.append(f"  - {s}")
    else:
        lines.append("  - (none)")

    lim_path = OUT_DIR / "limitations.txt"
    if lim_path.exists():
        lines.extend(["", "See also limitations.txt:", lim_path.read_text(encoding="utf-8")[:1500]])

    lines.extend(
        [
            "",
            "KEY FINDINGS",
            "------------",
            f"Patient-level validation: {'yes' if has_patient else 'no'} ({split_type})",
            f"Best ML model (macro F1 on test): {best_name}",
            f"Best model test metrics: {metrics_df[metrics_df['model']==best_name].to_dict('records')}",
            f"Top radiomic features (head): {top_feat}",
            f"Best feature group (ablation): {best_ablation_group}",
            "",
            "RECOMMENDED FIGURES/TABLES FOR CHAPTER 4",
            "----------------------------------------",
            "  - Table: patient_level_results.csv / full_model_metrics.csv",
            "  - Figure: best_radiomic_features.png",
            "  - Figure: shap/shap_summary_plot.png, shap_bar_plot.png",
            "  - Figure: ablation_study.png",
            "  - Figure: confusion_matrix.png",
            "  - Figure: roc_curves/ (per-class OvR)",
            "  - Figure: calibration_curve.png",
            "  - Figure: gradcam/ (if CNN weights available)",
            "  - Table: statistical_tests.csv, error_analysis.csv",
            "",
            "ALL GENERATED FILES:",
        ]
    )
    for f in sorted(set(GENERATED_FILES)):
        lines.append(f"  - {f}")

    save_path("A_PLUS_IMPLEMENTATION_SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")
    log("\n".join(lines[:40]))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 70)
    log("A+ Full Improvements Pipeline")
    log(f"Output directory: {OUT_DIR}")
    log("=" * 70)

    try:
        df, X, y, le, feature_cols, groups, source, has_patient, _ = load_data()
        task0_dataset_summary(df, feature_cols, groups, source, has_patient, le)

        models = build_ml_models()
        train_idx, test_idx, X_train, X_test, y_train, y_test, _, split_type = task1_patient_level(
            df, X, y, le, has_patient, models
        )

        imp_df, top20 = task2_feature_importance(X_train, y_train, feature_cols, groups)

        metrics_df = task5_full_metrics(models, X_train, y_train, X_test, y_test, le)
        best_name = metrics_df.sort_values("f1_macro", ascending=False).iloc[0]["model"]
        log(f"\nBest model by macro F1: {best_name}")

        task3_shap(X_train, y_train, X_test, y_test, feature_cols, best_name, models)

        abl_df, best_ablation_group = task4_ablation(
            X_train, y_train, X_test, y_test, feature_cols, groups, le
        )

        task6_statistical_tests(
            models, X_train, y_train, X_test, y_test, df.iloc[train_idx], has_patient, best_name, metrics_df
        )

        best_model = models[best_name]
        task7_error_analysis(best_model, X_train, y_train, X_test, y_test, le)

        task8_hyperparameter_tuning(X_train, y_train, X_test, y_test, metrics_df)

        task9_calibration(best_model, X_train, y_train, X_test, y_test, le, best_name)

        try:
            task10_gradcam()
        except Exception as e:
            SKIPPED.append(f"Task 10 Grad-CAM: {e}")
            append_limitation(f"Grad-CAM error: {e}\n")
            log(f"  Grad-CAM error (non-fatal): {e}")

        write_final_summary(
            best_name, top20, best_ablation_group, has_patient, split_type, metrics_df
        )

        log("\n" + "=" * 70)
        log(f"DONE — {len(set(GENERATED_FILES))} files under {OUT_DIR}")
        log("=" * 70)

    except Exception as e:
        log(f"FATAL ERROR: {e}")
        traceback.print_exc()
        append_limitation(f"Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
