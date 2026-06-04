#!/usr/bin/env python3
"""Fix SHAP bar/csv and calibration CSV if missing from main A+ run."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent
OUT = PROJECT_ROOT / "results" / "a_plus_full_improvements"
SHAP_DIR = OUT / "shap"
DATA = PROJECT_ROOT / "output" / "final_ultrasound_dataset.csv"
RANDOM_STATE = 42

METADATA = {"image_path", "patient_id", "disease", "severity", "severity_label", "dataset_source"}


def load_split():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["disease", "patient_id"])
    invalid = {"", "Unknown", "unknown", "nan", "NAN", "NaN"}
    df["disease"] = df["disease"].astype(str).str.strip()
    df = df[~df["disease"].isin(invalid)]
    feats = [c for c in df.columns if c not in METADATA]
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    le = LabelEncoder()
    y = le.fit_transform(df["disease"])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups=df["patient_id"].astype(str)))
    return X.iloc[tr], X.iloc[te], y[tr], y[te], feats, le


def fix_shap():
    import shap

    X_train, X_test, y_train, y_test, feats, _ = load_split()
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    n = min(200, len(X_test))
    idx = np.random.default_rng(RANDOM_STATE).choice(len(X_test), n, replace=False)
    Xt = pipe.named_steps["imputer"].transform(X_test.iloc[idx])
    clf = pipe.named_steps["model"]
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt)

    sv_arr = np.asarray(sv)
    if sv_arr.ndim == 3:
        # (samples, features, classes)
        mean_abs = np.abs(sv_arr).mean(axis=(0, 2))
    elif isinstance(sv, list):
        mean_abs = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0).flatten()
    else:
        mean_abs = np.abs(sv_arr).mean(axis=0).flatten()

    top = (
        pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(20)
    )
    top.to_csv(OUT / "shap_top_20_features.csv", index=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, y="feature", x="mean_abs_shap", color="coral")
    plt.title("Top 20 SHAP features (Random Forest)")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_bar_plot.png", dpi=150)
    plt.close()
    print("SHAP fix OK:", OUT / "shap_top_20_features.csv")


def fix_calibration():
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test, _, le = load_split()
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced")),
        ]
    )
    cal = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
    cal.fit(X_train, y_train)
    maj = int(np.bincount(y_test).argmax())
    proba = cal.predict_proba(X_test)[:, maj]
    y_bin = (y_test == maj).astype(int)
    prob_true, prob_pred = calibration_curve(
        y_bin, proba, n_bins=8, strategy="quantile"
    )
    brier = brier_score_loss(y_bin, proba)
    pd.DataFrame(
        {
            "mean_predicted_prob": prob_pred,
            "fraction_positive": prob_true,
            "brier_score": brier,
            "class": le.classes_[maj],
        }
    ).to_csv(OUT / "calibration_results.csv", index=False)
    print("Calibration CSV OK:", OUT / "calibration_results.csv")


if __name__ == "__main__":
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    fix_shap()
    fix_calibration()
