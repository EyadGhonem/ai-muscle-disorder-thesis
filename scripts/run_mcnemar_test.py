"""
McNemar's test for pairwise model comparison.
Pairs: SVM vs RF, SVM vs GB, GB vs RF.
Run: python scripts/run_mcnemar_test.py
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from statsmodels.stats.contingency_tables import mcnemar

# ── paths ────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_PATH = os.path.join(BASE, "output", "baseline_and_advanced_models", "trained_models.pkl")
CSV_PATH = os.path.join(BASE, "output", "final_ultrasound_dataset.csv")

EXCLUDE_COLS = ["image_path", "patient_id", "disease", "severity_label", "dataset_source"]

PAIRS = [
    ("SVM",               "Random Forest"),
    ("SVM",               "Gradient Boosting"),
    ("Gradient Boosting", "Random Forest"),
]

# ── load ─────────────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(CSV_PATH)
df = df[df["disease"].notna() & (df["disease"] != "NAN")].copy()

feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
X      = df[feature_cols].fillna(df[feature_cols].median())
groups = df["patient_id"].astype(str)

with open(PKL_PATH, "rb") as f:
    bundle = pickle.load(f)
models = bundle["models"]
scaler = bundle["scaler"]
le     = bundle["label_encoder"]

mask   = np.isin(df["disease"].values, le.classes_)
X      = X[mask]
y_raw  = df["disease"].values[mask]
groups = groups[mask]
y      = le.transform(y_raw)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, test_idx = next(gss.split(X, y, groups))
X_test = scaler.transform(X.iloc[test_idx].values)
y_test = y[test_idx]

print(f"Test set: {len(y_test)} samples\n")

# ── get predictions for all needed models ─────────────────────────────────────
needed = {m for pair in PAIRS for m in pair}
preds  = {}
for name in needed:
    if name not in models:
        print(f"WARNING: '{name}' not found in pkl — pairs involving it will be skipped.")
        continue
    preds[name] = models[name].predict(X_test)
    correct = (preds[name] == y_test).sum()
    print(f"  {name:<25} accuracy = {correct/len(y_test)*100:.2f}%")

# ── McNemar per pair ──────────────────────────────────────────────────────────
print()
results = []
for name_a, name_b in PAIRS:
    if name_a not in preds or name_b not in preds:
        print(f"Skipping {name_a} vs {name_b} (model missing)")
        continue

    correct_a = (preds[name_a] == y_test)
    correct_b = (preds[name_b] == y_test)

    # 2×2 contingency: rows = model A correct/wrong, cols = model B correct/wrong
    # cell[i,j]: i=A outcome (1=correct,0=wrong), j=B outcome
    n11 = int(np.sum( correct_a &  correct_b))   # both correct
    n10 = int(np.sum( correct_a & ~correct_b))   # A correct, B wrong
    n01 = int(np.sum(~correct_a &  correct_b))   # A wrong, B correct
    n00 = int(np.sum(~correct_a & ~correct_b))   # both wrong

    table = np.array([[n11, n10],
                      [n01, n00]])

    res       = mcnemar(table, exact=False)
    chi2      = res.statistic
    p_value   = res.pvalue
    sig       = "YES (p<0.05)" if p_value < 0.05 else "no"

    results.append((name_a, name_b, n11, n10, n01, n00, chi2, p_value, sig))

    print(f"{'─'*55}")
    print(f"  {name_a}  vs  {name_b}")
    print(f"  Contingency table:")
    print(f"                  {name_b} correct  {name_b} wrong")
    print(f"  {name_a} correct      {n11:>5}         {n10:>5}")
    print(f"  {name_a} wrong        {n01:>5}         {n00:>5}")
    print(f"  chi2 = {chi2:.4f}   p-value = {p_value:.4e}   Significant: {sig}")

# ── summary table ─────────────────────────────────────────────────────────────
print(f"\n{'═'*75}")
print(f"  {'Model A':<25} {'Model B':<25} {'chi2':>8}  {'p-value':>10}  Significant")
print(f"  {'-'*25} {'-'*25} {'-'*8}  {'-'*10}  {'-'*12}")
for name_a, name_b, _, _, _, _, chi2, p_value, sig in results:
    print(f"  {name_a:<25} {name_b:<25} {chi2:>8.4f}  {p_value:>10.4e}  {sig}")
print(f"{'═'*75}")
print("\nInterpretation: p < 0.05 means the two models make significantly different errors.")
