"""
run_cnn_confusion_matrix.py
----------------------------
Generates a normalised confusion matrix for the EfficientNetB0 4-class disease CNN.

Input:
  gui_demo/models/efficientnetb0_disease.keras   — trained CNN weights
  gui_demo/models/disease_label_classes.json     — class-index mapping (4 classes)
  data/images_extracted_from_mat_LABELED/        — MAT ultrasound images per class folder

Processing:
  1. Scan all PNG files in the MAT folder structure (including "IBM" → "Inclusion Body Myositis").
  2. Parse patient ID from each filename stem (characters before the first underscore).
  3. Apply GroupShuffleSplit (test_size=0.2, random_state=42) to create a patient-level
     test split — the same strategy used during CNN training.
  4. Run inference on every test image and collect true vs predicted labels.
  5. Plot a seaborn heatmap of the confusion matrix (normalised by true label,
     so each row sums to 1.0).
  6. Print a per-class classification report (precision, recall, F1).

Output (saved to output/aplus/):
  cnn_confusion_matrix.png  — normalised heatmap

Run: python scripts/run_cnn_confusion_matrix.py
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# ── paths ────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE, "gui_demo", "models", "efficientnetb0_disease.keras")
CLASSES_PATH = os.path.join(BASE, "gui_demo", "models", "disease_label_classes.json")
MAT_DIR      = os.path.join(BASE, "data", "images_extracted_from_mat_LABELED")
OUT_ROOT     = os.path.join(BASE, "output", "aplus", "cnn_confusion_matrix")
os.makedirs(OUT_ROOT, exist_ok=True)

IMG_SIZE = 224
plt.style.use("seaborn-v0_8-whitegrid")

# ── load model + classes ──────────────────────────────────────────────────────
print("Loading model …")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASSES_PATH) as f:
    meta = json.load(f)
classes     = meta["classes"]          # e.g. ["Dermatomyositis", "Inclusion Body Myositis", ...]
class_to_idx = meta["class_to_idx"]
print(f"Classes: {classes}")

# ── collect all images + labels ───────────────────────────────────────────────
print(f"Scanning {MAT_DIR} …")
rows = []

# Map known alternative folder names to canonical class labels
FOLDER_ALIASES = {
    "IBM": "Inclusion Body Myositis",
}

# Try subfolder-per-class layout first
for cls_name in classes:
    safe = cls_name.replace(" ", "_")
    for pattern in [
        os.path.join(MAT_DIR, cls_name, "*.png"),
        os.path.join(MAT_DIR, safe,     "*.png"),
    ]:
        hits = glob.glob(pattern)
        for p in hits:
            rows.append({"path": p, "disease": cls_name})

# Also scan subfolders by alias (e.g. "IBM" → "Inclusion Body Myositis")
for folder_name, mapped_cls in FOLDER_ALIASES.items():
    if mapped_cls not in classes:
        continue
    hits = glob.glob(os.path.join(MAT_DIR, folder_name, "*.png"))
    for p in hits:
        rows.append({"path": p, "disease": mapped_cls})

# Fall back: flat folder with class name embedded in filename
if not rows:
    all_pngs = glob.glob(os.path.join(MAT_DIR, "**", "*.png"), recursive=True)
    for p in all_pngs:
        stem = os.path.splitext(os.path.basename(p))[0]
        matched = None
        for cls_name in classes:
            if cls_name.replace(" ", "_").lower() in stem.lower() or \
               cls_name.lower() in stem.lower():
                matched = cls_name
                break
        if matched:
            rows.append({"path": p, "disease": matched})

if not rows:
    print(f"ERROR: No images found under {MAT_DIR}")
    raise SystemExit(1)

df = pd.DataFrame(rows)
print(f"Found {len(df)} images across {df['disease'].nunique()} classes")
print(df["disease"].value_counts().to_string())

# ── patient-level split ───────────────────────────────────────────────────────
# parse patient id from filename stem before first underscore
df["patient_id"] = df["path"].apply(
    lambda p: os.path.splitext(os.path.basename(p))[0].split("_")[0]
)

df["label_idx"] = df["disease"].map(class_to_idx)
X_paths = df["path"].values
y       = df["label_idx"].values
groups  = df["patient_id"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, test_idx = next(gss.split(X_paths, y, groups))
test_paths  = X_paths[test_idx]
y_test      = y[test_idx]

print(f"\nTest split: {len(test_paths)} images")

# ── inference ─────────────────────────────────────────────────────────────────
def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.cast(img, tf.float32) / 255.0

print("Running inference …")
BATCH = 32
y_pred = []
for i in range(0, len(test_paths), BATCH):
    batch_paths = test_paths[i:i+BATCH]
    imgs  = tf.stack([load_img(p) for p in batch_paths])
    probs = model.predict(imgs, verbose=0)
    y_pred.extend(np.argmax(probs, axis=1))
    if (i // BATCH) % 5 == 0:
        print(f"  {i+len(batch_paths)}/{len(test_paths)} images done")

y_pred = np.array(y_pred)

# ── metrics ───────────────────────────────────────────────────────────────────
acc = accuracy_score(y_test, y_pred) * 100
print(f"\nOverall accuracy: {acc:.2f}%")
print("\nClassification report:")
actual_classes = sorted(set(y_test))
actual_names   = [classes[i] for i in actual_classes]
print(classification_report(y_test, y_pred, labels=actual_classes, target_names=actual_names))

# ── confusion matrix heatmap ──────────────────────────────────────────────────
cm      = confusion_matrix(y_test, y_pred, labels=actual_classes)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # normalise by true label

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm_norm,
    annot=True, fmt=".2f",
    cmap="Blues",
    xticklabels=actual_names,
    yticklabels=actual_names,
    linewidths=0.5,
    vmin=0, vmax=1,
    ax=ax,
)
ax.set_xlabel("Predicted label", fontsize=11)
ax.set_ylabel("True label",      fontsize=11)
ax.set_title(
    f"EfficientNetB0 — Disease Confusion Matrix\n"
    f"(normalised by true class | test acc = {acc:.1f}%)",
    fontsize=11,
)
plt.xticks(rotation=25, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

out_path = os.path.join(OUT_ROOT, "cnn_confusion_matrix.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")
