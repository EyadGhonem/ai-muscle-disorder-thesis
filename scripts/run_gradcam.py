"""
Grad-CAM visualisations for EfficientNetB0 disease model.
Run: python scripts/run_gradcam.py
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# ── paths ────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE, "gui_demo", "models", "efficientnetb0_disease.keras")
CLASSES_PATH= os.path.join(BASE, "gui_demo", "models", "disease_label_classes.json")
MANIFEST    = os.path.join(BASE, "output", "gui_real_ultrasound_manifest.csv")
MAT_DIR     = os.path.join(BASE, "data", "images_extracted_from_mat_LABELED")
OUT_ROOT    = os.path.join(BASE, "output", "aplus", "run_gradcam")
os.makedirs(OUT_ROOT, exist_ok=True)

PALETTE = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]
plt.style.use("seaborn-v0_8-whitegrid")

IMG_SIZE = 224

# ── load model + classes ──────────────────────────────────────────────────────
print("Loading model …")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.summary(print_fn=lambda x: None)

with open(CLASSES_PATH) as f:
    meta = json.load(f)
classes = meta["classes"]
print(f"Classes: {classes}")


def find_last_conv(m):
    """Return name of last Conv2D layer."""
    name = None
    for layer in m.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            name = layer.name
        # handle nested models (EfficientNet backbone)
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if isinstance(sub, tf.keras.layers.Conv2D):
                    name = sub.name
    return name


def load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()


def compute_gradcam(model, img_array, class_idx):
    """Pure TF GradientTape Grad-CAM; returns heatmap (H,W) float32."""
    # find last conv layer in the backbone sub-model or top model
    grad_model = None
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):               # nested model (backbone)
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    try:
                        grad_model = tf.keras.Model(
                            inputs  = model.inputs,
                            outputs = [layer.output, model.output],
                        )
                    except Exception:
                        pass
                    if grad_model is not None:
                        break
            if grad_model is not None:
                break
        if isinstance(layer, tf.keras.layers.Conv2D):
            try:
                grad_model = tf.keras.Model(
                    inputs  = model.inputs,
                    outputs = [layer.output, model.output],
                )
            except Exception:
                pass
            if grad_model is not None:
                break

    if grad_model is None:
        raise RuntimeError("Could not build grad_model — no Conv2D found.")

    inp = tf.expand_dims(img_array, 0)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp, training=False)
        score = preds[:, class_idx]

    grads  = tape.gradient(score, conv_out)          # (1, H, W, C)
    weights = tf.reduce_mean(grads, axis=[0, 1, 2])  # (C,)
    cam    = tf.reduce_sum(conv_out[0] * weights, axis=-1)  # (H, W)
    cam    = tf.maximum(cam, 0)
    cam    = cam.numpy()
    if cam.max() > 0:
        cam /= cam.max()
    return cam


def overlay_heatmap(img_np, heatmap, alpha=0.4):
    """Resize heatmap to img size and blend."""
    from PIL import Image
    h, w = img_np.shape[:2]
    hm_uint8 = (heatmap * 255).astype(np.uint8)
    hm_resized = np.array(Image.fromarray(hm_uint8).resize((w, h), Image.BILINEAR))
    hm_color   = plt.cm.jet(hm_resized / 255.0)[..., :3]  # (H,W,3) float
    overlaid   = (1 - alpha) * img_np + alpha * hm_color
    overlaid   = np.clip(overlaid, 0, 1)
    return overlaid


# ── find one sample image per class ──────────────────────────────────────────
def find_samples():
    samples = {}  # class_name -> path

    # try manifest first
    if os.path.exists(MANIFEST):
        mf = pd.read_csv(MANIFEST)
        if "disease" in mf.columns and "image_path" in mf.columns:
            for cls in classes:
                sub = mf[mf["disease"] == cls]
                if len(sub):
                    p = sub.iloc[0]["image_path"]
                    if os.path.exists(p):
                        samples[cls] = p

    # fill from MAT folder for remaining classes
    for cls in classes:
        if cls in samples:
            continue
        safe_cls = cls.replace(" ", "_").replace("/", "_")
        patterns  = [
            os.path.join(MAT_DIR, cls, "*.png"),
            os.path.join(MAT_DIR, safe_cls, "*.png"),
            os.path.join(MAT_DIR, f"*{cls}*", "*.png"),
            os.path.join(MAT_DIR, "**", f"*{safe_cls}*.png"),
        ]
        for pat in patterns:
            hits = glob.glob(pat, recursive=True)
            if hits:
                samples[cls] = hits[0]
                break

    return samples

print("Searching for sample images …")
samples = find_samples()
print(f"Found samples for: {list(samples.keys())}")

if not samples:
    print("ERROR: No sample images found. Check data/images_extracted_from_mat_LABELED/ exists.")
    raise SystemExit(1)

# ── Grad-CAM per class ────────────────────────────────────────────────────────
results = {}  # class_name -> (orig_img, overlay, pred_class, cam)

for cls_name, img_path in samples.items():
    cls_idx = classes.index(cls_name) if cls_name in classes else 0
    print(f"  Processing {cls_name} …")

    img_np = load_and_preprocess(img_path)
    try:
        cam    = compute_gradcam(model, img_np, cls_idx)
        ovrly  = overlay_heatmap(img_np, cam)
    except Exception as e:
        print(f"    Grad-CAM failed: {e} — saving plain image only")
        cam   = np.zeros(img_np.shape[:2], dtype=np.float32)
        ovrly = img_np.copy()

    probs    = model.predict(np.expand_dims(img_np, 0), verbose=0)[0]
    pred_cls = classes[int(np.argmax(probs))]
    results[cls_name] = (img_np, ovrly, pred_cls, cam)

    # individual save
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(ovrly);  axes[1].set_title(f"Grad-CAM\nPredicted: {pred_cls}"); axes[1].axis("off")
    fig.suptitle(f"Grad-CAM — {cls_name}", fontsize=12)
    plt.tight_layout()
    safe = cls_name.replace(" ", "_")
    fig.savefig(os.path.join(OUT_ROOT, f"gradcam_{safe}.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ── grid figure ───────────────────────────────────────────────────────────────
n     = len(results)
fig   = plt.figure(figsize=(5 * n, 8))
gs    = gridspec.GridSpec(2, n, hspace=0.4, wspace=0.15)

for col, (cls_name, (orig, ovrly, pred_cls, _)) in enumerate(results.items()):
    ax_orig  = fig.add_subplot(gs[0, col])
    ax_ovrly = fig.add_subplot(gs[1, col])
    ax_orig.imshow(orig);  ax_orig.set_title(f"{cls_name}\n(original)", fontsize=9); ax_orig.axis("off")
    ax_ovrly.imshow(ovrly); ax_ovrly.set_title(f"Pred: {pred_cls}", fontsize=9); ax_ovrly.axis("off")

fig.suptitle("Grad-CAM Grid — EfficientNetB0 Disease Classifier", fontsize=13)
fig.savefig(os.path.join(OUT_ROOT, "gradcam_grid.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nAll Grad-CAM outputs saved to: {OUT_ROOT}")
