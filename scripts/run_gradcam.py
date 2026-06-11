"""
run_gradcam.py
--------------
Gradient-weighted Class Activation Mapping (Grad-CAM) visualisations for the
EfficientNetB0 MAT 4-class disease CNN.

Grad-CAM highlights the image regions that most influenced the CNN's prediction
by computing the gradient of the predicted class score with respect to the last
convolutional layer's feature maps, then creating a weighted average heatmap.

Input:
  gui_demo/models/efficientnetb0_disease.keras   — trained disease CNN weights
  gui_demo/models/disease_label_classes.json     — class-index mapping
  output/gui_real_ultrasound_manifest.csv        — (optional) image paths per class
  data/images_extracted_from_mat_LABELED/        — fallback image source

Processing:
  1. Load the EfficientNetB0 model and class names.
  2. Find one representative image per disease class (from manifest or MAT folder).
  3. For each image, run ``compute_gradcam`` to obtain the heatmap and overlay it
     on the original image using a JET colormap (``overlay_heatmap``).
  4. Run model inference and record the predicted class for each image.

Output (saved to output/aplus/run_gradcam/):
  gradcam_<ClassName>.png  — side-by-side original + Grad-CAM overlay per class
  gradcam_grid.png         — combined grid of all classes (2 rows × n_classes cols)

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
    """Compute a Grad-CAM heatmap for a single image and target class.

    Algorithm (pure TensorFlow GradientTape):
    1. Build a sub-model that outputs (last_conv_feature_map, predictions).
    2. Record gradients of the target class score w.r.t. the feature maps.
    3. Pool the gradients spatially (global average) to get per-channel weights.
    4. Compute the weighted sum of feature maps → ReLU → normalise to [0, 1].

    Handles both flat-layer and nested-backbone architectures (EfficientNet
    wraps its Conv2D layers inside a sub-model).

    Parameters
    ----------
    model      : loaded Keras model
    img_array  : (H, W, 3) float32 image array in [0, 1]
    class_idx  : index of the target class in the softmax output

    Returns
    -------
    np.ndarray of shape (H', W') normalised to [0, 1], where H'×W' is the
    spatial resolution of the last convolutional feature map.
    """
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
    """Resize the Grad-CAM heatmap to match the input image and blend with JET colormap.

    Parameters
    ----------
    img_np  : (H, W, 3) float32 original image in [0, 1]
    heatmap : (H', W') normalised Grad-CAM map in [0, 1]
    alpha   : blending weight for the heatmap overlay (default 0.4)

    Returns
    -------
    (H, W, 3) float32 blended image clipped to [0, 1]
    """
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
    """Locate one representative image per disease class.

    Search order:
    1. gui_real_ultrasound_manifest.csv (pre-built image list).
    2. Direct sub-folder scan under data/images_extracted_from_mat_LABELED/.

    Returns
    -------
    dict mapping class_name → absolute image file path
    """
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
