"""Generate Grad-CAM for ResNet50 on the IBM (Inclusion Body Myositis) class.

Saves output to:
  output/aplus/run_gradcam/gradcam_resnet50_Inclusion_Body_Myositis.png

Uses the same CLAHE-ROI preprocessing as the trained model.
Run from the project root:
  python scripts/generate_resnet50_gradcam_ibm.py
"""
import sys
import json
import numpy as np
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
GUI_DIR    = ROOT / "gui_demo"
MODEL_PATH = GUI_DIR / "models" / "resnet50_disease.keras"
META_PATH  = GUI_DIR / "models" / "disease_label_classes.json"
OUT_DIR    = ROOT / "output" / "aplus" / "run_gradcam"
IBM_IMGS   = ROOT / "demo_data" / "IBM"

sys.path.insert(0, str(GUI_DIR))

# ── load class info ───────────────────────────────────────────────────────────
with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)
classes   = meta["classes"]
ibm_idx   = meta["class_to_idx"].get("Inclusion Body Myositis", 1)
print(f"Classes: {classes}")
print(f"IBM index: {ibm_idx}")

# ── pick best IBM image (first one that loads cleanly) ────────────────────────
candidates = sorted(IBM_IMGS.glob("*.png"))
if not candidates:
    print("ERROR: No IBM demo images found in demo_data/IBM/")
    sys.exit(1)
image_path = candidates[0]
print(f"Using image: {image_path.name}")

# ── load model ────────────────────────────────────────────────────────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

print("Loading ResNet50 model ...")
model = tf.keras.models.load_model(str(MODEL_PATH))
print("Model loaded.")

# Find the last convolutional layer in ResNet50
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer.name
        break
if last_conv_layer is None:
    # fallback for ResNet50 standard layer name
    last_conv_layer = "conv5_block3_out"
print(f"Last conv layer: {last_conv_layer}")

# ── preprocess image (same CLAHE-ROI pipeline as training) ───────────────────
from image_pipeline import prepare_ultrasound_cnn_tensor
from model_registry import get_preprocess_fn

pre_fn = get_preprocess_fn("resnet50")
tensor = prepare_ultrasound_cnn_tensor(image_path, pre_fn, augment=False)
if tensor is None:
    print("ERROR: preprocessing returned None")
    sys.exit(1)
x = np.expand_dims(tensor, axis=0)
print(f"Input tensor shape: {x.shape}")

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
grad_model = tf.keras.models.Model(
    inputs  = model.inputs,
    outputs = [model.get_layer(last_conv_layer).output, model.output],
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(x)
    loss = predictions[:, ibm_idx]

grads      = tape.gradient(loss, conv_outputs)
pooled     = tf.reduce_mean(grads, axis=(0, 1, 2))
cam        = conv_outputs[0] @ pooled[..., tf.newaxis]
cam        = tf.squeeze(cam)
cam        = tf.nn.relu(cam)
cam_np     = cam.numpy()
if cam_np.max() > 0:
    cam_np = cam_np / cam_np.max()

print(f"Grad-CAM map shape: {cam_np.shape}, max: {cam_np.max():.4f}")

# ── overlay on original image ─────────────────────────────────────────────────
import cv2
from PIL import Image

orig_rgb = np.array(Image.open(image_path).convert("RGB"))
h, w     = orig_rgb.shape[:2]

# Resize heatmap to match image
cam_resized = cv2.resize(cam_np, (w, h))
heatmap     = np.uint8(255 * cam_resized)
heatmap_col = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
heatmap_rgb = cv2.cvtColor(heatmap_col, cv2.COLOR_BGR2RGB)

# Blend
alpha      = 0.5
overlay    = np.uint8(alpha * heatmap_rgb + (1 - alpha) * orig_rgb)

# ── annotate and save ─────────────────────────────────────────────────────────
from PIL import ImageDraw, ImageFont

final = Image.fromarray(overlay)
draw  = ImageDraw.Draw(final)
try:
    font = ImageFont.truetype("arial.ttf", 16)
except Exception:
    font = ImageFont.load_default()

probs  = model.predict(x, verbose=0)[0]
conf   = float(probs[ibm_idx]) * 100
draw.rectangle([0, 0, w - 1, 26], fill=(0, 0, 0, 180))
draw.text((6, 5), f"ResNet50  |  IBM  |  conf: {conf:.1f}%", fill=(255, 255, 255), font=font)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "gradcam_resnet50_Inclusion_Body_Myositis.png"
final.save(str(out_path))
print(f"\nSaved: {out_path}")
print("Done.")
