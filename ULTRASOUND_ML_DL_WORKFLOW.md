# Complete Workflow: MRI Radiomics + Ultrasound (ML & DL)

## Overview

Your project now combines:
- **MRI Analysis** (existing radiomics)
- **Ultrasound Analysis** (new machine learning & deep learning)

With both **machine learning** (radiomics features) and **deep learning** (neural networks):

```
┌─────────────────────────────┐
│   Your 246 Ultrasound Images      │
└───────────┬─────────────────┘
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
MACHINE LEARNING  DEEP LEARNING
(Radiomics)      (Neural Network)
    │               │
    ▼               ▼
  ┌──────┬──────┐
  │Radiomics│CNN Model│
  │Features │         │
  └──────┴──────┘
    │
    ▼
Classification Results
```

---

## Quick Start (5 Steps)

### Step 1: Prepare Your Data
```powershell
# Copy your 246 TIF images here:
mkdir -p data\ultrasound_images

# Then run setup:
python setup_ultrasound_data.py
```

### Step 2: Test Your Data
```powershell
python test_ultrasound_data.py
```
Creates visualization of sample images to verify data quality.

### Step 3: Extract Machine Learning Features (Radiomics)
```powershell
python extract_ultrasound_radiomics.py
```
Generates `ultrasound_radiomics_features.csv` with 100+ statistical features.

**What you get**: Each image analyzed for texture, shape, intensity patterns.

### Step 4: Setup Deep Learning Environment
```powershell
python setup_dl_environment.py
```
Installs TensorFlow and deep learning libraries (takes 5-10 minutes first time).

### Step 5: Train & Predict with Deep Learning

#### 5a. Prepare training data
```powershell
python prepare_training_data.py
```
Creates: `output/labels_template.csv`

**YOU NEED TO:** Fill in the `label` column:
- `0` = healthy muscle
- `1` = diseased/abnormal muscle

Save as `output/labels.csv`

#### 5b. Train classifier
```powershell
python train_ultrasound_classifier.py
```

Uses **transfer learning** with EfficientNet (pre-trained on millions of images).

- Training time: 5-15 minutes on GPU, 30-60 minutes on CPU
- Saves: `output/ultrasound_classifier.keras`

#### 5c. Get predictions
```powershell
python predict_ultrasound.py
```

Generates: `output/ultrasound_predictions.csv`
- Healthy/diseased classification
- Confidence scores (0-100%)

---

## Understanding the Approach

### Machine Learning (Radiomics) Phase
```
Image → Extract Features → CSV Table → Analyze
```

**What it does:**
- 100+ features per image (texture, shape, intensity)
- Fast to compute (seconds per image)
- Human-interpretable features
- Good for understanding WHAT the AI sees

**Output:** `ultrasound_radiomics_features.csv`

**When to use:**
- Small datasets (<100 images)
- Need explainability
- Want to combine with MRI features

---

### Deep Learning Phase
```
Image → Neural Network → Prediction → CSV Results
```

**What it does:**
- Learns patterns directly from pixels
- Uses pre-trained model (already learned from millions of images)
- Fine-tunes on YOUR data
- Makes classification predictions

**Output:** `ultrasound_predictions.csv`

**When to use:**
- Have labeled data
- Want higher accuracy
- Can spare 30-60 minutes for training

---

## Your Data Split (246 Images)

After running `prepare_training_data.py`:

```
246 Total Images
│
├─ Training: 172 images (70%)        ← Model learns from these
├─ Validation: 37 images (15%)       ← Model evaluated on these
└─ Test: 37 images (15%)             ← Final validation (reserved)
```

---

## Files Created & What They Mean

### Phase 1: Machine Learning (Radiomics)
```
setup_ultrasound_data.py          → Sets up directory structure
test_ultrasound_data.py           → Verifies image quality
extract_ultrasound_radiomics.py   → Extracts 100+ features
```

**Outputs:**
- `output/ultrasound_radiomics_features.csv` (246 rows × 107 columns)

### Phase 2: Deep Learning (Classification)
```
setup_dl_environment.py           → Installs TensorFlow
prepare_training_data.py          → Creates train/val/test split
train_ultrasound_classifier.py    → Trains neural network
predict_ultrasound.py             → Makes predictions
```

**Outputs:**
- `output/labels.csv` (you fill this in)
- `output/ultrasound_classifier.keras` (trained model)
- `output/training_history.png` (learning curves)
- `output/ultrasound_predictions.csv` (predictions)

---

## Example Results

### After Radiomics Extraction
```
image_name,original_shape_Elongation,original_shape_Volume,original_firstorder_Mean,...
img_00001.tif,0.842,15284.3,28.4,...
img_00002.tif,0.756,14892.1,31.2,...
```

### After Classification
```
image_name,model_score,predicted_class,predicted_label,confidence
img_00001.tif,0.87,1,diseased,87%
img_00002.tif,0.23,0,healthy,77%
img_00003.tif,0.45,0,healthy,55%
```

---

## Comparing ML vs DL Results

After both phases complete:

```powershell
python combine_radiomics_features.py  # Show feature comparison
```

Then analyze both CSV files in Excel or Python:
- Compare which ultrasound features matter (ML)
- Compare deep learning predictions (DL)
- Ensemble both for best accuracy

---

## Troubleshooting

### "No TIF images found"
- Check images are in `data/ultrasound_images/`
- Verify correct file extension (`.tif`, `.tiff`, `.TIF`)

### "Model not found" when predicting
- Run training first: `python train_ultrasound_classifier.py`
- Wait for "✅ TRAINING COMPLETE" message

### "Out of memory" during training
- Reduce batch size in `train_ultrasound_classifier.py`
- Change: `BATCH_SIZE = 32` → `BATCH_SIZE = 16`

### "No labels found"
- Run: `python prepare_training_data.py`
- Fill out: `output/labels.csv`

---

## Combining with MRI Radiomics

To compare MRI and ultrasound:

```powershell
python combine_radiomics_features.py
```

Creates: `output/combined_radiomics_features.csv`

Now you have:
- MRI radiomics (existing)
- Ultrasound radiomics (new)
- Side-by-side comparison

---

## Next: Full Ensemble Pipeline

After both MRI and ultrasound are working:

```
MRI Radiomics + MRI Deep Learning + Ultrasound Radiomics + Ultrasound Deep Learning
                              ↓
                        Ensemble Voting
                              ↓
                    Final Classification
```

Most confident classification wins!

---

## Timeline Estimate

| Step | Time | Hardware |
|------|------|----------|
| Data setup | 5 min | Any |
| Radiomics extraction | 10-20 min | Any |
| DL environment install | 10 min | Any |
| Training data prep | 5 min | Any |
| Model training | 30-60 min | GPU faster, CPU works |
| Predictions | 1-2 min | Any |
| **Total** | **~70-140 min** | Parallel where possible |

**GPU speeds up training by 10-20x!** Consider using Google Colab (free GPU) if your computer is slow.

---

## Key Differences: Your ML vs DL

| Aspect | Radiomics (ML) | Deep Learning (DL) |
|--------|---|---|
| **Time to extract features** | Once, then instant | Instant at prediction time |
| **Explainability** | High (can see which features matter) | Low (black box) |
| **Accuracy** | Good, needs domain knowledge | Better, learns automatically |
| **Data needed** | 50+ images | 200+ images (you have 246!) |
| **Your situation** | Best for comparing MRI+Ultrasound | Best for individual classification |
| **Ensemble together** | Yes! Use both for best results | Yes! Vote on final decision |

---

## Start Here
```powershell
# 1. Move your 246 tiff images to data/ultrasound_images/
# 2. Run this:
python setup_ultrasound_data.py
python test_ultrasound_data.py
python extract_ultrasound_radiomics.py

# Then fill out output/labels.csv with 0 or 1
# Then:
python setup_dl_environment.py
python prepare_training_data.py
python train_ultrasound_classifier.py
python predict_ultrasound.py
```

That's it! You now have ML + DL on ultrasound!
