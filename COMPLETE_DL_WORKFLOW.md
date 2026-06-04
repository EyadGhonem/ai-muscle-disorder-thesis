# COMPLETE DEEP LEARNING WORKFLOW EXPLANATION
## MRI and Ultrasound Deep Learning for Muscle Health Classification

### OVERVIEW

This document explains the complete step-by-step process of how we use **deep learning** (neural networks) to determine if muscle tissue is healthy or diseased. Unlike machine learning which uses extracted numbers (radiomics features), deep learning works directly with the raw medical images. The neural network learns patterns automatically from pixel data.

We use two types of medical images:
- **MRI:** 3D volumes (like a 3D cube of muscle data)
- **Ultrasound:** 2D images (flat pictures of muscle tissue)

The process is similar for both but with key architectural differences because MRI is 3D and ultrasound is 2D.

---

## MRI DEEP LEARNING PIPELINE

### STEP 1: IMAGE ACQUISITION
**Where we get the data:** `data/mri/raw/images/`

**What we have:** MRI scans stored as NIfTI files (.nii.gz format)
- Each subject has thigh and calf 3D scans
- Each scan is a complete 3D cube of muscle tissue
- Each scan has segmentation masks (optional, but useful for masking)
- Typical MRI size: ~200×200×200 voxels (can be much larger)

**Example MRI file:** `subject001_thigh.nii.gz`
- Physical file size: 50-100 MB
- Dimensions: 200 × 200 × 150 voxels  
- Data type: Integer (intensity values 0-32767)
- Represents: 3D volume with X, Y, Z coordinates

**Why MRI for DL:** 
- Contains rich spatial information that neural networks can learn
- 3D structure preserves anatomical context
- Small dataset? No problem - transfer learning or data augmentation helps

---

### STEP 2: LABEL LOADING
**Source:** `output/mri_labels.csv` (or generate using `clinical_validation.py`)

**What labels contain:**
```csv
image_name,label
subject001_thigh.nii.gz,0
subject002_thigh.nii.gz,1
subject001_calf.nii.gz,0
subject003_calf.nii.gz,1
...
```

**Label meanings:**
- **0 = Healthy:** Normal muscle tissue with no signs of disease
- **1 = Diseased:** Muscle tissue showing pathological changes

**Key requirement:** Balanced labels (roughly 50% healthy, 50% diseased) for best deep learning performance.

---

### STEP 3: IMAGE PREPROCESSING
**Script:** `train_mri_classifier.py` (function: `load_mri_image()` and `resize_3d_image()`)

**Step 3a: Load 3D Volume**
```python
import SimpleITK as sitk

# Load .nii.gz file
img = sitk.ReadImage("subject001_thigh.nii.gz")
img_array = sitk.GetArrayFromImage(img)

# Result: 3D numpy array
# Shape example: (150, 200, 200)
# Data type: float32
# Values: 0-32767 (intensity values)
```

**What happens:**
- Read the NIfTI file containing 3D voxel data
- Convert to numpy array for Python processing
- Preserve all 3D spatial information

**Step 3b: Intensity Normalization**
```python
# Normalize intensity to 0-1 range
img_normalized = (img_array - img_array.min()) / (img_array.max() - img_array.min())

# Before: Values range from 0-32767
# After:  Values range from 0-1.0
# Why:    Neural networks learn better with normalized inputs
```

**What this does:**
- Scales all pixel values to standard 0-1 range
- Makes training more stable
- Helps neural network learn faster
- Accounts for MRI intensity variations between scans

**Example normalization:**
```
Original image intensity: 0 to 32767
Min value in image: 50
Max value in image: 30000
Range: 29950

For a pixel with value 15000:
Normalized = (15000 - 50) / 29950 = 0.499
```

---

### STEP 4: IMAGE RESIZING TO STANDARD SIZE
**Step 4a: Why Resize?**

Problem: MRI images come in different sizes
- Subject A: 200×200×150 voxels
- Subject B: 210×195×160 voxels  
- Subject C: 190×205×145 voxels

Neural networks require fixed input sizes. We resize all to standard size.

**Step 4b: Resize Process**
```python
from scipy import ndimage

# Target size for all images
TARGET_SIZE = (64, 64, 64)

# Current image size: (150, 200, 200)
# Calculate zoom factors for each dimension
zoom_factors = [64/150, 64/200, 64/200]  # [0.427, 0.32, 0.32]

# Apply zoom using interpolation
img_resized = ndimage.zoom(img_array, zoom_factors, order=1)

# Result: (64, 64, 64) voxels
```

**What this does:**
- Downsamples large images (saves memory and computation)
- Upsamples small images (maintains important details)
- Uses bilinear interpolation (smooth resizing, preserves information)
- All images now uniform size for neural network input

**Why 64×64×64?**
```
Memory calculation:
- One 3D image: 64 × 64 × 64 × 4 bytes (float32) = 1 MB
- Batch of 4 images: 4 MB
- Batch of 32 images: 32 MB

Larger sizes (128×128×128):
- One image: 8 MB
- Batch of 4: 32 MB (requires larger GPU)

Trade-off: 64×64×64 is small enough to fit in memory while preserving spatial information.
```

**Example resizing:**
```
Original: 200×200×150
Target:   64×64×64

Volume calculation:
Original volume: 200 × 200 × 150 = 6,000,000 voxels
Resized volume:  64 × 64 × 64 = 262,144 voxels
Reduction: ~95.6% (huge memory savings!)

But all important information is preserved through interpolation.
```

---

### STEP 5: ADD CHANNEL DIMENSION
**What is a channel?**

Grayscale images have 1 channel (intensity only)
Color images have 3 channels (Red, Green, Blue)
MRI has 1 channel (we're treating it as grayscale)

```python
# After resizing: Shape (64, 64, 64)
# Add channel dimension for neural network input
img_with_channel = img_resized[..., np.newaxis]
# New shape: (64, 64, 64, 1)
```

**What this means:**
- Neural networks expect (Depth, Height, Width, Channels) format
- We're adding 1 channel to the end
- Result: (64, 64, 64, 1) - ready for 3D CNN

---

### STEP 6: DATA SPLITTING
**Script:** `train_mri_classifier.py` (function: `prepare_mri_dataset()`)

**Why split?**
- **Training set:** Used to train the neural network (70%)
- **Validation set:** Used to check performance during training (15%)
- **Test set:** Used to evaluate final model performance (15%)

Keep sets separate so model doesn't memorize test data.

**How splitting works:**
```python
from sklearn.model_selection import train_test_split

# Assume we have 120 total images
# All images preprocessed: Shape (120, 64, 64, 64, 1)
# All labels loaded: Shape (120,), values 0 or 1

# First split: 70% train, 30% temp (val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# Result: 84 train, 36 temp

# Second split: Split temp 50-50 into val and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
# Result: 18 val, 18 test

# Final: 84 train, 18 val, 18 test (total 120)
```

**COMPLETE INPUT FOR DL TRAINING:**

**X_train (Training Images):**
- **Shape:** 84 samples × 64 × 64 × 64 × 1 channel
- **Data type:** Float32
- **Value range:** 0.0 to 1.0 (normalized)
- **Example shape:** (84, 64, 64, 64, 1)
- **Memory size:** 84 × 262,144 voxels × 4 bytes = 88 MB
- **Each image:** 3D cube of normalized intensity values
- **Missing data:** Handled by interpolation during resizing

**y_train (Training Labels):**
- **Shape:** 84 values
- **Data type:** Float32
- **Values:** 0 (healthy) or 1 (diseased)
- **Distribution:** ~42 healthy (0), ~42 diseased (1) - balanced!
- **Example:** [0, 1, 1, 0, 1, 0, 0, 1, ...]

**X_val (Validation Images):**
- **Shape:** 18 samples × 64 × 64 × 64 × 1 channel
- **Purpose:** Check model performance during training
- **Used:** Every epoch, model is tested on this data
- **Not used:** Model does NOT learn from validation data

**y_val (Validation Labels):**
- **Shape:** 18 values
- **Purpose:** Score validation images during training
- **Example:** [1, 0, 1, 1, 0, 1, ...]

**Summary Table:**
```
Dataset     Count    Healthy    Diseased    % Diseased    Memory
─────────────────────────────────────────────────────────────────
Train         84        ~42         ~42         50%        88 MB
Validation    18        ~9          ~9          50%        19 MB
Test          18        ~9          ~9          50%        19 MB
─────────────────────────────────────────────────────────────────
TOTAL        120        ~60         ~60         50%       126 MB
```

---

## STEP 7: 3D CNN MODEL ARCHITECTURE
**Script:** `train_mri_classifier.py` (function: `create_3d_model()`)

**What is a 3D CNN?**

CNN = Convolutional Neural Network
- Learns patterns from images
- Uses convolution filters (small pattern detectors)
- 3D version works on 3D volumes (not flat images)

**Simple analogy:**
- Imagine a robot learning to identify healthy vs diseased muscle
- It scans the 3D image with small 3×3×3 pattern detectors
- Each detector looks for specific features (texture, edges, intensity changes)
- The robot combines all pattern detections to make final diagnosis

**Model Architecture:**

```
INPUT: (64, 64, 64, 1) - One preprocessed 3D MRI image
  ↓
BLOCK 1: 3D Convolution + BatchNorm + MaxPool + Dropout
  - Learns low-level features (edges, basic textures)
  - 32 filters (32 different pattern detectors)
  - Output: (32, 32, 32, 32)
  ↓
BLOCK 2: 3D Convolution + BatchNorm + MaxPool + Dropout
  - Learns mid-level features (combinations of edges)
  - 64 filters (64 different pattern detectors)
  - Output: (16, 16, 16, 64)
  ↓
BLOCK 3: 3D Convolution + BatchNorm + MaxPool + Dropout
  - Learns high-level features (anatomical structures)
  - 128 filters (128 different pattern detectors)
  - Output: (8, 8, 8, 128)
  ↓
FLATTEN: Convert 3D output to 1D vector
  - All values flattened: 65,536 values
  - Like reading out all pattern detection results
  ↓
DENSE LAYER 1: 256 neurons + Dropout
  - Like a decision-making layer
  - Makes connections between features
  ↓
DENSE LAYER 2: 128 neurons + Dropout
  - Further processes the feature combinations
  ↓
OUTPUT: 1 neuron + Sigmoid activation
  - Outputs single value between 0 and 1
  - 0 = Healthy, 1 = Diseased
  - Example: 0.78 = "78% confident it's diseased"
```

**Key Components:**

1. **Convolution Layers (Conv3D):**
   ```
   - Slide 3×3×3 filter across the image
   - Calculate average within each 3×3×3 region
   - Detect patterns like texture, edges, intensity changes
   - Each layer has multiple filters (32, 64, 128)
   ```

2. **Batch Normalization:**
   ```
   - Normalize layer outputs to mean 0, std 1
   - Makes training more stable and faster
   - Reduces sensitivity to initialization
   ```

3. **MaxPooling:**
   ```
   - Reduce image dimensions by 50% in each direction
   - Keep the maximum value in each region
   - Highlights most important features
   - Reduces computation in next layer
   ```

4. **Dropout:**
   ```
   - Randomly ignore some neurons during training (30%, 50%)
   - Forces redundancy (multiple pathways to same answer)
   - Prevents overfitting (memorizing training data)
   - Makes model more robust
   ```

**Total Parameters:** ~4.2 million

---

## STEP 8: MODEL COMPILATION
**Script:** `train_mri_classifier.py` (function: `create_3d_model()`)

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC()]
)
```

**What "compile" means:**
- Configure how model should learn
- Choose training algorithm
- Choose error measurement
- Choose performance metrics

**Components:**

1. **Optimizer (Adam):**
   - Algorithm that updates neural network weights
   - Adam = Adaptive Moment Estimation
   - Automatically adjusts learning rate for each parameter
   - Learning rate 0.001 = step size for weight updates
   - Smaller = slower but steadier learning
   - Larger = faster but might overshoot

2. **Loss Function (Binary Crossentropy):**
   - Measures how wrong the model is
   - Binary = 2 classes (healthy/diseased)
   - Crossentropy = standard for classification
   - Training goal: Minimize this value
   - Perfect prediction: Loss = 0
   - Wrong prediction: Loss = high value

3. **Metrics (Accuracy, AUC):**
   - **Accuracy:** % correct predictions (49/50 = 98%)
   - **AUC:** Area Under Curve, measures discrimination ability
   - AUC range: 0-1, where 1.0 is perfect

---

## STEP 9: MODEL TRAINING
**Script:** `train_mri_classifier.py` (function: `train_model()`)

**What is training?**
Neural network learns by making predictions, calculating errors, and adjusting weights.

**Training Process:**

```
Epoch 1:
  ├─ Batch 1: 4 images → predictions [0.45, 0.78, 0.12, 0.91]
  │            actual labels      [0,    1,    0,    1  ]
  │            loss = calculate error
  │            update weights based on error
  ├─ Batch 2: 4 images → update weights
  ├─ Batch 3: 4 images → update weights
  └─ [21 total batches for 84 training images]
  
  After Epoch 1:
  ├─ Training accuracy: 75%
  ├─ Validation accuracy: 72%
  └─ Validation loss: 0.48

Epoch 2:
  ├─ Batch 1: weights updated again
  ├─ Training accuracy: 82%
  ├─ Validation accuracy: 78%
  └─ Validation loss: 0.35

Epoch 3-20: Continue improving...
```

**Configuration:**

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=4,
    callbacks=[early_stopping, reduce_lr]
)
```

**Key settings for MRI DL:**

| Setting | Value | Why |
|---------|-------|-----|
| EPOCHS | 20 | Enough to learn, not so much to overfit |
| BATCH_SIZE | 4 | Small because 3D images are memory-intensive |
| Learning Rate | 0.001 | Slow but steady learning for stable training |
| Early Stopping | patience=5 | Stop if validation loss doesn't improve for 5 epochs |
| Reduce LR | factor=0.5 | Cut learning rate in half if stuck |

**Example Training Output:**
```
Epoch 1/20
21/21 [==============================] - 4s - loss: 0.6920 - accuracy: 0.5238 - val_loss: 0.6845 - val_accuracy: 0.5556
Epoch 2/20
21/21 [==============================] - 4s - loss: 0.6510 - accuracy: 0.6190 - val_loss: 0.5892 - val_accuracy: 0.7222
Epoch 3/20
21/21 [==============================] - 4s - loss: 0.5234 - accuracy: 0.7143 - val_loss: 0.4127 - val_accuracy: 0.8333
Epoch 4/20
21/21 [==============================] - 4s - loss: 0.3874 - accuracy: 0.8095 - val_loss: 0.3001 - val_accuracy: 0.8889
...
Epoch 17/20
21/21 [==============================] - 4s - loss: 0.1287 - accuracy: 0.9524 - val_loss: 0.2456 - val_accuracy: 0.9444
Early stopping triggered - val_loss hasn't improved for 5 epochs
```

**Understanding Training Metrics:**
- **Accuracy 0.9524** = 95.24% of training images classified correctly
- **Loss 0.1287** = Average error per image (lower is better)
- **Val Accuracy 0.9444** = 94.44% validation accuracy (good generalization!)
- **Val Loss 0.2456** = Validation error (slightly higher than training, which is normal)

---

## STEP 10: MODEL EVALUATION ON TEST SET
**What is the test set?**
- 18 images the model has NEVER seen
- True test of how well model generalizes
- Used only ONCE for final evaluation

**Evaluation Metrics:**

```python
# After training, evaluate on completely unseen test data
test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test)

# Example results:
test_accuracy = 0.8889  # 88.89% (16 out of 18 correct)
test_auc = 0.9167      # 0.9167 out of 1.0 (excellent discrimination)
test_loss = 0.3421     # Error measure
```

**Understanding Results:**
- **Test Accuracy 88.89%:** Model correctly predicts 16 out of 18 test images
- **Test AUC 0.9167:** Excellent ability to distinguish healthy from diseased
- **Comparison to training:** Similar to validation, shows no overfitting

---

## STEP 11: MAKING PREDICTIONS
**Script:** `predict_mri.py`

**How predictions work:**

```python
# New image arrives (not in training set)
new_image_path = "data/mri/raw/images/subject_new_thigh.nii.gz"

# Preprocess (same as training)
img = load_mri_image(new_image_path)          # Load
img = (img - img.min()) / (img.max() - img.min())  # Normalize
img = resize_3d_image(img, (64, 64, 64))     # Resize
img = img[..., np.newaxis]                     # Add channel

# Make prediction
prediction = model.predict(img[np.newaxis, :])  # Add batch dimension
# Result: [[0.23]]  (value between 0 and 1)

# Convert to class and confidence
score = prediction[0][0]  # 0.23
predicted_class = 1 if score >= 0.5 else 0  # 0 (healthy)
confidence = max(score, 1 - score) * 100    # 77% confident
predicted_label = "healthy"                   # Interpretation

# Result:
{
    'image_name': 'subject_new_thigh.nii.gz',
    'model_score': 0.23,
    'predicted_class': 0,
    'predicted_label': 'healthy',
    'confidence': 77.0  # 77% confident it's healthy
}
```

**Interpretation Guide:**

| Score | Class | Label | Confidence |
|-------|-------|-------|-----------|
| 0.05  | 0 | Healthy | 95% |
| 0.25  | 0 | Healthy | 75% |
| 0.48  | 0 | Healthy | 52% (uncertain) |
| 0.51  | 1 | Diseased | 51% (uncertain) |
| 0.75  | 1 | Diseased | 75% |
| 0.95  | 1 | Diseased | 95% |

---

## DATA SPECIFICATIONS SUMMARY

### MRI Deep Learning Workflow Data Specifications

**Raw Input:**
- Format: NIfTI (.nii.gz)
- Type: 3D medical imaging volume
- Typical size: 50-100 MB per image

**Processed Input to Model:**
- Format: Numpy array (float32)
- Shape: (64, 64, 64, 1)
- Value range: 0.0 to 1.0 (normalized)
- Size: ~1 MB per image

**Training Data:**
- Training images (X_train): 84 samples × 64 × 64 × 64 × 1
- Training labels (y_train): 84 binary values (0 or 1)
- Validation images (X_val): 18 samples
- Validation labels (y_val): 18 binary values
- Total size: ~126 MB

**Model Output:**
- Shape: 1 value per image
- Range: 0.0 to 1.0
- Interpretation: Probability of being diseased
- Standard threshold: 0.5 (≥0.5 = diseased, <0.5 = healthy)

---

## ULTRASOUND DEEP LEARNING PIPELINE

Deep learning for ultrasound is very similar but uses 2D images instead of 3D:

### Key Differences:

**1. Images are 2D instead of 3D:**
```
MRI:         (200, 200, 150)  → resize to (64, 64, 64, 1)
Ultrasound:  (512, 512)       → resize to (256, 256, 1)
```

**2. CNN Architecture is 2D instead of 3D:**
```
MRI Deep Learning:            Ultrasound Deep Learning:
Conv3D (3D convolutions)       Conv2D (2D convolutions)
MaxPooling3D                   MaxPooling2D
3×3×3 filters                  3×3 filters
(Depth, Height, Width, C)      (Height, Width, C)
```

**3. Model is simpler but deeper:**
- Fewer parameters needed (2D inherently simpler than 3D)
- Can use deeper networks (more layers)
- Faster training and inference

**4. Batch size can be larger:**
```
MRI:        batch_size = 4  (because 3D images are huge)
Ultrasound: batch_size = 16-32  (2D images are smaller)
```

**5. Input size is larger:**
```
MRI:        (64, 64, 64) = 262K voxels per image
Ultrasound: (256, 256) = 65K pixels per image
But ultrasound can afford larger 2D images
```

---

## COMPARISON: ML vs DL WORKFLOWS

### Input Data Representation:

**Machine Learning (Radiomics):**
```
MRI Image (3D, 50-100 MB)
              ↓
Apply Mask (isolate muscle)
              ↓
Extract 100+ Features (10 KB CSV)
              ↓
Input to Model: [1250.5, 127.3, 1.25, ..., 0.67]
              ↓
Output: 0 (healthy) or 1 (diseased)
```

**Deep Learning:**
```
MRI Image (3D, 50-100 MB)
              ↓
Normalize & Resize (1 MB)
              ↓
Input to Model: (64, 64, 64, 1) 3D tensor
              ↓
Convolutional Layers Learn Features Automatically
              ↓
Output: 0.78 (probability)
```

### Key Advantages of Each:

**Machine Learning (Radiomics):**
- ✓ Interpretable features (radiologists understand "volume", "texture")
- ✓ Small dataset OK (100 features, just need 50-100 samples)
- ✓ Small feature CSV (10 KB per scan)
- ✓ Fast training (seconds)
- ✗ Hand-crafted features may miss patterns
- ✗ Requires manual mask creation

**Deep Learning:**
- ✓ Learns features automatically
- ✓ Can discover complex patterns
- ✓ No manual mask needed
- ✓ Can use data augmentation (flip, rotate)
- ✗ Needs more data (100+ samples ideally)
- ✗ Black box (hard to interpret)
- ✗ Requires GPU for speed
- ✗ Larger model files (50-100 MB)

---

## COMPLETE EXAMPLE: FROM RAW IMAGE TO PREDICTION

### Step-by-Step Example with Real Numbers:

**Initial MRI File:**
```
File: subject042_calf.nii.gz
Location: data/mri/raw/images/subject042_calf.nii.gz
File size: 75 MB
Voxel dimensions: 0.5mm × 0.5mm × 1.0mm
Voxel matrix: 250 × 210 × 180 voxels
```

**After Normalization:**
```
Value range: 0-32767  →  0-1.0
Size: Still 75 MB (same raw data)
Min voxel value: 0 (background)
Max voxel value: 32767 (bright tissue)
Normalized:
  - 0 → 0.0
  - 16383 → 0.5
  - 32767 → 1.0
```

**After Resizing:**
```
Original shape: (180, 210, 250)
Target shape: (64, 64, 64)
Zoom factors: [0.356, 0.305, 0.256]
Result: Smooth downsampling preserving key features
File size reduction: 75 MB → 1 MB (99% reduction!)
```

**After Adding Channel:**
```
Shape before: (64, 64, 64)
Shape after: (64, 64, 64, 1)
Ready for neural network input
```

**Through Neural Network:**
```
Input: (64, 64, 64, 1) tensor of floats
↓
Conv3D layer 1: Detects edges, intensity changes → (32, 32, 32, 32)
↓
Conv3D layer 2: Combines edge patterns → (16, 16, 16, 64)
↓
Conv3D layer 3: High-level anatomy features → (8, 8, 8, 128)
↓
Flatten: 8×8×8×128 = 65,536 values
↓
Dense 256: Complex decision logic
↓
Dense 128: Final reasoning
↓
Output sigmoid: Single value 0-1
```

**Raw Output:**
```
Neural network output: [0.78]
This is a probability (78% confident)
```

**Final Prediction:**
```
Score: 0.78
Threshold: 0.5
Decision: 0.78 ≥ 0.5 → Predicted class = 1 (diseased)
Confidence: max(0.78, 1-0.78) × 100 = 78%
Label: "diseased with 78% confidence"

CSV Output:
image_name,model_score,predicted_class,predicted_label,confidence
subject042_calf.nii.gz,0.78,1,diseased,78.0
```

---

## TROUBLESHOOTING COMMON ISSUES

### "Model accuracy is 50%" (Random guessing)
- **Cause:** Model not learning anything
- **Fixes:**
  - Check labels are correct (50% healthy, 50% diseased)
  - Increase training epochs
  - Use smaller learning rate (0.0001 instead of 0.001)
  - Reduce dropout rates

### "Training loss decreases but validation loss increases" (Overfitting)
- **Cause:** Model memorizing training data instead of learning patterns
- **Fixes:**
  - Increase dropout (0.5 instead of 0.3)
  - Reduce model size (fewer filters)
  - Add L2 regularization
  - Collect more training data

### "Out of memory error"
- **Cause:** Batch size too large or images too big
- **Fixes:**
  - Reduce batch_size (4 instead of 8)
  - Reduce image size (32×32×32 instead of 64×64×64)
  - Use gradient accumulation

### "Model predictions are all 0 or all 1"
- **Cause:** Model overconfident or data imbalanced
- **Fixes:**
  - Check label distribution (should be ~50-50)
  - Adjust decision threshold (0.4 instead of 0.5)
  - Add class weights during training

---

## FILES GENERATED BY DEEP LEARNING PIPELINE

| File | Purpose | Size |
|------|---------|------|
| `output/mri_classifier.keras` | Trained model (all weights) | 50-100 MB |
| `output/mri_predictions.csv` | Predictions on all images | 1-5 KB |
| Training history plot | Accuracy/loss curves | Image file |

---

## READY TO RUN

**To train MRI deep learning classifier:**
```bash
python train_mri_classifier.py
```

**To make predictions with trained model:**
```bash
python predict_mri.py
```

**To train ultrasound deep learning classifier:**
```bash
python train_ultrasound_classifier.py
```

**To make predictions with trained ultrasound model:**
```bash
python predict_ultrasound.py
```

---

## SUMMARY

**What Deep Learning Does:**
- Takes raw 3D MRI images or 2D ultrasound images
- Automatically learns to distinguish healthy from diseased tissue
- Makes probability-based predictions on new images
- No manual feature engineering needed

**Why It's Powerful:**
- Doesn't require radiomics expertise
- Can discover hidden patterns humans might miss
- Works with raw pixel data directly
- Can be improved with more data and training

**When to Use:**
- When you have medium-sized dataset (100+ images)
- When you have GPU available for training
- When interpretability is less important than accuracy
- When you want to use the actual image structure

This workflow automatically handles preprocessing, training, validation, and prediction for complete end-to-end deep learning classification!
