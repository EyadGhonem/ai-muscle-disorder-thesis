# Expanded Project: MRI + Ultrasound with ML & DL

## Project Evolution

### Current State (MRI Only)
```
MRI Images → Radiomics Features → CSV Output
```

### New Integrated Approach (MRI + Ultrasound)
```
┌─────────────────────────────┐
│   Image Data                │
├─────────────┬───────────────┤
│   MRI       │   Ultrasound  │
├─────────────┼───────────────┤
│ Radiomics   │ Radiomics     │  ← Machine Learning (statistical features)
│ Features    │ Features      │
└─────────────┴───────────────┘
         ↓
    Feature Analysis
         ↓
┌─────────────────────────────┐
│   Deep Learning Models      │
├─────────────┬───────────────┤
│   MRI CNN   │ Ultrasound CNN│  ← Deep Learning (neural networks)
│             │               │
│ Classification trained on extracted features pattern
└─────────────┴───────────────┘
         ↓
    Predictions & Classification
```

## Detailed Pipeline

### **Part 1: Machine Learning - Radiomics on Both Modalities**

#### A) Ultrasound Radiomics (Mirror your MRI setup)
```bash
python extract_radiomics_ultrasound.py
```
- Extracts 100+ radiomics features from ultrasound images
- Requires:
  - Ultrasound images in `data/ultrasound/`
  - Binary masks in `data/ultrasound_masks/`
  - Format: NIfTI (.nii.gz) - may need conversion from DICOM (hospital format)
- Output: `output/ultrasound_features.csv`

#### B) Combine MRI + Ultrasound Features
```bash
python combine_radiomics_features.py
```
- Merges MRI and ultrasound radiomics outputs
- Enables side-by-side comparison
- Output: `output/combined_radiomics_features.csv`

---

### **Part 2: Deep Learning - Classification Models**

#### C) Ultrasound Deep Learning Classifier
```bash
python train_ultrasound_classifier.py
```
- **Model**: Pre-trained CNN (ResNet-50 or EfficientNet)
- **Task**: Classify muscle health/disease status
- **Input**: Ultrasound images directly (2D or 3D slices)
- **Output**: 
  - `output/ultrasound_classifier_model.keras`
  - `output/ultrasound_predictions.csv` (probabilities)

#### D) MRI Deep Learning Classifier
```bash
python train_mri_classifier.py
```
- **Model**: Pre-trained 3D CNN (suitable for volumetric data)
- **Task**: Classify muscle condition from full 3D volume
- **Input**: Full 3D MRI images
- **Output**:
  - `output/mri_classifier_model.keras`
  - `output/mri_predictions.csv` (probabilities)

#### E) Ensemble Prediction (Combine All Models)
```bash
python ensemble_predictions.py
```
- Combines results from:
  - MRI radiomics ML model
  - Ultrasound radiomics ML model
  - MRI deep learning model
  - Ultrasound deep learning model
- **Output**: `output/ensemble_predictions.csv` (most confident prediction)

---

## Project Structure (Expanded)

```
thesis_project/
│
├── data/
│   ├── images/                    # MRI images (existing)
│   ├── masks/                     # MRI masks (existing)
│   ├── images_small/              # Downsampled MRI (existing)
│   │
│   ├── ultrasound/                # NEW: Ultrasound 2D images
│   ├── ultrasound_masks/          # NEW: Ultrasound segmentation masks
│   ├── ultrasound_3d/             # NEW: Ultrasound converted to 3D (optional)
│   │
│   └── labels.csv                 # NEW: Disease labels/classifications
│
├── output/
│   ├── mri_features.csv           # Existing radiomics
│   ├── ultrasound_features.csv    # NEW: Radiomics
│   ├── combined_features.csv      # NEW: Combined analysis
│   │
│   ├── mri_classifier_model.keras         # NEW: Trained model
│   ├── ultrasound_classifier_model.keras  # NEW: Trained model
│   │
│   ├── mri_predictions.csv        # NEW: DL predictions
│   ├── ultrasound_predictions.csv # NEW: DL predictions
│   └── ensemble_predictions.csv   # NEW: Final ensemble
│
├── scripts/
│   ├── extract_mri_radiomics.py           # Refactored from existing
│   ├── extract_ultrasound_radiomics.py    # NEW
│   ├── combine_radiomics_features.py      # NEW
│   │
│   ├── train_mri_classifier.py            # NEW
│   ├── train_ultrasound_classifier.py     # NEW
│   ├── predict_mri.py                     # NEW
│   ├── predict_ultrasound.py              # NEW
│   ├── ensemble_predictions.py            # NEW
│   │
│   └── utils/
│       ├── image_processing.py            # Shared functions
│       ├── model_utils.py                 # NEW: Training helpers
│       └── ensemble_utils.py              # NEW: Voting/averaging
│
├── radiomics_env/                 # Existing Python environment
│
├── dl_env/                        # NEW: Deep learning environment
│   ├── (TensorFlow/PyTorch + dependencies)
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb      # NEW
│   ├── 02_model_training.ipynb            # NEW
│   ├── 03_results_comparison.ipynb        # NEW
│   └── 04_feature_importance.ipynb        # NEW
│
└── docs/
    ├── RADIOMICS_PIPELINE.md              # Existing
    ├── ULTRASOUND_PREPARATION.md          # NEW
    ├── DEEP_LEARNING_GUIDE.md             # NEW
    └── COMPLETE_WORKFLOW.md               # NEW
```

---

## Implementation Phases

### **Phase 1: Ultrasound Radiomics (1-2 weeks)**
- Set up ultrasound data directory structure
- Create `extract_ultrasound_radiomics.py` (copy + adapt existing MRI script)
- Extract radiomics features from ultrasound
- Combine with existing MRI radiomics

### **Phase 2: Data Preparation for Deep Learning (1 week)**
- Organize training/validation/test splits
- Convert ultrasound to consistent format (2D arrays)
- Create `labels.csv` with disease classification for each image
- Optional: Convert ultrasound to 3D volumes (using stacking technique)

### **Phase 3: Deep Learning Models (2-3 weeks)**
- Set up TensorFlow/PyTorch environment
- Create MRI classifier (using pre-trained 3D CNN like ResNet-50 3D)
- Create Ultrasound classifier (using pre-trained 2D CNN like EfficientNet)
- Train and validate both models

### **Phase 4: Ensemble & Analysis (1 week)**
- Implement ensemble voting system
- Compare predictions across methods
- Analyze which modality/method works best
- Generate comparative reports

---

## Technical Requirements

### New Dependencies

#### Deep Learning Environment
```
TensorFlow 2.13+ or PyTorch
    ├── keras (included in TF)
    ├── numpy
    ├── pandas
    └── matplotlib
    
Optional:
    ├── scikit-learn (for metrics)
    └── plotly (for visualization)
```

### Input Data Format

#### Ultrasound Images
```
Accepted Formats:
- PNG/JPG (single 2D slices) → Most common
- DICOM (.dcm) → Hospital standard (requires conversion)
- NIfTI (.nii.gz) → If already in medical format

Directory Structure:
data/ultrasound/
├── patient_001_frame_01.png
├── patient_001_frame_02.png
├── patient_002_frame_01.png
└── ...

data/ultrasound_masks/
├── patient_001_frame_01_mask.png
├── patient_001_frame_02_mask.png
└── ...
```

#### Labels File
```csv
image_name,label,disease_status
patient_001_frame_01.png,0,healthy
patient_001_frame_02.png,0,healthy
patient_002_frame_01.png,1,diseased
patient_002_frame_02.png,1,diseased
```

---

## Quick Comparison: ML vs DL

| Aspect | Radiomics (ML) | Deep Learning (DL) |
|--------|----------------|-------------------|
| **What it extracts** | 100+ hand-crafted features | Learned patterns from raw pixels |
| **Training time** | Minutes | Hours/Days |
| **Data needed** | 50+ images | 500+ images (better with more) |
| **Interpretability** | High (can see which features matter) | Low (black box) |
| **Speed** | Very fast at prediction | Slower at prediction |
| **Best for** | Few images, explainability | Many images, accuracy |
| **Your use** | Already doing on MRI | NEW: Add as second approach |

---

## Decision Tree: Which Method to Use?

```
Do you have <200 images?
├─ YES → Focus on Radiomics + small ensemble DL model
└─ NO (>200) → Full DL model training

Is model interpretability important?
├─ YES → Prioritize radiomics features
└─ NO → Use deep learning

Do you have labeled data?
├─ YES → Train supervised DL models
└─ NO → Use radiomics features + unsupervised clustering
```

---

## Next Steps

1. **Verify ultrasound data format**: DICOM, PNG, or something else?
2. **Check data availability**: How many ultrasound images? Any labels?
3. **Choose environment**: Use existing `radiomics_env` or create separate `dl_env`?
4. **Create test script**: Simple script to load and display first ultrasound image

Ready to start Phase 1 (Ultrasound Radiomics)?
