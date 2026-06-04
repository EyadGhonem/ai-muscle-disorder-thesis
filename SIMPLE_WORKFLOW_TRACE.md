# SIMPLE WORKFLOW TRACE
## MRI Machine Learning Pipeline - Step by Step

### STEP 1: GET MRI DATA
**Where:** `data/mri/raw/MRI_data/`
**What we get:** MRI files (.nii.gz format)
```
MRI_data/
├── 01/Thigh/In_phase.nii.gz
├── 01/Thigh/mask_muscles.nii.gz
├── 02/Thigh/In_phase.nii.gz
├── 02/Thigh/mask_muscles.nii.gz
└── ...
```

### STEP 2: GENERATE MASKS
**Script:** `extract_mri_radiomics.py`
**What happens:** 
- Load MRI volume
- Load segmentation mask
- Use mask to isolate muscle tissue
- **Output:** Masked muscle data

### STEP 3: EXTRACT FEATURES (RADIOMICS)
**Script:** `extract_mri_radiomics.py`
**What happens:** Extract numerical features from muscle tissue

**Simple Explanation of Radiomics:**
- **Shape Features:** How big, how round, how long the muscle is
- **Intensity Features:** Average brightness, brightness variation
- **Texture Features:** Smooth vs rough patterns, uniform vs mixed
- **Pattern Features:** Repeating patterns, edge information

**Example Features Extracted:**
```
For each MRI scan, we get numbers like:
- Volume: 1250.5 (how big the muscle is)
- Mean intensity: 127.3 (average brightness)
- Elongation: 1.25 (how long vs wide)
- Texture: 0.45 (how smooth/rough)
- Skewness: 0.15 (brightness distribution)
- Kurtosis: 2.8 (peakiness of distribution)
```

**Output file:** `output/mri_radiomics_features.csv`
```
image_name,volume,mean_intensity,elongation,texture,skewness,kurtosis
subject01_Thigh_In_phase.nii.gz,1250.5,127.3,1.25,0.45,0.15,2.8
subject02_Thigh_In_phase.nii.gz,1180.2,132.1,1.18,0.52,0.22,3.1
```

**How it works:**
1. Take MRI volume + muscle mask
2. Look only at muscle pixels (ignore everything else)
3. Calculate 100+ numerical features
4. Each feature describes one aspect of muscle health
5. Save all features in CSV file for ML training

### STEP 4: CREATE LABELS
**Script:** `clinical_validation.py`
**What happens:**
- Calculate disease score from features
- Assign labels: 0=Healthy, 1=Diseased
- Balance dataset (50% each)
- **Output file:** `output/clinical_labels.csv`
```
image_name,label
subject01_Thigh_In_phase.nii.gz,0
subject02_Thigh_In_phase.nii.gz,1
```

### STEP 5: PREPARE TRAINING DATA
**Script:** `clinical_validation.py`
**What happens:**
- Merge features with labels
- Split data: 80% training, 20% testing
- Handle missing values (fill with 0)
- **Output:** X_train, X_test, y_train, y_test

### STEP 6: TRAIN ML MODEL
**Script:** `clinical_validation.py`
**What happens:**
- Use RandomForestClassifier
- Train on training data (X_train, y_train)
- Learn patterns from features
- **Model:** Trained classifier

### STEP 7: MAKE PREDICTIONS
**Script:** `clinical_validation.py`
**What happens:**
- Test model on X_test
- Predict: Healthy (0) or Diseased (1)
- Get confidence scores
- **Output:** Predictions for all test samples

### STEP 8: EVALUATE RESULTS
**Script:** `clinical_validation.py`
**What happens:**
- Compare predictions to true labels
- Calculate: Accuracy, Precision, Recall, F1-Score
- **Output:** Performance metrics
```
Accuracy: 0.85
Precision: 0.82
Recall: 0.88
F1-Score: 0.85
```

### STEP 9: SAVE RESULTS
**Output files created:**
- `output/ml_vs_dl_comparison.csv` - ML results
- `output/ml_feature_importance.csv` - Important features
- `output/clinical_labels.csv` - Final labels

---

## ULTRASOUND MACHINE LEARNING PIPELINE - Step by Step

### STEP 1: GET ULTRASOUND IMAGES
**Where:** `data/ultrasound_images/`
**What we get:** JPEG/PNG image files
```
ultrasound_images/
├── image001.jpg
├── image002.jpg
├── image003.jpg
└── ...
```

### STEP 2: CREATE MASKS
**Script:** `extract_ultrasound_radiomics.py`
**What happens:**
- Load ultrasound image
- Convert to grayscale
- Create automatic mask using Otsu thresholding
- Clean mask with morphological operations
- **Output:** Binary mask for muscle tissue

### STEP 3: EXTRACT FEATURES (RADIOMICS)
**Script:** `extract_ultrasound_radiomics.py`
**What happens:** Extract numerical features from muscle tissue

**Simple Explanation of Radiomics:**
- **Shape Features:** How big, how round, muscle area and perimeter
- **Intensity Features:** Average brightness, brightness variation
- **Texture Features:** Smooth vs rough patterns, uniform vs mixed
- **Pattern Features:** Repeating patterns, edge information

**Example Features Extracted:**
```
For each ultrasound image, we get numbers like:
- Area: 1250.5 (how much muscle area)
- Mean intensity: 127.3 (average brightness)
- Elongation: 1.25 (how long vs wide)
- Texture: 0.45 (how smooth/rough)
- Perimeter: 180.2 (muscle boundary length)
- Compactness: 0.82 (how compact the shape is)
```

**Output file:** `output/ultrasound_radiomics_features.csv`
```
image_name,area,mean_intensity,elongation,texture,perimeter,compactness
image001.jpg,1250.5,127.3,1.25,0.45,180.2,0.82
image002.jpg,1180.2,132.1,1.18,0.52,165.8,0.78
```

**How it works:**
1. Take ultrasound image + automatic mask
2. Look only at muscle pixels (ignore everything else)
3. Calculate 100+ numerical features
4. Each feature describes one aspect of muscle health
5. Save all features in CSV file for ML training

### STEP 4: CREATE LABELS
**Script:** `prepare_training_data.py`
**What happens:**
- Create labels template
- Generate sample labels (50% healthy, 50% diseased)
- **Output file:** `output/labels.csv`
```
image_name,label
image001.jpg,0  # Healthy
image002.jpg,1  # Diseased
```

### STEP 5: PREPARE TRAINING DATA
**Script:** `simple_classifier.py`
**What happens:**
- Load features and labels
- Merge them together
- Split data: 80% training, 20% testing
- **Output:** X_train, X_test, y_train, y_test

### STEP 6: TRAIN ML MODEL
**Script:** `simple_classifier.py`
**What happens:**
- Use RandomForestClassifier
- Train on training data (X_train, y_train)
- Learn patterns from features
- **Model:** Trained classifier

### STEP 7: MAKE PREDICTIONS
**Script:** `simple_classifier.py`
**What happens:**
- Test model on X_test
- Predict: Healthy (0) or Diseased (1)
- Get confidence scores
- **Output file:** `output/simple_ultrasound_predictions.csv`
```
image_index,predicted_class,predicted_label,confidence
0,0,healthy,85.2
1,1,diseased,92.1
```

### STEP 8: EVALUATE RESULTS
**Script:** `simple_classifier.py`
**What happens:**
- Compare predictions to true labels
- Calculate: Accuracy, Precision, Recall, F1-Score
- **Output:** Performance metrics and confusion matrix

---

## SUMMARY: BOTH PIPELINES

### MRI ML Flow:
```
MRI Files → Generate Masks → Extract Features → Create Labels → Train Model → Predict → Results
```

### Ultrasound ML Flow:
```
Images → Create Masks → Extract Features → Create Labels → Train Model → Predict → Results
```

**Final Answer:** For each medical image (MRI or ultrasound), we predict whether muscle is HEALTHY or DISEASED based on radiomics features using machine learning.
