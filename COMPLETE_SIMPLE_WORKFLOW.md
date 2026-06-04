# COMPLETE WORKFLOW EXPLANATION
## MRI and Ultrasound Machine Learning for BMD/DMD Classification

### OVERVIEW

This document explains complete step-by-step process of how we use machine learning to differentiate between Becker Muscular Dystrophy (BMD) and Duchenne Muscular Dystrophy (DMD) using radiomics features from medical images. We use two types of medical images: MRI (3D volumes) and ultrasound (2D images). The process is similar for both but with some important differences.

⚠️ **IMPORTANT:** This pipeline currently uses **synthetic proxy labels** for development and testing. Real clinical classification requires confirmed BMD/DMD diagnoses from neurologists and genetic testing.

---

## MRI MACHINE LEARNING PIPELINE

### STEP 1: DATA ACQUISITION
**Where we get the data:** `data/mri/raw/MRI_data/`

**What we have:** MRI scans stored as NIfTI files (.nii.gz format)
- Each subject has thigh and calf scans
- Each scan has multiple sequences (In_phase, Opp_phase, Water, Fat)
- Each scan has corresponding segmentation masks

**Why MRI:** MRI provides detailed 3D views of muscle tissue, allowing us to see internal structure and volume.

### STEP 2: MASK APPLICATION
**Script:** `extract_mri_radiomics.py`

**How it works:** 
1. Load the 3D MRI volume (like a 3D cube of muscle data)
2. Load the corresponding segmentation mask (shows exactly where muscle tissue is)
3. Apply the mask to isolate only muscle pixels, ignoring bone, fat, and other tissues

**Why masks:** Masks tell the computer exactly which pixels contain muscle tissue, so we only analyze the relevant parts of the image.

### STEP 3: RADIOMICS FEATURE EXTRACTION
**Script:** `extract_mri_radiomics.py`

**What is radiomics:** Converting medical images into numbers that describe tissue characteristics.

**How it works:**
1. Take the masked muscle tissue (only muscle pixels)
2. Calculate mathematical features that describe the tissue:
   - **Shape features:** How big is the muscle, how round, how elongated
   - **Intensity features:** Average brightness, variation in brightness, brightness patterns
   - **Texture features:** How smooth vs rough, uniform vs mixed patterns
   - **Pattern features:** Repeating patterns, edge characteristics

**Example features extracted:**
- Volume: 1250.5 cubic millimeters (muscle size)
- Mean intensity: 127.3 (average brightness level)
- Elongation: 1.25 (how long vs wide, 1.25 means 25% longer than wide)
- Texture: 0.45 (smoothness score, 0 = very smooth, 1 = very rough)
- Skewness: 0.15 (brightness distribution, 0 = symmetric)
- Kurtosis: 2.8 (peakiness of brightness distribution)

**Output:** `output/mri_radiomics_features.csv` - One row per MRI scan with 100+ feature columns

### STEP 4: PROXY LABEL GENERATION
**Script:** `clinical_validation.py` 

**How it works:**
1. Take all extracted radiomics features for each MRI scan
2. Calculate a "severity score" by combining multiple features
3. Assign proxy labels: 0 = BMD (milder form), 1 = DMD (more severe form)
4. Ensure balanced dataset (50% BMD, 50% DMD)

**⚠️ CRITICAL WARNING:** These are **synthetic proxy labels** for pipeline testing only!
- **NOT real clinical diagnoses**
- **Based on radiomics patterns, not medical assessment**
- **Real classification requires:**
  - Genetic confirmation of dystrophin mutations
  - Clinical evaluation by neuromuscular specialists
  - Age of onset and disease progression patterns

**Why balanced:** Machine learning works better with balanced classes to avoid bias during pipeline development.

**Output:** `output/proxy_bmd_dmd_labels.csv` - Maps each MRI scan to proxy BMD/DMD classification

### STEP 5: DATA PREPARATION FOR TRAINING
**Script:** `clinical_validation.py` 

**How it works:**
1. **Merge:** Combine features with proxy labels by MRI scan name
2. **Clean data:** Fill missing values with 0
3. **Split:** 80% for training, 20% for testing
4. **Balance:** Ensure both sets have 50% BMD, 50% DMD

**Output:**
- **X_train:** Training features (96 samples × 100+ features)
- **X_test:** Testing features (24 samples × 100+ features)  
- **y_train:** Training proxy labels (96 samples: 0 or 1)
- **y_test:** Testing proxy labels (24 samples: 0 or 1)

**⚠️ NOTE:** All labels are synthetic proxy classifications for pipeline validation only.

### STEP 6: BMD/DMD CLASSIFICATION MODEL TRAINING
**Script:** `clinical_validation.py` 

**COMPLETE INPUT FOR MODEL TRAINING:**

**X_train (Features Data):**
- **Source:** `output/mri_radiomics_features.csv` (after merging with proxy labels)
- **Shape:** 96 samples × 100+ features
- **Data type:** Float64 numbers
- **Example row:** [1250.5, 127.3, 1.25, 0.45, 0.15, 2.8, 45.2, 0.67, ...]
- **Column names:** volume, mean_intensity, elongation, texture, skewness, kurtosis, glcm_correlation, glrlm_short_run_emphasis, ...
- **Missing values:** Filled with 0
- **Normalization:** Features already normalized during extraction

**y_train (Proxy Labels Data):**
- **Source:** `output/proxy_bmd_dmd_labels.csv` (after merging with features)
- **Shape:** 96 samples
- **Data type:** Integer (0 or 1)
- **Values:** 0 = BMD (Becker Muscular Dystrophy), 1 = DMD (Duchenne Muscular Dystrophy)
- **Example:** [0, 1, 0, 1, 0, 1, 0, 1, ...]
- **Balance:** 48 BMD, 48 DMD (50/50 split)
- **Mapping:** Each feature row corresponds to correct proxy label
- **⚠️ WARNING:** These are synthetic proxy labels, NOT real clinical diagnoses

**Training Parameters:**
```python
# RandomForestClassifier configuration
clf = RandomForestClassifier(
    n_estimators=100,        # Number of decision trees
    random_state=42,          # Reproducibility
    max_depth=None,            # Trees can grow deep
    min_samples_split=2,       # Minimum samples to split
    min_samples_leaf=1,         # Minimum samples per leaf
    bootstrap=True,             # Use bootstrap sampling
    n_jobs=-1                  # Use all CPU cores
)
```

**Training Process:**
1. **Input:** X_train (96×100+ features) + y_train (96 proxy labels)
2. **Model creation:** 
   ```python
   clf = RandomForestClassifier(n_estimators=100, random_state=42)
   clf.fit(X_train, y_train)
   ```
3. **Learning mechanism:** Creates 100 decision trees that learn different patterns

**How Each Tree Learns:**
- **Tree 1:** If volume > 1100 AND texture > 0.5 → DMD (proxy severe)
- **Tree 2:** If elongation < 1.3 AND intensity < 130 → BMD (proxy milder)
- **Tree 3:** If kurtosis > 2.5 AND correlation < 0.6 → DMD (proxy severe)
- **Tree 4:** If area < 1000 AND perimeter < 150 → BMD (proxy milder)
- **Tree 5:** If glcm_contrast > 80 AND glrlm_long_run_emphasis < 0.7 → DMD (proxy severe)
- ... (100 trees with different rules)

**Voting Process:**
- Each tree votes for BMD (0) or DMD (1)
- Majority vote wins (51+ votes needed)
- Confidence = percentage of trees agreeing

**Training Output:**
- **Trained model:** RandomForest object with 100 fitted trees
- **Feature importance:** Array showing which features matter most
- **Training score:** How well model fits training data

**⚠️ IMPORTANT:** Model learns patterns from synthetic proxy labels only.
**Real clinical performance requires actual BMD/DMD patient data.**

**Why RandomForest:** 
- Handles complex patterns in medical data
- Resistant to overfitting (many trees average out errors)
- Provides feature importance (knows which features matter most)
- Works well with limited medical datasets
- Robust to noisy medical data

### STEP 7: PREDICTION AND EVALUATION
**Script:** `clinical_validation.py`

**FOCUS: ML EVALUATION AND RESULTS**

**Input for Evaluation:**
- **X_test:** Unseen test features (20% of data, never used in training)
- **y_test:** True labels for test data (correct answers)
- **Trained model:** RandomForest that learned from training data

**How Evaluation Works:**
1. **Testing:** Give trained model X_test (unseen data)
2. **Prediction:** Model predicts BMD (0) or DMD (1) for each test sample
3. **Comparison:** Compare model predictions to y_test (true proxy labels)
4. **Scoring:** Calculate how well model performed

**Example Evaluation:**
```
Test Sample 1:
- Features: [1180.2, 132.1, 1.18, 0.52, ...]
- True proxy label: 1 (DMD)
- Model prediction: 1 (DMD) ✓ CORRECT

Test Sample 2:
- Features: [950.3, 115.8, 0.95, 0.32, ...]
- True proxy label: 0 (BMD)
- Model prediction: 0 (BMD) ✓ CORRECT

Test Sample 3:
- Features: [1420.7, 140.2, 1.35, 0.68, ...]
- True proxy label: 0 (BMD)
- Model prediction: 1 (DMD) ✗ INCORRECT
```

**⚠️ NOTE:** All evaluations use synthetic proxy labels.

**Metrics Calculated and What They Mean:**

**Accuracy:**
- Formula: (Correct predictions) / (Total predictions)
- Example: 85/100 = 85% accuracy
- Meaning: Overall, how often is model correct

**Precision:**
- Formula: (True DMD predictions) / (All DMD predictions)
- Example: 42/50 = 84% precision
- Meaning: When model says DMD, how often is it right

**Recall (Sensitivity):**
- Formula: (True DMD predictions) / (All actual DMD cases)
- Example: 42/50 = 84% recall
- Meaning: Of all DMD cases, how many did model find

**F1-Score:**
- Formula: 2 × (Precision × Recall) / (Precision + Recall)
- Example: 2 × (0.84 × 0.84) / (0.84 + 0.84) = 0.84
- Meaning: Balance between precision and recall

**Specificity:**
- Formula: (True BMD predictions) / (All actual BMD cases)
- Example: 43/50 = 86% specificity
- Meaning: Of all BMD cases, how many did model correctly identify

**AUC-ROC:**
- Range: 0.5 (random) to 1.0 (perfect)
- Example: 0.89
- Meaning: Overall ability to distinguish BMD vs DMD

**Output Files:**
- `output/ml_vs_dl_comparison.csv` - All performance metrics
- `output/ml_feature_importance.csv` - Which features matter most
- `output/proxy_bmd_dmd_labels.csv` - Final proxy labels used

⚠️ **WARNING:** All outputs use synthetic proxy labels, not real clinical data.

**Why Evaluation Matters:**
- Tests model on unseen data (real-world performance)
- Shows if model is overfitting (memorizing training data)
- Provides confidence in model's pipeline usefulness
- Allows comparison between different approaches

⚠️ **IMPORTANT:** Current evaluation uses proxy labels only.
**Real clinical validation requires actual BMD/DMD patient data.**

---

## ULTRASOUND MACHINE LEARNING PIPELINE

### STEP 1: DATA ACQUISITION
**Where we get the data:** `data/ultrasound_images/`

**What we have:** Ultrasound images stored as JPEG/PNG files
- 309 total images
- Variable sizes (typically 400×600 to 800×1200 pixels)
- RGB color format

**Why ultrasound:** Ultrasound provides real-time, cost-effective muscle imaging with good tissue differentiation.

### STEP 2: AUTOMATIC MASK GENERATION
**Script:** `extract_ultrasound_radiomics.py`

**How it works:**
1. Load the ultrasound image
2. Convert to grayscale (simplifies processing)
3. Apply Otsu thresholding (automatically finds best brightness threshold)
4. Create binary mask (white = muscle, black = background)
5. Clean mask with morphological operations (remove noise, fill holes)

**Why automatic:** Ultrasound images don't come with pre-made masks like MRI, so we create them automatically.

### STEP 3: RADIOMICS FEATURE EXTRACTION
**Script:** `extract_ultrasound_radiomics.py`

**How it works:** Same concept as MRI but for 2D images
1. Take the masked ultrasound image (only muscle pixels)
2. Calculate mathematical features:
   - **Shape features:** Area, perimeter, compactness, elongation
   - **Intensity features:** Mean brightness, standard deviation, brightness range
   - **Texture features:** Smoothness, roughness, pattern uniformity
   - **Pattern features:** Edge strength, repeating patterns

**Example features extracted:**
- Area: 1250.5 square millimeters (muscle area)
- Mean intensity: 127.3 (average brightness)
- Elongation: 1.25 (how long vs wide)
- Texture: 0.45 (smoothness score)
- Perimeter: 180.2 mm (muscle boundary length)
- Compactness: 0.82 (how compact the shape is, 1 = perfect circle)

**Output:** `output/ultrasound_radiomics_features.csv`

### STEP 4: LABELS PREPARATION
**Script:** `prepare_training_data.py`

**How it works:**
1. Create a template file with all image names
2. Generate sample labels (50% healthy, 50% diseased)
3. Save as `output/labels.csv`

**Why sample labels:** We need ground truth to train the machine learning model.

### STEP 5: DATA PREPARATION FOR TRAINING
**Script:** `simple_classifier.py`

**How it works:**
1. Load ultrasound features and labels
2. Merge data by image name
3. Split into training (80%) and testing (20%) sets
4. Prepare data for sklearn format

### STEP 6: MACHINE LEARNING MODEL TRAINING
**Script:** `simple_classifier.py`

**Algorithm:** RandomForestClassifier (same as MRI)

**How it works:** Same process as MRI but learns ultrasound-specific patterns
1. Creates 100 decision trees
2. Each tree learns different ultrasound feature patterns
3. Trees vote on healthy vs diseased classification
4. Learns which ultrasound features indicate muscle health

### STEP 7: PREDICTION AND RESULTS
**Script:** `simple_classifier.py`

**How it works:**
1. Test trained model on ultrasound test images
2. Make predictions with confidence scores
3. Save results to `output/simple_ultrasound_predictions.csv`
4. Calculate performance metrics

**Output format:**
```
image_index,predicted_class,predicted_label,confidence
0,0,healthy,85.2
1,1,diseased,92.1
```

---

## COMPARISON AND VALIDATION

### STEP 8: CLINICAL VALIDATION
**Script:** `clinical_validation.py`

**How it works:**
1. Collect results from both MRI and ultrasound ML models
2. Calculate comprehensive metrics for both
3. Compare performance between modalities
4. Generate comparison reports and visualizations

**Metrics compared:**
- Accuracy, Precision, Recall, F1-Score
- Specificity, Sensitivity
- AUC-ROC curves
- Feature importance analysis

**Outputs:**
- `output/ml_vs_dl_comparison.csv` - Performance comparison
- `output/ml_vs_dl_comparison.png` - Visual comparison charts
- `output/ml_feature_importance.csv` - Most important features

---

## KEY CONCEPTS SIMPLIFIED

### What is Machine Learning?
Teaching computers to recognize patterns by showing them many examples with known answers.

### What are Radiomics Features?
Numbers that describe medical image characteristics:
- **Shape:** Size, roundness, elongation
- **Brightness:** Average, variation, distribution
- **Texture:** Smoothness, roughness, patterns
- **Patterns:** Repeating structures, edges

### Why Use Machine Learning?
- Can find complex patterns humans might miss
- Consistent and unbiased
- Can process many cases quickly
- Improves with more data

### How Does Classification Work?
1. **Training:** Show model many examples with labels (healthy/diseased)
2. **Learning:** Model finds patterns in the features
3. **Prediction:** Model predicts health status for new, unseen cases
4. **Evaluation:** Check how well model performs on test data

---

## FINAL OUTCOME

**Goal:** Automatically classify muscle tissue as healthy or diseased
**Input:** Medical images (MRI volumes or ultrasound images)
**Process:** Extract numerical features → Train ML model → Make predictions
**Output:** Health classification with confidence scores
**Benefit:** Fast, consistent, objective muscle health assessment

This complete workflow allows us to take medical images, convert them to meaningful numbers, train machine learning models to recognize disease patterns, and provide automated muscle health assessments that can assist medical professionals in diagnosis and treatment planning.
