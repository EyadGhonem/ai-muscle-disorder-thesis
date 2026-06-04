# Thesis Draft for Claude — Paste into MET Template

**Instructions for Claude:**  
- **KEEP AS-IS:** Chapters 1 (Introduction), 2 (Background, Concepts Overview, Literature Review), Bibliography, title page, declaration.  
- **UPDATE:** Abstract (use Section A below — replaces MRI-heavy wording).  
- **REPLACE / COMPLETE:** Chapter 3 (minor edits for ultrasound-only final scope), Chapter 4 (full), Chapter 5 (full).  
- **Insert figures** at marked placeholders using files from `thesis_project/output/`.  
- **Tone:** Formal bachelor thesis (GUC MET style), third person, compare models in tables, discuss limitations honestly.

**Author:** Eyad Ghonem  
**Title:** AI-Powered Radiomics for Assessment of Muscle Disorders  
**Final experimental scope:** Ultrasound only; multi-class disease classification + severity; no DMD/BMD (data unavailable).

---

## Section A — Revised Abstract (replace current Abstract)

Muscle disorders are a group of diseases that affect skeletal muscles and often lead to progressive weakness and loss of muscle function. Early and accurate assessment is important to support clinical decision-making and monitor disease progression. Ultrasound is widely used in musculoskeletal imaging because it is non-invasive, accessible, and suitable for repeated examinations.

This thesis investigates an AI-powered radiomics framework for automated analysis of muscle ultrasound images. The proposed approach includes image preprocessing, quantitative feature extraction (intensity, texture, shape, and gradient descriptors), and classification using both machine learning and deep learning methods. Experiments were conducted on a combined dataset of approximately 8,017 labeled ultrasound samples across five categories: Facioscapulohumeral muscular dystrophy (FSHD), Inclusion Body Myositis (IBM), Dermatomyositis, Polymyositis, and Normal controls.

Multiple machine learning classifiers were trained and compared, including Support Vector Machine (SVM), Random Forest, Logistic Regression, Gradient Boosting, XGBoost, LightGBM, CatBoost, Extra Trees, and a stacking ensemble. Deep learning was investigated using EfficientNetB0 (transfer learning on a subset of images) and a multilayer perceptron (MLP) neural network on radiomics feature vectors. Models were evaluated using accuracy, precision, recall, F1-score, and confusion matrices, including patient-level held-out testing to reduce leakage across scans from the same subject.

Results show high overall classification accuracy on the combined dataset (up to approximately 99% for advanced ensemble models on stratified splits). Patient-level evaluation and macro-averaged F1-scores indicate weaker performance on minority disease classes, partly due to class imbalance and correlation between dataset source and disease label. Feature importance analysis highlighted gradient and morphological descriptors as influential predictors. The study demonstrates the feasibility of ultrasound-based radiomics and machine learning for multi-class muscle disease assessment, while identifying limitations related to dataset bias, ROI definition, and the need for full PyRadiomics extraction across all cohorts.

**Keywords:** radiomics, ultrasound, muscle disorders, machine learning, deep learning, classification, FSHD

---

## Chapter 3 — Methodology (edits + keep structure)

*Note for Claude: Most of Chapter 3 in the template is acceptable. Apply these corrections when merging:*

### 3.0 Scope clarification (add at start of Chapter 3 or in 3.3)

The final experimental work reported in Chapter 4 focuses on **ultrasound imaging only**. Although the methodology chapter describes both MRI and ultrasound pipelines (as initially planned), MRI experiments were not included in the final comparative analysis because of dataset and labeling constraints. Duchenne muscular dystrophy (DMD) and Becker muscular dystrophy (BMD) differentiation was not performed because labeled DMD/BMD ultrasound data were not available; instead, the study uses the disease categories present in the acquired datasets (FSHD, IBM, Dermatomyositis, Polymyositis, and Normal).

### 3.1 Proposed Model — keep template text with these replacements

**Remove or shorten:** Sections that present MRI 3D CNN as a primary result. Keep as “exploratory / not reported in Chapter 4” if the supervisor requires methodology completeness.

**Keep and emphasize:**
- 3.1.1 Radiomics-Based Machine Learning (SVM, RF, LR + extended models in experiments)
- 3.1.2 Deep Learning — EfficientNetB0 for ultrasound; MLP on radiomics features for multi-class comparison
- 3.1.4 Pipeline figures (Figures 3.1, 3.2)

**Figure 3.1 — Radiomics ML pipeline**  
`[INSERT FIGURE: Block diagram — Data → Preprocess → Auto mask → Feature extraction → CSV → ML models → Metrics]`  
*Claude: draw simple flowchart matching template style.*

**Figure 3.2 — Deep learning pipeline**  
`[INSERT FIGURE: Ultrasound images → Resize 224×224 → EfficientNetB0 → Classification]`  
*Optional second branch: Feature vector → MLP → Classification*

### 3.1.5 Training Configuration — use this table

**Table 3.2: Training configurations used in ultrasound experiments**

| Component | Setting |
|-----------|---------|
| ML classifiers | SVM (RBF), Random Forest (300 trees), Logistic Regression, Gradient Boosting, XGBoost, LightGBM, CatBoost, Extra Trees, Stacking |
| ML preprocessing | Median imputation; StandardScaler for SVM/LR |
| DL (EfficientNetB0) | ImageNet weights; input 224×224×3; Adam lr=0.001; batch 32; epochs 20; binary cross-entropy |
| DL (MLP baseline) | Hidden layers (256, 128); Adam; early stopping; multi-class on radiomics features |
| Split (main ML) | Stratified train/test; 5-fold CV in advanced experiments |
| Split (thesis_final) | Patient-level 80/20 (GroupShuffleSplit) |
| Random seed | 42 |

### 3.3 Dataset — replace MRI-primary narrative with this

#### 3.3.1 Combined ultrasound dataset

The master dataset is stored as `final_ultrasound_dataset.csv` and contains approximately **8,017** samples after cleaning (invalid severity labels removed). Data come from two sources:

1. **ULTRASOUND_LABELD_1 (FSHD cohort):** ~4,775 samples with real PNG images on disk, patient identifiers, and Heckmatt-based severity labels (e.g., Normal/Mild vs Moderate/Severe).

2. **ULTRASOUND_LABELD_2 (multi-disease cohort):** ~3,323 samples with clinical labels from spreadsheet metadata (Normal, Dermatomyositis, Polymyositis, IBM). Image files were not consistently available on disk for this cohort; engineered feature vectors were used to maintain a unified table for multi-class experiments (limitation discussed in Chapter 5).

**Table 3.3: Disease class distribution (approximate)**

| Disease class | Approx. samples | Primary source |
|---------------|-----------------|----------------|
| FSHD | 4,775 | ULTRASOUND_LABELD_1 |
| Normal | 1,337 | ULTRASOUND_LABELD_2 |
| Inclusion Body Myositis | 796 | ULTRASOUND_LABELD_2 |
| Dermatomyositis | 555 | ULTRASOUND_LABELD_2 |
| Polymyositis | 554 | ULTRASOUND_LABELD_2 |

**Figure 3.4 — Class distribution**  
`[INSERT GRAPH: output/01_class_distribution.png]`  
*Caption: Distribution of disease and severity labels in the combined ultrasound dataset.*

**Figure 3.5 — Dataset source vs disease (bias analysis)**  
`[INSERT GRAPH: output/02_dataset_source_bias.png]`  
*Caption: Relationship between dataset source and disease class, illustrating potential confounding.*

#### 3.3.2 Feature extraction

Features were extracted to represent radiomics-style quantitative descriptors:

- **First-order:** mean, median, std, percentiles, skewness, kurtosis, entropy  
- **Texture (GLCM):** contrast, dissimilarity, homogeneity, energy, correlation, ASM  
- **Morphology:** area, perimeter, circularity, aspect ratio, solidity, equivalent diameter  
- **Gradient:** mean, std, max, energy  

For FSHD images, features were computed from real ultrasound intensities using OpenCV-based pipelines (`extract_custom_features.py`). PyRadiomics (`original_*` features) was applied to a smaller labeled set (`ultrasound_radiomics_features.csv`, ~309 images) and can be extended to the full FSHD cohort via `extract_pyradiomics_labeled1.py`. Ultrasound regions of interest were obtained using **automatic threshold-based foreground masks**, not expert manual segmentation.

#### 3.3.3 Data splitting

- **Stratified random split** (70/15/15 or 80/20 depending on script) for main model benchmarking.  
- **Patient-level split** (20% of patients held out) for `run_final_thesis_evaluation.py` to reduce patient leakage.  
- **5-fold cross-validation** for robust metrics in `output/advanced_results/`.

### 3.4 Summary

Chapter 3 described preprocessing, radiomics-inspired feature extraction, machine learning and deep learning models, dataset composition, splitting strategies, and evaluation metrics. Chapter 4 presents the experimental results and comparisons.

---

## Chapter 4 — Results (REPLACE placeholder Chapter 4 entirely)

### 4.1 Experimental Setup

#### 4.1.1 Tools and environment

Experiments were implemented in **Python 3.9+** using:

- **scikit-learn** — classical ML and metrics  
- **TensorFlow/Keras** — EfficientNetB0 ultrasound classifier  
- **OpenCV, NumPy, Pandas** — image I/O and feature engineering  
- **PyRadiomics, SimpleITK** — standardized radiomics on subset  
- **XGBoost, LightGBM, CatBoost** — advanced gradient boosting models  

Project code and outputs are organized under `thesis_project/` with results in `thesis_project/output/`.

#### 4.1.2 Evaluation protocol

All models were compared using:

- Accuracy  
- Precision, recall, F1-score (per class, macro, and weighted)  
- Confusion matrix  
- AUC-ROC where applicable (binary CNN vs RF experiment)  

Two evaluation settings are reported:

1. **Main benchmark** — stratified split on full combined dataset (8,017 samples).  
2. **Patient-level benchmark** — 80% train / 20% test by `patient_id` (`output/thesis_final/`).

---

### 4.2 Disease classification — machine learning comparison (main benchmark)

Nine machine learning models were trained on the combined feature table. Table 4.1 summarizes test-set performance (from `output/baseline_and_advanced_models/model_comparison.csv`).

**Table 4.1: Machine learning model comparison (disease classification, combined dataset)**

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|------|-------|----------|-----------|--------|----------|
| 1 | Gradient Boosting | 99.10% | 98.92% | 99.10% | 98.89% |
| 2 | Random Forest | 98.99% | 98.62% | 98.99% | 98.74% |
| 3 | Extra Trees | 98.99% | 98.62% | 98.99% | 98.74% |
| 4 | Stacking Ensemble | 98.99% | 98.62% | 98.99% | 98.74% |
| 5 | LightGBM | 98.89% | 98.68% | 98.89% | 98.76% |
| 6 | CatBoost | 98.89% | 98.68% | 98.89% | 98.76% |
| 7 | SVM | 98.80% | 98.78% | 98.80% | 98.76% |
| 8 | XGBoost | 98.80% | 98.56% | 98.80% | 98.64% |
| 9 | Logistic Regression | 98.49% | 98.47% | 98.49% | 98.48% |

**Discussion of Table 4.1:** Gradient Boosting achieved the highest F1-score among single models, while stacking ensembles matched Random Forest and Extra Trees. All models exceeded 98% accuracy, indicating strong separability on the combined feature representation. However, this must be interpreted alongside dataset-source confounding (FSHD predominantly from one source), as analyzed in Section 4.5.

**Figure 4.1 — Model comparison bar chart**  
`[INSERT GRAPH: output/baseline_and_advanced_models/model_comparison.png if exists, else generate from Table 4.1]`  
*Caption: Comparison of classification accuracy across nine machine learning models.*

---

### 4.3 Severity classification (FSHD)

For the FSHD subset, binary severity classification (e.g., Mild vs Moderate/Severe based on Heckmatt grading) was evaluated. Random Forest and Extra Trees achieved approximately **93.6%** accuracy in project summaries; cross-validated results are stored in `output/advanced_results/`. Severity assessment supports the thesis objective of monitoring disease progression, although longitudinal time-series analysis was not performed.

**Figure 4.2 — Severity results (optional)**  
`[INSERT GRAPH: from output/advanced_results/ if severity plot exists]`  
*Caption: Severity classification performance for FSHD ultrasound samples.*

---

### 4.4 Radiomics + ML vs deep learning

#### 4.4.1 Patient-level multi-class evaluation

To better approximate clinical deployment, models were re-evaluated with a **patient-level held-out test set** (849 test samples, 7,168 train; five disease classes). Table 4.2 compares classical ML and an MLP neural network on the same feature vectors (`output/thesis_final/thesis_final_comparison_table.csv`).

**Table 4.2: Patient-level evaluation (five disease classes)**

| Model family | Model | Accuracy | F1 (macro) | F1 (weighted) |
|--------------|-------|----------|------------|---------------|
| Radiomics + ML | SVM | 98.35% | 0.409 | 0.984 |
| Radiomics + ML | Random Forest | 98.23% | 0.280 | 0.979 |
| Radiomics + ML | Logistic Regression | 97.88% | 0.264 | 0.979 |
| Deep Learning | MLP Neural Network | 98.23% | 0.280 | 0.979 |

**Interpretation:** Overall accuracy remains high because the test set is dominated by FSHD samples. **Macro F1 is substantially lower** (0.26–0.41), showing poor balanced performance on minority classes (IBM, DM, PM, Normal). This is a critical honest result for the thesis: the system is effective at detecting the majority class but requires improvement for rare diseases.

**Figure 4.3 — Confusion matrix (Random Forest, patient-level)**  
`[INSERT GRAPH: heatmap from output/thesis_final/confusion_matrix_Random_Forest.csv]`  
*Caption: Confusion matrix for Random Forest under patient-level split.*

**Figure 4.4 — Confusion matrix (SVM, patient-level)**  
`[INSERT GRAPH: output/thesis_final/confusion_matrix_SVM.csv]`

#### 4.4.2 Binary ultrasound subset: Random Forest vs EfficientNet CNN

On a smaller binary-labeled set (`data/ultrasound_images`, ~309 images), Random Forest on radiomics features was compared to EfficientNetB0 (`output/ml_vs_dl_comparison.csv`).

**Table 4.3: Binary classification — ML vs CNN**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest (radiomics + ML) | 96.77% | 96.77% | 96.77% | 96.77% |
| EfficientNetB0 CNN | 80.33% | 100%* | 80.33% | 89.09% |

*CNN precision is computed on positive predictions; review confusion matrix for class imbalance.

**Figure 4.5 — ML vs DL comparison**  
`[INSERT GRAPH: output/ml_vs_dl_comparison.png if exists]`  
*Caption: Comparison of Random Forest (radiomics features) and EfficientNetB0 on the binary ultrasound subset.*

On this subset, **classical radiomics + Random Forest outperformed the CNN** in accuracy, supporting the thesis comparison narrative that hand-crafted radiomics features remain competitive when training data are limited.

---

### 4.5 Feature importance and clinical interpretation

Random Forest feature importance (`output/thesis_final/feature_importance.csv`) ranked the following among the top predictors:

1. gradient_mean  
2. perimeter  
3. area  
4. glcm_homogeneity  
5. gradient_max  
6. equivalent_diameter  
7. glcm_energy  

**Clinical interpretation (for Discussion):** Higher gradient and morphological variation may reflect disorganized muscle architecture and altered echogenicity patterns associated with dystrophic and inflammatory myopathies. GLCM homogeneity captures local texture uniformity; reduced homogeneity may indicate structural heterogeneity in diseased muscle. These associations are hypothesis-generating and require validation with expert ultrasound readers.

**Figure 4.6 — Feature importance**  
`[INSERT GRAPH: bar chart from output/thesis_final/feature_importance.csv — top 15 features]`  
*Caption: Top 15 radiomics-inspired features by Random Forest importance.*

---

### 4.6 Data quality and dataset-source bias

A data leakage and quality report (`output/data_leakage_report.txt`) documented:

- No duplicate images in the master table  
- Severe class imbalance (ratio up to ~59:1)  
- **Significant correlation between dataset source and disease class** (FSHD almost exclusively from ULTRASOUND_LABELD_1; other diseases from ULTRASOUND_LABELD_2)

Per-source test metrics are saved as `output/thesis_final/dataset_source_metrics_*.csv` for SVM, Random Forest, and MLP. The thesis should report that high accuracy may partially reflect source-specific patterns rather than purely disease biology.

**Figure 4.7 — Per-source performance (optional)**  
`[INSERT GRAPH: grouped bar chart from dataset_source_metrics_Random_Forest.csv]`

---

### 4.7 Summary of results

| Task | Best model (reported) | Key metric | Caveat |
|------|----------------------|------------|--------|
| Multi-class disease (stratified) | Gradient Boosting | ~99% accuracy | Source-disease confounding |
| Multi-class disease (patient-level) | SVM | 98.35% acc., F1 macro 0.41 | Weak on minority classes |
| Severity (FSHD) | Extra Trees / RF | ~93.6% accuracy | FSHD-only labels |
| Binary (small set) | Random Forest | 96.77% vs CNN 80.33% | Small N (~309) |

---

## Chapter 5 — Conclusion and Future Work (REPLACE placeholder Chapter 5)

### 5.1 Conclusion

This bachelor thesis presented an AI-powered radiomics framework for assessing muscle disorders using ultrasound imaging. The work addressed the need for objective, quantitative analysis beyond subjective visual interpretation by extracting texture, intensity, shape, and gradient features from muscle ultrasound and training multiple machine learning and deep learning classifiers.

The main contributions are:

1. **Integration of two labeled ultrasound sources** into a unified dataset of approximately 8,017 samples spanning five disease categories (FSHD, IBM, Dermatomyositis, Polymyositis, and Normal), enabling multi-class classification experiments not limited to binary healthy-vs-diseased screening.

2. **Comprehensive model benchmarking** of nine machine learning algorithms (including Gradient Boosting, Random Forest, SVM, XGBoost, LightGBM, CatBoost, and stacking ensembles), achieving up to approximately 99% accuracy on stratified evaluation.

3. **Comparative analysis of radiomics + ML versus deep learning**, including EfficientNetB0 on raw images and an MLP on feature vectors, with patient-level testing to reduce patient leakage.

4. **Feature importance and bias analysis**, linking quantitative descriptors to interpretable ultrasound phenomena and documenting dataset-source confounding as a limitation.

The results support the feasibility of ultrasound radiomics combined with machine learning for muscle disease classification, particularly for FSHD-dominated cohorts. At the same time, macro F1-scores under patient-level splitting demonstrate that balanced performance across all disease types remains an open challenge.

### 5.2 Limitations

- **Modality scope:** Final reported experiments used ultrasound only; MRI pipeline was not included in Chapter 4 results.  
- **Disease scope:** DMD/BMD were not studied due to unavailable labeled data.  
- **Features:** Not all samples used full PyRadiomics extraction; part of the multi-disease cohort relies on engineered features when images were unavailable.  
- **ROI:** Automatic masks approximate muscle foreground but are not expert segmentations.  
- **Bias:** Dataset source correlates with disease class, which may inflate accuracy metrics.  
- **Progression:** Longitudinal monitoring over multiple time points was not modeled.

### 5.3 Future Work

1. Extract **full PyRadiomics** features from all available FSHD images and recover multi-disease images from `PatientData.mat` (HDF5) for real feature extraction on ULTRASOUND_LABELD_2.  
2. Apply **patient-level cross-validation** and external validation on an independent hospital dataset.  
3. Improve minority-class performance using **class balancing**, cost-sensitive learning, or focal loss in deep models.  
4. Train **end-to-end CNNs** (EfficientNet or similar) on the full multi-class image set once all images are available, with fair comparison to radiomics + ML on identical splits.  
5. Add **explainability** (SHAP, Grad-CAM) and clinician-in-the-loop evaluation.  
6. Extend to **severity progression** using repeated scans per patient.

---

## Figure and file checklist for Claude

| Figure | File path |
|--------|-----------|
| 3.4 Class distribution | `output/01_class_distribution.png` |
| 3.5 Source bias | `output/02_dataset_source_bias.png` |
| 4.1 ML comparison | `output/baseline_and_advanced_models/model_comparison.png` |
| 4.3 Confusion matrix RF | `output/thesis_final/confusion_matrix_Random_Forest.csv` → heatmap |
| 4.5 Feature importance | `output/thesis_final/feature_importance.csv` → bar chart |
| 4.5 ML vs DL | `output/ml_vs_dl_comparison.png` |

---

## Tables to paste directly

- Table 4.1 — Section 4.2  
- Table 4.2 — Section 4.4.1  
- Table 4.3 — Section 4.4.2  
- Table 3.3 — Section 3.3.1  

---

*End of draft — merge into `MET_Thesis_Template__Spring2025___Copy_.pdf` structure.*
