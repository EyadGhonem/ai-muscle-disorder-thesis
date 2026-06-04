# Bachelor Thesis Source Pack — Copy This to Claude for Writing

**Topic:** AI-Powered Radiomics for Assessment of Muscle Disorders  
**Student project path:** `c:\Users\Lenovo\Desktop\thesis_project`  
**Final scope:** Ultrasound only (MRI excluded). DMD/BMD not available — used FSHD, IBM, Dermatomyositis, Polymyositis, Normal.

---

## 1. Project aim

Build an AI pipeline that:
1. Extracts quantitative imaging features (radiomics-style) from muscle ultrasound.
2. Trains machine learning and deep learning models to classify muscle disease type and severity.
3. Compares Radiomics+ML vs deep learning approaches with proper metrics.

Inspired by radiomics literature (e.g. MRI for muscular dystrophy differentiation), adapted to **ultrasound** and **available disease labels**.

---

## 2. Datasets used

| Source | Samples (approx.) | Diseases | Notes |
|--------|-------------------|----------|--------|
| ULTRASOUND_LABELD_1 | ~4,775 | FSHD | Real PNG images on disk; Heckmatt severity grades |
| ULTRASOUND_LABELD_2 | ~3,323 | Normal, Dermatomyositis, Polymyositis, IBM | Labels from Excel; image paths often missing on disk |
| Combined master file | ~8,017 | 5 classes | `output/final_ultrasound_dataset.csv` |
| Small set | ~309 | Binary healthy/diseased | `data/ultrasound_images/`, `output/labels.csv` |

**Important limitation:** Combined dataset features are mixed:
- LABELD_1: **custom OpenCV features** from real images (area, GLCM, gradient, intensity stats) — not full PyRadiomics `original_*` on all rows.
- LABELD_2: **synthetic placeholder features** generated in `build_master_dataset.py` when images were unavailable.
- Real **PyRadiomics** extraction exists for ~309 images: `output/ultrasound_radiomics_features.csv`.
- Script added for real PyRadiomics on LABELD_1: `extract_pyradiomics_labeled1.py` → `output/pyradiomics_labeled1_features.csv` (run if needed).

**Dataset bias:** FSHD strongly linked to dataset source 1; other diseases to source 2. Must discuss in limitations (`output/data_leakage_report.txt`).

---

## 3. Feature extraction

**Feature types:**
- First-order: mean/median/std intensity, percentiles, skewness, kurtosis, entropy
- Texture: GLCM (contrast, homogeneity, energy, correlation, dissimilarity, ASM)
- Morphology: area, perimeter, circularity, aspect ratio, solidity, equivalent diameter
- Gradient: mean, std, max, energy

**Methods in codebase:**
- `data_processing/extract_custom_features.py` — OpenCV + numpy (main FSHD pipeline)
- `extract_ultrasound_radiomics.py` — PyRadiomics on `data/ultrasound_images/`
- `extract_pyradiomics_labeled1.py` — PyRadiomics on LABELD_1 images from master CSV
- `extract_real_features_final.py` / HDF5 — attempted extraction from `PatientData.mat` (~100 images)

**ROI:** Automatic foreground mask (threshold + morphology), not expert manual segmentation.

---

## 4. Models trained (complete list)

### Machine learning (tabular / radiomics features)
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Logistic Regression
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- Stacking ensemble (combines multiple models)

### Deep learning
- **EfficientNetB0** + dense layers (`train_ultrasound_classifier.py`) — binary classification on small image set; saved as `output/ultrasound_classifier.keras`
- **MLP Neural Network** (`run_final_thesis_evaluation.py`) — multi-class on same tabular features as ML
- ~~3D CNN for MRI~~ — excluded from final thesis scope

### Tasks trained
1. **Multi-class disease classification** (5 classes)
2. **Severity classification** (e.g. Mild vs Moderate/Severe for FSHD)
3. **Binary** healthy vs diseased (small ultrasound_images set)

---

## 5. Validation methodology

- Stratified train/test splits
- 5-fold cross-validation (advanced/baseline training scripts)
- **Patient-level** hold-out split in `run_final_thesis_evaluation.py` (20% patients test)
- Metrics: accuracy, precision, recall, F1 (macro and weighted), confusion matrix, AUC where computed
- Data leakage check: `output/data_leakage_report.txt`

---

## 6. Key results (where to find numbers)

### Main ML comparison (9 models, combined dataset)
**File:** `output/baseline_and_advanced_models/model_comparison.csv`

| Model | Accuracy (approx.) |
|-------|-------------------|
| Gradient Boosting | 99.10% |
| Random Forest | 98.99% |
| Extra Trees | 98.99% |
| Stacking | 98.99% |
| LightGBM | 98.89% |
| CatBoost | 98.89% |
| SVM | 98.80% |
| XGBoost | 98.80% |
| Logistic Regression | 98.49% |

### Final thesis evaluation (patient-level split, 5 diseases)
**Folder:** `output/thesis_final/`  
**Table:** `thesis_final_comparison_table.md`

| Model family | Model | Accuracy | F1 macro |
|--------------|-------|----------|----------|
| Radiomics+ML | SVM | 98.35% | 0.41 |
| Radiomics+ML | Random Forest | 98.23% | 0.28 |
| Radiomics+ML | Logistic Regression | 97.88% | 0.26 |
| Deep Learning | MLP Neural Network | 98.23% | 0.28 |

**Note:** High accuracy but lower macro F1 → strong on majority class (FSHD), weaker on rare classes. Discuss honestly.

### ML vs CNN (small binary set)
**File:** `output/ml_vs_dl_comparison.csv`
- Random Forest (ML): ~96.8% accuracy
- CNN (EfficientNet): ~80.3% accuracy

### Severity classification
- ~92–94% accuracy (Random Forest / Extra Trees) in project summaries and `output/advanced_results/`

### Feature importance
- `output/thesis_final/feature_importance.csv` (top: gradient_mean, perimeter, area, glcm_homogeneity, …)

---

## 7. Main code scripts (pipeline)

| Step | Script |
|------|--------|
| Build combined dataset | `build_master_dataset.py`, `create_final_ultrasound_dataset.py` |
| Custom features | `data_processing/extract_custom_features.py` |
| PyRadiomics (small set) | `extract_ultrasound_radiomics.py` |
| PyRadiomics (LABELD_1) | `extract_pyradiomics_labeled1.py` |
| Train many ML models | `train_all_models.py`, `baseline_models_proper.py`, `advanced_radiomics_models.py` |
| Train CNN | `train_ultrasound_classifier.py` |
| Predict CNN | `predict_ultrasound.py` |
| Final evaluation | `run_final_thesis_evaluation.py` |
| Prepare labels template | `prepare_training_data.py` |

---

## 8. Output files index (for tables/figures)

```
output/
├── final_ultrasound_dataset.csv          # Master dataset (~8017 rows)
├── final_ultrasound_dataset_REAL_features.csv  # Subset with real extracted features
├── ultrasound_radiomics_features.csv     # PyRadiomics, ~309 images
├── baseline_and_advanced_models/
│   ├── model_comparison.csv              # 9 ML models
│   └── trained_models.pkl
├── advanced_results/                     # CV + test metrics, trained .joblib models
├── baseline_results/
├── thesis_final/                         # Patient-level ML vs MLP comparison
│   ├── thesis_final_comparison_table.md
│   ├── confusion_matrix_*.csv
│   ├── feature_importance.csv
│   ├── dataset_source_metrics_*.csv
│   └── THESIS_SCOPE_AND_LIMITATIONS.md
├── thesis_reports/THESIS_MODEL_RESULTS.txt
├── data_leakage_report.txt
├── ml_vs_dl_comparison.csv
└── ultrasound_classifier.keras             # EfficientNet binary model
```

---

## 9. What to state in the thesis (honest framing)

**Do write:**
- Ultrasound radiomics-inspired feature extraction and ML/DL classification of muscle diseases.
- Five disease categories available in data (not DMD/BMD).
- Comparison of multiple ML algorithms and a neural network baseline.
- Patient-level validation and dataset-source bias analysis.
- Severity grading for FSHD (Heckmatt-based).

**Do not overclaim:**
- Do not say "100% complete" or "perfect clinical deployment" without caveats.
- Do not claim full PyRadiomics on all 8017 samples unless `pyradiomics_labeled1_features.csv` was run on all images.
- Do not claim DMD vs BMD differentiation.
- Do not claim MRI was used in the final study.
- Acknowledge synthetic features for part of multi-disease data and dataset-source confounding.

---

## 10. Suggested thesis chapter outline for Claude

1. **Introduction** — muscle disorders, ultrasound, radiomics, AI; scope (ultrasound, available diseases).
2. **Background** — radiomics, ML/DL in medical imaging; related work (Chen et al. style, adapted to ultrasound).
3. **Materials & Methods** — datasets, labeling, feature extraction, ROI, models, validation splits.
4. **Implementation** — pipeline diagram, tools (Python, scikit-learn, TensorFlow, OpenCV, PyRadiomics).
5. **Results** — class distribution, model comparison tables, confusion matrices, feature importance, ML vs DL.
6. **Discussion** — clinical interpretation, bias/limitations, synthetic features, future work (real multi-disease images, progression).
7. **Conclusion** — contributions and limitations.

---

## 11. Other documentation files (optional extra context)

| File | Use |
|------|-----|
| `PROJECT_SUMMARY_WHAT_WE_DID.md` | Shorter accomplishments + limitations |
| `FINAL_THESIS_COMPLETE_DOCUMENTATION.md` | Long outline (some claims are optimistic — verify against Section 9) |
| `output/README_THESIS_OUTPUTS.md` | Detailed output index |
| `THESIS_RUN_ORDER.md` | How to re-run pipeline |
| `Chapter_3_Methodology.md` | Draft methodology if present |

---

## 12. One-paragraph abstract draft (for Claude to expand)

This bachelor thesis presents an AI-powered radiomics framework for assessing muscle disorders using ultrasound imaging. Quantitative features describing tissue intensity, texture, shape, and gradients were extracted from labeled ultrasound datasets comprising Facioscapulohumeral muscular dystrophy (FSHD) and other myopathies (inclusion body myositis, dermatomyositis, polymyositis, and normal controls). Multiple machine learning classifiers (Random Forest, Gradient Boosting, SVM, Logistic Regression, XGBoost, LightGBM, CatBoost, and stacking ensembles) and deep learning approaches (EfficientNet-based CNN and multilayer perceptron) were trained and compared for disease-type and severity classification. Models were evaluated using cross-validation, held-out test sets, and patient-level splitting. Results show high classification accuracy on the combined dataset, with important limitations including dataset-source bias and incomplete image availability for some disease cohorts. The work demonstrates feasibility of ultrasound radiomics for multi-class muscle disease assessment while identifying directions for clinical validation and full PyRadiomics extraction across all cohorts.

---

*End of source pack — give this entire file to Claude when asking to write the bachelor thesis.*
