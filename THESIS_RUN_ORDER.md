# Thesis run order (ultrasound-only)

## 1. Build / verify dataset
- Master file: `output/final_ultrasound_dataset.csv`

## 2. Extract REAL PyRadiomics (FSHD images with files on disk)
```powershell
& .\radiomics_env\Scripts\Activate.ps1
python extract_pyradiomics_labeled1.py
```
This creates `output/pyradiomics_labeled1_features.csv` (~4,775 images).
Takes time (can set `MAX_IMAGES = 200` in script for a quick test).

## 3. Run final evaluation (uses PyRadiomics file if present)
```powershell
python run_final_thesis_evaluation.py
```

## 4. Outputs to use in Chapter 4
| File | Purpose |
|------|---------|
| `output/thesis_final/thesis_final_comparison_table.md` | Main results table |
| `output/thesis_final/model_comparison.csv` | All metrics |
| `output/thesis_final/confusion_matrix_*.csv` | Per-model confusion matrices |
| `output/thesis_final/feature_importance.csv` | Top radiomics features |
| `output/thesis_final/dataset_source_metrics_*.csv` | Bias analysis |
| `output/thesis_final/THESIS_SCOPE_AND_LIMITATIONS.md` | Scope (no MRI, no DMD/BMD) |

## 4. Optional (radiomics from raw images)
```powershell
python extract_ultrasound_radiomics.py
```

## 5. Train CNN models (ResNet50, DenseNet121) — same pipeline as other models
```powershell
# Trains BOTH binary + severity (default), saves models + val splits
python train_ultrasound_cnn_models.py --models resnet50 densenet121 efficientnetb0

# Evaluate CNN with same metrics (accuracy, F1, confusion matrix)
python evaluate_cnn_models.py

# Or run full thesis evaluation (ML + MLP + CNN in one table)
python run_final_thesis_evaluation.py
```
Outputs:
- `output/dl_models/resnet50_binary.keras`, `resnet50_severity.keras`, etc.
- `output/thesis_final/cnn_model_comparison.csv`
- `output/thesis_final/thesis_final_comparison_table.csv` (all models combined)

## 6. Optional (legacy EfficientNet only)
```powershell
python train_ultrasound_classifier.py
python predict_ultrasound.py
```
