# Radiomics Pipeline - Quick Reference Guide

## Environment Setup

```powershell
# Activate environment
& .\radiomics_env\Scripts\Activate.ps1

# Verify Python 3.9.13
python --version

# Check packages
python test_environment.py
```

## File Organization

```
thesis_project/
├── extract_radiomics.py              # MAIN SCRIPT - Start here
├── extract_radiomics_advanced.py     # Advanced features & batch processing
├── test_environment.py               # Validate setup
├── RADIOMICS_SETUP.md                # Full documentation
├── QUICK_REFERENCE.md                # This file
├── data/
│   ├── images/                       # Place .nii.gz files here
│   └── masks/                        # Place *_mask.nii.gz files here
└── output/                           # CSV files saved here
```

## Quick Start (3 Steps)

### Step 1: Prepare Data
```powershell
mkdir -p data/images
mkdir -p data/masks
# Copy your .nii.gz files to these directories
```

### Step 2: Run Extraction
```powershell
& .\radiomics_env\Scripts\Activate.ps1
python extract_radiomics.py
```

### Step 3: Check Output
```
output/case1_features.csv  ← Your extracted features!
```

## Common Operations

### Extract features from a single case
```python
from radiomics import featureextractor

extractor = featureextractor.RadiomicsFeatureExtractor()
features = extractor.execute("data/images/patient1.nii.gz", 
                             "data/masks/patient1_mask.nii.gz")
```

### Filter diagnostic keys
```python
features = {k: v for k, v in features.items() 
           if not (k.startswith('diagnostics_') or k.startswith('setting_'))}
```

### Convert to DataFrame
```python
import pandas as pd
df = pd.DataFrame([features])
df.to_csv("output/features.csv", index=False)
```

### Batch process multiple cases
```python
# Run from extract_radiomics_advanced.py:
df = batch_extract_features("data/images", "data/masks", "output/batch.csv")
```

### Load and analyze features
```python
import pandas as pd
df = pd.read_csv("output/features.csv")
print(df.describe())
print(df.info())
```

### Get specific feature types
```python
# Shape features
shape_features = {k: v for k, v in features.items() if '_shape_' in k}

# First-order statistics
firstorder = {k: v for k, v in features.items() if '_firstorder_' in k}

# GLCM texture
glcm = {k: v for k, v in features.items() if '_glcm_' in k}

# GLRLM texture
glrlm = {k: v for k, v in features.items() if '_glrlm_' in k}
```

## Feature Categories

| Feature Class | Count | Purpose |
|--------------|-------|---------|
| **Shape** | ~14 | Geometric properties of ROI |
| **First-order** | ~19 | Histogram statistics |
| **GLCM** | ~24 | Gray Level Co-occurrence Matrix (texture) |
| **GLRLM** | ~16 | Gray Level Run Length Matrix (texture) |
| **GLSZM** | ~16 | Gray Level Size Zone Matrix (texture) |
| **NGTDM** | ~5 | Neighbouring Gray Tone Difference Matrix |
| **GLDM** | ~14 | Gray Level Dependence Matrix |
| **Total** | **~150+** | All features combined |

## Feature Name Format

```
{imageType}_{featureClass}_{featureName}

Examples:
- original_shape_Volume          → Original image shape feature
- original_firstorder_Mean       → Mean intensity in ROI
- wavelet_glcm_Correlation       → GLCM texture on wavelet
- original_glrlm_ShortRunEmphasis → Run-length texture feature
```

## Image Types

- **original**: Raw image
- **wavelet**: Wavelet decomposition
- **LoG**: Laplacian of Gaussian

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: radiomics` | Activate environment: `& .\radiomics_env\Scripts\Activate.ps1` |
| `FileNotFoundError` | Check data directory structure and file naming |
| Memory error on large images | Resample images to lower resolution first |
| Mask dimension mismatch | Ensure image and mask have same size |
| Empty features | Ensure mask has non-zero voxels |

## Key Code Snippets

### Load image and mask
```python
import SimpleITK as sitk

image = sitk.ReadImage("data/images/case1.nii.gz")
mask = sitk.ReadImage("data/masks/case1_mask.nii.gz")

print(f"Image size: {image.GetSize()}")
print(f"Image spacing: {image.GetSpacing()}")
```

### Extract features with custom verbosity
```python
import logging
logging.getLogger('radiomics').setLevel(logging.WARNING)  # Suppress output

from radiomics import featureextractor
extractor = featureextractor.RadiomicsFeatureExtractor()
features = extractor.execute(image_path, mask_path)
```

### Process and save features
```python
import pandas as pd

# Extract from multiple cases
all_cases = []
for case_id in ["case1", "case2", "case3"]:
    features = extractor.execute(
        f"data/images/{case_id}.nii.gz",
        f"data/masks/{case_id}_mask.nii.gz"
    )
    features['case_id'] = case_id
    all_cases.append(features)

# Save to CSV
df = pd.DataFrame(all_cases)
df.to_csv("output/all_features.csv", index=False)
```

## Performance Tips

1. **Parallel processing**: Process multiple cases on separate cores
2. **Batch processing**: Extract from all cases in one run
3. **Memory**: Close other applications when processing large datasets
4. **Disk space**: Medical images can be large; ensure adequate storage

## Pipeline Continuation

After feature extraction:

1. **Preprocessing**
   - Normalization (StandardScaler, MinMaxScaler)
   - Handle missing values
   - Remove highly correlated features

2. **Feature Selection**
   - Univariate selection (SelectKBest)
   - Recursive elimination (RFE)
   - Principal Component Analysis (PCA)

3. **Modeling**
   - Classification (SVM, Random Forest, Neural Networks)
   - Regression (Linear, Ridge, Lasso)
   - Cross-validation

4. **Evaluation**
   - Metrics (AUC, Sensitivity, Specificity)
   - Feature importance analysis
   - Model interpretability

## Resources

- **PyRadiomics**: https://pyradiomics.readthedocs.io/
- **Features**: https://pyradiomics.readthedocs.io/en/latest/features.html
- **Radiomics Journal**: https://www.radiomics.io/
- **SimpleITK**: https://simpleitk.org/

## Notes for Your Thesis

Remember to document:
- ✓ PyRadiomics version used
- ✓ Feature classes and image types selected
- ✓ Preprocessing steps (resampling, normalization)
- ✓ Number of cases and features
- ✓ Feature selection strategy
- ✓ Model validation approach
