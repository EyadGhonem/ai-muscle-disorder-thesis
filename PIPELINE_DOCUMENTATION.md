# PyRadiomics Feature Extraction Pipeline

## Project Overview

This Python 3.9.13 project implements a **radiomics feature extraction pipeline** using the PyRadiomics library. Radiomics involves the extraction of quantitative features from medical images to enable computational analysis of disease characteristics.

## Environment Setup

- **Python Version**: 3.9.13
- **Virtual Environment**: `radiomics_env/`
- **Key Packages**:
  - PyRadiomics 3.1.0 - Feature extraction
  - SimpleITK 2.5.3 - Medical image I/O
  - NumPy 1.26.4 - Array operations
  - Pandas 2.3.3 - Data processing
  - OpenCV 4.7.0 - Image format conversion

### Activate Environment
```powershell
& .\radiomics_env\Scripts\Activate.ps1
```

## Project Structure

```
thesis_project/
├── data/
│   ├── images/         # NIfTI format medical images (.nii.gz)
│   ├── images_small/   # Downsampled versions (1/3 resolution)
│   ├── masks/          # Binary ROI masks matching image dimensions
│   └── masks_small/    # Downsampled masks
├── output/             # CSV results with extracted features
├── extract_full.py     # Main full-resolution extraction script
├── extract_small.py    # Quick test on downsampled data
├── downsample_images.py # Create downsampled versions for testing
└── convert_images_to_nifti.py # Convert 2D images to 3D NIfTI format
```

## Feature Extraction Scripts

### 1. **extract_small.py** - Quick Testing (takes ~30 seconds)
```bash
python extract_small.py
```
- Extracts features from downsampled image (1/3 resolution)
- Great for testing and validation
- Produces `output/mri_features_small.csv`

### 2. **extract_full.py** - Full Resolution (takes 15-30 minutes)
```bash
python extract_full.py
```
- Extracts features from full-resolution 3D MRI
- Computationally intensive but comprehensive
- Produces `output/mri_features.csv`

## Radiomics Features (107+ total)

| Category | Count | Examples |
|----------|-------|----------|
| **Shape** | 14 | Elongation, Flatness, Compactness, Volume |
| **First-Order** | 19 | Mean, Std Dev, Minimum, Maximum, Median, Skewness |
| **GLCM** | 24 | Contrast, Correlation, Energy, Homogeneity |
| **GLRLM** | 16 | Short-run emphasis, Long-run emphasis, Run entropy |
| **GLSZM** | 16 | Size zone emphasis, Small area emphasis, Zone entropy |
| **NGTDM** | 5 | Coarseness, Contrast, Busyness, Complexity, Strength |
| **GLDM** | 14 | Dependence count, Large dependence emphasis |

## Input Image Requirements

- **Format**: NIfTI (.nii or .nii.gz)
- **Dimensions**: 3D volume (Z, Y, X)
- **Bit Depth**: Float32 for intensity, Uint8 for masks
- **Mask Labels**: Binary (0 = background, 1 = ROI)

## Image Conversion (2D → 3D)

To convert 2D images (PNG, JPG) to 3D NIfTI format:

```bash
python convert_images_to_nifti.py
```

- Automatically stacks 2D images into 3D volumes
- Creates binary masks from image content
- Outputs files to `data/images/` and `data/masks/`

## Usage Example

```python
from radiomics import featureextractor
import pandas as pd

# Create extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

# Extract features
features = extractor.execute("data/images/mri.nii.gz", 
                             "data/masks/mri_mask.nii.gz")

# Save to CSV
df = pd.DataFrame([features])
df.to_csv("output/features.csv", index=False)
```

## Output Format

The CSV file contains:
- **case_id**: Image identifier
- **original_shape_***: Geometric features
- **original_firstorder_***: Intensity statistics
- **original_glcm_***: Gray-level co-occurrence
- **original_glrlm_***: Gray-level run-length
- **original_glszm_***: Gray-level size zone
- **original_ngtdm_***: Neighborhood gray-tone difference
- **original_gldm_***: Gray-level dependence

## Performance Notes

- **Downsampled (1/3 res)**: ~30 seconds
- **Full resolution**: 15-30 minutes depending on image size
- **Memory Usage**: ~1-2 GB for large 3D volumes

## Troubleshooting

| Issue | Solution |
|-------|----------|
| PyRadiomics hanging | Try downsampled version first; check image dimensions match mask |
| Import errors | Ensure virutal env is activated: `& .\radiomics_env\Scripts\Activate.ps1` |
| Out of memory | Use downsampled images or reduce image size |
| Shape extraction stuck | Very normal for large images; give it time |

## Thesis/Research Usage

This pipeline is designed for:
- ✓ Automated radiomics feature extraction
- ✓ Batch processing multiple images
- ✓ Machine learning model training
- ✓ Quantitative medical image analysis
- ✓ Clinical decision support systems

## Next Steps

After feature extraction:
1. Load CSV results: `pd.read_csv('output/mri_features.csv')`
2. Explore statistics: `df.describe()`
3. Visualize distributions
4. Train ML models using extracted features
5. Validate model performance

---

Generated: April 2, 2026
Pipeline Status: ✓ Testing Complete, Full Extraction Running
