# Radiomics Feature Extraction Pipeline Setup Guide

## Overview
This directory contains Python scripts for the first step of an AI-powered radiomics pipeline for muscle disorder assessment.

## Project Structure

```
thesis_project/
├── extract_radiomics.py              # Main feature extraction script (start here)
├── extract_radiomics_advanced.py     # Advanced features (batch processing, custom settings)
├── data/
│   ├── images/                       # MRI images in NIfTI format (.nii.gz)
│   │   ├── case1.nii.gz
│   │   ├── case2.nii.gz
│   │   └── ...
│   └── masks/                        # Corresponding ROI masks
│       ├── case1_mask.nii.gz
│       ├── case2_mask.nii.gz
│       └── ...
└── output/                           # Output directory for CSV files
    ├── case1_features.csv
    ├── batch_features.csv
    └── ...
```

## Quick Start

### 1. Prepare Your Data
Create the required directory structure:
```powershell
# From thesis_project directory
mkdir -p data/images
mkdir -p data/masks
mkdir -p output
```

Place your MRI images and masks in the appropriate folders:
- Place `.nii.gz` or `.nii` files in `data/images/`
- Place corresponding mask files in `data/masks/`
- Ensure mask filenames match: `{case_name}_mask.nii.gz`

### 2. Run Basic Feature Extraction
```powershell
# Activate environment
& .\radiomics_env\Scripts\Activate.ps1

# Run the main extraction script
python extract_radiomics.py
```

Output will include:
- Sample features printed to console
- CSV file with all features saved to `output/case1_features.csv`

### 3. Run Advanced Batch Processing
```powershell
# Process multiple cases at once
python extract_radiomics_advanced.py
```

This will process all images in `data/images/` and create a combined CSV file.

## Features Extracted by PyRadiomics

The extraction includes features from multiple categories:

### 1. **Shape Features**
- Volume, surface area, compactness, sphericity
- Geometric properties of the ROI

### 2. **First-order Statistics**
- Mean, median, std, skewness, kurtosis
- Histogram-based texture measures

### 3. **Texture Features (GLCM)**
- Gray Level Co-occurrence Matrix
- Contrast, correlation, energy, homogeneity

### 4. **Texture Features (GLRLM)**
- Gray Level Run Length Matrix
- Short-run emphasis, long-run emphasis

### 5. **Texture Features (GLSZM)**
- Gray Level Size Zone Matrix
- Zone-related texture properties

## Using the Scripts

### Basic Script: `extract_radiomics.py`
- Load a single image and mask
- Extract radiomics features
- Display sample features
- Save to CSV
- **Best for**: Testing with individual cases

**Usage:**
```python
python extract_radiomics.py
```

**Customization:**
Edit the file paths in the `main()` function:
```python
image_path = "data/images/your_image.nii.gz"
mask_path = "data/masks/your_mask.nii.gz"
output_csv = "output/your_features.csv"
```

### Advanced Script: `extract_radiomics_advanced.py`
- Batch process multiple cases
- Custom feature extraction settings
- Feature analysis and statistics
- **Best for**: Processing multiple patients

**Key functions:**
```python
# Extract with custom settings
features = extract_with_custom_settings(image_path, mask_path)

# Batch process all images
df = batch_extract_features("data/images", "data/masks", "output/batch_features.csv")

# Analyze extracted features
df_analysis = analyze_features("output/batch_features.csv")
```

## Understanding the Output

### Features CSV File
Each row = one patient case
Each column = one radiomics feature

Example columns:
- `case_id`: Patient identifier
- `original_shape_Volume`: Tumor volume
- `original_firstorder_Mean`: Mean intensity
- `original_glcm_Correlation`: Texture correlation
- `original_glrlm_ShortRunEmphasis`: Run-length texture
- ... (150+ features total)

### Feature Names Format
`{imageType}_{featureClass}_{featureName}`

Examples:
- `original_shape_Volume` - Original image shape
- `wavelet_firstorder_Mean` - Wavelet transformed statistics
- `original_glcm_Correlation` - GLCM texture feature

## Common Tasks

### Filter out diagnostic information
```python
features = {k: v for k, v in features.items() 
           if not (k.startswith('diagnostics_') or k.startswith('setting_'))}
```

### Convert features to DataFrame
```python
df = pd.DataFrame([features])  # Single case
df = pd.concat([pd.DataFrame([f]) for f in features_list])  # Multiple cases
```

### Access specific feature types
```python
# Get only shape features
shape_features = {k: v for k, v in features.items() if 'shape' in k}

# Get only texture features
texture_features = {k: v for k, v in features.items() if 'glcm' in k or 'glrlm' in k}
```

## Troubleshooting

### Import Error: No module named 'radiomics'
```powershell
# Activate the correct virtual environment
& .\radiomics_env\Scripts\Activate.ps1

# Verify packages
pip list | grep radiomics
```

### File not found error
- Check that image and mask paths are correct
- Verify files are in NIfTI format (.nii or .nii.gz)
- Ensure image and mask are registered (same spatial dimensions)

### Memory issues with large images
- Try resampling images to lower resolution
- Process one case at a time
- Check available RAM

### Mask validation errors
- Ensure mask contains binary values (0 and 1 or >0)
- Verify mask and image have same dimensions
- Mask should contain the ROI as non-zero values

## Next Steps

After extracting features, the pipeline continues with:
1. **Feature Selection**: Reduce dimensionality
2. **Normalization**: Scale features appropriately
3. **Classification/Regression**: Apply ML models
4. **Validation**: Cross-validation and testing

## References

- **PyRadiomics Documentation**: https://pyradiomics.readthedocs.io/
- **Feature Definitions**: https://pyradiomics.readthedocs.io/en/latest/features.html
- **Radiomics Literature**: https://www.radiomics.io/

## Requirements

- Python 3.9.13
- pyradiomics >= 3.0
- SimpleITK >= 2.5
- pandas >= 1.0
- numpy, scipy, scikit-learn

Check installation:
```powershell
& .\radiomics_env\Scripts\Activate.ps1
pip list
```
