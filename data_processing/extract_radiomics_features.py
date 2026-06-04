#!/usr/bin/env python3
"""
Extract radiomics features from ultrasound images using masks
"""

import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import radiomics
from radiomics import featureextractor
import warnings
from tqdm import tqdm
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('radiomics').setLevel(logging.CRITICAL)

def setup_radiomics_extractor():
    """Setup radiomics feature extractor with optimized settings for ultrasound"""
    settings = {
        'binWidth': 25,
        'interpolator': 'sitkBSpline',
        'resampledPixelSpacing': None,
        'padDistance': 10,
        'distances': [1],
        'force2D': True,
        'force2Ddimension': 0,
        'gldm_a': 0,
        'gldm_b': 0
    }
    
    # Enable feature classes
    enabled_features = {
        'firstorder': None,  # All first-order features
        'glcm': None,        # All GLCM features
        'glrlm': None,       # All GLRLM features
        'glszm': None,       # All GLSZM features
        'gldm': None,        # All GLDM features
        'ngtdm': None        # All NGTDM features
    }
    
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableFeatures(enabled_features)
    
    return extractor

def load_and_preprocess_image(image_path, mask_path):
    """Load and preprocess ultrasound image and mask"""
    try:
        # Load image and mask
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            return None, None
        
        # Ensure mask is binary (0 and 255)
        mask = (mask > 127).astype(np.uint8) * 255
        
        # Check if mask has valid region
        if np.sum(mask > 0) == 0:
            return None, None
        
        # Normalize image to [0, 255] range
        if image.max() > 0:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        return image, mask
        
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None, None

def extract_features_for_image(extractor, image_path, mask_path, image_info):
    """Extract radiomics features for a single image"""
    try:
        # Load and preprocess
        image, mask = load_and_preprocess_image(image_path, mask_path)
        if image is None or mask is None:
            return None
        
        # Save temporary files for radiomics (SimpleITK format)
        temp_dir = Path("temp_radiomics")
        temp_dir.mkdir(exist_ok=True)
        
        temp_image_path = temp_dir / f"temp_image_{image_info['filename']}.png"
        temp_mask_path = temp_dir / f"temp_mask_{image_info['filename']}.png"
        
        cv2.imwrite(str(temp_image_path), image)
        cv2.imwrite(str(temp_mask_path), mask)
        
        # Extract features
        features = extractor.execute(str(temp_image_path), str(temp_mask_path))
        
        # Clean up temp files
        temp_image_path.unlink(missing_ok=True)
        temp_mask_path.unlink(missing_ok=True)
        
        # Clean up feature dict and add metadata
        clean_features = {}
        for key, value in features.items():
            # Remove 'original_' prefix and clean key names
            clean_key = key.replace('original_', '').replace('_', ' ').title()
            if not any(x in clean_key for x in ['General', 'Image', 'Dimension']):
                clean_features[clean_key] = value
        
        # Add metadata
        clean_features.update({
            'filename': image_info['filename'],
            'subject': image_info['subject'],
            'muscle_code': image_info['muscle_code'],
            'side': image_info['side'],
            'instance': image_info['instance'],
            'binary_label': image_info['binary_label'],
            'grade_category': image_info['grade_category'],
            'original_grade': image_info['original_grade']
        })
        
        return clean_features
        
    except Exception as e:
        print(f"Error extracting features for {image_info['filename']}: {e}")
        return None

def extract_radiomics_features():
    """Main function to extract radiomics features from all labeled images"""
    
    print("=== Radiomics Feature Extraction ===")
    
    # Setup paths
    data_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/data/final_ultrasound_labeled")
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    processed_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/processed_data")
    
    # Load image-label mapping
    mapping_file = processed_dir / "image_label_mapping.csv"
    if not mapping_file.exists():
        raise FileNotFoundError(f"Image label mapping not found: {mapping_file}")
    
    print("Loading image-label mapping...")
    image_mapping = pd.read_csv(mapping_file)
    print(f"Found {len(image_mapping)} labeled images")
    
    # Setup radiomics extractor
    print("Setting up radiomics extractor...")
    extractor = setup_radiomics_extractor()
    
    # Extract features for each image
    all_features = []
    failed_extractions = 0
    
    print("Extracting radiomics features...")
    for idx, row in tqdm(image_mapping.iterrows(), total=len(image_mapping), desc="Processing images"):
        
        # Construct file paths
        image_path = images_dir / f"{row['filename']}.png"
        mask_path = masks_dir / f"{row['filename']}.png"
        
        # Check if files exist
        if not image_path.exists() or not mask_path.exists():
            failed_extractions += 1
            continue
        
        # Extract features
        image_info = {
            'filename': row['filename'],
            'subject': row['subject'],
            'muscle_code': row['muscle_code'],
            'side': row['side'],
            'instance': row['instance'],
            'binary_label': row['binary_label'],
            'grade_category': row['grade_category'],
            'original_grade': row['original_grade']
        }
        
        features = extract_features_for_image(extractor, image_path, mask_path, image_info)
        
        if features is not None:
            all_features.append(features)
        else:
            failed_extractions += 1
    
    print(f"\nFeature extraction completed!")
    print(f"Successfully processed: {len(all_features)} images")
    print(f"Failed extractions: {failed_extractions} images")
    
    if len(all_features) == 0:
        raise ValueError("No features were successfully extracted!")
    
    # Create features dataframe
    features_df = pd.DataFrame(all_features)
    
    # Remove any remaining non-numeric columns (except metadata)
    metadata_cols = ['filename', 'subject', 'muscle_code', 'side', 'instance', 'binary_label', 'grade_category', 'original_grade']
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    # Convert feature columns to numeric, replacing any remaining non-numeric values with NaN
    for col in feature_cols:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
    
    # Remove features with too many NaN values
    nan_threshold = 0.1  # Remove features with >10% NaN values
    valid_features = []
    for col in feature_cols:
        nan_ratio = features_df[col].isna().sum() / len(features_df)
        if nan_ratio <= nan_threshold:
            valid_features.append(col)
    
    print(f"\nFeature filtering:")
    print(f"Total features extracted: {len(feature_cols)}")
    print(f"Features after NaN filtering: {len(valid_features)}")
    
    # Final feature selection
    final_cols = metadata_cols + valid_features
    features_df = features_df[final_cols]
    
    # Fill remaining NaN values with median
    for col in valid_features:
        if features_df[col].isna().any():
            median_val = features_df[col].median()
            features_df[col].fillna(median_val, inplace=True)
    
    # Save features
    output_file = processed_dir / "radiomics_features.csv"
    features_df.to_csv(output_file, index=False)
    print(f"Saved features to: {output_file}")
    
    # Create feature summary
    summary_file = processed_dir / "radiomics_features_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"""Radiomics Feature Extraction Summary
=====================================

Dataset Information:
- Total labeled images: {len(image_mapping)}
- Successfully processed: {len(all_features)}
- Failed extractions: {failed_extractions}
- Success rate: {len(all_features)/len(image_mapping)*100:.1f}%

Feature Statistics:
- Total features extracted: {len(feature_cols)}
- Features after filtering: {len(valid_features)}
- Features per image: {len(valid_features)}

Label Distribution:
- Normal/Mild (0): {(features_df['binary_label'] == 0).sum()} cases
- Moderate/Severe (1): {(features_df['binary_label'] == 1).sum()} cases

Feature Categories:
- First-order statistics: Intensity-based features
- GLCM: Gray-Level Co-occurrence Matrix features
- GLRLM: Gray-Level Run Length Matrix features
- GLSZM: Gray-Level Size Zone Matrix features
- GLDM: Gray-Level Dependence Matrix features
- NGTDM: Neighboring Gray Tone Difference Matrix features

Files Generated:
- radiomics_features.csv: Main feature dataset
- radiomics_features_summary.txt: This summary file

Next Steps:
1. Split data into training/testing sets
2. Train machine learning models
3. Evaluate model performance
""")
    
    print(f"Saved summary to: {summary_file}")
    
    # Clean up temp directory
    temp_dir = Path("temp_radiomics")
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
    
    print(f"\n=== Feature Extraction Complete ===")
    return features_df

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import radiomics
    except ImportError:
        print("Installing radiomics package...")
        os.system("pip install pyradiomics")
        import radiomics
    
    features_df = extract_radiomics_features()
