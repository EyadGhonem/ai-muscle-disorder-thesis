#!/usr/bin/env python3
"""
Extract custom features from ultrasound images using OpenCV and numpy
Alternative to pyradiomics for ultrasound image analysis
"""

import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def extract_first_order_features(image, mask):
    """Extract first-order statistical features from masked region"""
    # Apply mask to image
    masked_image = image[mask > 0]
    
    if len(masked_image) == 0:
        return {}
    
    features = {
        'mean_intensity': np.mean(masked_image),
        'std_intensity': np.std(masked_image),
        'min_intensity': np.min(masked_image),
        'max_intensity': np.max(masked_image),
        'median_intensity': np.median(masked_image),
        'q25_intensity': np.percentile(masked_image, 25),
        'q75_intensity': np.percentile(masked_image, 75),
        'skewness': float(pd.Series(masked_image).skew()),
        'kurtosis': float(pd.Series(masked_image).kurtosis()),
        'entropy': -np.sum(np.histogram(masked_image, bins=256, density=True)[0] * 
                           np.log2(np.histogram(masked_image, bins=256, density=True)[0] + 1e-10))
    }
    
    return features

def extract_texture_features(image, mask):
    """Extract texture features using GLCM"""
    # Convert to uint8 for GLCM
    masked_image = image.copy()
    masked_image[mask == 0] = 0
    
    # Normalize to 0-255 range
    if masked_image.max() > 0:
        masked_image = cv2.normalize(masked_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Calculate GLCM
    try:
        # Reduce number of gray levels for computation efficiency
        levels = 32
        masked_image_reduced = (masked_image * (levels-1) // 255).astype(np.uint8)
        
        glcm = cv2.calcGLCM(masked_image_reduced, [1], [0, 45, 90, 135], levels=levels, symmetric=True, normed=True)
        
        features = {
            'glcm_contrast': np.mean(glcm[:, :, 0, 0]),  # Contrast
            'glcm_dissimilarity': np.mean(glcm[:, :, 0, 1]),  # Dissimilarity
            'glcm_homogeneity': np.mean(glcm[:, :, 0, 2]),  # Homogeneity
            'glcm_energy': np.mean(glcm[:, :, 0, 3]),  # Energy
            'glcm_correlation': np.mean(glcm[:, :, 0, 4]),  # Correlation
            'glcm_asm': np.mean(glcm[:, :, 0, 5])  # Angular Second Moment
        }
    except:
        # Fallback if GLCM fails
        features = {
            'glcm_contrast': 0,
            'glcm_dissimilarity': 0,
            'glcm_homogeneity': 0,
            'glcm_energy': 0,
            'glcm_correlation': 0,
            'glcm_asm': 0
        }
    
    return features

def extract_shape_features(mask):
    """Extract shape features from mask"""
    if np.sum(mask > 0) == 0:
        return {}
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {}
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate shape features
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
    else:
        circularity = 0
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h if h > 0 else 0
    extent = area / (w * h) if w * h > 0 else 0
    
    # Convex hull
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    features = {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'extent': extent,
        'solidity': solidity,
        'equivalent_diameter': 2 * np.sqrt(area / np.pi) if area > 0 else 0
    }
    
    return features

def extract_gradient_features(image, mask):
    """Extract gradient-based features"""
    # Apply mask
    masked_image = image.copy()
    masked_image[mask == 0] = 0
    
    if np.sum(masked_image) == 0:
        return {}
    
    # Calculate gradients
    grad_x = cv2.Sobel(masked_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(masked_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Apply mask to gradients
    grad_magnitude_masked = grad_magnitude[mask > 0]
    
    if len(grad_magnitude_masked) == 0:
        return {}
    
    features = {
        'gradient_mean': np.mean(grad_magnitude_masked),
        'gradient_std': np.std(grad_magnitude_masked),
        'gradient_max': np.max(grad_magnitude_masked),
        'gradient_energy': np.sum(grad_magnitude_masked**2) / len(grad_magnitude_masked)
    }
    
    return features

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

def extract_features_for_image(image_path, mask_path, image_info):
    """Extract all features for a single image"""
    try:
        # Load and preprocess
        image, mask = load_and_preprocess_image(image_path, mask_path)
        if image is None or mask is None:
            return None
        
        # Extract all feature types
        first_order = extract_first_order_features(image, mask)
        texture = extract_texture_features(image, mask)
        shape = extract_shape_features(mask)
        gradient = extract_gradient_features(image, mask)
        
        # Combine all features
        all_features = {}
        all_features.update(first_order)
        all_features.update(texture)
        all_features.update(shape)
        all_features.update(gradient)
        
        # Add metadata
        all_features.update({
            'filename': image_info['filename'],
            'subject': image_info['subject'],
            'muscle_code': image_info['muscle_code'],
            'side': image_info['side'],
            'instance': image_info['instance'],
            'binary_label': image_info['binary_label'],
            'grade_category': image_info['grade_category'],
            'original_grade': image_info['original_grade']
        })
        
        return all_features
        
    except Exception as e:
        print(f"Error extracting features for {image_info['filename']}: {e}")
        return None

def extract_custom_features():
    """Main function to extract custom features from all labeled images"""
    
    print("=== Custom Feature Extraction ===")
    
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
    
    # Extract features for each image
    all_features = []
    failed_extractions = 0
    
    print("Extracting custom features...")
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
        
        features = extract_features_for_image(image_path, mask_path, image_info)
        
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
    output_file = processed_dir / "custom_features.csv"
    features_df.to_csv(output_file, index=False)
    print(f"Saved features to: {output_file}")
    
    # Create feature summary
    summary_file = processed_dir / "custom_features_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"""Custom Feature Extraction Summary
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

Feature Categories:
- First-order statistics: Intensity-based features (mean, std, skewness, kurtosis, etc.)
- Texture features: GLCM-based texture descriptors (contrast, homogeneity, etc.)
- Shape features: Geometric properties of muscle region (area, circularity, etc.)
- Gradient features: Edge and gradient-based descriptors

Label Distribution:
- Normal/Mild (0): {(features_df['binary_label'] == 0).sum()} cases
- Moderate/Severe (1): {(features_df['binary_label'] == 1).sum()} cases

Files Generated:
- custom_features.csv: Main feature dataset
- custom_features_summary.txt: This summary file

Next Steps:
1. Split data into training/testing sets
2. Train machine learning models
3. Train deep learning models
4. Evaluate model performance
""")
    
    print(f"Saved summary to: {summary_file}")
    
    print(f"\n=== Feature Extraction Complete ===")
    return features_df

if __name__ == "__main__":
    features_df = extract_custom_features()
