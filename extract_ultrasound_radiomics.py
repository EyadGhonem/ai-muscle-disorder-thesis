"""
Extract radiomics features from ultrasound images
Adapted from extract_full.py for 2D ultrasound slices
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from radiomics import featureextractor
import SimpleITK as sitk
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("radiomics").setLevel(logging.WARNING)

# Directories
DATA_DIR = Path("data")
ULTRASOUND_DIR = DATA_DIR / "ultrasound_images"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

def setup_radiomics_extractor():
    """Initialize PyRadiomics feature extractor"""
    settings = {
        "force2D": True,
        "force2Ddimension": 0,
        "label": 1,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    logger.info("PyRadiomics extractor initialized (2D mode)")
    return extractor

def load_ultrasound_image(image_path):
    """Load single ultrasound image as numpy array"""
    try:
        # Try OpenCV first
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback to PIL
            from PIL import Image
            img = np.array(Image.open(image_path).convert('L'))
        
        return img
    except Exception as e:
        logger.warning(f"Error loading {image_path}: {e}")
        return None

def create_binary_mask(image_array):
    """
    Create binary mask of muscle region
    Thresholding to separate muscle from background
    """
    # Apply threshold to separate tissue from noise/background
    _, mask = cv2.threshold(image_array, image_array.mean() * 0.5, 1, cv2.THRESH_BINARY)
    
    # Clean up mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask.astype(np.uint8)

def numpy_to_sitk(image_array, mask_array):
    """Convert numpy arrays to SimpleITK format for PyRadiomics"""
    # Image must be float32
    image = sitk.GetImageFromArray(image_array.astype(np.float32))
    mask = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    
    return image, mask

def extract_features_from_image(extractor, image_path):
    """Extract radiomics features from single ultrasound image"""
    try:
        # Load image
        img_array = load_ultrasound_image(image_path)
        if img_array is None:
            return None
        
        # Normalize intensity
        img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
        
        # Create binary mask directly in 2D.
        mask_2d = create_binary_mask(img_array)
        if int(mask_2d.sum()) == 0:
            logger.warning(f"Empty mask for {image_path.name}, skipping")
            return None

        # Convert to SimpleITK (2D)
        image_sitk, mask_sitk = numpy_to_sitk(img_array, mask_2d)
        
        # Extract features
        features = extractor.execute(image_sitk, mask_sitk)
        
        # Extract only radiomics features (skip diagnostics)
        radiomics_features = {
            key: float(value) 
            for key, value in features.items() 
            if key.startswith('original_')
        }
        
        return radiomics_features
    
    except Exception as e:
        logger.error(f"Error extracting features from {image_path.name}: {e}")
        return None

def extract_all_features(extractor, image_dir):
    """Extract features from all ultrasound images"""
    
    # Get all TIF files
    image_files = (
        list(image_dir.glob("*.tif"))
        + list(image_dir.glob("*.TIF"))
        + list(image_dir.glob("*.tiff"))
        + list(image_dir.glob("*.TIFF"))
    )
    # Deduplicate on lowercase filename; avoids duplicate processing on
    # case-insensitive file systems (e.g., Windows).
    unique_files = {}
    for file_path in image_files:
        unique_files[file_path.name.lower()] = file_path
    image_files = sorted(unique_files.values(), key=lambda p: p.name.lower())
    
    logger.info(f"Found {len(image_files)} ultrasound images")
    
    all_features = []
    failed_images = []
    
    for i, image_path in enumerate(tqdm(image_files, desc="Extracting features")):
        features = extract_features_from_image(extractor, image_path)
        
        if features is not None:
            features['image_name'] = image_path.name
            all_features.append(features)
        else:
            failed_images.append(image_path.name)
    
    logger.info(f"\n✓ Successfully extracted: {len(all_features)}/{len(image_files)} images")
    
    if failed_images:
        logger.warning(f"⚠️  Failed: {len(failed_images)} images")
        for name in failed_images[:5]:
            logger.warning(f"   - {name}")
        if len(failed_images) > 5:
            logger.warning(f"   ... and {len(failed_images) - 5} more")
    
    return all_features

def save_features_to_csv(features_list, output_path):
    """Save extracted features to CSV"""
    df = pd.DataFrame(features_list)
    
    # Move image_name to first column
    cols = df.columns.tolist()
    cols = ['image_name'] + [col for col in cols if col != 'image_name']
    df = df[cols]
    
    df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Features saved to {output_path}")
    logger.info(f"   Shape: {df.shape[0]} images × {df.shape[1]} features")
    
    return df

def print_feature_summary(df):
    """Print summary statistics of extracted features"""
    print("\n" + "="*60)
    print("ULTRASOUND RADIOMICS EXTRACTION SUMMARY")
    print("="*60)
    print(f"\nDataset: {len(df)} images")
    print(f"Features: {len(df.columns) - 1} radiomics features")
    
    # Feature categories
    feature_categories = {}
    for col in df.columns:
        if col != 'image_name':
            # Parse feature name: original_<category>_<name>
            parts = col.split('_')
            if len(parts) >= 2:
                category = parts[1]  # e.g., 'shape', 'firstorder', 'glcm'
                feature_categories[category] = feature_categories.get(category, 0) + 1
    
    print("\nFeature breakdown:")
    for category in sorted(feature_categories.keys()):
        count = feature_categories[category]
        print(f"  - {category}: {count} features")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("🔬 Extracting radiomics features from ultrasound images...\n")
    
    # Check if images exist
    if not ULTRASOUND_DIR.exists() or len(list(ULTRASOUND_DIR.glob("*"))) == 0:
        print(f"❌ No images found in {ULTRASOUND_DIR}")
        print("Run setup_ultrasound_data.py first")
        exit(1)
    
    # Setup extractor
    extractor = setup_radiomics_extractor()
    
    # Extract features
    features_list = extract_all_features(extractor, ULTRASOUND_DIR)
    
    if features_list:
        # Save to CSV
        output_csv = OUTPUT_DIR / "ultrasound_radiomics_features.csv"
        df = save_features_to_csv(features_list, output_csv)
        
        # Print summary
        print_feature_summary(df)
        
        print("\n✅ Extraction complete!")
    else:
        print("\n❌ No features extracted")
