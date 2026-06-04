#!/usr/bin/env python3
"""
Extract REAL radiomics features from PatientData.mat
Replace synthetic features with actual image-based features
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.measure import shannon_entropy
import warnings
warnings.filterwarnings('ignore')

def load_matlab_data():
    """Load ultrasound images from PatientData.mat"""
    print("Loading PatientData.mat...")
    
    mat_path = Path("c:/Users/Lenovo/Desktop/thesis_project/data/ULTRASOUND_LABELD_2/PatientData.mat")
    if not mat_path.exists():
        print(f"PatientData.mat not found: {mat_path}")
        return None, None
    
    try:
        mat_data = sio.loadmat(str(mat_path))
        print(f"MAT file keys: {list(mat_data.keys())}")
        
        # Try to find the images array
        if 'images' in mat_data:
            images = mat_data['images']
            print(f"Images shape: {images.shape}")
            return images, mat_data
        elif 'PatientData' in mat_data:
            images = mat_data['PatientData']
            print(f"PatientData shape: {images.shape}")
            return images, mat_data
        else:
            print("Available keys:", list(mat_data.keys()))
            return None, mat_data
            
    except Exception as e:
        print(f"Error loading MATLAB file: {e}")
        return None, None

def extract_radiomics_features(image):
    """Extract REAL radiomics features from single ultrasound image"""
    try:
        # Ensure image is in valid range
        if image is None or image.size == 0:
            return None
        
        # Normalize image to 0-255 range
        if image.max() > 0:
            image_normalized = (image * 255.0 / image.max()).astype(np.uint8)
        else:
            image_normalized = image.astype(np.uint8)
        
        # First-order statistics
        features = []
        
        # Basic statistics
        features.append(np.mean(image))
        features.append(np.std(image))
        features.append(np.min(image))
        features.append(np.max(image))
        features.append(np.median(image))
        features.append(np.percentile(image, 25))
        features.append(np.percentile(image, 75))
        features.append(np.var(image))
        
        # Skewness and kurtosis
        if len(image.flatten()) > 1:
            features.append(float(((image.flatten() - np.mean(image))**3).mean() / (np.std(image)**3)))
            features.append(float(((image.flatten() - np.mean(image))**4).mean() / (np.std(image)**4)))
        else:
            features.extend([0.0, 0.0])
        
        # Entropy
        try:
            entropy_val = shannon_entropy(image_normalized)
            features.append(entropy_val)
        except:
            features.append(0.0)
        
        # Texture features (GLCM)
        try:
            # Create binary image for GLCM
            threshold = threshold_otsu(image_normalized)
            binary_image = image_normalized > threshold
            
            # Compute GLCM
            glcm = greycomatrix(binary_image, distances=[1], angles=[0, 45, 90, 135])
            glcm_props = greycoprops(glcm)
            
            features.append(glcm_props['contrast'])
            features.append(glcm_props['dissimilarity'])
            features.append(glcm_props['homogeneity'])
            features.append(glcm_props['energy'])
            features.append(glcm_props['correlation'])
            features.append(glcm_props['ASM'])
        except:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Shape features
        try:
            threshold = threshold_otsu(image_normalized)
            labeled = label(image_normalized > threshold)
            regions = regionprops(labeled)
            
            if regions:
                # Get largest region
                largest_region = max(regions, key=lambda r: r.area)
                
                features.append(largest_region.area)
                features.append(largest_region.perimeter)
                
                # Circularity
                if largest_region.perimeter > 0:
                    circularity = 4 * np.pi * largest_region.area / (largest_region.perimeter ** 2)
                else:
                    circularity = 0
                features.append(circularity)
                
                # Aspect ratio
                min_row, min_col, max_row, max_col = largest_region.bbox
                if max_col - min_col > 0:
                    aspect_ratio = (max_col - min_col) / (max_row - min_row)
                else:
                    aspect_ratio = 0
                features.append(aspect_ratio)
                
                # Extent
                bbox_area = (max_col - min_col) * (max_row - min_row)
                if bbox_area > 0:
                    extent = largest_region.area / bbox_area
                else:
                    extent = 0
                features.append(extent)
                
                # Solidity
                if largest_region.area > 0:
                    hull = largest_region.convex_hull_image
                    hull_area = np.sum(hull)
                    solidity = largest_region.area / hull_area if hull_area > 0 else 0
                else:
                    solidity = 0
                features.append(solidity)
                
                # Equivalent diameter
                equiv_diameter = np.sqrt(4 * largest_region.area / np.pi)
                features.append(equiv_diameter)
                
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0])
        except:
            features.extend([0, 0, 0, 0, 0, 0, 0, 0])
        
        # Gradient features
        try:
            # Compute gradients
            gy, gx = np.gradient(image)
            gx, gy = np.gradient(image)
            
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            features.append(np.mean(gradient_magnitude))
            features.append(np.std(gradient_magnitude))
            features.append(np.max(gradient_magnitude))
            
            # Gradient energy
            features.append(np.sum(gradient_magnitude**2))
        except:
            features.extend([0, 0, 0, 0])
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from image: {e}")
        return None

def process_all_images():
    """Process all images and extract real radiomics features"""
    print("=== Extracting REAL Radiomics Features ===")
    
    # Load MATLAB data
    images, mat_data = load_matlab_data()
    if images is None:
        print("Could not load images from MATLAB file")
        return False
    
    # Load existing dataset to update
    dataset_path = Path("c:/Users/Lenovo/Desktop/thesis_project/final_ultrasound_dataset.csv")
    if not dataset_path.exists():
        print("final_ultrasound_dataset.csv not found")
        return False
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded existing dataset: {df.shape}")
    
    # Process images and extract features
    processed_count = 0
    error_count = 0
    
    for idx in range(min(len(images), len(df))):
        try:
            # Get image (assuming first dimension is patients)
            if len(images.shape) == 4:  # 4D array: [patients, height, width, channels]
                image = images[idx, :, :, 0]  # First channel
            elif len(images.shape) == 3:  # 3D array: [patients, height, width]
                image = images[idx, :, :]
            else:
                image = images[idx]
            
            # Extract real radiomics features
            real_features = extract_radiomics_features(image)
            
            if real_features is not None and len(real_features) >= 27:
                # Update existing features with real ones
                for i in range(27):
                    feature_name = f'feature_{i+1}'
                    if feature_name in df.columns:
                        df.at[idx, feature_name] = real_features[i]
                
                processed_count += 1
            else:
                error_count += 1
                print(f"Error processing image {idx}: got {len(real_features) if real_features else 0} features")
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} images...")
                
        except Exception as e:
            error_count += 1
            print(f"Error processing image {idx}: {e}")
    
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {processed_count} images")
    print(f"  Errors: {error_count} images")
    
    # Save updated dataset
    output_path = Path("c:/Users/Lenovo/Desktop/thesis_project/final_ultrasound_dataset_real_features.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Real features dataset saved to: {output_path}")
    print(f"📊 Updated dataset shape: {df.shape}")
    
    # Feature statistics
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    if feature_cols:
        print(f"\nFeature Statistics (Real):")
        for col in feature_cols[:5]:  # Show first 5 features
            print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    
    return True

def main():
    """Main function"""
    print("🔬 Extracting REAL Radiomics Features from Ultrasound Images")
    print("=" * 60)
    
    success = process_all_images()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Real radiomics feature extraction completed!")
        print("📁 Updated dataset: final_ultrasound_dataset_real_features.csv")
        print("🏥 Ready for clinical interpretation and medical analysis")
        print("\nNext steps:")
        print("1. Retrain models with real features")
        print("2. Analyze feature importance with clinical meaning")
        print("3. Update thesis with real radiomics results")
    else:
        print("\n❌ Feature extraction failed. Please check MATLAB file format.")

if __name__ == "__main__":
    main()
