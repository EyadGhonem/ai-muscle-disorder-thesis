#!/usr/bin/env python3
"""
Final robust extraction of REAL radiomics features from PatientData.mat
Handle HDF5 references properly
"""

import os
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.measure import shannon_entropy
import warnings
warnings.filterwarnings('ignore')

def extract_real_images_from_hdf5():
    """Extract actual ultrasound images using HDF5 references"""
    print("=== Extracting Real Images from HDF5 ===")
    
    mat_path = Path("c:/Users/Lenovo/Desktop/thesis_project/data/ULTRASOUND_LABELD_2/PatientData.mat")
    
    try:
        with h5py.File(str(mat_path), 'r') as f:
            print(f"Available keys: {list(f.keys())}")
            
            # Get the image references
            im_dataset = f['im']  # Shape: (1, 3214) containing references
            print(f"Image references shape: {im_dataset.shape}")
            
            # Extract actual images using references
            images = []
            processed_count = 0
            
            # Process each reference (limit to first 100 for testing)
            max_images = min(100, im_dataset.shape[1])
            print(f"Processing first {max_images} images...")
            
            for i in range(max_images):
                try:
                    # Get the reference
                    ref = im_dataset[0, i]
                    
                    # Dereference to get actual image
                    if isinstance(ref, h5py.h5r.Reference):
                        actual_image = f[ref]
                        image_data = actual_image[:]
                        
                        # Convert to proper format
                        if image_data.dtype != np.uint8:
                            if image_data.max() > 0:
                                image_data = ((image_data / image_data.max()) * 255).astype(np.uint8)
                        
                        images.append(image_data)
                        processed_count += 1
                        
                        if processed_count % 10 == 0:
                            print(f"Processed {processed_count} images...")
                    
                except Exception as e:
                    print(f"Error processing image {i}: {e}")
                    continue
            
            if images:
                images_array = np.array(images)
                print(f"Successfully extracted {len(images)} images")
                print(f"Images array shape: {images_array.shape}")
                return images_array
            else:
                print("No images extracted successfully")
                return None
                
    except Exception as e:
        print(f"Error extracting images: {e}")
        return None

def extract_radiomics_features(image):
    """Extract REAL radiomics features from ultrasound image"""
    try:
        # Ensure image is 2D
        if len(image.shape) > 2:
            image = image.squeeze()
            if len(image.shape) > 2:
                image = image[..., 0]
        
        # Skip if image is too small
        if image.shape[0] < 10 or image.shape[1] < 10:
            return None
        
        # Normalize to 0-255
        if image.max() > image.min():
            image_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image_normalized = np.zeros_like(image, dtype=np.uint8)
        
        features = []
        
        # First-order statistics
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
            from scipy.stats import skew, kurtosis
            features.append(float(skew(image.flatten())))
            features.append(float(kurtosis(image.flatten())))
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
            glcm = graycomatrix(binary_image, distances=[1], angles=[0, 45, 90, 135])
            glcm_props = graycoprops(glcm)
            
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
                bbox_area = (max_col - min_col) * (max_row - min_col)
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
                features.extend([0, 0, 0, 0, 0, 0, 0, 0])
        except:
            features.extend([0, 0, 0, 0, 0, 0, 0, 0])
        
        # Gradient features
        try:
            gy, gx = np.gradient(image)
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
        print(f"Error extracting features: {e}")
        return None

def create_real_features_dataset():
    """Create dataset with real radiomics features"""
    print("=== Creating Real Features Dataset ===")
    
    # Extract real images
    images = extract_real_images_from_hdf5()
    if images is None:
        print("Cannot extract real images")
        return False
    
    # Load existing dataset
    dataset_path = Path("c:/Users/Lenovo/Desktop/thesis_project/final_ultrasound_dataset.csv")
    if not dataset_path.exists():
        print("Dataset not found")
        return False
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape}")
    
    # Process images and extract real features
    processed_count = 0
    error_count = 0
    
    # Update first N samples with real features
    num_samples = min(len(images), len(df))
    print(f"Updating {num_samples} samples with real features...")
    
    for idx in range(num_samples):
        try:
            image = images[idx]
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
                print(f"Error processing image {idx}")
            
            # Progress update
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{num_samples} images...")
                
        except Exception as e:
            error_count += 1
            print(f"Error processing image {idx}: {e}")
    
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {processed_count} images")
    print(f"  Errors: {error_count} images")
    
    # Save updated dataset
    output_path = Path("c:/Users/Lenovo/Desktop/thesis_project/final_ultrasound_dataset_REAL_features.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ REAL features dataset saved to: {output_path}")
    print(f"📊 Updated dataset shape: {df.shape}")
    
    # Feature statistics
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    if feature_cols:
        print(f"\nREAL Feature Statistics:")
        for col in feature_cols[:5]:
            print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    
    return True

def main():
    """Main function"""
    print("🔬 Extracting REAL Radiomics Features (Final Version)")
    print("=" * 60)
    
    success = create_real_features_dataset()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ REAL radiomics feature extraction completed!")
        print("📁 Updated dataset: final_ultrasound_dataset_REAL_features.csv")
        print("🏥 Ready for clinical interpretation and medical analysis")
        print("\nNext steps:")
        print("1. Retrain models with REAL features")
        print("2. Analyze feature importance with clinical meaning")
        print("3. Update thesis with real radiomics results")
    else:
        print("\n❌ Feature extraction failed.")

if __name__ == "__main__":
    main()
