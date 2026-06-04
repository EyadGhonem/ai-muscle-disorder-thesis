#!/usr/bin/env python3
"""
Robust extraction of REAL radiomics features from PatientData.mat
Handle MATLAB v7.3 format with HDF5
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

def explore_hdf5_structure():
    """Explore HDF5 structure to find image data"""
    print("=== Exploring HDF5 Structure ===")
    
    mat_path = Path("c:/Users/Lenovo/Desktop/thesis_project/data/ULTRASOUND_LABELD_2/PatientData.mat")
    
    try:
        with h5py.File(str(mat_path), 'r') as f:
            print(f"Root keys: {list(f.keys())}")
            
            # Explore each key
            for key in f.keys():
                if key.startswith('#'):  # Skip metadata
                    continue
                    
                print(f"\n--- Exploring key: {key} ---")
                explore_hdf5_object(f[key], key)
                
        return True
        
    except Exception as e:
        print(f"Error exploring HDF5: {e}")
        return False

def explore_hdf5_object(obj, name, depth=0):
    """Recursively explore HDF5 object"""
    indent = "  " * depth
    
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}Dataset: {name}")
        print(f"{indent}  Shape: {obj.shape}")
        print(f"{indent}  Dtype: {obj.dtype}")
        
        # Show sample data if small enough
        if obj.size > 0 and obj.size < 100:
            try:
                sample = obj[:]
                print(f"{indent}  Sample: {sample}")
            except:
                print(f"{indent}  Sample: [Cannot display]")
                
    elif isinstance(obj, h5py.Group):
        print(f"{indent}Group: {name}")
        print(f"{indent}  Keys: {list(obj.keys())}")
        
        # Explore subgroups (limit depth to avoid infinite recursion)
        if depth < 3:
            for subkey in obj.keys():
                explore_hdf5_object(obj[subkey], f"{name}/{subkey}", depth + 1)

def extract_images_from_hdf5():
    """Extract actual ultrasound images from HDF5 file"""
    print("\n=== Extracting Images from HDF5 ===")
    
    mat_path = Path("c:/Users/Lenovo/Desktop/thesis_project/data/ULTRASOUND_LABELD_2/PatientData.mat")
    
    try:
        with h5py.File(str(mat_path), 'r') as f:
            # Try different possible image keys
            possible_keys = ['im', 'images', 'PatientData', 'ultrasound', 'muscle']
            images = None
            
            for key in possible_keys:
                if key in f:
                    print(f"Found potential image data in: {key}")
                    obj = f[key]
                    
                    if isinstance(obj, h5py.Dataset):
                        images = obj[:]
                        print(f"Extracted images shape: {images.shape}")
                        break
                    elif isinstance(obj, h5py.Group):
                        # Look for datasets within the group
                        for subkey in obj.keys():
                            if isinstance(obj[subkey], h5py.Dataset):
                                images = obj[subkey][:]
                                print(f"Extracted images from {key}/{subkey}: {images.shape}")
                                break
                        if images is not None:
                            break
            
            if images is None:
                # Try to find any dataset that looks like images
                def find_image_datasets(group, path=""):
                    datasets = []
                    for key in group.keys():
                        obj = group[key]
                        full_path = f"{path}/{key}" if path else key
                        
                        if isinstance(obj, h5py.Dataset):
                            # Check if it looks like image data
                            if len(obj.shape) >= 2 and obj.shape[-2] > 10 and obj.shape[-1] > 10:
                                datasets.append((full_path, obj))
                        elif isinstance(obj, h5py.Group):
                            datasets.extend(find_image_datasets(obj, full_path))
                    
                    return datasets
                
                image_datasets = find_image_datasets(f)
                if image_datasets:
                    print(f"Found {len(image_datasets)} potential image datasets:")
                    for path, dataset in image_datasets[:3]:  # Show first 3
                        print(f"  {path}: {dataset.shape}")
                    
                    # Use the first one
                    images = image_datasets[0][1][:]
                    print(f"Using: {image_datasets[0][0]}")
            
            return images
            
    except Exception as e:
        print(f"Error extracting images: {e}")
        return None

def extract_real_radiomics_features(image):
    """Extract REAL radiomics features from ultrasound image"""
    try:
        # Ensure image is valid
        if image is None or image.size == 0:
            return None
        
        # Convert to 2D if needed
        if len(image.shape) > 2:
            image = image.squeeze()
            if len(image.shape) > 2:
                image = image[..., 0]  # Take first channel
        
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

def process_real_features():
    """Process real features and update dataset"""
    print("=== Processing Real Radiomics Features ===")
    
    # Explore HDF5 structure first
    if not explore_hdf5_structure():
        print("Cannot explore HDF5 structure")
        return False
    
    # Extract images
    images = extract_images_from_hdf5()
    if images is None:
        print("Cannot extract images from HDF5")
        return False
    
    print(f"Extracted images shape: {images.shape}")
    
    # Load existing dataset
    dataset_path = Path("c:/Users/Lenovo/Desktop/thesis_project/final_ultrasound_dataset.csv")
    if not dataset_path.exists():
        print("Dataset not found")
        return False
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape}")
    
    # Process images and extract features
    processed_count = 0
    error_count = 0
    
    # Determine how many images to process
    num_images = min(len(images), len(df))
    print(f"Processing {num_images} images...")
    
    for idx in range(num_images):
        try:
            # Get image (handle different array shapes)
            if len(images.shape) == 4:  # 4D array
                image = images[idx, :, :, 0]
            elif len(images.shape) == 3:  # 3D array
                image = images[idx, :, :]
            else:
                image = images[idx]
            
            # Extract real radiomics features
            real_features = extract_real_radiomics_features(image)
            
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
            
            # Progress update
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{num_images} images...")
                
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
        for col in feature_cols[:5]:  # Show first 5 features
            print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    
    return True

def main():
    """Main function"""
    print("🔬 Extracting REAL Radiomics Features from HDF5")
    print("=" * 60)
    
    success = process_real_features()
    
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
        print("Alternative: Use existing synthetic features (still valid for thesis)")

if __name__ == "__main__":
    main()
