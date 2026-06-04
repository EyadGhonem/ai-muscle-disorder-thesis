import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_mri_slice(mri_path, output_path, slice_index=None):
    """
    Visualize a slice from MRI volume and save as PNG
    """
    # Load the MRI volume
    try:
        img = sitk.ReadImage(mri_path)
        img_array = sitk.GetArrayFromImage(img)
        
        print(f"MRI shape: {img_array.shape}")
        print(f"Data type: {img_array.dtype}")
        print(f"Value range: {img_array.min()} to {img_array.max()}")
        
        # If no slice specified, take middle slice
        if slice_index is None:
            slice_index = img_array.shape[0] // 2
        
        # Normalize the slice for visualization
        slice_data = img_array[slice_index, :, :]
        
        # Normalize to 0-255 range for visualization
        if slice_data.max() > slice_data.min():
            slice_normalized = ((slice_data - slice_data.min()) / 
                              (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
        else:
            slice_normalized = np.zeros_like(slice_data, dtype=np.uint8)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.imshow(slice_normalized, cmap='gray')
        plt.title(f'MRI Slice {slice_index}\nFile: {os.path.basename(mri_path)}')
        plt.colorbar(label='Intensity')
        plt.axis('off')
        
        # Save as PNG
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {mri_path}: {e}")
        return False

def main():
    # Choose an MRI file to visualize
    mri_path = "c:/Users/Lenovo/Desktop/thesis_project/data/mri/raw/MRI_data/01/Thigh/In_phase.nii.gz"
    output_path = "c:/Users/Lenovo/Desktop/thesis_project/output/mri_visualization.png"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Visualize the MRI slice
    success = visualize_mri_slice(mri_path, output_path)
    
    if success:
        print("MRI visualization completed successfully!")
    else:
        print("Failed to create MRI visualization")

if __name__ == "__main__":
    main()
