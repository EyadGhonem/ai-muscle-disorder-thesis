"""
Make predictions on MRI images using trained 3D CNN classifier
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm

# Directories
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

# Model configuration
IMG_SIZE = (64, 64, 64)


def resolve_mri_dir():
    """Resolve MRI image directory from supported dataset layouts."""
    candidates = [
        DATA_DIR / "mri" / "raw" / "images",
        DATA_DIR / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

def load_trained_model(model_path):
    """Load trained MRI classifier"""
    try:
        model = keras.models.load_model(model_path)
        print(f"✓ Loaded MRI model: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def load_and_preprocess_mri(image_path, target_size=IMG_SIZE):
    """Load and preprocess MRI image"""
    try:
        # Load image
        img = sitk.ReadImage(str(image_path))
        img_array = sitk.GetArrayFromImage(img).astype(np.float32)
        
        # Normalize
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        
        # Resize
        zoom_factors = [target_size[i] / img_array.shape[i] for i in range(3)]
        img_resized = ndimage.zoom(img_array, zoom_factors, order=1)
        
        # Add channel dimension
        img_resized = img_resized[..., np.newaxis]
        
        return img_resized.astype(np.float32)
    except Exception as e:
        print(f"Error preprocessing {image_path.name}: {e}")
        return None

def predict_batch(model, image_dir):
    """Make predictions on all MRI images"""
    
    # Get all MRI files
    image_files = sorted(list(image_dir.glob("*.nii.gz")) + list(image_dir.glob("*.nii")))
    
    print(f"\n🔮 Making predictions on {len(image_files)} MRI images...\n")
    
    predictions = []
    
    for img_path in tqdm(image_files, desc="Predicting"):
        img = load_and_preprocess_mri(img_path)
        
        if img is not None:
            # Predict
            prediction = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
            
            # Round to 2 decimals
            conf = float(prediction)
            predicted_class = 1 if conf >= 0.5 else 0
            predicted_label = "diseased" if conf >= 0.5 else "healthy"
            
            predictions.append({
                'image_name': img_path.name,
                'model_score': conf,
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': max(conf, 1 - conf) * 100
            })
    
    return predictions

def save_predictions(predictions, output_path):
    """Save predictions to CSV"""
    df = pd.DataFrame(predictions)
    df = df.sort_values('model_score', ascending=False)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to {output_path}")
    return df

def print_prediction_summary(df):
    """Print summary of predictions"""
    
    print("\n" + "="*60)
    print("MRI PREDICTION SUMMARY")
    print("="*60)
    
    total = len(df)
    healthy = (df['predicted_class'] == 0).sum()
    diseased = (df['predicted_class'] == 1).sum()
    avg_confidence = df['confidence'].mean()
    
    print(f"\nTotal MRI scans: {total}")
    print(f"Healthy: {healthy} ({healthy/total*100:.1f}%)")
    print(f"Diseased: {diseased} ({diseased/total*100:.1f}%)")
    print(f"Average confidence: {avg_confidence:.1f}%")
    
    if len(df[df['predicted_class'] == 1]) > 0:
        print("\nMost confident DISEASED predictions:")
        diseased_preds = df[df['predicted_class'] == 1].head(5)
        for idx, row in diseased_preds.iterrows():
            print(f"  {row['image_name']}: {row['confidence']:.1f}%")
    
    if len(df[df['predicted_class'] == 0]) > 0:
        print("\nMost confident HEALTHY predictions:")
        healthy_preds = df[df['predicted_class'] == 0].head(5)
        for idx, row in healthy_preds.iterrows():
            print(f"  {row['image_name']}: {row['confidence']:.1f}%")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("="*60)
    print("MRI CLASSIFICATION PREDICTIONS")
    print("="*60 + "\n")
    
    mri_dir = resolve_mri_dir()
    print(f"Using MRI image directory: {mri_dir}")

    # Check model exists
    model_path = OUTPUT_DIR / "mri_classifier.keras"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Run train_mri_classifier.py first")
        exit(1)
    
    # Check images exist
    if not mri_dir.exists() or not any(mri_dir.glob("*.nii*")):
        print(f"❌ No MRI images in {mri_dir}")
        exit(1)
    
    # Load model
    model = load_trained_model(model_path)
    if model is None:
        exit(1)
    
    # Make predictions
    predictions = predict_batch(model, mri_dir)
    
    if predictions:
        # Save
        output_csv = OUTPUT_DIR / "mri_predictions.csv"
        df_pred = save_predictions(predictions, output_csv)
        
        # Summary
        print_prediction_summary(df_pred)
        
        print("\n✅ MRI predictions complete!")
    else:
        print("\n❌ No predictions generated")
