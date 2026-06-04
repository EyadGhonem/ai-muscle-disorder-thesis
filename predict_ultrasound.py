"""
Make predictions on new ultrasound images using trained classifier
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm

# Directories
DATA_DIR = Path("data")
ULTRASOUND_DIR = DATA_DIR / "ultrasound_images"
OUTPUT_DIR = Path("output")

# Model configuration
IMG_SIZE = (224, 224)

def load_trained_model(model_path):
    """Load trained classifier model"""
    try:
        model = keras.models.load_model(model_path)
        print(f"✓ Loaded model: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def load_and_preprocess_image(image_path, target_size=IMG_SIZE):
    """Load and preprocess single image"""
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            from PIL import Image
            img = Image.open(image_path)
            img = np.array(img)
        
        # Convert to RGB if grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        return img
    except Exception as e:
        print(f"Error preprocessing {image_path.name}: {e}")
        return None

def predict_single_image(model, image_path):
    """Get prediction for single image"""
    img = load_and_preprocess_image(image_path)
    if img is None:
        return None
    
    # Predict
    prediction = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
    
    return prediction

def predict_batch(model, image_dir):
    """Make predictions on all images"""
    
    # Get all image files
    image_extensions = ['*.tif', '*.TIF', '*.tiff', '*.TIFF', '*.png', '*.PNG', '*.jpg', '*.JPG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(sorted(image_dir.glob(ext)))
    
    print(f"\n🔮 Making predictions on {len(image_files)} images...\n")
    
    predictions = []
    
    for img_path in tqdm(image_files, desc="Predicting"):
        prediction = predict_single_image(model, img_path)
        
        if prediction is not None:
            # Round to 2 decimals and create label
            conf = float(prediction)
            predicted_class = 1 if conf >= 0.5 else 0
            predicted_label = "diseased" if conf >= 0.5 else "healthy"
            
            predictions.append({
                'image_name': img_path.name,
                'model_score': conf,
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': max(conf, 1 - conf) * 100  # Confidence percentage
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
    print("PREDICTION SUMMARY")
    print("="*60)
    
    total = len(df)
    healthy = (df['predicted_class'] == 0).sum()
    diseased = (df['predicted_class'] == 1).sum()
    avg_confidence = df['confidence'].mean()
    
    print(f"\nTotal images: {total}")
    print(f"Healthy: {healthy} ({healthy/total*100:.1f}%)")
    print(f"Diseased: {diseased} ({diseased/total*100:.1f}%)")
    print(f"Average confidence: {avg_confidence:.1f}%")
    
    print("\nTop 10 most confident DISEASED predictions:")
    diseased_preds = df[df['predicted_class'] == 1].head(10)
    for idx, row in diseased_preds.iterrows():
        print(f"  {row['image_name']}: {row['confidence']:.1f}%")
    
    print("\nTop 10 most confident HEALTHY predictions:")
    healthy_preds = df[df['predicted_class'] == 0].head(10)
    for idx, row in healthy_preds.iterrows():
        print(f"  {row['image_name']}: {row['confidence']:.1f}%")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("="*60)
    print("ULTRASOUND CLASSIFICATION PREDICTIONS")
    print("="*60 + "\n")
    
    # Check model exists
    model_path = OUTPUT_DIR / "ultrasound_classifier.keras"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Run train_ultrasound_classifier.py first")
        exit(1)
    
    # Check images exist
    if not ULTRASOUND_DIR.exists() or not any(ULTRASOUND_DIR.glob("*")):
        print(f"❌ No images in {ULTRASOUND_DIR}")
        exit(1)
    
    # Load model
    model = load_trained_model(model_path)
    if model is None:
        exit(1)
    
    # Make predictions
    predictions = predict_batch(model, ULTRASOUND_DIR)
    
    if predictions:
        # Save
        output_csv = OUTPUT_DIR / "ultrasound_predictions.csv"
        df_pred = save_predictions(predictions, output_csv)
        
        # Summary
        print_prediction_summary(df_pred)
        
        print("\n✅ Predictions complete!")
    else:
        print("\n❌ No predictions generated")
