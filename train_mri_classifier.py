"""
Train deep learning classifier for MRI images (3D)
Uses a custom 3D CNN model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

# Model configuration
IMG_SIZE = (64, 64, 64)  # 3D patch size for MRI
BATCH_SIZE = 4  # Smaller batch for 3D data
EPOCHS = 20


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

def load_mri_labels():
    """Load real MRI labels from dataset annotations"""
    labels_path = OUTPUT_DIR / "mri_labels.csv"
    
    if labels_path.exists():
        df = pd.read_csv(labels_path)
        logger.info(f"✓ Loaded MRI labels from {labels_path}")
    else:
        logger.error("⚠️  Real MRI labels not found at output/mri_labels.csv")
        logger.error("Create mri_labels.csv with real labels before training.")
        return None

    if "label" not in df.columns or "image_name" not in df.columns:
        logger.error("mri_labels.csv must contain 'image_name' and 'label' columns.")
        return None

    return df

def load_mri_image(image_path):
    """Load MRI image as 3D numpy array"""
    try:
        img = sitk.ReadImage(str(image_path))
        img_array = sitk.GetArrayFromImage(img).astype(np.float32)
        
        # Normalize intensity
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        
        return img_array
    except Exception as e:
        logger.warning(f"Error loading {image_path.name}: {e}")
        return None

def resize_3d_image(img_array, target_size=(64, 64, 64)):
    """Resize 3D image to target size"""
    from scipy import ndimage
    
    # Calculate zoom factors
    zoom_factors = [target_size[i] / img_array.shape[i] for i in range(3)]
    
    # Resize
    resized = ndimage.zoom(img_array, zoom_factors, order=1)
    
    return resized.astype(np.float32)

def create_3d_model():
    """Create 3D CNN model for MRI classification"""
    
    model = models.Sequential([
        # 3D Convolution blocks
        layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', 
                     input_shape=(*IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    logger.info("✓ 3D CNN model created for MRI")
    return model

def prepare_mri_dataset(df):
    """Prepare MRI dataset"""
    mri_dir = resolve_mri_dir()
    logger.info(f"Using MRI image directory: {mri_dir}")

    mri_files = sorted(list(mri_dir.glob("*.nii.gz")) + list(mri_dir.glob("*.nii")))
    
    if not mri_files:
        logger.error(f"No MRI files found in {mri_dir}")
        return None, None, None, None
    
    print(f"\n📊 Preparing MRI dataset ({len(mri_files)} images)...\n")
    
    # Load images
    images = []
    labels_list = []
    
    for i, img_path in enumerate(mri_files):
        if i % 5 == 0:
            print(f"   Loading MRI {i+1}/{len(mri_files)}...")
        
        img_array = load_mri_image(img_path)
        if img_array is not None:
            # Resize to target size
            img_resized = resize_3d_image(img_array, IMG_SIZE)
            images.append(img_resized)
            
            # Get label
            label = df[df['image_name'] == img_path.name]['label'].values
            if len(label) > 0:
                labels_list.append(label[0])
    
    if not images:
        logger.error("No images loaded")
        return None, None, None, None
    
    # Convert to arrays
    X = np.array(images)
    X = X[..., np.newaxis]  # Add channel dimension (Z, Y, X, C)
    y = np.array(labels_list, dtype=np.float32)
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    return X_train, X_val, y_train, y_val

def train_model(model, X_train, X_val, y_train, y_val):
    """Train the model"""
    
    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001
    )
    
    # Train
    logger.info("\n🔄 Training MRI classifier...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    return history

def save_model(model, model_path):
    """Save trained model"""
    model.save(model_path)
    logger.info(f"✓ Model saved to {model_path}")

def plot_training_history(history, output_path):
    """Plot and save training history"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('MRI Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('MRI Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Training history saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print("TRAINING MRI DEEP LEARNING CLASSIFIER (3D CNN)")
    print("="*60 + "\n")
    
    # Load labels
    df = load_mri_labels()
    if df is None:
        exit(1)
    
    # Create model
    model = create_3d_model()
    print(f"\nModel parameters: {model.count_params():,}")
    
    # Prepare dataset
    X_train, X_val, y_train, y_val = prepare_mri_dataset(df)
    if X_train is None:
        exit(1)
    
    # Train
    history = train_model(model, X_train, X_val, y_train, y_val)
    
    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = OUTPUT_DIR / "mri_classifier.keras"
    save_model(model, model_path)
    
    # Plot
    history_plot = OUTPUT_DIR / "mri_training_history.png"
    plot_training_history(history, history_plot)
    
    # Summary
    print("\n" + "="*60)
    print("✅ MRI TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved: {model_path}")
    print(f"Training history: {history_plot}")
    print("\nNext: python predict_mri.py")
    print("="*60)
