"""
Train deep learning classifier for ultrasound images
Uses transfer learning with pre-trained EfficientNet model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path("data")
ULTRASOUND_DIR = DATA_DIR / "ultrasound_images"
OUTPUT_DIR = Path("output")

# Model configuration
IMG_SIZE = (224, 224)  # EfficientNetB0 input size
BATCH_SIZE = 32
EPOCHS = 20

def load_labels():
    """Load real training labels from dataset annotations"""
    labels_path = OUTPUT_DIR / "labels.csv"
    
    if labels_path.exists():
        df = pd.read_csv(labels_path)
        logger.info(f"✓ Loaded labels from {labels_path}")
    else:
        logger.error("⚠️  Real labels not found at output/labels.csv")
        logger.error("Create labels.csv with real dataset labels before training.")
        return None

    if "label" not in df.columns or "image_name" not in df.columns:
        logger.error("labels.csv must contain 'image_name' and 'label' columns.")
        return None
    
    return df

def create_data_generators():
    """Create image data generators for training/validation"""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_datagen

def create_transfer_learning_model():
    """Create model using transfer learning (EfficientNetB0)"""
    
    # Load pre-trained model
    base_model = EfficientNetB0(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model weights
    base_model.trainable = False
    
    # Add custom layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )
    
    logger.info("✓ Transfer learning model created (EfficientNetB0)")
    return model

def prepare_data_generators(train_datagen, val_datagen, df):
    """Create generators for training and validation data"""
    
    # Convert labels to strings (required by flow_from_dataframe)
    df = df.copy()
    df['label'] = df['label'].astype(str)
    
    # Split data
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=ULTRASOUND_DIR,
        x_col='image_name',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb'
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=ULTRASOUND_DIR,
        x_col='image_name',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb'
    )
    
    logger.info(f"✓ Training samples: {len(train_df)}")
    logger.info(f"✓ Validation samples: {len(val_df)}")
    
    return train_generator, val_generator

def train_model(model, train_generator, val_generator):
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
    logger.info("\n🔄 Training model...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
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
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Training history saved to {output_path}")
    plt.close()

def print_model_summary(model):
    """Print model architecture"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("TRAINING ULTRASOUND CLASSIFIER")
    print("="*60 + "\n")
    
    # Load labels
    df = load_labels()
    if df is None:
        exit(1)
    
    # Setup
    train_datagen, val_datagen = create_data_generators()
    model = create_transfer_learning_model()
    print_model_summary(model)
    
    # Prepare data
    print("\n📊 Preparing data generators...")
    train_gen, val_gen = prepare_data_generators(train_datagen, val_datagen, df)
    
    # Train
    history = train_model(model, train_gen, val_gen)
    
    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = OUTPUT_DIR / "ultrasound_classifier.keras"
    save_model(model, model_path)
    
    # Plot
    history_plot = OUTPUT_DIR / "training_history.png"
    plot_training_history(history, history_plot)
    
    # Summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved: {model_path}")
    print(f"Training history: {history_plot}")
    print("\nNext: python predict_ultrasound.py")
    print("="*60)
