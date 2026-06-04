#!/usr/bin/env python3
"""
Quick CNN Model Evaluation: ResNet50, DenseNet121, EfficientNetB0
Trains for just 2 epochs on a subset to get initial accuracy estimates
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers, models

PROJECT_ROOT = Path(__file__).resolve().parent
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 2  # Quick training
RANDOM_STATE = 42

def get_preprocess_fn(architecture):
    if architecture == "resnet50":
        return resnet_preprocess
    if architecture == "densenet121":
        return densenet_preprocess
    if architecture == "efficientnetb0":
        return efficientnet_preprocess

def build_model(architecture, num_classes, learning_rate=1e-3):
    """Transfer-learning head on ImageNet backbones."""
    if architecture == "resnet50":
        base = ResNet50(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    elif architecture == "densenet121":
        base = DenseNet121(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    elif architecture == "efficientnetb0":
        base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))

    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(base.input, outputs, name=architecture)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model

def load_data():
    """Load FSHD severity data"""
    master = PROJECT_ROOT / "output" / "final_ultrasound_dataset.csv"
    df = pd.read_csv(master)
    df = df[df["dataset_source"] == "ULTRASOUND_LABELD_1"].copy()
    df["filepath"] = df["image_path"].apply(lambda p: str(PROJECT_ROOT / p))
    df = df[df["filepath"].apply(lambda p: Path(p).exists())]
    df = df.dropna(subset=["severity"])
    df["label"] = df["severity"].astype(int)
    
    # Use subset for quick training
    df = df.sample(n=min(800, len(df)), random_state=RANDOM_STATE)
    
    return df

def train_model(arch, X_train, y_train, X_val, y_val):
    """Quick model training"""
    print(f"\nTraining {arch.upper()}")
    model = build_model(arch, 2)
    
    preprocess_fn = get_preprocess_fn(arch)
    
    # Use simple data augmentation
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    # Quick training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    return model, history

def main():
    print("="*60)
    print("QUICK CNN EVALUATION - ResNet50, DenseNet121")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} samples")
    
    # Split train/val
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss.split(df, groups=df['patient_id']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Load images
    def load_images(df_subset):
        images = []
        labels = []
        for idx, row in df_subset.iterrows():
            try:
                img = keras.preprocessing.image.load_img(row['filepath'], target_size=IMG_SIZE)
                img_array = keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(row['label'])
            except:
                pass
        return np.array(images), np.array(labels)
    
    X_train, y_train = load_images(train_df)
    X_val, y_val = load_images(val_df)
    
    print(f"Loaded X_train: {X_train.shape}, X_val: {X_val.shape}")
    
    results = []
    
    for arch in ['resnet50', 'densenet121']:
        model, history = train_model(arch, X_train, y_train, X_val, y_val)
        
        # Evaluate
        y_pred_proba = model.predict(X_val, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        print(f"\n{arch.upper()} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        
        results.append({
            'Model': arch.upper(),
            'Accuracy': f"{accuracy:.2%}",
            'Precision': f"{precision:.4f}",
            'Recall': f"{recall:.4f}",
            'F1-Score': f"{f1:.4f}",
            'AUC': f"{auc:.4f}"
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = PROJECT_ROOT / "output" / "cnn_quick_results.csv"
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print(f"Results saved to: {output_file}")
    print("="*60)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
