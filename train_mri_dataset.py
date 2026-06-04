#!/usr/bin/env python3
"""
Training pipeline for MRI_LABELED dataset
Same approach as ultrasound but adapted for MRI images
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report, roc_curve)
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_mri_dataset():
    """Load MRI dataset from Dataset.json"""
    print("Loading MRI dataset...")
    
    dataset_path = Path("c:/Users/Lenovo/Desktop/thesis_project/data/MRI_LABELED")
    json_file = dataset_path / "Dataset.json"
    
    if not json_file.exists():
        raise FileNotFoundError(f"Dataset.json not found: {json_file}")
    
    with open(json_file, 'r') as f:
        dataset = json.load(f)
    
    # Combine training and testing data
    all_data = dataset.get('training', []) + dataset.get('testing', [])
    
    print(f"Found {len(all_data)} MRI samples")
    
    # Extract features and create binary labels
    features = []
    labels = []
    filenames = []
    
    for i, sample in enumerate(all_data):
        try:
            # Load MRI image
            image_path = dataset_path / sample['image']
            label_path = dataset_path / sample['label']
            
            if not image_path.exists() or not label_path.exists():
                print(f"Missing files for sample {i}: {sample}")
                continue
            
            # Load image and segmentation
            img_nii = nib.load(image_path)
            label_nii = nib.load(label_path)
            
            img_data = img_nii.get_fdata()
            label_data = label_nii.get_fdata()
            
            # Extract simple radiomics features
            mask = label_data > 0
            if np.sum(mask) == 0:
                continue
                
            masked_img = img_data[mask]
            
            # First-order statistics
            features.append([
                np.mean(masked_img),
                np.std(masked_img),
                np.percentile(masked_img, 25),
                np.percentile(masked_img, 75),
                np.min(masked_img),
                np.max(masked_img),
                np.percentile(masked_img, 10),
                np.percentile(masked_img, 90),
                np.percentile(masked_img, 50),
                np.var(masked_img),
                np.mean(masked_img) / (np.std(masked_img) + 1e-8),  # SNR
                np.sum(masked_img),  # Volume
                len(masked_img),  # Number of voxels
            ])
            
            # Create binary label based on segmentation characteristics
            # Simple heuristic: larger muscle area = more severe
            volume = np.sum(masked_img)
            binary_label = 1 if volume > np.median([np.sum(label_nii.get_fdata()[label_nii.get_fdata() > 0]) for _ in range(1)]) else 0
            
            labels.append(binary_label)
            filenames.append(sample['image'])
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    X = np.array(features)
    y = np.array(labels)
    
    print(f"Successfully processed {len(X)} samples")
    print(f"Features shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    return X, y, filenames

def train_ml_models(X_train, y_train, X_test, y_test):
    """Train ML models on MRI data"""
    print("\n=== Training ML Models on MRI Dataset ===")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name} - Accuracy: {results[name]['accuracy']:.4f}, "
              f"Precision: {results[name]['precision']:.4f}, "
              f"Recall: {results[name]['recall']:.4f}, "
              f"F1: {results[name]['f1']:.4f}, "
              f"AUC: {results[name]['auc']:.4f}")
    
    return results

def create_dl_model(input_dim):
    """Create deep learning model for MRI"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def train_dl_model(X_train, y_train, X_test, y_test):
    """Train deep learning model on MRI data"""
    print("\n=== Training Deep Learning Model on MRI Dataset ===")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = create_dl_model(X_train_scaled.shape[1])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ]
    
    class_weight = {0: 1, 1: len(y_train[y_train==0])/len(y_train[y_train==1])}
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weight
    )
    
    y_pred_proba = model.predict(X_test_scaled).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    dl_results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"DL Model - Accuracy: {dl_results['accuracy']:.4f}, "
          f"Precision: {dl_results['precision']:.4f}, "
          f"Recall: {dl_results['recall']:.4f}, "
          f"F1: {dl_results['f1']:.4f}, "
          f"AUC: {dl_results['auc']:.4f}")
    
    return dl_results

def save_results(ml_results, dl_results, output_dir):
    """Save training results"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Combine all results
    all_results = {**ml_results, 'Deep Learning': dl_results}
    
    # Save comparison table
    comparison_data = []
    for name, results in all_results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1'],
            'AUC': results['auc']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_path / 'mri_model_comparison.csv', index=False)
    
    print(f"\nResults saved to {output_path}")
    print(comparison_df)

def main():
    """Main training pipeline for MRI dataset"""
    print("=== MRI Dataset Training Pipeline ===")
    
    try:
        # Load data
        X, y, filenames = load_mri_dataset()
        
        if len(X) == 0:
            print("No data loaded. Exiting.")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train models
        ml_results = train_ml_models(X_train, y_train, X_test, y_test)
        dl_results = train_dl_model(X_train, y_train, X_test, y_test)
        
        # Save results
        save_results(ml_results, dl_results, "output/mri_results")
        
        print("\n=== MRI Training Complete ===")
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
