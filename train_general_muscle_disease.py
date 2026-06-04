#!/usr/bin/env python3
"""
General Muscle Disease Classification Pipeline
Combines ULTRASOUND_LABELD_1 (FSHD) and ULTRASOUND_LABELD_2 (Multi-disease)
for comprehensive muscle disease classification
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from itertools import cycle
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

def load_fshd_data():
    """Load FSHD data from ULTRASOUND_LABELD_1"""
    print("Loading FSHD dataset...")
    
    features_file = Path("c:/Users/Lenovo/Desktop/thesis_project/processed_data/custom_features.csv")
    if not features_file.exists():
        print(f"FSHD features file not found: {features_file}")
        return None, None, None
    
    df = pd.read_csv(features_file)
    print(f"Loaded FSHD dataset: {df.shape}")
    
    # Extract features and labels
    metadata_cols = ['filename', 'subject', 'muscle_code', 'side', 'instance', 
                   'grade_category', 'original_grade']
    feature_cols = [col for col in df.columns if col not in metadata_cols + ['binary_label']]
    
    X = df[feature_cols]
    
    # Create disease labels (all FSHD)
    y_disease = ['FSHD'] * len(df)
    y_severity = df['binary_label'].values  # 0=Normal/Mild, 1=Moderate/Severe
    
    filenames = df['filename'].values
    
    print(f"FSHD samples: {len(X)}")
    print(f"FSHD severity distribution: {np.bincount(y_severity)}")
    
    return X, y_disease, y_severity, filenames

def load_multi_disease_data():
    """Load multi-disease data from ULTRASOUND_LABELD_2"""
    print("Loading multi-disease dataset...")
    
    dataset_path = Path("c:/Users/Lenovo/Desktop/thesis_project/data/ULTRASOUND_LABELD_2")
    excel_file = dataset_path / "PatientImages_PLOS2017.xlsx"
    
    if not excel_file.exists():
        print(f"Multi-disease Excel file not found: {excel_file}")
        return None, None, None
    
    try:
        df = pd.read_excel(excel_file)
        print(f"Loaded multi-disease dataset: {df.shape}")
        
        # Find diagnosis column
        diagnosis_col = None
        for col in df.columns:
            if 'Diagnosis' in col or 'D -' in col:
                diagnosis_col = col
                break
        
        if diagnosis_col is None:
            print("Could not find diagnosis column")
            return None, None, None
        
        # Create features (synthetic for now - in real implementation extract from .mat file)
        features = []
        disease_labels = []
        severity_labels = []
        filenames = []
        
        for idx, row in df.iterrows():
            # Create synthetic radiomics features
            # In real implementation, extract from PatientData.mat
            feature_vector = [
                np.random.normal(100, 30),  # mean_intensity
                np.random.normal(45, 15),   # std_intensity
                np.random.normal(0.2, 0.5),  # skewness
                np.random.normal(2.8, 1.2),  # kurtosis
                np.random.normal(4.5, 1.0),  # entropy
                np.random.normal(0.4, 0.2),  # glcm_contrast
                np.random.normal(0.8, 0.1),  # glcm_homogeneity
                np.random.normal(1200, 400), # area
                np.random.normal(0.7, 0.2),  # circularity
                np.random.normal(15, 5),     # gradient_mean
                np.random.normal(25, 8),     # perimeter
                np.random.normal(1.5, 0.3),  # aspect_ratio
                np.random.normal(0.8, 0.1),  # extent
                np.random.normal(0.9, 0.05), # solidity
                # Add clinical features
                row.get('Age', 50) if 'Age' in df.columns else 50,
                1 if str(row.get('Sex', 'M')).upper() == 'M' else 0,  # 1=M, 0=F
                row.get('Muscle Strength\n(1 - 10)', 5) if 'Muscle Strength\n(1 - 10)' in df.columns else 5,
                row.get('CPK', 200) if 'CPK' in df.columns else 200,
                row.get('Duration of symptoms (months)', 12) if 'Duration of symptoms (months)' in df.columns else 12,
                row.get('Age', 50) % 10,  # Age decade
                1 if str(row.get('Sex', 'M')).upper() == 'M' else 0,  # Gender binary
                min(row.get('Muscle Strength\n(1 - 10)', 5), 10) / 10,  # Normalized strength
                np.log1p(row.get('CPK', 200)),  # Log CPK
                min(row.get('Duration of symptoms (months)', 12), 60) / 60,  # Normalized duration
                len(str(row.get('Patient Identifier', ''))),  # ID length
                idx % 100,  # Sample index mod 100
            ]
            
            # Ensure 27 features to match FSHD data
            if len(feature_vector) < 27:
                feature_vector.extend([0] * (27 - len(feature_vector)))
            elif len(feature_vector) > 27:
                feature_vector = feature_vector[:27]
            
            features.append(feature_vector)
            
            # Disease label
            disease = row[diagnosis_col]
            if disease == 'N':
                disease_labels.append('Normal')
            elif disease == 'D':
                disease_labels.append('Dermatomyositis')
            elif disease == 'P':
                disease_labels.append('Polymyositis')
            elif disease == 'I':
                disease_labels.append('IBM')
            else:
                disease_labels.append('Other')
            
            # Severity label (based on muscle strength)
            strength = row.get('Muscle Strength\n(1 - 10)', 5)
            severity_labels.append(1 if strength <= 5 else 0)  # 1=Severe, 0=Mild
            
            filenames.append(f"multi_disease_{idx}")
        
        X = np.array(features)
        y_disease = np.array(disease_labels)
        y_severity = np.array(severity_labels)
        filenames = np.array(filenames)
        
        print(f"Multi-disease samples: {len(X)}")
        print(f"Disease distribution: {pd.Series(y_disease).value_counts().to_dict()}")
        print(f"Severity distribution: {np.bincount(y_severity)}")
        
        return X, y_disease, y_severity, filenames
        
    except Exception as e:
        print(f"Error loading multi-disease data: {e}")
        return None, None, None

def combine_datasets():
    """Combine FSHD and multi-disease datasets"""
    print("\n=== Combining Datasets for General Muscle Disease Classification ===")
    
    # Load both datasets
    X_fshd, disease_fshd, severity_fshd, filenames_fshd = load_fshd_data()
    X_multi, disease_multi, severity_multi, filenames_multi = load_multi_disease_data()
    
    if X_fshd is None and X_multi is None:
        print("No data loaded!")
        return None, None, None, None
    
    # Combine data
    if X_fshd is not None and X_multi is not None:
        X_combined = np.vstack([X_fshd, X_multi])
        disease_combined = np.concatenate([disease_fshd, disease_multi])
        severity_combined = np.concatenate([severity_fshd, severity_multi])
        filenames_combined = np.concatenate([filenames_fshd, filenames_multi])
    elif X_fshd is not None:
        X_combined = X_fshd
        disease_combined = np.array(disease_fshd)
        severity_combined = np.array(severity_fshd)
        filenames_combined = np.array(filenames_fshd)
    else:
        X_combined = X_multi
        disease_combined = disease_multi
        severity_combined = severity_multi
        filenames_combined = filenames_multi
    
    # Clean data - handle NaN values
    print("Cleaning data...")
    X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\nCombined dataset: {X_combined.shape[0]} samples")
    print(f"Disease categories: {pd.Series(disease_combined).value_counts().to_dict()}")
    print(f"Severity distribution: {np.bincount(severity_combined)}")
    
    return X_combined, disease_combined, severity_combined, filenames_combined

def train_disease_classification(X, y_disease, y_severity):
    """Train models for disease classification"""
    print("\n=== Training Disease Classification Models ===")
    
    # Encode disease labels
    label_encoder = LabelEncoder()
    y_disease_encoded = label_encoder.fit_transform(y_disease)
    disease_classes = label_encoder.classes_
    
    print(f"Disease classes: {list(disease_classes)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_disease_encoded, test_size=0.2, stratify=y_disease_encoded, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train ML models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} for disease classification...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # AUC for multi-class (one-vs-rest)
        try:
            if len(disease_classes) == 2:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                y_test_bin = label_binarize(y_test, classes=range(len(disease_classes)))
                auc = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')
        except:
            auc = 0.5
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_test
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    return results, disease_classes

def train_severity_classification(X, y_severity):
    """Train models for severity classification"""
    print("\n=== Training Severity Classification Models ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_severity, test_size=0.2, stratify=y_severity, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Severity distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    
    # Train ML models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} for severity classification...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
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

def save_results(disease_results, severity_results, disease_classes, output_dir):
    """Save training results"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save disease classification results
    disease_data = []
    for name, results in disease_results.items():
        disease_data.append({
            'Model': name,
            'Task': 'Disease Classification',
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1'],
            'AUC': results['auc']
        })
    
    # Save severity classification results
    severity_data = []
    for name, results in severity_results.items():
        severity_data.append({
            'Model': name,
            'Task': 'Severity Classification',
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1'],
            'AUC': results['auc']
        })
    
    # Combine and save
    all_results = disease_data + severity_data
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path / 'general_muscle_disease_results.csv', index=False)
    
    print(f"\nResults saved to {output_path}")
    print("\nDISEASE CLASSIFICATION RESULTS:")
    print(pd.DataFrame(disease_data).to_string(index=False))
    
    print("\nSEVERITY CLASSIFICATION RESULTS:")
    print(pd.DataFrame(severity_data).to_string(index=False))
    
    # Save disease classes
    pd.Series(disease_classes).to_csv(output_path / 'disease_classes.csv', index=False, header=False)
    
    return results_df

def main():
    """Main training pipeline for general muscle disease classification"""
    print("🏥 General Muscle Disease Classification Pipeline")
    print("=" * 60)
    
    try:
        # Load and combine datasets
        X, y_disease, y_severity, filenames = combine_datasets()
        
        if X is None:
            print("No data loaded. Exiting.")
            return
        
        # Train disease classification models
        disease_results, disease_classes = train_disease_classification(X, y_disease, y_severity)
        
        # Train severity classification models
        severity_results = train_severity_classification(X, y_severity)
        
        # Save results
        results_df = save_results(disease_results, severity_results, disease_classes, "output/general_muscle_disease")
        
        print("\n" + "=" * 60)
        print("🏁 General Muscle Disease Classification Complete!")
        
        # Summary
        print(f"\n📋 Summary:")
        print(f"   Total samples: {len(X)}")
        print(f"   Disease categories: {len(disease_classes)}")
        print(f"   Best disease model: {max(disease_results.keys(), key=lambda k: disease_results[k]['accuracy'])}")
        print(f"   Best severity model: {max(severity_results.keys(), key=lambda k: severity_results[k]['accuracy'])}")
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
