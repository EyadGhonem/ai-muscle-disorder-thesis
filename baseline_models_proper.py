#!/usr/bin/env python3
"""
Proper baseline ML models with cross-validation
Using final_ultrasound_dataset.csv with proper validation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report, roc_curve)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

def load_and_clean_data():
    """Load and clean the final dataset"""
    print("Loading final ultrasound dataset...")
    
    dataset_path = Path("c:/Users/Lenovo/Desktop/thesis_project/final_ultrasound_dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape}")
    
    # Remove rows with Unknown severity
    df = df[df['severity_label'] != 'Unknown'].copy()
    print(f"After removing Unknown severity: {df.shape}")
    
    # Handle missing values in features
    feature_cols = [col for col in df.columns if col.startswith('feature_') or col in 
                  ['mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 
                   'median_intensity', 'q25_intensity', 'q75_intensity', 'skewness', 
                   'kurtosis', 'entropy', 'glcm_contrast', 'glcm_dissimilarity',
                   'glcm_homogeneity', 'glcm_energy', 'glcm_correlation', 'glcm_asm',
                   'area', 'perimeter', 'circularity', 'aspect_ratio', 'extent', 
                   'solidity', 'equivalent_diameter', 'gradient_mean', 'gradient_std',
                   'gradient_max', 'gradient_energy']]
    
    # Fill missing values with median
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    print(f"Missing values after cleaning: {df[feature_cols].isnull().sum().sum()}")
    
    return df, feature_cols

def analyze_data_quality(df):
    """Analyze data quality issues"""
    print("\n=== Data Quality Analysis ===")
    
    # Class distribution
    print("\nDisease Distribution:")
    disease_counts = df['label'].value_counts()
    for disease, count in disease_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {disease}: {count} ({percentage:.1f}%)")
    
    print("\nSeverity Distribution:")
    severity_counts = df['severity_label'].value_counts()
    for severity, count in severity_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {severity}: {count} ({percentage:.1f}%)")
    
    # Patient ID analysis
    print(f"\nPatient ID Analysis:")
    print(f"Total unique patient IDs: {df['patient_id'].nunique()}")
    print(f"Total records: {len(df)}")
    print(f"Duplicate patient IDs: {len(df) - df['patient_id'].nunique()}")
    
    # Feature statistics
    feature_cols = [col for col in df.columns if col.startswith('feature_') or col in 
                  ['mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity']]
    if feature_cols:
        print(f"\nFeature Statistics:")
        print(f"Number of features: {len(feature_cols)}")
        print(f"Feature range: [{df[feature_cols].min().min():.2f}, {df[feature_cols].max().max():.2f}]")
        print(f"Mean feature values: {df[feature_cols].mean().mean():.2f}")

def train_baseline_models(X, y_disease, y_severity):
    """Train baseline models with proper validation"""
    print("\n=== Training Baseline Models ===")
    
    # Disease classification
    print("\n--- Disease Classification ---")
    label_encoder = LabelEncoder()
    y_disease_encoded = label_encoder.fit_transform(y_disease)
    disease_classes = label_encoder.classes_
    
    # Stratified 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    disease_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }
    
    disease_results = {}
    
    for name, model in disease_models.items():
        print(f"\nTraining {name}...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        # Cross-validation scores
        cv_scores = cross_val_score(pipeline, X, y_disease_encoded, cv=skf, 
                                  scoring='accuracy', n_jobs=-1)
        
        # Train final model on full data
        pipeline.fit(X, y_disease_encoded)
        
        disease_results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': pipeline
        }
        
        print(f"{name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Severity classification
    print("\n--- Severity Classification ---")
    severity_encoder = LabelEncoder()
    y_severity_encoded = severity_encoder.fit_transform(y_severity)
    severity_classes = severity_encoder.classes_
    
    severity_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }
    
    severity_results = {}
    
    for name, model in severity_models.items():
        print(f"\nTraining {name}...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        # Cross-validation scores
        cv_scores = cross_val_score(pipeline, X, y_severity_encoded, cv=skf, 
                                  scoring='accuracy', n_jobs=-1)
        
        # Train final model
        pipeline.fit(X, y_severity_encoded)
        
        severity_results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': pipeline
        }
        
        print(f"{name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return disease_results, severity_results, disease_classes, severity_classes

def evaluate_on_test_split(X, y_disease, y_severity, disease_results, severity_results):
    """Evaluate models on stratified test split"""
    print("\n=== Test Set Evaluation ===")
    
    # Create test split
    X_train, X_test, y_disease_train, y_disease_test, y_severity_train, y_severity_test = train_test_split(
        X, y_disease, y_severity, test_size=0.2, stratify=y_disease, random_state=42
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Disease classification evaluation
    print("\n--- Disease Classification Test Results ---")
    label_encoder = LabelEncoder()
    y_disease_train_encoded = label_encoder.fit_transform(y_disease_train)
    y_disease_test_encoded = label_encoder.transform(y_disease_test)
    
    disease_test_results = {}
    
    for name, result in disease_results.items():
        pipeline = result['pipeline']
        pipeline.fit(X_train, y_disease_train_encoded)
        
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_disease_test_encoded, y_pred)
        precision = precision_score(y_disease_test_encoded, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_disease_test_encoded, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_disease_test_encoded, y_pred, average='weighted', zero_division=0)
        
        try:
            if len(label_encoder.classes_) == 2:
                auc = roc_auc_score(y_disease_test_encoded, y_pred_proba[:, 1])
            else:
                from sklearn.preprocessing import label_binarize
                y_test_bin = label_binarize(y_disease_test_encoded, classes=range(len(label_encoder.classes_)))
                auc = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')
        except:
            auc = 0.5
        
        disease_test_results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': confusion_matrix(y_disease_test_encoded, y_pred)
        }
        
        print(f"{name}: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # Severity classification evaluation
    print("\n--- Severity Classification Test Results ---")
    severity_encoder = LabelEncoder()
    y_severity_train_encoded = severity_encoder.fit_transform(y_severity_train)
    y_severity_test_encoded = severity_encoder.transform(y_severity_test)
    
    severity_test_results = {}
    
    for name, result in severity_results.items():
        pipeline = result['pipeline']
        pipeline.fit(X_train, y_severity_train_encoded)
        
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_severity_test_encoded, y_pred)
        precision = precision_score(y_severity_test_encoded, y_pred, zero_division=0)
        recall = recall_score(y_severity_test_encoded, y_pred, zero_division=0)
        f1 = f1_score(y_severity_test_encoded, y_pred, zero_division=0)
        auc = roc_auc_score(y_severity_test_encoded, y_pred_proba)
        
        severity_test_results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': confusion_matrix(y_severity_test_encoded, y_pred)
        }
        
        print(f"{name}: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    return disease_test_results, severity_test_results, label_encoder, severity_encoder

def save_results(disease_cv, disease_test, severity_cv, severity_test, 
                disease_classes, severity_classes):
    """Save all results"""
    print("\n=== Saving Results ===")
    
    output_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/output/baseline_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save cross-validation results
    cv_data = []
    for name, result in disease_cv.items():
        cv_data.append({
            'Model': name,
            'Task': 'Disease_Classification',
            'CV_Mean': result['cv_mean'],
            'CV_Std': result['cv_std']
        })
    
    for name, result in severity_cv.items():
        cv_data.append({
            'Model': name,
            'Task': 'Severity_Classification',
            'CV_Mean': result['cv_mean'],
            'CV_Std': result['cv_std']
        })
    
    cv_df = pd.DataFrame(cv_data)
    cv_df.to_csv(output_dir / 'cross_validation_results.csv', index=False)
    
    # Save test results
    test_data = []
    for name, result in disease_test.items():
        test_data.append({
            'Model': name,
            'Task': 'Disease_Classification',
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1_Score': result['f1'],
            'AUC': result['auc']
        })
    
    for name, result in severity_test.items():
        test_data.append({
            'Model': name,
            'Task': 'Severity_Classification',
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1_Score': result['f1'],
            'AUC': result['auc']
        })
    
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(output_dir / 'test_results.csv', index=False)
    
    # Save classes
    pd.Series(disease_classes).to_csv(output_dir / 'disease_classes.csv', index=False, header=False)
    pd.Series(severity_classes).to_csv(output_dir / 'severity_classes.csv', index=False, header=False)
    
    print(f"Results saved to: {output_dir}")
    
    return output_dir

def create_plots(output_dir, disease_test, severity_test, disease_classes, severity_classes):
    """Create confusion matrices and performance plots"""
    print("\n=== Creating Plots ===")
    
    # Disease confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Disease Classification Confusion Matrices', fontsize=16)
    
    models = list(disease_test.keys())
    for idx, (name, result) in enumerate(disease_test.items()):
        row, col = idx // 3, idx % 3
        if row >= 2:
            break
            
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col])
        axes[row, col].set_title(f'{name}')
        axes[row, col].set_xlabel('Predicted')
        axes[row, col].set_ylabel('Actual')
    
    # Remove empty subplots
    for idx in range(len(models), 6):
        row, col = idx // 3, idx % 3
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'disease_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Severity confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Severity Classification Confusion Matrices', fontsize=16)
    
    for idx, (name, result) in enumerate(severity_test.items()):
        row, col = idx // 3, idx % 3
        if row >= 2:
            break
            
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col])
        axes[row, col].set_title(f'{name}')
        axes[row, col].set_xlabel('Predicted')
        axes[row, col].set_ylabel('Actual')
    
    # Remove empty subplots
    for idx in range(len(severity_test), 6):
        row, col = idx // 3, idx % 3
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'severity_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plots saved successfully")

def main():
    """Main function"""
    print("🔬 Baseline Models with Proper Validation")
    print("=" * 50)
    
    try:
        # Load and clean data
        df, feature_cols = load_and_clean_data()
        analyze_data_quality(df)
        
        # Prepare features and labels
        X = df[feature_cols].values
        y_disease = df['label'].values
        y_severity = df['severity_label'].values
        
        # Train baseline models
        disease_cv, severity_cv, disease_classes, severity_classes = train_baseline_models(X, y_disease, y_severity)
        
        # Evaluate on test split
        disease_test, severity_test, disease_encoder, severity_encoder = evaluate_on_test_split(
            X, y_disease, y_severity, disease_cv, severity_cv)
        
        # Save results
        output_dir = save_results(disease_cv, disease_test, severity_cv, severity_test, 
                                disease_classes, severity_classes)
        
        # Create plots
        create_plots(output_dir, disease_test, severity_test, disease_classes, severity_classes)
        
        print("\n" + "=" * 50)
        print("✅ Baseline models completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error in baseline modeling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
