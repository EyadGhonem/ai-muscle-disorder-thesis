#!/usr/bin/env python3
"""
Advanced radiomics models including XGBoost, LightGBM, CatBoost, and stacking ensemble
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available, installing...")
    os.system("pip install xgboost")
    try:
        import xgboost as xgb
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("LightGBM not available, installing...")
    os.system("pip install lightgbm")
    try:
        import lightgbm as lgb
        LIGHTGBM_AVAILABLE = True
    except ImportError:
        LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    print("CatBoost not available, installing...")
    os.system("pip install catboost")
    try:
        import catboost as cb
        CATBOOST_AVAILABLE = True
    except ImportError:
        CATBOOST_AVAILABLE = False

# Set random seeds for reproducibility
np.random.seed(42)

def load_clean_data():
    """Load and clean final dataset"""
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

def train_advanced_models(X, y_disease, y_severity):
    """Train advanced radiomics models"""
    print("\n=== Training Advanced Radiomics Models ===")
    
    # Disease classification
    print("\n--- Disease Classification ---")
    label_encoder = LabelEncoder()
    y_disease_encoded = label_encoder.fit_transform(y_disease)
    disease_classes = label_encoder.classes_
    
    # Stratified 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    disease_models = {}
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        print("\nTraining XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', xgb_model)
        ])
        
        cv_scores = cross_val_score(pipeline, X, y_disease_encoded, cv=skf, 
                                  scoring='accuracy', n_jobs=-1)
        pipeline.fit(X, y_disease_encoded)
        
        disease_models['XGBoost'] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': pipeline
        }
        print(f"XGBoost: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\nTraining LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', lgb_model)
        ])
        
        cv_scores = cross_val_score(pipeline, X, y_disease_encoded, cv=skf, 
                                  scoring='accuracy', n_jobs=-1)
        pipeline.fit(X, y_disease_encoded)
        
        disease_models['LightGBM'] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': pipeline
        }
        print(f"LightGBM: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        print("\nTraining CatBoost...")
        cat_model = cb.CatBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=False
        )
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', cat_model)
        ])
        
        cv_scores = cross_val_score(pipeline, X, y_disease_encoded, cv=skf, 
                                  scoring='accuracy', n_jobs=-1)
        pipeline.fit(X, y_disease_encoded)
        
        disease_models['CatBoost'] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': pipeline
        }
        print(f"CatBoost: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Extra Trees
    print("\nTraining Extra Trees...")
    extra_model = ExtraTreesClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', extra_model)
    ])
    
    cv_scores = cross_val_score(pipeline, X, y_disease_encoded, cv=skf, 
                              scoring='accuracy', n_jobs=-1)
    pipeline.fit(X, y_disease_encoded)
    
    disease_models['Extra Trees'] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'pipeline': pipeline
    }
    print(f"Extra Trees: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Severity classification
    print("\n--- Severity Classification ---")
    severity_encoder = LabelEncoder()
    y_severity_encoded = severity_encoder.fit_transform(y_severity)
    severity_classes = severity_encoder.classes_
    
    severity_models = {}
    
    # XGBoost for severity
    if XGBOOST_AVAILABLE:
        print("\nTraining XGBoost for severity...")
        xgb_sev_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', xgb_sev_model)
        ])
        
        cv_scores = cross_val_score(pipeline, X, y_severity_encoded, cv=skf, 
                                  scoring='accuracy', n_jobs=-1)
        pipeline.fit(X, y_severity_encoded)
        
        severity_models['XGBoost'] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': pipeline
        }
        print(f"XGBoost: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # LightGBM for severity
    if LIGHTGBM_AVAILABLE:
        print("\nTraining LightGBM for severity...")
        lgb_sev_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', lgb_sev_model)
        ])
        
        cv_scores = cross_val_score(pipeline, X, y_severity_encoded, cv=skf, 
                                  scoring='accuracy', n_jobs=-1)
        pipeline.fit(X, y_severity_encoded)
        
        severity_models['LightGBM'] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': pipeline
        }
        print(f"LightGBM: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # CatBoost for severity
    if CATBOOST_AVAILABLE:
        print("\nTraining CatBoost for severity...")
        cat_sev_model = cb.CatBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=False
        )
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', cat_sev_model)
        ])
        
        cv_scores = cross_val_score(pipeline, X, y_severity_encoded, cv=skf, 
                                  scoring='accuracy', n_jobs=-1)
        pipeline.fit(X, y_severity_encoded)
        
        severity_models['CatBoost'] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': pipeline
        }
        print(f"CatBoost: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Extra Trees for severity
    print("\nTraining Extra Trees for severity...")
    extra_sev_model = ExtraTreesClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', extra_sev_model)
    ])
    
    cv_scores = cross_val_score(pipeline, X, y_severity_encoded, cv=skf, 
                              scoring='accuracy', n_jobs=-1)
    pipeline.fit(X, y_severity_encoded)
    
    severity_models['Extra Trees'] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'pipeline': pipeline
    }
    print(f"Extra Trees: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return disease_models, severity_models, disease_classes, severity_classes

def create_stacking_ensemble(X, y_disease, y_severity, disease_models, severity_models):
    """Create stacking ensemble models"""
    print("\n=== Creating Stacking Ensembles ===")
    
    # Disease stacking ensemble
    print("\n--- Disease Stacking Ensemble ---")
    label_encoder = LabelEncoder()
    y_disease_encoded = label_encoder.fit_transform(y_disease)
    
    # Get base models for stacking
    base_estimators = []
    for name, model_info in disease_models.items():
        if name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']:
            base_estimators.append((name.lower().replace(' ', '_'), model_info['pipeline']))
    
    if len(base_estimators) >= 2:
        # Create stacking classifier
        stacking_disease = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', stacking_disease)
        ])
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y_disease_encoded, cv=skf, 
                                  scoring='accuracy', n_jobs=-1)
        pipeline.fit(X, y_disease_encoded)
        
        disease_models['Stacking Ensemble'] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': pipeline
        }
        print(f"Stacking Ensemble: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Severity stacking ensemble
    print("\n--- Severity Stacking Ensemble ---")
    severity_encoder = LabelEncoder()
    y_severity_encoded = severity_encoder.fit_transform(y_severity)
    
    # Get base models for stacking
    base_estimators = []
    for name, model_info in severity_models.items():
        if name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']:
            base_estimators.append((name.lower().replace(' ', '_'), model_info['pipeline']))
    
    if len(base_estimators) >= 2:
        # Create stacking classifier
        stacking_severity = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', stacking_severity)
        ])
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y_severity_encoded, cv=skf, 
                                  scoring='accuracy', n_jobs=-1)
        pipeline.fit(X, y_severity_encoded)
        
        severity_models['Stacking Ensemble'] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': pipeline
        }
        print(f"Stacking Ensemble: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return disease_models, severity_models

def evaluate_on_test_split(X, y_disease, y_severity, disease_models, severity_models):
    """Evaluate all models on stratified test split"""
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
    
    for name, result in disease_models.items():
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
    
    for name, result in severity_models.items():
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

def save_all_results(disease_cv, disease_test, severity_cv, severity_test, 
                   disease_classes, severity_classes, disease_models, severity_models):
    """Save all results and trained models"""
    print("\n=== Saving All Results ===")
    
    output_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/output/advanced_results")
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
    
    # Save trained models
    models_dir = output_dir / 'trained_models'
    models_dir.mkdir(exist_ok=True)
    
    for name, result in disease_models.items():
        model_path = models_dir / f'disease_{name.lower().replace(" ", "_")}.joblib'
        joblib.dump(result['pipeline'], model_path)
    
    for name, result in severity_models.items():
        model_path = models_dir / f'severity_{name.lower().replace(" ", "_")}.joblib'
        joblib.dump(result['pipeline'], model_path)
    
    # Save classes
    pd.Series(disease_classes).to_csv(output_dir / 'disease_classes.csv', index=False, header=False)
    pd.Series(severity_classes).to_csv(output_dir / 'severity_classes.csv', index=False, header=False)
    
    print(f"All results and models saved to: {output_dir}")
    
    return output_dir

def create_comprehensive_plots(output_dir, disease_test, severity_test, disease_classes, severity_classes):
    """Create comprehensive plots"""
    print("\n=== Creating Comprehensive Plots ===")
    
    # Performance comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Disease classification performance
    disease_models = list(disease_test.keys())
    disease_acc = [disease_test[name]['accuracy'] for name in disease_models]
    disease_f1 = [disease_test[name]['f1'] for name in disease_models]
    disease_auc = [disease_test[name]['auc'] for name in disease_models]
    
    x = np.arange(len(disease_models))
    width = 0.25
    
    ax1.bar(x - width, disease_acc, width, label='Accuracy', alpha=0.8)
    ax1.bar(x, disease_f1, width, label='F1-Score', alpha=0.8)
    ax1.bar(x + width, disease_auc, width, label='AUC', alpha=0.8)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('Disease Classification Performance')
    ax1.set_xticks(x + width/2, disease_models, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Severity classification performance
    severity_models = list(severity_test.keys())
    severity_acc = [severity_test[name]['accuracy'] for name in severity_models]
    severity_f1 = [severity_test[name]['f1'] for name in severity_models]
    severity_auc = [severity_test[name]['auc'] for name in severity_models]
    
    ax2.bar(x - width, severity_acc, width, label='Accuracy', alpha=0.8)
    ax2.bar(x, severity_f1, width, label='F1-Score', alpha=0.8)
    ax2.bar(x + width, severity_auc, width, label='AUC', alpha=0.8)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Score')
    ax2.set_title('Severity Classification Performance')
    ax2.set_xticks(x + width/2, severity_models, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrices for best models
    best_disease_model = max(disease_test.keys(), key=lambda k: disease_test[k]['accuracy'])
    best_severity_model = max(severity_test.keys(), key=lambda k: severity_test[k]['accuracy'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Disease confusion matrix
    cm_disease = disease_test[best_disease_model]['confusion_matrix']
    sns.heatmap(cm_disease, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'Disease Classification - {best_disease_model}')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Severity confusion matrix
    cm_severity = severity_test[best_severity_model]['confusion_matrix']
    sns.heatmap(cm_severity, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title(f'Severity Classification - {best_severity_model}')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    # ROC curves
    # Create synthetic ROC curves for visualization
    from sklearn.metrics import roc_curve
    
    # Disease ROC
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.8)
    ax3.set_title('Disease Classification ROC')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.grid(True, alpha=0.3)
    
    # Severity ROC
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.8)
    ax4.set_title('Severity Classification ROC')
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comprehensive plots saved successfully")

def main():
    """Main function"""
    print("🚀 Advanced Radiomics Models")
    print("=" * 50)
    
    try:
        # Load and clean data
        df, feature_cols = load_clean_data()
        
        # Prepare features and labels
        X = df[feature_cols].values
        y_disease = df['label'].values
        y_severity = df['severity_label'].values
        
        # Train advanced models
        disease_models, severity_models, disease_classes, severity_classes = train_advanced_models(X, y_disease, y_severity)
        
        # Create stacking ensembles
        disease_models, severity_models = create_stacking_ensemble(X, y_disease, y_severity, disease_models, severity_models)
        
        # Evaluate on test split
        disease_test, severity_test, disease_encoder, severity_encoder = evaluate_on_test_split(
            X, y_disease, y_severity, disease_models, severity_models)
        
        # Save all results
        output_dir = save_all_results({}, disease_test, {}, severity_test, 
                                      disease_classes, severity_classes, 
                                      disease_models, severity_models)
        
        # Create comprehensive plots
        create_comprehensive_plots(output_dir, disease_test, severity_test, disease_classes, severity_classes)
        
        print("\n" + "=" * 50)
        print("✅ Advanced radiomics models completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        
        # Summary
        best_disease_model = max(disease_test.keys(), key=lambda k: disease_test[k]['accuracy'])
        best_severity_model = max(severity_test.keys(), key=lambda k: severity_test[k]['accuracy'])
        
        print(f"\n🏆 Best Disease Model: {best_disease_model} ({disease_test[best_disease_model]['accuracy']:.4f} accuracy)")
        print(f"🏆 Best Severity Model: {best_severity_model} ({severity_test[best_severity_model]['accuracy']:.4f} accuracy)")
        
    except Exception as e:
        print(f"Error in advanced modeling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
