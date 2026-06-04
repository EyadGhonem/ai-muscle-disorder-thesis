#!/usr/bin/env python3
"""
Retrain models with REAL radiomics features and add clinical interpretation
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
                           roc_auc_score, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP for clinical interpretation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not available, installing...")
    os.system("pip install shap")
    try:
        import shap
        SHAP_AVAILABLE = True
    except ImportError:
        SHAP_AVAILABLE = False

# Try to import advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Set random seeds for reproducibility
np.random.seed(42)

def load_real_features_data():
    """Load dataset with real radiomics features"""
    print("Loading dataset with REAL radiomics features...")
    
    dataset_path = Path("c:/Users/Lenovo/Desktop/thesis_project/final_ultrasound_dataset_REAL_features.csv")
    if not dataset_path.exists():
        print(f"Real features dataset not found: {dataset_path}")
        return None, None, None
    
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

def create_clinical_feature_mapping():
    """Create mapping of features to clinical meaning"""
    clinical_mapping = {
        'feature_1': 'Mean Intensity - Average echogenicity (fat vs muscle)',
        'feature_2': 'Standard Deviation - Intensity variation',
        'feature_3': 'Minimum Intensity - Darkest tissue regions',
        'feature_4': 'Maximum Intensity - Brightest tissue regions',
        'feature_5': 'Median Intensity - Central echogenicity',
        'feature_6': '25th Percentile - Lower intensity range',
        'feature_7': '75th Percentile - Upper intensity range',
        'feature_8': 'Variance - Intensity spread',
        'feature_9': 'Skewness - Asymmetry of intensity distribution',
        'feature_10': 'Kurtosis - Tailedness of intensity distribution',
        'feature_11': 'Entropy - Tissue organization/disorganization',
        'feature_12': 'GLCM Contrast - Muscle texture variation',
        'feature_13': 'GLCM Dissimilarity - Local intensity differences',
        'feature_14': 'GLCM Homogeneity - Tissue uniformity',
        'feature_15': 'GLCM Energy - Texture uniformity',
        'feature_16': 'GLCM Correlation - Spatial dependency',
        'feature_17': 'GLCM ASM - Angular second moment',
        'feature_18': 'Area - Muscle cross-sectional area',
        'feature_19': 'Perimeter - Muscle boundary length',
        'feature_20': 'Circularity - Muscle shape compactness',
        'feature_21': 'Aspect Ratio - Muscle elongation',
        'feature_22': 'Extent - Muscle space utilization',
        'feature_23': 'Solidity - Muscle convexity',
        'feature_24': 'Equivalent Diameter - Circle equivalent size',
        'feature_25': 'Gradient Mean - Average edge strength',
        'feature_26': 'Gradient Std - Edge variation',
        'feature_27': 'Gradient Max - Strongest edge intensity'
    }
    
    return clinical_mapping

def train_models_with_real_features(X, y_disease, y_severity, feature_names):
    """Train models with real radiomics features"""
    print("\n=== Training Models with REAL Radiomics Features ===")
    
    # Disease classification
    print("\n--- Disease Classification ---")
    label_encoder = LabelEncoder()
    y_disease_encoded = label_encoder.fit_transform(y_disease)
    disease_classes = label_encoder.classes_
    
    # Stratified 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    disease_models = {}
    
    # Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', rf_model)])
    cv_scores = cross_val_score(pipeline, X, y_disease_encoded, cv=skf, scoring='accuracy', n_jobs=-1)
    pipeline.fit(X, y_disease_encoded)
    disease_models['Random Forest'] = {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(), 'pipeline': pipeline}
    print(f"Random Forest: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb_model = GradientBoostingClassifier(random_state=42)
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', gb_model)])
    cv_scores = cross_val_score(pipeline, X, y_disease_encoded, cv=skf, scoring='accuracy', n_jobs=-1)
    pipeline.fit(X, y_disease_encoded)
    disease_models['Gradient Boosting'] = {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(), 'pipeline': pipeline}
    print(f"Gradient Boosting: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        print("\nTraining XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')
        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', xgb_model)])
        cv_scores = cross_val_score(pipeline, X, y_disease_encoded, cv=skf, scoring='accuracy', n_jobs=-1)
        pipeline.fit(X, y_disease_encoded)
        disease_models['XGBoost'] = {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(), 'pipeline': pipeline}
        print(f"XGBoost: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Severity classification
    print("\n--- Severity Classification ---")
    severity_encoder = LabelEncoder()
    y_severity_encoded = severity_encoder.fit_transform(y_severity)
    severity_classes = severity_encoder.classes_
    
    severity_models = {}
    
    # Random Forest for severity
    print("\nTraining Random Forest for severity...")
    rf_sev_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', rf_sev_model)])
    cv_scores = cross_val_score(pipeline, X, y_severity_encoded, cv=skf, scoring='accuracy', n_jobs=-1)
    pipeline.fit(X, y_severity_encoded)
    severity_models['Random Forest'] = {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(), 'pipeline': pipeline}
    print(f"Random Forest: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Gradient Boosting for severity
    print("\nTraining Gradient Boosting for severity...")
    gb_sev_model = GradientBoostingClassifier(random_state=42)
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', gb_sev_model)])
    cv_scores = cross_val_score(pipeline, X, y_severity_encoded, cv=skf, scoring='accuracy', n_jobs=-1)
    pipeline.fit(X, y_severity_encoded)
    severity_models['Gradient Boosting'] = {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(), 'pipeline': pipeline}
    print(f"Gradient Boosting: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return disease_models, severity_models, disease_classes, severity_classes

def create_shap_analysis(model, X, feature_names, clinical_mapping, task_name):
    """Create SHAP analysis for clinical interpretation"""
    if not SHAP_AVAILABLE:
        print("SHAP not available for clinical interpretation")
        return None
    
    print(f"\n=== SHAP Analysis for {task_name} ===")
    
    try:
        # Get the classifier from pipeline
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['classifier']
            X_scaled = model.named_steps['scaler'].transform(X)
        else:
            classifier = model
            X_scaled = X
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_scaled)
        
        # For binary classification, get positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        # Create feature importance summary
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })
        importance_df['clinical_meaning'] = importance_df['feature'].map(clinical_mapping)
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features for {task_name}:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
            print(f"    Clinical: {row['clinical_meaning']}")
        
        return importance_df, shap_values, explainer
        
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        return None

def evaluate_on_test_split(X, y_disease, y_severity, disease_models, severity_models):
    """Evaluate models on test split"""
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
    
    return disease_test_results, severity_test_results, X_test, y_disease_test_encoded, y_severity_test_encoded

def save_all_results(disease_test, severity_test, disease_models, severity_models, 
                    disease_classes, severity_classes, clinical_mapping):
    """Save all results with clinical interpretation"""
    print("\n=== Saving Results with Clinical Interpretation ===")
    
    output_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/output/real_features_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
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
    test_df.to_csv(output_dir / 'real_features_test_results.csv', index=False)
    
    # Save trained models
    models_dir = output_dir / 'trained_models'
    models_dir.mkdir(exist_ok=True)
    
    for name, result in disease_models.items():
        model_path = models_dir / f'disease_{name.lower().replace(" ", "_")}_real.joblib'
        joblib.dump(result['pipeline'], model_path)
    
    for name, result in severity_models.items():
        model_path = models_dir / f'severity_{name.lower().replace(" ", "_")}_real.joblib'
        joblib.dump(result['pipeline'], model_path)
    
    # Save clinical mapping
    mapping_df = pd.DataFrame.from_dict(clinical_mapping, orient='index', columns=['Clinical_Meaning'])
    mapping_df.to_csv(output_dir / 'clinical_feature_mapping.csv')
    
    # Save classes
    pd.Series(disease_classes).to_csv(output_dir / 'disease_classes.csv', index=False, header=False)
    pd.Series(severity_classes).to_csv(output_dir / 'severity_classes.csv', index=False, header=False)
    
    print(f"Results saved to: {output_dir}")
    
    return output_dir

def main():
    """Main function"""
    print("🏥 Training Models with REAL Radiomics Features & Clinical Interpretation")
    print("=" * 70)
    
    try:
        # Load real features data
        df, feature_cols = load_real_features_data()
        if df is None:
            return
        
        # Create clinical mapping
        clinical_mapping = create_clinical_feature_mapping()
        
        # Prepare features and labels
        X = df[feature_cols].values
        y_disease = df['label'].values
        y_severity = df['severity_label'].values
        
        print(f"\nDataset Summary:")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Disease classes: {np.unique(y_disease)}")
        print(f"  Severity classes: {np.unique(y_severity)}")
        
        # Train models with real features
        disease_models, severity_models, disease_classes, severity_classes = train_models_with_real_features(
            X, y_disease, y_severity, feature_cols)
        
        # Evaluate on test split
        disease_test, severity_test, X_test, y_disease_test, y_severity_test = evaluate_on_test_split(
            X, y_disease, y_severity, disease_models, severity_models)
        
        # SHAP analysis for best models
        best_disease_model = max(disease_test.keys(), key=lambda k: disease_test[k]['accuracy'])
        best_severity_model = max(severity_test.keys(), key=lambda k: severity_test[k]['accuracy'])
        
        print(f"\nBest Disease Model: {best_disease_model}")
        print(f"Best Severity Model: {best_severity_model}")
        
        # SHAP analysis
        if SHAP_AVAILABLE:
            disease_shap = create_shap_analysis(
                disease_models[best_disease_model]['pipeline'], 
                X_test, feature_cols, clinical_mapping, "Disease Classification")
            
            severity_shap = create_shap_analysis(
                severity_models[best_severity_model]['pipeline'], 
                X_test, feature_cols, clinical_mapping, "Severity Classification")
        
        # Save all results
        output_dir = save_all_results(disease_test, severity_test, disease_models, severity_models,
                                     disease_classes, severity_classes, clinical_mapping)
        
        print("\n" + "=" * 70)
        print("✅ Models trained with REAL radiomics features completed!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"🏥 Clinical interpretation available with SHAP analysis")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"  Best Disease Model: {best_disease_model} ({disease_test[best_disease_model]['accuracy']:.4f} accuracy)")
        print(f"  Best Severity Model: {best_severity_model} ({severity_test[best_severity_model]['accuracy']:.4f} accuracy)")
        print(f"  Features: REAL radiomics from actual ultrasound images")
        print(f"  Clinical Interpretation: SHAP feature importance analysis")
        
    except Exception as e:
        print(f"Error in training with real features: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
