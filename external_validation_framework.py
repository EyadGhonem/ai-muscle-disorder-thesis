#!/usr/bin/env python3
"""
External validation framework for muscle disease classification
Test models on independent datasets and cross-dataset validation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

def load_trained_models():
    """Load trained models for external validation"""
    print("Loading trained models...")
    
    models_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/output/real_features_results/trained_models")
    
    models = {}
    
    # Load disease classification models
    disease_files = list(models_dir.glob("disease_*_real.joblib"))
    for model_file in disease_files:
        model_name = model_file.stem.replace("disease_", "").replace("_real", "")
        try:
            models[f"disease_{model_name}"] = joblib.load(model_file)
            print(f"Loaded disease model: {model_name}")
        except Exception as e:
            print(f"Error loading {model_file}: {e}")
    
    # Load severity classification models
    severity_files = list(models_dir.glob("severity_*_real.joblib"))
    for model_file in severity_files:
        model_name = model_file.stem.replace("severity_", "").replace("_real", "")
        try:
            models[f"severity_{model_name}"] = joblib.load(model_file)
            print(f"Loaded severity model: {model_name}")
        except Exception as e:
            print(f"Error loading {model_file}: {e}")
    
    return models

def create_external_validation_dataset():
    """Create external validation dataset from FSHD-only data"""
    print("\n=== Creating External Validation Dataset ===")
    
    # Load original FSHD dataset (for external validation)
    fshd_features_path = Path("c:/Users/Lenovo/Desktop/thesis_project/processed_data/custom_features.csv")
    if not fshd_features_path.exists():
        print("Original FSHD features not found")
        return None, None, None
    
    fshd_df = pd.read_csv(fshd_features_path)
    print(f"Loaded original FSHD dataset: {fshd_df.shape}")
    
    # Prepare external validation data
    # Use only FSHD samples as external validation
    external_df = fshd_df.copy()
    
    # Create labels for external validation
    external_df['label'] = 'FSHD'  # All FSHD
    external_df['severity_label'] = external_df['binary_label'].map({0: 'Mild', 1: 'Severe'})
    
    # Get feature columns (match real features format)
    feature_cols = [col for col in external_df.columns if col.startswith('feature_') or col in 
                   ['mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 
                    'median_intensity', 'q25_intensity', 'q75_intensity', 'skewness', 
                    'kurtosis', 'entropy', 'glcm_contrast', 'glcm_dissimilarity',
                    'glcm_homogeneity', 'glcm_energy', 'glcm_correlation', 'glcm_asm',
                    'area', 'perimeter', 'circularity', 'aspect_ratio', 'extent', 
                    'solidity', 'equivalent_diameter', 'gradient_mean', 'gradient_std',
                    'gradient_max', 'gradient_energy']]
    
    # Fill missing values
    for col in feature_cols:
        if col in external_df.columns:
            external_df[col] = external_df[col].fillna(external_df[col].median())
    
    print(f"External validation dataset: {external_df.shape}")
    print(f"Feature columns: {len(feature_cols)}")
    
    return external_df, feature_cols, 'FSHD_External'

def create_cross_dataset_validation():
    """Create cross-dataset validation scenarios"""
    print("\n=== Creating Cross-Dataset Validation ===")
    
    # Load real features dataset
    real_features_path = Path("c:/Users/Lenovo/Desktop/thesis_project/final_ultrasound_dataset_REAL_features.csv")
    if not real_features_path.exists():
        print("Real features dataset not found")
        return None
    
    df = pd.read_csv(real_features_path)
    print(f"Loaded real features dataset: {df.shape}")
    
    # Create cross-dataset validation scenarios
    validation_scenarios = {}
    
    # Scenario 1: Train on FSHD, test on Other diseases
    fshd_data = df[df['label'] == 'FSHD']
    other_data = df[df['label'] == 'Other']
    
    if len(fshd_data) > 0 and len(other_data) > 0:
        validation_scenarios['FSHD_to_Other'] = {
            'train_data': fshd_data,
            'test_data': other_data,
            'description': 'Train on FSHD, Test on Other Diseases'
        }
        
        validation_scenarios['Other_to_FSHD'] = {
            'train_data': other_data,
            'test_data': fshd_data,
            'description': 'Train on Other Diseases, Test on FSHD'
        }
    
    # Scenario 2: Train on combined, test on hold-out
    train_data, test_data = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    
    validation_scenarios['Combined_Holdout'] = {
        'train_data': train_data,
        'test_data': test_data,
        'description': 'Train on Combined, Test on Hold-out'
    }
    
    print(f"Created {len(validation_scenarios)} validation scenarios:")
    for name, scenario in validation_scenarios.items():
        print(f"  {name}: {scenario['description']}")
        print(f"    Train: {len(scenario['train_data'])}, Test: {len(scenario['test_data'])}")
    
    return validation_scenarios

def validate_model_on_external_data(model, X_test, y_test, task_name, model_name):
    """Validate single model on external data"""
    try:
        # Predict
        y_pred = model.predict(X_test)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            if task_name == 'disease':
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # AUC calculation
        if y_pred_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc = 0.5
        else:
            auc = 0.5
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'model_name': model_name,
            'task': task_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm
        }
        
        return results
        
    except Exception as e:
        print(f"Error validating {model_name}: {e}")
        return None

def run_external_validation():
    """Run comprehensive external validation"""
    print("🔍 EXTERNAL VALIDATION FRAMEWORK")
    print("=" * 50)
    
    # Load trained models
    models = load_trained_models()
    if not models:
        print("No models loaded")
        return
    
    # Create external validation dataset
    external_df, feature_cols, dataset_name = create_external_validation_dataset()
    if external_df is None:
        print("Cannot create external validation dataset")
        return
    
    # Create cross-dataset validation scenarios
    validation_scenarios = create_cross_dataset_validation()
    
    # Run validation scenarios
    all_results = []
    
    print("\n=== Running External Validation ===")
    
    # Validate on external FSHD dataset
    if external_df is not None:
        print(f"\n--- External Validation on {dataset_name} ---")
        
        X_external = external_df[feature_cols].values
        y_disease_external = external_df['label'].values
        y_severity_external = external_df['severity_label'].values
        
        # Encode labels
        disease_encoder = LabelEncoder()
        y_disease_external_encoded = disease_encoder.fit_transform(y_disease_external)
        
        severity_encoder = LabelEncoder()
        y_severity_external_encoded = severity_encoder.fit_transform(y_severity_external)
        
        # Test disease models
        for model_key in models.keys():
            if model_key.startswith('disease_'):
                model = models[model_key]
                model_name = model_key.replace('disease_', '')
                
                results = validate_model_on_external_data(
                    model, X_external, y_disease_external_encoded, 
                    'disease', model_name
                )
                
                if results:
                    results['validation_type'] = 'external'
                    results['dataset'] = dataset_name
                    all_results.append(results)
        
        # Test severity models
        for model_key in models.keys():
            if model_key.startswith('severity_'):
                model = models[model_key]
                model_name = model_key.replace('severity_', '')
                
                results = validate_model_on_external_data(
                    model, X_external, y_severity_external_encoded, 
                    'severity', model_name
                )
                
                if results:
                    results['validation_type'] = 'external'
                    results['dataset'] = dataset_name
                    all_results.append(results)
    
    # Run cross-dataset validation
    if validation_scenarios:
        print("\n--- Cross-Dataset Validation ---")
        
        for scenario_name, scenario in validation_scenarios.items():
            print(f"\nScenario: {scenario['description']}")
            
            train_data = scenario['train_data']
            test_data = scenario['test_data']
            
            # Prepare data
            X_train = train_data[feature_cols].values
            X_test = test_data[feature_cols].values
            
            y_train_disease = train_data['label'].values
            y_test_disease = test_data['label'].values
            
            y_train_severity = train_data['severity_label'].values
            y_test_severity = test_data['severity_label'].values
            
            # Encode labels
            disease_encoder = LabelEncoder()
            y_train_disease_encoded = disease_encoder.fit_transform(y_train_disease)
            y_test_disease_encoded = disease_encoder.transform(y_test_disease)
            
            severity_encoder = LabelEncoder()
            y_train_severity_encoded = severity_encoder.fit_transform(y_train_severity)
            y_test_severity_encoded = severity_encoder.transform(y_test_severity)
            
            # Train and test models
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.pipeline import Pipeline
            
            # Disease classification
            rf_disease = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf_disease_pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', rf_disease)])
            rf_disease_pipeline.fit(X_train, y_train_disease_encoded)
            
            results = validate_model_on_external_data(
                rf_disease_pipeline, X_test, y_test_disease_encoded, 
                'disease', 'RandomForest'
            )
            
            if results:
                results['validation_type'] = 'cross_dataset'
                results['dataset'] = scenario_name
                all_results.append(results)
            
            # Severity classification
            rf_severity = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf_severity_pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', rf_severity)])
            rf_severity_pipeline.fit(X_train, y_train_severity_encoded)
            
            results = validate_model_on_external_data(
                rf_severity_pipeline, X_test, y_test_severity_encoded, 
                'severity', 'RandomForest'
            )
            
            if results:
                results['validation_type'] = 'cross_dataset'
                results['dataset'] = scenario_name
                all_results.append(results)
    
    return all_results

def save_validation_results(results):
    """Save external validation results"""
    print("\n=== Saving Validation Results ===")
    
    if not results:
        print("No results to save")
        return
    
    output_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/output/external_validation")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_dir / 'external_validation_results.csv', index=False)
    
    # Create summary
    print("\nExternal Validation Summary:")
    print("=" * 50)
    
    for validation_type in results_df['validation_type'].unique():
        print(f"\n{validation_type.upper()} VALIDATION:")
        subset = results_df[results_df['validation_type'] == validation_type]
        
        for task in subset['task'].unique():
            task_subset = subset[subset['task'] == task]
            print(f"\n{task.upper()} Classification:")
            
            for _, row in task_subset.iterrows():
                print(f"  {row['model_name']}: Acc={row['accuracy']:.4f}, F1={row['f1']:.4f}, AUC={row['auc']:.4f}")
    
    # Create performance comparison plot
    plt.figure(figsize=(12, 8))
    
    validation_types = results_df['validation_type'].unique()
    tasks = results_df['task'].unique()
    
    x = np.arange(len(validation_types))
    width = 0.35
    
    for i, task in enumerate(tasks):
        task_results = results_df[results_df['task'] == task]
        accuracies = []
        
        for val_type in validation_types:
            val_results = task_results[task_results['validation_type'] == val_type]
            if len(val_results) > 0:
                accuracies.append(val_results['accuracy'].mean())
            else:
                accuracies.append(0)
        
        plt.bar(x + i*width, accuracies, width, label=f'{task} Classification')
    
    plt.xlabel('Validation Type')
    plt.ylabel('Accuracy')
    plt.title('External Validation Performance Comparison')
    plt.xticks(x + width/2, validation_types)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'external_validation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ External validation results saved to: {output_dir}")
    
    return output_dir

def main():
    """Main function"""
    print("🔍 External Validation Framework")
    print("=" * 50)
    
    try:
        # Run external validation
        results = run_external_validation()
        
        # Save results
        if results:
            output_dir = save_validation_results(results)
            
            print("\n" + "=" * 50)
            print("✅ External validation completed!")
            print(f"📁 Results saved to: {output_dir}")
            
            # Summary statistics
            results_df = pd.DataFrame(results)
            print(f"\n📊 Summary:")
            print(f"  Total validation tests: {len(results)}")
            print(f"  Validation types: {results_df['validation_type'].nunique()}")
            print(f"  Tasks evaluated: {results_df['task'].nunique()}")
            print(f"  Average external accuracy: {results_df['accuracy'].mean():.4f}")
        else:
            print("❌ No validation results generated")
    
    except Exception as e:
        print(f"Error in external validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
