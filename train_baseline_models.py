#!/usr/bin/env python3
"""
Baseline Machine Learning Models with Proper Validation
- Stratified split by patient (avoid patient leakage)
- 5-fold cross-validation
- Confusion matrix, accuracy, precision, recall, F1, AUC
- Class weighting for imbalanced data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report, roc_curve,
                           auc, make_scorer)
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)


class BaselineModelTrainer:
    """Train baseline ML models with proper validation"""
    
    def __init__(self, output_dir="output/03_baseline_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.models = {}
        
    def load_data(self):
        """Load and prepare data"""
        print("\n" + "="*80)
        print("LOADING AND PREPARING DATA")
        print("="*80)
        
        master_file = Path("output/final_ultrasound_dataset.csv")
        if not master_file.exists():
            print(f"❌ Master dataset not found: {master_file}")
            return None
        
        df = pd.read_csv(master_file)
        print(f"✓ Loaded dataset: {df.shape}")
        
        # Remove NaN disease labels
        df = df[df['disease'] != 'NAN'].copy()
        print(f"✓ After removing NAN labels: {df.shape}")
        
        # Extract radiomics features
        base_cols = ['image_path', 'patient_id', 'disease', 'severity_label', 'dataset_source']
        feature_cols = [col for col in df.columns if col not in base_cols]
        
        X = df[feature_cols].fillna(0)
        y = df['disease']
        patient_ids = df['patient_id']
        
        print(f"✓ Features: {X.shape}, Labels: {y.shape}")
        print(f"Disease distribution:\n{y.value_counts()}")
        
        return X, y, patient_ids, df
    
    def stratified_patient_split(self, X, y, patient_ids, test_size=0.2):
        """Split data by patient to avoid patient leakage"""
        print("\n" + "="*80)
        print("STRATIFIED TRAIN/TEST SPLIT (by patient)")
        print("="*80)
        
        # Get unique patients
        patient_disease = pd.DataFrame({
            'patient_id': patient_ids,
            'disease': y
        }).drop_duplicates('patient_id')
        
        # Stratified split at patient level
        encoder = LabelEncoder()
        patient_disease['disease_encoded'] = encoder.fit_transform(patient_disease['disease'])
        
        train_patients, test_patients = train_test_split(
            patient_disease,
            test_size=test_size,
            stratify=patient_disease['disease_encoded'],
            random_state=42
        )
        
        # Get indices for train/test
        train_idx = patient_ids.isin(train_patients['patient_id']).values
        test_idx = patient_ids.isin(test_patients['patient_id']).values
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"✓ Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
        print(f"✓ Train patients: {len(train_patients)}, Test patients: {len(test_patients)}")
        print(f"\nTrain disease distribution:\n{y_train.value_counts()}")
        print(f"\nTest disease distribution:\n{y_test.value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def train_baseline_models(self, X_train, X_test, y_train, y_test):
        """Train baseline models with class weighting"""
        print("\n" + "="*80)
        print("TRAINING BASELINE MODELS")
        print("="*80)
        
        # Define models with class weights
        models_config = {
            'Random Forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'params': {}
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=42,
                    learning_rate=0.1
                ),
                'params': {}
            },
            'SVM': {
                'model': SVC(
                    kernel='rbf',
                    probability=True,
                    class_weight='balanced',
                    random_state=42
                ),
                'params': {}
            },
            'Logistic Regression': {
                'model': LogisticRegression(
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=42
                ),
                'params': {}
            }
        }
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        for name, config in models_config.items():
            print(f"\n🔹 Training {name}...")
            
            model = config['model']
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Get probabilities for multiclass
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)
            else:
                y_pred_proba = model.decision_function(X_test_scaled)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # For multiclass AUC
            try:
                from sklearn.preprocessing import label_binarize
                y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
                if y_test_bin.shape[1] > 2:
                    roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='weighted', labels=np.unique(y_train))
                else:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                roc_auc = 0.0
            
            self.results[name] = {
                'model': model,
                'scaler': scaler,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'cm': confusion_matrix(y_test, y_pred)
            }
            
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  AUC:       {roc_auc:.4f}")
            
            self.models[name] = model
        
        return y_pred, y_test
    
    def cross_validation_analysis(self, X, y, patient_ids):
        """Perform 5-fold cross-validation"""
        print("\n" + "="*80)
        print("5-FOLD CROSS-VALIDATION ANALYSIS")
        print("="*80)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use patient-based stratified fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_results_by_model = {}
        
        for model_name in self.results.keys():
            print(f"\n🔹 {model_name}:")
            
            model = self.models[model_name]
            
            # Define scoring metrics
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score, average='weighted', zero_division=0),
                'recall': make_scorer(recall_score, average='weighted', zero_division=0),
                'f1': make_scorer(f1_score, average='weighted', zero_division=0),
            }
            
            # Cross-validate
            cv_results = cross_validate(model, X_scaled, y, cv=skf, scoring=scoring, n_jobs=-1)
            
            cv_results_by_model[model_name] = cv_results
            
            # Print results
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                scores = cv_results[f'test_{metric}']
                print(f"  {metric.capitalize():10s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_results_by_model
    
    def generate_confusion_matrices(self, y_test, classes):
        """Generate confusion matrices for all models"""
        print("\n" + "="*80)
        print("GENERATING CONFUSION MATRICES")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
        
        for idx, (name, results) in enumerate(self.results.items()):
            if idx < 4:
                cm = results['cm']
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                           xticklabels=classes, yticklabels=classes, cbar=True)
                axes[idx].set_title(f'{name} - Confusion Matrix\nAccuracy: {results["accuracy"]:.4f}',
                                   fontsize=11, fontweight='bold')
                axes[idx].set_ylabel('True Label')
                axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        output_file = self.output_dir / "confusion_matrices.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def generate_metrics_comparison(self):
        """Generate metrics comparison plot"""
        print("\n" + "="*80)
        print("GENERATING METRICS COMPARISON")
        print("="*80)
        
        metrics_data = []
        for model_name, results in self.results.items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1'],
                'AUC': results['auc']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        df_metrics.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
            kind='bar', ax=axes[0], edgecolor='black', width=0.8)
        axes[0].set_title('Model Comparison - Classification Metrics', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Score')
        axes[0].set_xlabel('Model')
        axes[0].legend(loc='lower right')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Radar chart would go here, but let's do a heatmap instead
        df_metrics_norm = df_metrics.set_index('Model').apply(lambda x: x / x.max())
        sns.heatmap(df_metrics_norm.T, annot=df_metrics.set_index('Model').T.round(3),
                   fmt='g', cmap='RdYlGn', ax=axes[1], cbar_kws={'label': 'Normalized Score'},
                   vmin=0, vmax=1)
        axes[1].set_title('Model Comparison - Normalized Heatmap', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_file = self.output_dir / "metrics_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
        
        # Save to CSV
        csv_file = self.output_dir / "baseline_models_results.csv"
        df_metrics.to_csv(csv_file, index=False)
        print(f"✓ Saved: {csv_file}")
        
        # Print table
        print("\n📊 BASELINE MODELS COMPARISON:")
        print(df_metrics.to_string(index=False))
    
    def save_detailed_results(self, y_test):
        """Save detailed results and classification reports"""
        print("\n" + "="*80)
        print("SAVING DETAILED RESULTS")
        print("="*80)
        
        # Get unique classes
        classes = sorted(np.unique(y_test))
        
        # Classification reports
        report_file = self.output_dir / "classification_reports.txt"
        with open(report_file, 'w') as f:
            for model_name, results in self.results.items():
                f.write(f"\n{'='*80}\n{model_name}\n{'='*80}\n")
                f.write(f"Accuracy:  {results['accuracy']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall:    {results['recall']:.4f}\n")
                f.write(f"F1-Score:  {results['f1']:.4f}\n")
                f.write(f"AUC:       {results['auc']:.4f}\n\n")
                f.write(classification_report(y_test, results['y_pred'], target_names=classes))
        
        print(f"✓ Saved: {report_file}")
        
        # Save models
        import pickle
        models_file = self.output_dir / "trained_models.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"✓ Saved: {models_file}")
    
    def run(self):
        """Run complete baseline training pipeline"""
        print("\n" + "="*80)
        print("BASELINE ML MODELS WITH PROPER VALIDATION")
        print("="*80)
        
        # Load data
        X, y, patient_ids, df = self.load_data()
        if X is None:
            return False
        
        # Stratified split
        X_train, X_test, y_train, y_test = self.stratified_patient_split(X, y, patient_ids)
        
        # Train models
        self.train_baseline_models(X_train, X_test, y_train, y_test)
        
        # Cross-validation
        cv_results = self.cross_validation_analysis(X, y, patient_ids)
        
        # Generate visualizations
        classes = sorted(np.unique(y))
        self.generate_confusion_matrices(y_test, classes)
        self.generate_metrics_comparison()
        
        # Save results
        self.save_detailed_results(y_test)
        
        print("\n" + "="*80)
        print("✓ BASELINE MODELS TRAINING COMPLETE")
        print("="*80)
        
        return True


def main():
    trainer = BaselineModelTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
