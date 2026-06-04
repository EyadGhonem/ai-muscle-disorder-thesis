#!/usr/bin/env python3
"""
Comprehensive ML Model Training: Baseline + Advanced Models
- Proper stratified split by patient
- Baseline: RF, GB, SVM, LR
- Advanced: XGBoost, LightGBM, CatBoost, Extra Trees, Stacking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report)
import pickle
import warnings
warnings.filterwarnings('ignore')

# Install advanced models if not available
try:
    import xgboost as xgb
except:
    print("Installing XGBoost...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost', '-q'])
    import xgboost as xgb

try:
    import lightgbm as lgb
except:
    print("Installing LightGBM...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'lightgbm', '-q'])
    import lightgbm as lgb

try:
    import catboost as cb
except:
    print("Installing CatBoost...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'catboost', '-q'])
    import catboost as cb

np.random.seed(42)


class ComprehensiveModelTrainer:
    """Train baseline and advanced models"""
    
    def __init__(self, output_dir="output/baseline_and_advanced_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.models = {}
        self.scaler = StandardScaler()
    
    def load_and_prepare_data(self):
        """Load master dataset and prepare features"""
        print("\n" + "="*80)
        print("LOADING AND PREPARING DATA")
        print("="*80)
        
        master_file = Path("output/final_ultrasound_dataset.csv")
        df = pd.read_csv(master_file)
        print(f"✓ Loaded: {df.shape}")
        
        # Remove NAN labels
        df = df[df['disease'] != 'NAN'].copy()
        print(f"✓ After cleaning: {df.shape}")
        
        # Extract features
        base_cols = ['image_path', 'patient_id', 'disease', 'severity_label', 'dataset_source']
        feature_cols = [col for col in df.columns if col not in base_cols]
        
        X = df[feature_cols].fillna(0)
        y = df['disease']
        patient_ids = df['patient_id']
        
        print(f"✓ Features: {X.shape}, Labels: {y.shape}")
        print(f"✓ Classes: {y.nunique()}")
        print(f"Classes distribution:\n{y.value_counts()}\n")
        
        return X, y, patient_ids
    
    def stratified_split(self, X, y, patient_ids):
        """Split by patient to avoid leakage"""
        print("="*80)
        print("STRATIFIED TRAIN/TEST SPLIT (by patient)")
        print("="*80)
        
        # Create patient-disease mapping
        patient_disease = pd.DataFrame({
            'patient_id': patient_ids,
            'disease': y
        }).drop_duplicates('patient_id')
        
        # Encode classes for stratification
        self.le = LabelEncoder()
        patient_disease['disease_encoded'] = self.le.fit_transform(patient_disease['disease'])
        
        # Split
        train_pts, test_pts = train_test_split(
            patient_disease, test_size=0.2,
            stratify=patient_disease['disease_encoded'],
            random_state=42
        )
        
        train_idx = patient_ids.isin(train_pts['patient_id']).values
        test_idx = patient_ids.isin(test_pts['patient_id']).values
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Encode labels for XGBoost compatibility
        y_train_encoded = self.le.transform(y_train)
        y_test_encoded = self.le.transform(y_test)
        
        print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
        print(f"Train classes:\n{y_train.value_counts()}\n")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, y_test
    
    def train_baseline_models(self, X_train, X_test, y_train, y_test):
        """Train baseline models"""
        print("\n" + "="*80)
        print("TRAINING BASELINE MODELS")
        print("="*80)
        
        baseline_models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'SVM': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, class_weight='balanced', random_state=42
            )
        }
        
        for name, model in baseline_models.items():
            print(f"\n🔹 {name}...", end=" ", flush=True)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            pre = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            self.results[name] = {
                'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1,
                'y_pred': y_pred, 'cm': confusion_matrix(y_test, y_pred)
            }
            self.models[name] = model
            
            print(f"Acc={acc:.4f} F1={f1:.4f}")
    
    def train_advanced_models(self, X_train, X_test, y_train, y_test):
        """Train advanced models"""
        print("\n" + "="*80)
        print("TRAINING ADVANCED MODELS")
        print("="*80)
        
        # XGBoost
        print("\n🔹 XGBoost...", end=" ", flush=True)
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, random_state=42, tree_method='hist', device='cpu'
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        self.results['XGBoost'] = {'accuracy': acc, 'f1': f1, 'y_pred': y_pred}
        self.models['XGBoost'] = xgb_model
        print(f"Acc={acc:.4f} F1={f1:.4f}")
        
        # LightGBM
        print("🔹 LightGBM...", end=" ", flush=True)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100, random_state=42, verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        y_pred = lgb_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        self.results['LightGBM'] = {'accuracy': acc, 'f1': f1, 'y_pred': y_pred}
        self.models['LightGBM'] = lgb_model
        print(f"Acc={acc:.4f} F1={f1:.4f}")
        
        # CatBoost
        print("🔹 CatBoost...", end=" ", flush=True)
        cat_model = cb.CatBoostClassifier(
            iterations=100, random_state=42, verbose=0
        )
        cat_model.fit(X_train, y_train)
        y_pred = cat_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        self.results['CatBoost'] = {'accuracy': acc, 'f1': f1, 'y_pred': y_pred}
        self.models['CatBoost'] = cat_model
        print(f"Acc={acc:.4f} F1={f1:.4f}")
        
        # Extra Trees
        print("🔹 Extra Trees...", end=" ", flush=True)
        et_model = ExtraTreesClassifier(
            n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
        )
        et_model.fit(X_train, y_train)
        y_pred = et_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        self.results['Extra Trees'] = {'accuracy': acc, 'f1': f1, 'y_pred': y_pred}
        self.models['Extra Trees'] = et_model
        print(f"Acc={acc:.4f} F1={f1:.4f}")
        
        # Stacking Ensemble
        print("🔹 Stacking Ensemble...", end=" ", flush=True)
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=50, random_state=42, verbose=0))
        ]
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42)
        )
        stacking_model.fit(X_train, y_train)
        y_pred = stacking_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        self.results['Stacking'] = {'accuracy': acc, 'f1': f1, 'y_pred': y_pred}
        self.models['Stacking'] = stacking_model
        print(f"Acc={acc:.4f} F1={f1:.4f}")
    
    def generate_summary_report(self, y_test):
        """Generate summary report"""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        summary_data = []
        for name, metrics in self.results.items():
            acc = metrics['accuracy']
            f1 = metrics.get('f1', f1_score(y_test, metrics['y_pred'], average='weighted'))
            precision = precision_score(y_test, metrics['y_pred'], average='weighted', zero_division=0)
            recall = recall_score(y_test, metrics['y_pred'], average='weighted', zero_division=0)
            
            summary_data.append({
                'Model': name,
                'Accuracy': acc,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
        
        df_summary = pd.DataFrame(summary_data).sort_values('Accuracy', ascending=False)
        
        print("\n" + df_summary.to_string(index=False))
        
        # Save CSV
        csv_file = self.output_dir / "model_comparison.csv"
        df_summary.to_csv(csv_file, index=False)
        print(f"\n✓ Saved: {csv_file}")
        
        # Save plot
        fig, ax = plt.subplots(figsize=(12, 6))
        df_summary.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
            kind='bar', ax=ax, width=0.8, edgecolor='black'
        )
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xlabel('Model')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0.5, 1.0])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_file = self.output_dir / "model_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {plot_file}")
        plt.close()
    
    def save_models(self):
        """Save trained models"""
        models_file = self.output_dir / "trained_models.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scaler': self.scaler,
                'results': self.results
            }, f)
        print(f"✓ Saved models: {models_file}")
    
    def run(self):
        """Run complete pipeline"""
        X, y, patient_ids = self.load_and_prepare_data()
        X_train, X_test, y_train, y_test, y_test_orig = self.stratified_split(X, y, patient_ids)
        
        self.train_baseline_models(X_train, X_test, y_train, y_test)
        self.train_advanced_models(X_train, X_test, y_train, y_test)
        
        self.generate_summary_report(y_test)
        self.save_models()
        
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETE")
        print("="*80)


def main():
    trainer = ComprehensiveModelTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
