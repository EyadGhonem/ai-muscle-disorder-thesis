#!/usr/bin/env python3
"""
Train both ML and DL models on ultrasound image features
Compare performance and generate results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

def load_and_prepare_data():
    """Load features and prepare for training"""
    print("Loading and preparing data...")
    
    # Load features
    features_file = Path("c:/Users/Lenovo/Desktop/thesis_project/processed_data/custom_features.csv")
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    df = pd.read_csv(features_file)
    print(f"Loaded dataset with shape: {df.shape}")
    
    # Separate features and labels
    metadata_cols = ['filename', 'subject', 'muscle_code', 'side', 'instance', 
                   'grade_category', 'original_grade']
    feature_cols = [col for col in df.columns if col not in metadata_cols + ['binary_label']]
    
    X = df[feature_cols]
    y = df['binary_label']
    
    print(f"Features: {X.shape}, Labels: {y.shape}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    return X, y, df, metadata_cols, feature_cols

def train_ml_models(X_train, y_train, X_test, y_test):
    """Train multiple ML models and evaluate them"""
    print("\n=== Training ML Models ===")
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': pipeline,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    return results

def create_dl_model(input_dim):
    """Create deep learning model"""
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
    """Train deep learning model"""
    print("\n=== Training DL Model ===")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    model = create_dl_model(X_train_scaled.shape[1])
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ]
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
        class_weight={0: 1, 1: len(y_train[y_train==0])/len(y_train[y_train==1])}  # Handle class imbalance
    )
    
    # Predictions
    y_pred_proba = model.predict(X_test_scaled).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    dl_results = {
        'model': model,
        'scaler': scaler,
        'history': history,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"DL Model - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    return dl_results

def evaluate_models(ml_results, dl_results, y_test):
    """Compare all models and create evaluation plots"""
    print("\n=== Model Comparison ===")
    
    # Combine results
    all_results = {**ml_results, 'Deep Learning': dl_results}
    
    # Create comparison table
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
    print("\nModel Performance Comparison:")
    print(comparison_df.round(4))
    
    # Create output directory
    output_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/results")
    output_dir.mkdir(exist_ok=True)
    
    # Save comparison table
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        values = [results[metric] for results in all_results.values()]
        models = list(all_results.keys())
        
        bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(models)])
        ax.set_title(metric_name)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Rotate x labels if needed
        if len(models) > 3:
            ax.tick_params(axis='x', rotation=45)
    
    # Remove empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create ROC curves
    plt.figure(figsize=(10, 8))
    
    for name, results in all_results.items():
        fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
        auc = results['auc']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_df, all_results

def create_confusion_matrices(all_results, y_test):
    """Create confusion matrices for all models"""
    output_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/results")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Confusion Matrices', fontsize=16)
    
    models = list(all_results.keys())
    
    for i, (name, results) in enumerate(all_results.items()):
        if i >= 6:  # Maximum 6 subplots
            break
            
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        cm = confusion_matrix(y_test, results['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal/Mild', 'Moderate/Severe'],
                   yticklabels=['Normal/Mild', 'Moderate/Severe'])
        ax.set_title(f'{name}\nAccuracy: {results["accuracy"]:.3f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Remove empty subplots
    for i in range(len(all_results), 6):
        row, col = i // 3, i % 3
        axes[row, col].remove()
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_predictions(all_results, y_test, df_test):
    """Save predictions for further analysis"""
    output_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/results")
    
    # Create predictions dataframe
    predictions_data = []
    
    for name, results in all_results.items():
        for i, (actual, pred, prob) in enumerate(zip(y_test, results['predictions'], results['probabilities'])):
            predictions_data.append({
                'Model': name,
                'Actual': actual,
                'Predicted': pred,
                'Probability': prob,
                'Correct': actual == pred
            })
    
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    
    # Create detailed classification reports
    for name, results in all_results.items():
        report = classification_report(y_test, results['predictions'], 
                                   target_names=['Normal/Mild', 'Moderate/Severe'])
        
        with open(output_dir / f"classification_report_{name.lower().replace(' ', '_')}.txt", 'w') as f:
            f.write(f"Classification Report - {name}\n")
            f.write("="*50 + "\n\n")
            f.write(report)

def main():
    """Main training and evaluation pipeline"""
    print("=== ML and DL Model Training Pipeline ===")
    
    # Load data
    X, y, df, metadata_cols, feature_cols = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train ML models
    ml_results = train_ml_models(X_train, y_train, X_test, y_test)
    
    # Train DL model
    dl_results = train_dl_model(X_train, y_train, X_test, y_test)
    
    # Evaluate and compare models
    comparison_df, all_results = evaluate_models(ml_results, dl_results, y_test)
    
    # Create confusion matrices
    create_confusion_matrices(all_results, y_test)
    
    # Save predictions
    save_predictions(all_results, y_test, None)
    
    # Create final summary
    output_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/results")
    summary_text = f"""
Model Training Summary
=====================

Dataset Information:
- Total samples: {len(X)}
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}
- Features: {X.shape[1]}
- Class distribution: {y.value_counts().to_dict()}

Best Performing Models:
- Best Accuracy: {comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']} ({comparison_df['Accuracy'].max():.4f})
- Best Precision: {comparison_df.loc[comparison_df['Precision'].idxmax(), 'Model']} ({comparison_df['Precision'].max():.4f})
- Best Recall: {comparison_df.loc[comparison_df['Recall'].idxmax(), 'Model']} ({comparison_df['Recall'].max():.4f})
- Best F1-Score: {comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']} ({comparison_df['F1-Score'].max():.4f})
- Best AUC: {comparison_df.loc[comparison_df['AUC'].idxmax(), 'Model']} ({comparison_df['AUC'].max():.4f})

Files Generated:
- model_comparison.csv: Performance comparison table
- model_comparison.png: Performance comparison plots
- roc_curves.png: ROC curves for all models
- confusion_matrices.png: Confusion matrices
- predictions.csv: Detailed predictions
- classification_report_*.txt: Individual classification reports

Next Steps:
1. Analyze feature importance
2. Perform hyperparameter tuning
3. Test on external validation set
4. Deploy best model
"""
    
    with open(output_dir / "training_summary.txt", 'w') as f:
        f.write(summary_text)
    
    print(f"\n=== Training Complete ===")
    print(f"Results saved to: {output_dir}")
    print(summary_text)

if __name__ == "__main__":
    # Install TensorFlow if needed
    try:
        import tensorflow
    except ImportError:
        print("Installing TensorFlow...")
        os.system("pip install tensorflow")
        import tensorflow
    
    main()
