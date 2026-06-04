"""
BMD/DMD Classification using Radiomics Features
Proxy labels for pipeline validation - NOT real clinical diagnosis
Classifies: 0 = BMD (Becker Muscular Dystrophy), 1 = DMD (Duchenne Muscular Dystrophy)

WARNING: These are synthetic/proxy labels for development only.
Real classification requires clinically labeled MRI data from neurologists.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_ultrasound_data():
    """Load ultrasound radiomics features and proxy BMD/DMD labels
    
    NOTE: Uses synthetic proxy labels for pipeline testing.
    Real clinical data should replace proxy labels when available.
    """
    features_path = Path("output/ultrasound_radiomics_features.csv")
    labels_path = Path("output/labels.csv")
    
    if not features_path.exists():
        logger.error("Ultrasound radiomics features not found!")
        return None, None
    
    if not labels_path.exists():
        logger.warning("Proxy labels not found! Run prepare_training_data.py first")
        logger.warning("NOTE: These are synthetic labels, not real clinical diagnoses")
        return None, None
    
    # Load features
    features_df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(features_df)} ultrasound features")
    
    # Load labels
    labels_df = pd.read_csv(labels_path)
    logger.info(f"Loaded {len(labels_df)} proxy labels (SYNTHETIC FOR TESTING)")
    
    # Merge on image name
    merged_df = features_df.merge(labels_df[['image_name', 'label']], on='image_name')
    logger.info(f"Merged dataset: {len(merged_df)} samples")
    
    # Prepare features and labels
    feature_cols = [col for col in merged_df.columns if col not in ['image_name', 'label', 'diagnostics']]
    X = merged_df[feature_cols].fillna(0)
    y = merged_df['label']
    
    return X, y

def train_and_classify():
    """Train simple classifier and make predictions"""
    
    # Load data
    X, y = load_ultrasound_data()
    if X is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"BMD samples: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    logger.info(f"DMD samples: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Print results
    logger.info("\n=== BMD/DMD CLASSIFICATION RESULTS (PROXY LABELS) ===")
    logger.warning("SYNTHETIC PROXY LABELS - NOT FOR CLINICAL USE")
    logger.info("\nConfusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    
    logger.info("\nClassification Report (Proxy BMD vs DMD):")
    logger.info(classification_report(y_test, y_pred, target_names=['BMD', 'DMD']))
    
    # Create predictions file
    test_images = X_test.index if hasattr(X_test, 'index') else range(len(X_test))
    results_df = pd.DataFrame({
        'image_index': test_images,
        'predicted_class': y_pred,
        'predicted_label': ['BMD' if p == 0 else 'DMD' for p in y_pred],
        'confidence': [max(p, 1-p) * 100 for p in y_proba]
    })
    
    results_df.to_csv('output/proxy_bmd_dmd_predictions.csv', index=False)
    logger.info(f"\nProxy predictions saved to output/proxy_bmd_dmd_predictions.csv")
    logger.warning("WARNING: These are proxy classifications using synthetic labels")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Important Features (Proxy BMD/DMD Classification):")
    logger.info(feature_importance.head(10))
    
    return clf

if __name__ == "__main__":
    train_and_classify()
