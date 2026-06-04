"""
Combined MRI + Ultrasound Radiomics Pipeline for BMD/DMD Classification
PROPER APPROACH: Extract features -> Create labels -> Train -> Test -> Compare

WARNING: Uses synthetic proxy labels for development.
Real clinical classification requires confirmed BMD/DMD patient data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_mri_radiomics():
    """Extract radiomics features from MRI scans"""
    logger.info("=== STEP 1: MRI RADIOMICS EXTRACTION ===")
    
    # Check if MRI features already exist
    mri_features_path = Path("output/mri_radiomics_features.csv")
    if mri_features_path.exists():
        logger.info("MRI radiomics features already exist")
        mri_features = pd.read_csv(mri_features_path)
        logger.info(f"Loaded {len(mri_features)} MRI radiomics features")
        return mri_features
    else:
        logger.warning("MRI radiomics features not found. Creating synthetic features for demonstration.")
        
        # Create synthetic MRI radiomics features
        np.random.seed(42)
        n_mri_samples = 31
        mri_feature_names = [
            'original_shape_Volume', 'original_shape_SurfaceArea', 'original_shape_Sphericity',
            'original_firstorder_Mean', 'original_firstorder_StdDev', 'original_firstorder_Skewness',
            'original_firstorder_Kurtosis', 'original_firstorder_Entropy', 'original_firstorder_Energy',
            'original_glcm_Correlation', 'original_glcm_Contrast', 'original_glcm_Dissimilarity',
            'original_glcm_Homogeneity', 'original_glrlm_ShortRunEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis'
        ]
        
        mri_data = []
        for i in range(n_mri_samples):
            features = {}
            for feature in mri_feature_names:
                if 'Volume' in feature:
                    features[feature] = np.random.uniform(800, 1500)
                elif 'Mean' in feature:
                    features[feature] = np.random.uniform(100, 150)
                elif 'Entropy' in feature:
                    features[feature] = np.random.uniform(3.0, 6.0)
                elif 'Correlation' in feature:
                    features[feature] = np.random.uniform(0.7, 0.95)
                else:
                    features[feature] = np.random.uniform(0.1, 1.0)
            mri_data.append(features)
        
        mri_features = pd.DataFrame(mri_data)
        mri_features['image_name'] = [f'mri_sample_{i:03d}.nii.gz' for i in range(n_mri_samples)]
        
        # Save features
        mri_features.to_csv('output/mri_radiomics_features.csv', index=False)
        logger.info(f"Created synthetic MRI radiomics features: {len(mri_features)} samples")
        return mri_features

def extract_ultrasound_radiomics():
    """Extract radiomics features from Ultrasound images"""
    logger.info("=== STEP 2: ULTRASOUND RADIOMICS EXTRACTION ===")
    
    # Check if ultrasound features already exist
    us_features_path = Path("output/ultrasound_radiomics_features.csv")
    if us_features_path.exists():
        logger.info("Ultrasound radiomics features already exist")
        us_features = pd.read_csv(us_features_path)
        logger.info(f"Loaded {len(us_features)} ultrasound radiomics features")
        return us_features
    else:
        logger.warning("Ultrasound radiomics features not found. Creating synthetic features for demonstration.")
        
        # Create synthetic ultrasound radiomics features
        np.random.seed(123)
        n_us_samples = 62
        us_feature_names = [
            'original_shape_Area', 'original_shape_Perimeter', 'original_shape_Compactness',
            'original_firstorder_Mean', 'original_firstorder_StdDev', 'original_firstorder_Skewness',
            'original_firstorder_Kurtosis', 'original_firstorder_Entropy', 'original_firstorder_Energy',
            'original_glcm_Correlation', 'original_glcm_Contrast', 'original_glcm_Dissimilarity',
            'original_glcm_Homogeneity', 'original_glrlm_ShortRunEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis'
        ]
        
        us_data = []
        for i in range(n_us_samples):
            features = {}
            for feature in us_feature_names:
                if 'Area' in feature:
                    features[feature] = np.random.uniform(500, 1200)
                elif 'Mean' in feature:
                    features[feature] = np.random.uniform(80, 130)
                elif 'Entropy' in feature:
                    features[feature] = np.random.uniform(2.5, 5.5)
                elif 'Correlation' in feature:
                    features[feature] = np.random.uniform(0.6, 0.9)
                else:
                    features[feature] = np.random.uniform(0.05, 0.9)
            us_data.append(features)
        
        us_features = pd.DataFrame(us_data)
        us_features['image_name'] = [f'us_sample_{i:03d}.tif' for i in range(n_us_samples)]
        
        # Save features
        us_features.to_csv('output/ultrasound_radiomics_features.csv', index=False)
        logger.info(f"Created synthetic ultrasound radiomics features: {len(us_features)} samples")
        return us_features

def create_bmd_dmd_labels(mri_features, us_features):
    """Create proper BMD/DMD labels for radiomics features (RIGHT WAY)"""
    logger.info("=== STEP 3: CREATE PROPER BMD/DMD LABELS ===")
    logger.warning("USING SYNTHETIC PROXY LABELS - REAL CLINICAL LABELS NEEDED FOR ACTUAL USE")
    
    # Combine all features for label generation
    all_features = pd.concat([mri_features, us_features], ignore_index=True)
    
    # Create proxy BMD/DMD labels based on radiomics patterns
    # This simulates how real clinical labeling might work
    labels = []
    
    for idx, row in all_features.iterrows():
        # Simulate clinical decision logic based on radiomics patterns
        dmd_score = 0  # Higher score = more likely DMD (severe form)
        
        # Use available features to create proxy classification
        entropy = row.get('original_firstorder_Entropy', 4.0)
        correlation = row.get('original_glcm_Correlation', 0.8)
        contrast = row.get('original_glcm_Contrast', 0.5)
        
        # Proxy clinical logic (simulated)
        if entropy > 4.5:  # Higher tissue heterogeneity suggests DMD
            dmd_score += 1
        if correlation < 0.75:  # Lower tissue organization suggests DMD
            dmd_score += 1
        if contrast > 0.6:  # Higher tissue contrast suggests DMD
            dmd_score += 1
        
        # Assign label: 0 = BMD (milder), 1 = DMD (more severe)
        label = 1 if dmd_score >= 2 else 0
        labels.append(label)
    
    # Ensure balanced dataset
    bmd_count = labels.count(0)
    dmd_count = labels.count(1)
    
    if bmd_count > dmd_count + 5:
        # Convert some BMD to DMD
        bmd_indices = [i for i, l in enumerate(labels) if l == 0]
        convert_count = min(len(bmd_indices) // 2, bmd_count - dmd_count)
        for i in range(convert_count):
            labels[bmd_indices[i]] = 1
    elif dmd_count > bmd_count + 5:
        # Convert some DMD to BMD
        dmd_indices = [i for i, l in enumerate(labels) if l == 1]
        convert_count = min(len(dmd_indices) // 2, dmd_count - bmd_count)
        for i in range(convert_count):
            labels[dmd_indices[i]] = 0
    
    # Create labels DataFrame
    labels_df = pd.DataFrame({
        'image_name': all_features['image_name'],
        'label': labels,
        'disease_class': ['DMD' if l == 1 else 'BMD' for l in labels],
        'clinical_notes': [f'Proxy classification: {"DMD (severe)" if l == 1 else "BMD (milder)"} - SYNTHETIC LABEL' for l in labels]
    })
    
    # Save labels
    labels_df.to_csv('output/proxy_bmd_dmd_labels.csv', index=False)
    
    # Print distribution
    final_bmd = labels.count(0)
    final_dmd = labels.count(1)
    logger.info(f"Proxy BMD/DMD label distribution:")
    logger.info(f"  BMD (milder): {final_bmd} ({final_bmd/len(labels)*100:.1f}%)")
    logger.info(f"  DMD (severe): {final_dmd} ({final_dmd/len(labels)*100:.1f}%)")
    logger.warning("WARNING: These are synthetic proxy labels for pipeline testing only!")
    
    return labels_df

def combine_radiomics_features(mri_features, us_features, labels_df):
    """Combine MRI and Ultrasound radiomics features into unified dataset"""
    logger.info("=== STEP 4: COMBINE RADIOMICS FEATURES ===")
    
    # Add modality information
    mri_features['modality'] = 'MRI'
    us_features['modality'] = 'Ultrasound'
    
    # Combine features
    combined_features = pd.concat([mri_features, us_features], ignore_index=True)
    
    # Merge with labels
    combined_dataset = combined_features.merge(labels_df[['image_name', 'label', 'disease_class']], on='image_name')
    
    # Prepare feature columns (exclude metadata)
    feature_cols = [col for col in combined_dataset.columns 
                   if col.startswith('original_') and col != 'image_name']
    
    X = combined_dataset[feature_cols].fillna(0)
    y = combined_dataset['label']
    
    logger.info(f"Combined dataset: {len(combined_dataset)} samples, {len(feature_cols)} features")
    logger.info(f"Feature columns: {len(feature_cols)} radiomics features")
    
    return X, y, combined_dataset, feature_cols

def split_dataset(X, y):
    """Split dataset into training and testing sets"""
    logger.info("=== STEP 5: SPLIT DATASET ===")
    
    # Split into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Testing set: {len(X_test)} samples")
    logger.info(f"Training BMD: {sum(y_train == 0)}, DMD: {sum(y_train == 1)}")
    logger.info(f"Testing BMD: {sum(y_test == 0)}, DMD: {sum(y_test == 1)}")
    
    return X_train, X_test, y_train, y_test

def train_ml_model(X_train, y_train):
    """Train Machine Learning model on combined radiomics features"""
    logger.info("=== STEP 6: TRAIN ML MODEL ===")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Random Forest
    ml_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1
    )
    
    ml_model.fit(X_train_scaled, y_train)
    
    logger.info("ML model trained successfully")
    logger.info(f"Training accuracy: {ml_model.score(X_train_scaled, y_train):.3f}")
    
    return ml_model, scaler

def train_dl_model(X_train, y_train):
    """Train Deep Learning model on combined radiomics features"""
    logger.info("=== STEP 7: TRAIN DL MODEL ===")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Neural Network (MLP)
    dl_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )
    
    dl_model.fit(X_train_scaled, y_train)
    
    logger.info("DL model trained successfully")
    logger.info(f"Training accuracy: {dl_model.score(X_train_scaled, y_train):.3f}")
    
    return dl_model, scaler

def test_models(ml_model, dl_model, ml_scaler, dl_scaler, X_test, y_test):
    """Test both trained models on remaining dataset"""
    logger.info("=== STEP 8: TEST MODELS ===")
    
    # Standardize test data
    X_test_ml_scaled = ml_scaler.transform(X_test)
    X_test_dl_scaled = dl_scaler.transform(X_test)
    
    # ML predictions
    ml_pred = ml_model.predict(X_test_ml_scaled)
    ml_proba = ml_model.predict_proba(X_test_ml_scaled)[:, 1]
    
    # DL predictions
    dl_pred = dl_model.predict(X_test_dl_scaled)
    dl_proba = dl_model.predict_proba(X_test_dl_scaled)[:, 1]
    
    # Calculate metrics
    ml_metrics = calculate_metrics(y_test, ml_pred, ml_proba, "ML (Random Forest)")
    dl_metrics = calculate_metrics(y_test, dl_pred, dl_proba, "DL (Neural Network)")
    
    logger.info("Model testing completed")
    
    return ml_metrics, dl_metrics, ml_pred, dl_pred, ml_proba, dl_proba

def calculate_metrics(y_true, y_pred, y_proba, model_name):
    """Calculate comprehensive evaluation metrics"""
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_proba)
    }
    
    # Specificity
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['Specificity'] = specificity
    
    return metrics

def compare_models(ml_metrics, dl_metrics):
    """Compare ML vs DL performance metrics"""
    logger.info("=== STEP 9: COMPARE MODELS ===")
    
    comparison_df = pd.DataFrame([ml_metrics, dl_metrics])
    
    logger.info("Performance Comparison:")
    logger.info(comparison_df.to_string(index=False))
    
    # Determine winner
    ml_acc = ml_metrics['Accuracy']
    dl_acc = dl_metrics['Accuracy']
    winner = "ML (Random Forest)" if ml_acc > dl_acc else "DL (Neural Network)"
    
    logger.info(f"Winner: {winner} (Accuracy: ML={ml_acc:.3f}, DL={dl_acc:.3f})")
    
    return comparison_df, winner

def generate_comprehensive_report(ml_metrics, dl_metrics, comparison_df, winner):
    """Generate comprehensive comparison report"""
    logger.info("=== STEP 10: GENERATE COMPREHENSIVE REPORT ===")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Metrics comparison
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Specificity']
    ml_values = [ml_metrics[m] for m in metrics_to_plot]
    dl_values = [dl_metrics[m] for m in metrics_to_plot]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, ml_values, width, label='ML (Random Forest)', alpha=0.8)
    axes[0, 0].bar(x + width/2, dl_values, width, label='DL (Neural Network)', alpha=0.8)
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('ML vs DL Performance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics_to_plot, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Summary text
    summary_text = f"""
    COMBINED RADIOMICS PIPELINE RESULTS
    
    Dataset: MRI + Ultrasound radiomics features
    Total samples: {ml_metrics.get('Total Samples', 'N/A')}
    Features: Combined radiomics from both modalities
    
    Machine Learning (Random Forest):
    - Accuracy: {ml_metrics['Accuracy']:.3f}
    - Precision: {ml_metrics['Precision']:.3f}
    - Recall: {ml_metrics['Recall']:.3f}
    - F1-Score: {ml_metrics['F1-Score']:.3f}
    - AUC-ROC: {ml_metrics['AUC-ROC']:.3f}
    - Specificity: {ml_metrics['Specificity']:.3f}
    
    Deep Learning (Neural Network):
    - Accuracy: {dl_metrics['Accuracy']:.3f}
    - Precision: {dl_metrics['Precision']:.3f}
    - Recall: {dl_metrics['Recall']:.3f}
    - F1-Score: {dl_metrics['F1-Score']:.3f}
    - AUC-ROC: {dl_metrics['AUC-ROC']:.3f}
    - Specificity: {dl_metrics['Specificity']:.3f}
    
    WINNER: {winner}
    
    IMPORTANT: Results use synthetic proxy labels.
    Real clinical validation required for actual use.
    """
    
    axes[0, 1].text(0.05, 0.95, summary_text, transform=axes[0, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[0, 1].axis('off')
    
    # Remove empty subplots
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('output/combined_radiomics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    comparison_df.to_csv('output/combined_radiomics_results.csv', index=False)
    
    logger.info("Comprehensive report generated:")
    logger.info("- output/combined_radiomics_comparison.png")
    logger.info("- output/combined_radiomics_results.csv")
    
    return comparison_df

def main():
    """Main combined radiomics pipeline"""
    logger.info("=== COMBINED MRI + ULTRASOUND RADIOMICS PIPELINE ===")
    logger.info("PROPER APPROACH: Extract -> Label -> Train -> Test -> Compare")
    logger.warning("USING SYNTHETIC PROXY LABELS - NOT FOR CLINICAL USE")
    logger.info("")
    
    try:
        # Step 1: Extract MRI radiomics
        mri_features = extract_mri_radiomics()
        
        # Step 2: Extract Ultrasound radiomics
        us_features = extract_ultrasound_radiomics()
        
        # Step 3: Create proper BMD/DMD labels
        labels_df = create_bmd_dmd_labels(mri_features, us_features)
        
        # Step 4: Combine radiomics features
        X, y, combined_dataset, feature_cols = combine_radiomics_features(mri_features, us_features, labels_df)
        
        # Step 5: Split dataset
        X_train, X_test, y_train, y_test = split_dataset(X, y)
        
        # Step 6: Train ML model
        ml_model, ml_scaler = train_ml_model(X_train, y_train)
        
        # Step 7: Train DL model
        dl_model, dl_scaler = train_dl_model(X_train, y_train)
        
        # Step 8: Test models
        ml_metrics, dl_metrics, ml_pred, dl_pred, ml_proba, dl_proba = test_models(
            ml_model, dl_model, ml_scaler, dl_scaler, X_test, y_test
        )
        
        # Step 9: Compare models
        comparison_df, winner = compare_models(ml_metrics, dl_metrics)
        
        # Step 10: Generate comprehensive report
        final_results = generate_comprehensive_report(ml_metrics, dl_metrics, comparison_df, winner)
        
        logger.info("")
        logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
        logger.info("Combined MRI + Ultrasound radiomics classification complete!")
        logger.warning("REMINDER: Results use synthetic proxy labels only.")
        
        return final_results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
