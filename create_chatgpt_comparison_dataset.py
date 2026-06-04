#!/usr/bin/env python3
"""
Create test dataset for ChatGPT comparison
50 samples with separate labels file
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil

def create_test_dataset():
    """Create 50 sample test dataset for ChatGPT comparison"""
    print("Creating ChatGPT Comparison Test Dataset")
    print("=" * 50)
    
    # Load the real features dataset
    dataset_path = Path("c:/Users/Lenovo/Desktop/thesis_project/final_ultrasound_dataset_REAL_features.csv")
    if not dataset_path.exists():
        print("Dataset not found")
        return
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {df.shape}")
    
    # Remove Unknown severity
    df = df[df['severity_label'] != 'Unknown'].copy()
    print(f"After removing Unknown: {df.shape}")
    
    # Sample 50 random cases (balanced across diseases if possible)
    np.random.seed(42)
    
    # Get samples from each disease type
    fshd_samples = df[df['label'] == 'FSHD'].sample(25, random_state=42)
    other_samples = df[df['label'] == 'Other'].sample(25, random_state=42)
    
    # Combine
    test_df = pd.concat([fshd_samples, other_samples], ignore_index=True)
    
    # Shuffle
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Test dataset: {test_df.shape}")
    print(f"Disease distribution:")
    print(test_df['label'].value_counts())
    
    # Create unlabeled version (remove label columns)
    unlabeled_df = test_df.drop(columns=['label', 'severity_label'])
    
    # Create labels file
    labels_df = test_df[['image_path', 'label', 'severity_label']].copy()
    
    # Save files
    output_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/chatgpt_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save unlabeled test data
    unlabeled_path = output_dir / "chatgpt_test_dataset_unlabeled.csv"
    unlabeled_df.to_csv(unlabeled_path, index=False)
    print(f"\n✅ Unlabeled test data saved: {unlabeled_path}")
    
    # Save labels (keep for later comparison)
    labels_path = output_dir / "chatgpt_test_dataset_labels.csv"
    labels_df.to_csv(labels_path, index=False)
    print(f"✅ Labels saved: {labels_path}")
    
    # Create instructions for ChatGPT
    instructions = """
# ChatGPT Comparison Test Instructions

## Dataset Information
- Total samples: 50 ultrasound images
- Features: 54 radiomics features per image
- Task: Classify as diseased/not diseased AND identify specific disease

## Files Provided
1. **chatgpt_test_dataset_unlabeled.csv** - Test data WITHOUT labels
2. **chatgpt_test_dataset_labels.csv** - Ground truth labels (DO NOT SHOW CHATGPT)

## Instructions for ChatGPT
"I have a dataset of 50 ultrasound images with radiomics features. 
Please classify each image as:
1. Diseased or Not Diseased
2. If diseased, identify the specific disease type (FSHD, Dermatomyositis, Polymyositis, IBM, Normal)

The dataset contains the following columns:
- image_path: Path to the ultrasound image
- feature_1 to feature_27: Radiomics features (intensity, texture, shape, gradient)
- Additional feature columns for clinical data

Please provide your predictions in a CSV format with columns:
- image_path
- prediction (Diseased/Not Diseased)
- disease_type (if diseased)
- confidence_score (0-1)"

## After ChatGPT Predictions
1. Save ChatGPT's predictions to a file
2. Compare with ground truth labels using the labels file
3. Calculate accuracy, precision, recall, F1-score
4. Compare with our model's performance on the same 50 samples
"""
    
    instructions_path = output_dir / "README_INSTRUCTIONS.md"
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    print(f"✅ Instructions saved: {instructions_path}")
    
    # Also create a version with just the essential features for easier ChatGPT input
    essential_features = ['image_path'] + [col for col in unlabeled_df.columns if col.startswith('feature_')]
    essential_df = unlabeled_df[essential_features]
    
    essential_path = output_dir / "chatgpt_test_essential_features.csv"
    essential_df.to_csv(essential_path, index=False)
    print(f"✅ Essential features saved: {essential_path}")
    
    # Create summary statistics
    print("\n" + "=" * 50)
    print("TEST DATASET SUMMARY")
    print("=" * 50)
    print(f"Total samples: {len(test_df)}")
    print(f"\nDisease distribution:")
    print(test_df['label'].value_counts())
    print(f"\nSeverity distribution:")
    print(test_df['severity_label'].value_counts())
    print(f"\nDataset saved to: {output_dir}")
    
    # Test our model on these 50 samples for comparison
    print("\n" + "=" * 50)
    print("TESTING OUR MODEL ON THESE 50 SAMPLES")
    print("=" * 50)
    
    return output_dir, test_df, labels_df

def test_our_model_on_samples(test_df):
    """Test our trained model on these 50 samples"""
    print("\nTesting our model on the 50 test samples...")
    
    try:
        import joblib
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Load trained model
        models_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/output/real_features_results/trained_models")
        model_path = models_dir / "disease_random_forest_real.joblib"
        
        if not model_path.exists():
            print("Trained model not found")
            return None
        
        model = joblib.load(model_path)
        print(f"Loaded model: {model_path}")
        
        # Prepare features - use all feature columns
        feature_cols = [col for col in test_df.columns if col.startswith('feature_') or col in 
                       ['mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 
                        'median_intensity', 'q25_intensity', 'q75_intensity', 'skewness', 
                        'kurtosis', 'entropy', 'glcm_contrast', 'glcm_dissimilarity',
                        'glcm_homogeneity', 'glcm_energy', 'glcm_correlation', 'glcm_asm',
                        'area', 'perimeter', 'circularity', 'aspect_ratio', 'extent', 
                        'solidity', 'equivalent_diameter', 'gradient_mean', 'gradient_std',
                        'gradient_max', 'gradient_energy']]
        X_test = test_df[feature_cols].values
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Get actual labels
        y_true = test_df['label'].values
        
        # Calculate accuracy
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"Our model accuracy on these 50 samples: {accuracy:.4f}")
        
        # Create results dataframe
        results_df = test_df[['image_path', 'label']].copy()
        results_df['our_prediction'] = y_pred
        results_df['our_confidence'] = y_pred_proba.max(axis=1)
        
        # Save our results
        output_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/chatgpt_comparison")
        our_results_path = output_dir / "our_model_predictions.csv"
        results_df.to_csv(our_results_path, index=False)
        print(f"✅ Our predictions saved: {our_results_path}")
        
        return accuracy, results_df
        
    except Exception as e:
        print(f"Error testing our model: {e}")
        return None

def main():
    """Main function"""
    print("🔬 Creating ChatGPT Comparison Test Dataset")
    print("=" * 50)
    
    # Create test dataset
    output_dir, test_df, labels_df = create_test_dataset()
    
    # Test our model on these samples
    our_accuracy, our_results = test_our_model_on_samples(test_df)
    
    print("\n" + "=" * 50)
    print("✅ COMPARISON TEST DATASET READY")
    print("=" * 50)
    print(f"\n📁 Files created in: {output_dir}")
    print(f"  1. chatgpt_test_dataset_unlabeled.csv - Give to ChatGPT")
    print(f"  2. chatgpt_test_dataset_labels.csv - Keep for comparison")
    print(f"  3. chatgpt_test_essential_features.csv - Simplified version")
    print(f"  4. README_INSTRUCTIONS.md - Instructions for ChatGPT")
    if our_accuracy:
        print(f"  5. our_model_predictions.csv - Our model's results")
        print(f"\n📊 Our model accuracy on these 50 samples: {our_accuracy:.4f}")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Give 'chatgpt_test_dataset_unlabeled.csv' to ChatGPT")
    print("2. Ask ChatGPT to classify each sample")
    print("3. Save ChatGPT's predictions")
    print("4. Compare with 'chatgpt_test_dataset_labels.csv'")
    print("5. Compare with 'our_model_predictions.csv'")

if __name__ == "__main__":
    main()
