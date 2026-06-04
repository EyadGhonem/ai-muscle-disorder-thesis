#!/usr/bin/env python3
"""
Master training script for all three datasets:
1. ULTRASOUND_LABELD_1 (original dataset, renamed)
2. MRI_LABELED (new MRI dataset)
3. ULTRASOUND_LABELD_2 (new ultrasound dataset)
"""

import os
import sys
import subprocess
from pathlib import Path

def train_ultrasound_1():
    """Train on original ultrasound dataset (ULTRASOUND_LABELD_1)"""
    print("=== Training on ULTRASOUND_LABELD_1 (Original Dataset) ===")
    
    # Update the original training script to use the new folder name
    script_path = Path("c:/Users/Lenovo/Desktop/thesis_project/models/train_ml_dl_models.py")
    
    # Run the original training script
    try:
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, cwd="c:/Users/Lenovo/Desktop/thesis_project")
        
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print("✅ ULTRASOUND_LABELD_1 training completed")
        return True
        
    except Exception as e:
        print(f"❌ Error training ULTRASOUND_LABELD_1: {e}")
        return False

def train_mri_dataset():
    """Train on MRI dataset"""
    print("\n=== Training on MRI_LABELED Dataset ===")
    
    script_path = Path("c:/Users/Lenovo/Desktop/thesis_project/train_mri_dataset.py")
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, cwd="c:/Users/Lenovo/Desktop/thesis_project")
        
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print("✅ MRI_LABELED training completed")
        return True
        
    except Exception as e:
        print(f"❌ Error training MRI_LABELED: {e}")
        return False

def train_ultrasound_2():
    """Train on second ultrasound dataset"""
    print("\n=== Training on ULTRASOUND_LABELD_2 Dataset ===")
    
    script_path = Path("c:/Users/Lenovo/Desktop/thesis_project/train_ultrasound_2_dataset.py")
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, cwd="c:/Users/Lenovo/Desktop/thesis_project")
        
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print("✅ ULTRASOUND_LABELD_2 training completed")
        return True
        
    except Exception as e:
        print(f"❌ Error training ULTRASOUND_LABELD_2: {e}")
        return False

def compare_all_results():
    """Compare results across all three datasets"""
    print("\n=== Comparing Results Across All Datasets ===")
    
    import pandas as pd
    
    # Load results from all datasets
    results_files = [
        ("c:/Users/Lenovo/Desktop/thesis_project/results/model_comparison.csv", "ULTRASOUND_LABELD_1"),
        ("c:/Users/Lenovo/Desktop/thesis_project/output/mri_results/mri_model_comparison.csv", "MRI_LABELED"),
        ("c:/Users/Lenovo/Desktop/thesis_project/output/ultrasound_2_results/ultrasound_2_model_comparison.csv", "ULTRASOUND_LABELD_2")
    ]
    
    all_results = []
    
    for file_path, dataset_name in results_files:
        if Path(file_path).exists():
            try:
                df = pd.read_csv(file_path)
                df['Dataset'] = dataset_name
                all_results.append(df)
                print(f"✅ Loaded results for {dataset_name}")
            except Exception as e:
                print(f"❌ Error loading {dataset_name}: {e}")
        else:
            print(f"⚠️  Results file not found for {dataset_name}: {file_path}")
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        output_path = Path("c:/Users/Lenovo/Desktop/thesis_project/output/all_datasets_comparison.csv")
        output_path.parent.mkdir(exist_ok=True)
        combined_results.to_csv(output_path, index=False)
        
        print(f"\n📊 Combined results saved to: {output_path}")
        print("\n📈 Performance Comparison:")
        
        # Display summary
        for dataset in combined_results['Dataset'].unique():
            dataset_data = combined_results[combined_results['Dataset'] == dataset]
            best_model = dataset_data.loc[dataset_data['Accuracy'].idxmax()]
            
            print(f"\n🏆 {dataset}:")
            print(f"   Best Model: {best_model['Model']}")
            print(f"   Accuracy: {best_model['Accuracy']:.4f}")
            print(f"   Precision: {best_model['Precision']:.4f}")
            print(f"   Recall: {best_model['Recall']:.4f}")
            print(f"   F1-Score: {best_model['F1-Score']:.4f}")
            print(f"   AUC: {best_model['AUC']:.4f}")
        
        return True
    else:
        print("❌ No results to compare")
        return False

def main():
    """Main function to train all datasets"""
    print("🚀 Starting Training Pipeline for All Three Datasets")
    print("=" * 60)
    
    # Train each dataset
    results = {}
    
    results['ultrasound_1'] = train_ultrasound_1()
    results['mri'] = train_mri_dataset()
    results['ultrasound_2'] = train_ultrasound_2()
    
    # Compare results
    if any(results.values()):
        compare_all_results()
    
    print("\n" + "=" * 60)
    print("🏁 Training Pipeline Complete!")
    
    # Summary
    print("\n📋 Training Summary:")
    for dataset, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {dataset}: {status}")

if __name__ == "__main__":
    main()
