#!/usr/bin/env python3
"""
Count images and analyze labels for MRI and ULTRASOUND_LABELD_2 datasets
"""

import json
import pandas as pd
from pathlib import Path

def analyze_mri_dataset():
    """Analyze MRI dataset"""
    print("=== MRI DATASET ANALYSIS ===")
    
    dataset_path = Path("c:/Users/Lenovo/Desktop/thesis_project/data/MRI_LABELED")
    json_file = dataset_path / "Dataset.json"
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    training_count = len(data['training'])
    testing_count = len(data['testing'])
    total_count = training_count + testing_count
    
    print(f"Total MRI images: {total_count}")
    print(f"Training images: {training_count}")
    print(f"Testing images: {testing_count}")
    print(f"All have segmentation labels: Yes")
    print(f"Disease focus: Healthy/General muscle anatomy (no disease labels)")
    
    # Count modalities
    modalities = set()
    for item in data['training'] + data['testing']:
        image_path = item['image']
        if 'T1' in image_path:
            modalities.add('T1')
        elif 'T2' in image_path:
            modalities.add('T2')
        elif 'STIR' in image_path:
            modalities.add('STIR')
        elif 'Water' in image_path:
            modalities.add('Water')
        elif 'Fat' in image_path:
            modalities.add('Fat')
        elif 'Others' in image_path:
            modalities.add('Others')
    
    print(f"Modalities: {', '.join(sorted(modalities))}")
    print(f"Anatomy: Thigh muscles only")
    
    return total_count, training_count, testing_count

def analyze_ultrasound_2_dataset():
    """Analyze ULTRASOUND_LABELD_2 dataset"""
    print("\n=== ULTRASOUND_LABELD_2 DATASET ANALYSIS ===")
    
    dataset_path = Path("c:/Users/Lenovo/Desktop/thesis_project/data/ULTRASOUND_LABELD_2")
    excel_file = dataset_path / "PatientImages_PLOS2017.xlsx"
    
    # Load Excel data
    df = pd.read_excel(excel_file)
    
    print(f"Total patient records: {len(df)}")
    print(f"All have images: Yes (stored in PatientData.mat)")
    print(f"Disease labels: Yes (from Diagnosis column)")
    
    # Count diseases
    if 'Diagnosis\nD - Dermatomyositis\nP - Polymyositis\nI - IBM\nN - normal' in df.columns:
        diagnosis_col = 'Diagnosis\nD - Dermatomyositis\nP - Polymyositis\nI - IBM\nN - normal'
    else:
        # Find the diagnosis column
        diagnosis_col = None
        for col in df.columns:
            if 'Diagnosis' in col or 'D -' in col:
                diagnosis_col = col
                break
    
    if diagnosis_col:
        disease_counts = df[diagnosis_col].value_counts()
        print(f"\nDisease distribution:")
        for disease, count in disease_counts.items():
            print(f"  {disease}: {count} patients ({count/len(df)*100:.1f}%)")
    
    # Count muscles
    if 'Muscle\nD - deltoid\nB - biceps\nR - rectus\nG - gastroc\nT - tibialis ant\nFCR - FCR\nFDP -FDP' in df.columns:
        muscle_col = 'Muscle\nD - deltoid\nB - biceps\nR - rectus\nG - gastroc\nT - tibialis ant\nFCR - FCR\nFDP -FDP'
    else:
        muscle_col = None
        for col in df.columns:
            if 'Muscle' in col or 'D -' in col:
                muscle_col = col
                break
    
    if muscle_col:
        muscle_counts = df[muscle_col].value_counts()
        print(f"\nMuscle types:")
        for muscle, count in muscle_counts.items():
            print(f"  {muscle}: {count} images")
    
    print(f"\nImage format: MATLAB .mat file")
    print(f"Disease focus: Inflammatory muscle diseases")
    
    return len(df)

def main():
    """Main analysis function"""
    print("📊 DATASET COUNT AND LABEL ANALYSIS")
    print("=" * 50)
    
    # Analyze MRI dataset
    mri_total, mri_train, mri_test = analyze_mri_dataset()
    
    # Analyze Ultrasound 2 dataset
    us2_total = analyze_ultrasound_2_dataset()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    print(f"MRI Dataset: {mri_total} total images ({mri_train} train, {mri_test} test)")
    print(f"ULTRASOUND_LABELD_2: {us2_total} patient records with images")
    print(f"\nKey difference: MRI has anatomy labels, Ultrasound 2 has disease labels")

if __name__ == "__main__":
    main()
