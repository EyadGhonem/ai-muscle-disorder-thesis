#!/usr/bin/env python3
"""
Convert FSHD Heckmatt grades to binary classification
Grades 1-2: Normal/Mild (0)
Grades 3-4: Moderate/Severe (1)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def convert_heckmatt_to_binary():
    """Convert Heckmatt grades from 1-4 scale to binary classification"""
    
    # Set paths
    data_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/data/final_ultrasound_labeled")
    subjects_file = data_dir / "SubjectsInfo.xlsx"
    output_dir = Path("c:/Users/Lenovo/Desktop/thesis_project/processed_data")
    output_dir.mkdir(exist_ok=True)
    
    # Load the subjects data
    print("Loading SubjectsInfo.xlsx...")
    df = pd.read_excel(subjects_file)
    print(f"Loaded data with shape: {df.shape}")
    
    # Identify Heckmatt grade columns (format: muscle_code_side)
    heckmatt_cols = [col for col in df.columns if '_' in col and col not in ['Code', 'Sex', 'age', 'BMI', 'Weight', 'Length']]
    print(f"Found {len(heckmatt_cols)} Heckmatt grade columns")
    
    # Create binary classification columns
    binary_cols = []
    for col in heckmatt_cols:
        binary_col = f"{col}_binary"
        binary_cols.append(binary_col)
        
        # Convert to binary: grades 1-2 -> 0, grades 3-4 -> 1
        df[binary_col] = df[col].apply(lambda x: 0 if pd.isna(x) or x <= 2 else 1)
    
    # Create summary statistics
    print("\n=== Binary Classification Summary ===")
    
    # Overall distribution
    all_binary = []
    for col in binary_cols:
        all_binary.extend(df[col].dropna().tolist())
    
    binary_counts = pd.Series(all_binary).value_counts().sort_index()
    print(f"Overall binary distribution:")
    print(f"  Normal/Mild (0): {binary_counts.get(0, 0)} cases")
    print(f"  Moderate/Severe (1): {binary_counts.get(1, 0)} cases")
    print(f"  Total: {len(all_binary)} cases")
    
    # Per muscle distribution
    print(f"\n=== Per Muscle Binary Distribution ===")
    for i, (orig_col, binary_col) in enumerate(zip(heckmatt_cols, binary_cols)):
        valid_count = df[binary_col].notna().sum()
        if valid_count > 0:
            counts = df[binary_col].value_counts().sort_index()
            normal_pct = (counts.get(0, 0) / valid_count) * 100
            severe_pct = (counts.get(1, 0) / valid_count) * 100
            print(f"{orig_col}: {counts.get(0, 0)} normal ({normal_pct:.1f}%), {counts.get(1, 0)} severe ({severe_pct:.1f}%)")
    
    # Create a mapping file for image filenames to binary labels
    print(f"\n=== Creating Image-Label Mapping ===")
    
    # Get all image files
    images_dir = data_dir / "images"
    image_files = list(images_dir.glob("*.png"))
    print(f"Found {len(image_files)} image files")
    
    # Parse image filenames to extract subject, muscle, side, and instance
    # Format: subject_muscle_side_instance.png
    image_label_data = []
    
    for img_file in image_files:
        filename = img_file.stem  # Remove .png extension
        parts = filename.split('_')
        
        if len(parts) >= 4:
            subject = parts[0]
            muscle = parts[1]
            side = parts[2]
            instance = parts[3]
            
            # Create the column name for Heckmatt grade
            heckmatt_col = f"{muscle}_{side}"
            
            if heckmatt_col in heckmatt_cols:
                # Find the subject row
                subject_row = df[df['Code'] == int(subject)]
                
                if not subject_row.empty:
                    # Get the binary label
                    binary_col = f"{heckmatt_col}_binary"
                    binary_label = subject_row[binary_col].iloc[0]
                    original_grade = subject_row[heckmatt_col].iloc[0]
                    
                    image_label_data.append({
                        'filename': filename,
                        'filepath': str(img_file),
                        'subject': int(subject),
                        'muscle_code': muscle,
                        'side': side,
                        'instance': instance,
                        'heckmatt_col': heckmatt_col,
                        'original_grade': original_grade,
                        'binary_label': binary_label,
                        'grade_category': 'Normal/Mild' if binary_label == 0 else 'Moderate/Severe'
                    })
    
    # Create image-label dataframe
    image_labels_df = pd.DataFrame(image_label_data)
    print(f"Created mapping for {len(image_labels_df)} images with valid labels")
    
    # Save the processed data
    output_files = {
        'subjects_with_binary_labels.xlsx': df,
        'image_label_mapping.csv': image_labels_df,
        'binary_classification_summary.txt': f"""
Binary Classification Summary
=============================

Conversion Rule:
- Grades 1-2: Normal/Mild (0)
- Grades 3-4: Moderate/Severe (1)

Overall Statistics:
- Total subjects: {df.shape[0]}
- Total grade entries: {len(all_binary)}
- Normal/Mild cases: {binary_counts.get(0, 0)} ({binary_counts.get(0, 0)/len(all_binary)*100:.1f}%)
- Moderate/Severe cases: {binary_counts.get(1, 0)} ({binary_counts.get(1, 0)/len(all_binary)*100:.1f}%)

Image Mapping:
- Total images found: {len(image_files)}
- Images with valid labels: {len(image_labels_df)}
- Images without labels: {len(image_files) - len(image_labels_df)}

Muscle Codes:
From SubjectsInfo.xlsx README:
001: Biceps_brachii
002: Deltoideus
003: Depressor_anguli_oris
004: Digastricus
005: Extensor_digitorum_brevis
006: Flexor_carpi_radialis
007: Flexor_digitorum_profundus
008: Gastrocnemius_medial_head
009: Geniohyoideus
010: Levator_labii_superior
011: Masseter
012: Mentalis
013: Orbicularis_oris
014: Peroneus_tertius
015: Rectus_abdominis
016: Rectus_femoris
017: Temporalis
018: Tibialis_anterior
019: Trapezius
020: Vastus_lateralis
021: Zygomaticus

Side Codes:
00: Left
01: Right
"""
    }
    
    for filename, data in output_files.items():
        output_path = output_dir / filename
        if filename.endswith('.xlsx'):
            data.to_excel(output_path, index=False)
        elif filename.endswith('.csv'):
            data.to_csv(output_path, index=False)
        else:
            # Text file
            with open(output_path, 'w') as f:
                f.write(data)
        print(f"Saved: {output_path}")
    
    print(f"\n=== Conversion Complete ===")
    print(f"Processed data saved to: {output_dir}")
    
    return df, image_labels_df

if __name__ == "__main__":
    subjects_df, image_labels_df = convert_heckmatt_to_binary()
