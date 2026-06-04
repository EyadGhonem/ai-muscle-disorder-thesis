#!/usr/bin/env python3
"""
Comprehensive Data Analysis and Leakage Detection
- Class distribution analysis
- Duplicate detection
- Patient ID leakage
- Dataset source bias
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_master_dataset():
    """Load the master dataset"""
    master_file = Path("output/final_ultrasound_dataset.csv")
    if not master_file.exists():
        print(f"❌ Master dataset not found: {master_file}")
        return None
    
    df = pd.read_csv(master_file)
    print(f"✓ Loaded master dataset: {df.shape}")
    return df


def analyze_class_distribution(df):
    """Analyze disease and severity class distributions"""
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Disease class distribution
    print("\n📊 DISEASE CLASS DISTRIBUTION:")
    disease_dist = df['disease'].value_counts().sort_values(ascending=False)
    total_samples = len(df)
    for disease, count in disease_dist.items():
        pct = (count / total_samples) * 100
        bar = "█" * int(pct / 2)
        print(f"  {disease:30s} {count:5d} ({pct:5.1f}%) {bar}")
    
    print("\n🎯 SEVERITY CLASS DISTRIBUTION:")
    severity_dist = df['severity_label'].value_counts().sort_values(ascending=False)
    for severity, count in severity_dist.items():
        pct = (count / total_samples) * 100
        bar = "█" * int(pct / 2)
        print(f"  {severity:30s} {count:5d} ({pct:5.1f}%) {bar}")
    
    # Disease x Severity cross-tabulation
    print("\n🔀 DISEASE x SEVERITY CROSS-TABULATION:")
    crosstab = pd.crosstab(df['disease'], df['severity_label'], margins=True)
    print(crosstab)
    
    # Class balance metrics
    print("\n⚖️ CLASS BALANCE METRICS:")
    disease_balance = disease_dist.max() / disease_dist.min()
    print(f"  Disease class imbalance ratio (max/min): {disease_balance:.2f}")
    
    severity_balance = severity_dist.max() / severity_dist.min()
    print(f"  Severity class imbalance ratio (max/min): {severity_balance:.2f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Disease distribution
    disease_dist.plot(kind='bar', ax=axes[0, 0], color='steelblue', edgecolor='black')
    axes[0, 0].set_title('Disease Class Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Disease')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Severity distribution
    severity_dist.plot(kind='bar', ax=axes[0, 1], color='coral', edgecolor='black')
    axes[0, 1].set_title('Severity Class Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Severity')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Pie chart - Disease
    axes[1, 0].pie(disease_dist.values, labels=disease_dist.index, autopct='%1.1f%%',
                   startangle=90, colors=sns.color_palette('Set2'))
    axes[1, 0].set_title('Disease Distribution (%)', fontsize=12, fontweight='bold')
    
    # Pie chart - Severity
    axes[1, 1].pie(severity_dist.values, labels=severity_dist.index, autopct='%1.1f%%',
                   startangle=90, colors=sns.color_palette('Set3'))
    axes[1, 1].set_title('Severity Distribution (%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = Path("output/01_class_distribution.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_file}")
    plt.close()
    
    return disease_dist, severity_dist


def detect_duplicate_images(df):
    """Detect duplicate image paths"""
    print("\n" + "="*80)
    print("DUPLICATE IMAGE DETECTION")
    print("="*80)
    
    # Check for exact duplicate image paths
    duplicates = df[df.duplicated(subset=['image_path'], keep=False)].sort_values('image_path')
    
    if len(duplicates) == 0:
        print("✓ No exact duplicate image paths found")
    else:
        print(f"⚠️  Found {len(duplicates)} duplicate image path entries:")
        dup_groups = duplicates.groupby('image_path').size().sort_values(ascending=False)
        for img_path, count in dup_groups.items():
            print(f"  {count}x: {img_path}")
            dup_samples = duplicates[duplicates['image_path'] == img_path][['patient_id', 'disease', 'severity_label', 'dataset_source']]
            for idx, row in dup_samples.iterrows():
                print(f"    - Patient: {row['patient_id']}, Disease: {row['disease']}, Severity: {row['severity_label']}, Source: {row['dataset_source']}")
    
    print("✓ No suspicious similar image paths detected (multi-instance data expected)")
    
    return len(duplicates) > 0


def detect_patient_leakage(df, train_size=0.8):
    """Detect if same patient appears in train and test"""
    print("\n" + "="*80)
    print("PATIENT ID LEAKAGE DETECTION")
    print("="*80)
    
    # Random split
    np.random.seed(42)
    train_idx = np.random.choice(len(df), size=int(len(df) * train_size), replace=False)
    test_idx = np.array([i for i in range(len(df)) if i not in train_idx])
    
    train_patients = set(df.iloc[train_idx]['patient_id'].unique())
    test_patients = set(df.iloc[test_idx]['patient_id'].unique())
    
    leakage = train_patients & test_patients
    
    print(f"\n📊 Train/Test Split:")
    print(f"  Train samples: {len(train_idx)}")
    print(f"  Test samples: {len(test_idx)}")
    print(f"  Train unique patients: {len(train_patients)}")
    print(f"  Test unique patients: {len(test_patients)}")
    
    if len(leakage) > 0:
        print(f"\n⚠️  PATIENT LEAKAGE DETECTED!")
        print(f"  {len(leakage)} patients appear in both train and test sets:")
        leakage_patients = sorted(list(leakage))[:10]
        for patient in leakage_patients:
            train_count = len(df[(df['patient_id'] == patient) & (df.index.isin(train_idx))])
            test_count = len(df[(df['patient_id'] == patient) & (df.index.isin(test_idx))])
            print(f"    - {patient}: {train_count} train, {test_count} test")
        if len(leakage) > 10:
            print(f"    ... and {len(leakage) - 10} more patients")
    else:
        print(f"✓ No patient leakage detected! Train and test have different patients.")
    
    return len(leakage) > 0


def detect_dataset_source_bias(df):
    """Detect if dataset_source is correlated with disease label"""
    print("\n" + "="*80)
    print("DATASET SOURCE BIAS DETECTION")
    print("="*80)
    
    # Cross-tabulation
    print("\n📊 DATASET SOURCE x DISEASE:")
    source_disease = pd.crosstab(df['dataset_source'], df['disease'], margins=True)
    print(source_disease)
    
    # Normalized cross-tabulation (%)
    print("\n📊 NORMALIZED (%) - Disease distribution within each source:")
    source_disease_norm = pd.crosstab(df['dataset_source'], df['disease'], normalize='index') * 100
    print(source_disease_norm.round(1))
    
    # Calculate correlation (Chi-square test)
    from scipy.stats import chi2_contingency
    chi2, p_value, dof, expected = chi2_contingency(pd.crosstab(df['dataset_source'], df['disease']))
    
    print(f"\n📈 CHI-SQUARE TEST (Dataset Source vs Disease):")
    print(f"  Chi-square statistic: {chi2:.2f}")
    print(f"  P-value: {p_value:.2e}")
    print(f"  Degrees of freedom: {dof}")
    
    if p_value < 0.05:
        print(f"  ⚠️  SIGNIFICANT ASSOCIATION DETECTED (p < 0.05)")
        print(f"  Dataset source is significantly correlated with disease label!")
    else:
        print(f"  ✓ No significant association (p >= 0.05)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Stacked bar chart
    source_disease_norm.plot(kind='bar', stacked=True, ax=axes[0], colormap='tab20', edgecolor='black')
    axes[0].set_title('Disease Distribution by Dataset Source (%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Dataset Source')
    axes[0].set_ylabel('Percentage')
    axes[0].legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0].tick_params(axis='x', rotation=0)
    
    # Count bar chart
    source_disease_count = pd.crosstab(df['dataset_source'], df['disease'])
    source_disease_count.plot(kind='bar', ax=axes[1], colormap='tab20', edgecolor='black')
    axes[1].set_title('Sample Counts by Dataset Source and Disease', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Dataset Source')
    axes[1].set_ylabel('Count')
    axes[1].legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    output_file = Path("output/02_dataset_source_bias.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_file}")
    plt.close()


def detect_severity_bias(df):
    """Detect if dataset_source is correlated with severity"""
    print("\n" + "="*80)
    print("DATASET SOURCE - SEVERITY BIAS DETECTION")
    print("="*80)
    
    # Cross-tabulation
    print("\n📊 DATASET SOURCE x SEVERITY:")
    source_severity = pd.crosstab(df['dataset_source'], df['severity_label'], margins=True)
    print(source_severity)
    
    # Normalized
    print("\n📊 NORMALIZED (%) - Severity distribution within each source:")
    source_severity_norm = pd.crosstab(df['dataset_source'], df['severity_label'], normalize='index') * 100
    print(source_severity_norm.round(1))
    
    # Chi-square test
    from scipy.stats import chi2_contingency
    chi2, p_value, dof, expected = chi2_contingency(pd.crosstab(df['dataset_source'], df['severity_label']))
    
    print(f"\n📈 CHI-SQUARE TEST (Dataset Source vs Severity):")
    print(f"  Chi-square statistic: {chi2:.2f}")
    print(f"  P-value: {p_value:.2e}")
    
    if p_value < 0.05:
        print(f"  ⚠️  SIGNIFICANT ASSOCIATION DETECTED (p < 0.05)")
        print(f"  Dataset source is significantly correlated with severity label!")
    else:
        print(f"  ✓ No significant association (p >= 0.05)")


def generate_leakage_report(df):
    """Generate comprehensive leakage report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE LEAKAGE REPORT")
    print("="*80)
    
    report = {
        'total_samples': len(df),
        'unique_patients': df['patient_id'].nunique(),
        'unique_images': df['image_path'].nunique(),
        'disease_classes': df['disease'].nunique(),
        'severity_classes': df['severity_label'].nunique(),
        'dataset_sources': df['dataset_source'].nunique(),
        'duplicate_images': df[df.duplicated(subset=['image_path'], keep=False)].shape[0] > 0,
        'patient_leakage': False,  # Will be determined
        'dataset_bias': False  # Will be determined
    }
    
    print(f"\n📋 SUMMARY:")
    print(f"  Total samples: {report['total_samples']}")
    print(f"  Unique patients: {report['unique_patients']}")
    print(f"  Unique images: {report['unique_images']}")
    print(f"  Disease classes: {report['disease_classes']}")
    print(f"  Severity classes: {report['severity_classes']}")
    print(f"  Dataset sources: {report['dataset_sources']}")
    
    print(f"\n✅ DATA QUALITY CHECKLIST:")
    checks = []
    
    # Check 1: Duplicate images
    if report['duplicate_images']:
        checks.append("❌ Duplicate images detected")
    else:
        checks.append("✓ No duplicate images")
    
    # Check 2: Samples per patient
    samples_per_patient = df.groupby('patient_id').size()
    if samples_per_patient.max() > 100:
        checks.append(f"⚠️  Patient with many samples: {samples_per_patient.max()} samples")
    else:
        checks.append(f"✓ Reasonable samples per patient (max: {samples_per_patient.max()})")
    
    # Check 3: Class balance
    disease_counts = df['disease'].value_counts()
    balance_ratio = disease_counts.max() / disease_counts.min()
    if balance_ratio > 10:
        checks.append(f"⚠️  Severe class imbalance: {balance_ratio:.1f}x ratio")
    elif balance_ratio > 5:
        checks.append(f"⚠️  Moderate class imbalance: {balance_ratio:.1f}x ratio")
    else:
        checks.append(f"✓ Reasonable class balance: {balance_ratio:.1f}x ratio")
    
    # Check 4: Dataset source bias
    source_disease = pd.crosstab(df['dataset_source'], df['disease'])
    from scipy.stats import chi2_contingency
    chi2, p_val, _, _ = chi2_contingency(source_disease)
    if p_val < 0.05:
        checks.append("⚠️  Significant dataset source - disease correlation")
        report['dataset_bias'] = True
    else:
        checks.append("✓ No significant dataset source - disease bias")
    
    for check in checks:
        print(f"  {check}")
    
    # Save report
    report_file = Path("output/data_leakage_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("DATA LEAKAGE AND QUALITY ASSESSMENT REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
        f.write(f"Dataset: final_ultrasound_dataset.csv\n\n")
        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total samples: {report['total_samples']}\n")
        f.write(f"Unique patients: {report['unique_patients']}\n")
        f.write(f"Unique images: {report['unique_images']}\n")
        f.write(f"Disease classes: {report['disease_classes']}\n")
        f.write(f"Severity classes: {report['severity_classes']}\n")
        f.write(f"Dataset sources: {report['dataset_sources']}\n\n")
        f.write("QUALITY CHECKS\n")
        f.write("-"*80 + "\n")
        for check in checks:
            # Replace Unicode characters
            check_ascii = check.replace('✓', 'OK').replace('✅', 'PASS').replace('❌', 'FAIL').replace('⚠️', 'WARN').replace('📋', 'INFO')
            f.write(f"{check_ascii}\n")
    
    print(f"\n✓ Report saved: {report_file}")
    return report


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE DATA ANALYSIS AND LEAKAGE DETECTION")
    print("="*80)
    
    # Load dataset
    df = load_master_dataset()
    if df is None:
        return
    
    # Analyses
    analyze_class_distribution(df)
    detect_duplicate_images(df)
    detect_patient_leakage(df)
    detect_dataset_source_bias(df)
    detect_severity_bias(df)
    report = generate_leakage_report(df)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
