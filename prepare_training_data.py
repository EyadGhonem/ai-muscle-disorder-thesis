"""
Prepare training data - split images into train/validation/test sets
Create labels.csv for classification task
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

DATA_DIR = Path("data")
ULTRASOUND_DIR = DATA_DIR / "ultrasound_images"
OUTPUT_DIR = Path("output")

def get_image_list():
    """Get all ultrasound images"""
    image_extensions = ['*.tif', '*.TIF', '*.tiff', '*.TIFF', '*.png', '*.PNG', '*.jpg', '*.JPG']
    images = []
    
    for ext in image_extensions:
        images.extend(ULTRASOUND_DIR.glob(ext))
    
    # Deduplicate by lowercase name (important on Windows where glob can
    # return the same file for case-variant patterns like *.tif and *.TIF).
    unique_images = {}
    for img in images:
        unique_images[img.name.lower()] = img.name
    return sorted(unique_images.values())

def create_labels_template(images):
    """Create template labels.csv for user to fill"""
    
    df = pd.DataFrame({
        'image_name': images,
        'label': [None] * len(images),  # User fills this
        'disease_class': [''] * len(images),  # Optional: healthy/diseased/etc
        'notes': [''] * len(images)  # Optional: any notes
    })
    
    return df

def generate_sample_labels(images, random_split=None):
    """
    Generate sample labels for demo purposes
    
    For production: User should fill labels manually based on medical assessment
    """
    
    if random_split is None:
        random_split = 0.5  # 50-50 healthy/diseased for demo
    
    n_diseased = int(len(images) * random_split)
    labels = [1] * n_diseased + [0] * (len(images) - n_diseased)
    random.shuffle(labels)
    
    disease_map = {0: 'healthy', 1: 'diseased'}
    
    df = pd.DataFrame({
        'image_name': images,
        'label': labels,
        'disease_class': [disease_map[l] for l in labels],
        'notes': [''] * len(images)
    })
    
    return df

def split_train_val_test(images, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split images into train/validation/test sets
    
    With 246 images:
    - Train: 172 (70%)
    - Val: 37 (15%)
    - Test: 37 (15%)
    """
    
    # First split: train + temp (val + test)
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        images, labels, train_size=train_ratio, random_state=42
    )
    
    # Second split: temp into val and test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, train_size=val_ratio_adjusted, random_state=42
    )
    
    splits = {
        'train': {'images': train_imgs, 'labels': train_labels},
        'val': {'images': val_imgs, 'labels': val_labels},
        'test': {'images': test_imgs, 'labels': test_labels}
    }
    
    return splits

def create_split_csv(splits, output_path):
    """Create CSV with train/val/test assignment"""
    
    data = []
    
    for split_name, split_data in splits.items():
        for img, label in zip(split_data['images'], split_data['labels']):
            data.append({
                'image_name': img,
                'label': label,
                'split': split_name
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    return df

def print_split_summary(splits):
    """Print summary of train/val/test split"""
    
    print("\n📊 Train/Validation/Test Split Summary")
    print("-" * 60)
    
    for split_name, split_data in splits.items():
        n_images = len(split_data['images'])
        n_diseased = sum(split_data['labels'])
        n_healthy = n_images - n_diseased
        
        print(f"\n{split_name.upper()}:")
        print(f"  Total images: {n_images}")
        print(f"  Healthy: {n_healthy} ({n_healthy/n_images*100:.1f}%)")
        print(f"  Diseased: {n_diseased} ({n_diseased/n_images*100:.1f}%)")

if __name__ == "__main__":
    print("="*60)
    print("PREPARING TRAINING DATA")
    print("="*60 + "\n")
    
    # Check if images exist
    images = get_image_list()
    if not images:
        print(f"❌ No images found in {ULTRASOUND_DIR}")
        exit(1)
    
    print(f"✓ Found {len(images)} ultrasound images\n")
    
    # Create labels template
    print("Creating labels template...\n")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    labels_path = OUTPUT_DIR / "labels_template.csv"
    df_template = create_labels_template(images)
    df_template.to_csv(labels_path, index=False)
    print(f"✓ Created template: {labels_path}")
    print("\n📝 IMPORTANT:")
    print("   1. Fill out labels in the CSV file")
    print("   2. 'label' column: 0 = healthy, 1 = diseased")
    print("   3. Save as 'labels.csv' when done\n")
    
    # For demo: create sample labels
    print("Creating sample labels for demonstration...\n")
    images_list = [img.name if hasattr(img, 'name') else img for img in images]
    df_sample = generate_sample_labels(images_list, random_split=0.4)
    
    labels_sample_path = OUTPUT_DIR / "labels_sample.csv"
    df_sample.to_csv(labels_sample_path, index=False)
    print(f"✓ Sample labels: {labels_sample_path}")
    
    # Create train/val/test split using sample labels
    print("\n\nCreating train/validation/test split...\n")
    try:
        splits = split_train_val_test(
            df_sample['image_name'].tolist(),
            df_sample['label'].tolist()
        )
        
        split_csv_path = OUTPUT_DIR / "data_split.csv"
        df_split = create_split_csv(splits, split_csv_path)
        
        print(f"✓ Split CSV created: {split_csv_path}")
        print_split_summary(splits)
        
        print("\n" + "="*60)
        print("✅ Data preparation complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Review labels: output/labels_template.csv")
        print("2. Fill in disease labels (0 or 1)")
        print("3. Save as: output/labels.csv")
        print("4. Run: python train_ultrasound_classifier.py")
        
    except Exception as e:
        print(f"Error creating split: {e}")
