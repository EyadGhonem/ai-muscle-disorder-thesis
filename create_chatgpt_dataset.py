import pandas as pd
import random

# Load the dataset
df = pd.read_csv('final_ultrasound_dataset_REAL_features.csv')

# Shuffle and select 50 random samples
random.seed(42)
sample_indices = random.sample(range(len(df)), min(50, len(df)))
sample_df = df.iloc[sample_indices].reset_index(drop=True)

# Create ChatGPT dataset (without label and severity_label columns)
chatgpt_df = sample_df.drop(columns=['label', 'severity_label']).copy()
chatgpt_df.insert(0, 'sample_id', [f'SAMPLE_{i+1:03d}' for i in range(len(chatgpt_df))])

# Create labels file for comparison (keep only key columns)
labels_df = sample_df[['image_path', 'label', 'severity_label', 'patient_id']].copy()
labels_df.insert(0, 'sample_id', [f'SAMPLE_{i+1:03d}' for i in range(len(labels_df))])

# Save both files
chatgpt_df.to_csv('chatgpt_comparison_dataset_50samples.csv', index=False)
labels_df.to_csv('chatgpt_comparison_labels_ANSWERS.csv', index=False)

print(f"✓ Created ChatGPT dataset: chatgpt_comparison_dataset_50samples.csv")
print(f"  - Rows: {len(chatgpt_df)}")
print(f"  - Columns: {len(chatgpt_df.columns)}")
print(f"\n✓ Created Labels file: chatgpt_comparison_labels_ANSWERS.csv")
print(f"  - Rows: {len(labels_df)}")
print(f"\nLabels breakdown:")
print(labels_df['label'].value_counts())
print(f"\nSeverity breakdown:")
print(labels_df['severity_label'].value_counts())
