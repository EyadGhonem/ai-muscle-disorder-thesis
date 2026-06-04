
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
