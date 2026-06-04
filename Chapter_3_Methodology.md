# Chapter 3
## Methodology

A well-structured methodology was adopted to ensure a comprehensive approach to model development, evaluation, and interpretability for muscle health classification using MRI and ultrasound imaging modalities. Below is a detailed description of the proposed model architecture and workflow, illustrated by a block diagram. Additionally, the training configuration and evaluation metrics employed in this experiment are outlined, as well as the dataset used.

### 3.1 Proposed Model

#### 3.1.1 Transfer Learning

Transfer learning was adopted, which employed pre-trained deep neural networks on large-scale datasets. Transfer learning is a technique that utilizes the knowledge of a trained model to learn another set of data, thus eliminating the time and effort needed to design and train such networks from scratch. Transfer learning aims to improve learning in the target domain by leveraging the knowledge from the source domain and learning tasks [61]. The use of transfer learning for this study allows for more efficient training despite the complexity and variability of medical imaging data by reducing the risk of overfitting and accelerating convergence through utilizing previously learned mid- and low-level features [62].

For this muscle health classification study, transfer learning was particularly valuable due to the limited availability of annotated medical imaging datasets and the high computational cost of training deep networks from scratch. By leveraging pre-trained models on ImageNet, we could utilize their learned feature extraction capabilities for medical image analysis tasks.

#### 3.1.2 Architectural Analysis of Investigated Models

This study employed deep CNNs to perform binary muscle health classification (healthy vs diseased) based on medical images. Specifically, two architectures were explored: a custom 3D CNN for MRI data and EfficientNetB0 for ultrasound images. According to the literature review presented in Section 2.2 of Chapter 2, these models were identified as suitable architectures for medical image classification tasks.

Both architectures were initially pretrained on the ImageNet dataset, allowing them to generalize visual features such as shapes, textures, and edges, which are critical for medical image analysis.

**MRI Classification Architecture:**
For MRI data, a custom 3D CNN was developed to handle the volumetric nature of MRI scans. The architecture was designed specifically for medical imaging tasks, incorporating 3D convolutional layers to capture spatial relationships across all three dimensions of the MRI volumes.

**Ultrasound Classification Architecture:**
For ultrasound images, EfficientNetB0 was adopted as the primary model due to its proven efficiency and accuracy in medical imaging tasks. EfficientNetB0 provided a better balance between accuracy and computational efficiency compared to other architectures considered.

**Architecture Overview:**
A detailed overview of the two architectures considered in this study is presented below:

*   **Custom 3D CNN for MRI:** A specialized architecture designed to handle volumetric medical imaging data. The model incorporates 3D convolutional layers, batch normalization, and dropout layers to prevent overfitting. The architecture was specifically optimized for muscle tissue analysis in MRI volumes.

*   **EfficientNetB0 for Ultrasound:** A highly efficient CNN model utilizing depthwise separable convolutions and squeeze-and-excitation (SE) blocks, designed through neural architecture search techniques (AutoML). EfficientNet employs compound scaling to optimize accuracy and efficiency simultaneously [20].

**Architectural Analysis of EfficientNet**

EfficientNet is a family of deep CNNs that achieve state-of-the-art accuracy across multiple classification tasks while being much more efficient, up to ten times smaller and faster, compared to previous models. The core innovation of EfficientNet lies in its use of AutoML techniques for efficiently scaling up CNNs by employing a compound scaling strategy. This strategy scales the network's depth, width, and resolution simultaneously using a single predefined coefficient, as opposed to separately scaling these dimensions [20].

Two key innovations underpin EfficientNet's architecture: depthwise separable convolutions [65] and squeeze-and-excitation (SE) blocks [66]. These elements significantly lower the computational cost while maintaining or even improving model accuracy.

*   **Depthwise Separable Convolution:**
    This technique splits the standard convolution operation into two stages: (1) depthwise convolution, where a single filter is used for every channel independently, extracting channel-specific spatial features; and (2) pointwise convolution, where a 1×1 convolution combines these features across channels.

*   **Squeeze-and-Excitation Unit:**
    First, a squeeze operation applies global average pooling to each channel, resulting in a 1×1×C vector (where C is the number of channels). Then, an excitation phase, comprising a fully connected layer -> ReLU -> fully connected -> sigmoid, is applied. This process recalibrates the channel-wise feature maps, capturing interdependencies between channels. The final excitation output is multiplied by the original input.

#### 3.1.3 Proposed Model Pipeline

The proposed model pipeline starts with the collection of comprehensive datasets that includes both MRI and ultrasound imaging modalities for muscle health assessment. The data is then subjected to preprocessing and feature extraction, which involves different approaches for each modality. For MRI data, volumetric preprocessing is applied, while ultrasound images undergo radiomics feature extraction. The preprocessed data is then fed into respective models, where both the custom 3D CNN (for MRI) and EfficientNetB0 (for ultrasound) are trained and tested. Additionally, a machine learning approach using radiomics features with Random Forest classifier is implemented for comparison. Each approach's performance is evaluated and compared, providing comprehensive analysis for muscle health classification.

Figure 3.1 illustrates the complete workflow of the proposed muscle health classification system.

```
[Figure 3.1: Block diagram of the proposed muscle health classification model]

MRI Data Collection -> 3D CNN -> MRI Classification Results
                     |
                     v
              Feature Extraction
                     |
                     v
              Clinical Validation

Ultrasound Data Collection -> Radiomics Extraction -> ML/DL Models -> Ultrasound Classification Results
                                                     |
                                                     v
                                              Clinical Validation
```

#### 3.1.4 Training Configuration and Hyperparameters

The compiled models used the Adam optimizer with a learning rate of 0.0001 and were trained using the binary cross-entropy loss function, suitable for binary muscle health classification tasks. The training configuration was standardized across both modalities as follows:

*   **MRI Model Configuration:**
    *   Epochs: 50
    *   Optimizer: Adam
    *   Batch Size: 8 (smaller batch size due to 3D volumetric data)
    *   Image Size: 128×128×128 (volumetric)
    *   Loss Function: Binary Cross-Entropy

*   **Ultrasound Model Configuration:**
    *   Epochs: 20
    *   Optimizer: Adam
    *   Batch Size: 32
    *   Image Size: 224×224×3
    *   Loss Function: Binary Cross-Entropy

*   **Machine Learning Configuration (Radiomics-based):**
    *   Algorithm: Random Forest Classifier
    *   Number of Estimators: 100
    *   Test Size: 20%
    *   Random State: 42

A separate instance of each model was trained for each imaging modality, allowing the networks to specialize in distinguishing between healthy and diseased muscle tissue characteristics unique to each imaging technique.

### 3.2 Evaluation Metrics

A set of standard evaluation metrics was used to evaluate the performance of the proposed models across both imaging modalities. These metrics are commonly used in medical image classification tasks and provide insights into the model's precision, generalization ability, and robustness. The mathematical definitions of the evaluation metrics used in this study adhere to standard formalizations thoroughly analyzed and examined by Sokolova et al. [67] and Rainio et al. [68].

The primary evaluation metrics used in this study are:

#### 3.2.1 Accuracy

Accuracy is a basic yet important metric that measures the overall correctness of the model by dividing the number of correctly classified instances by the total number of instances. The formula for accuracy is:

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Where:
*   TP: True Positives (correctly identified diseased cases)
*   TN: True Negatives (correctly identified healthy cases)
*   FP: False Positives (healthy cases incorrectly identified as diseased)
*   FN: False Negatives (diseased cases incorrectly identified as healthy)

#### 3.2.2 Confusion Matrix

To represent the distribution of correct and incorrect predictions across both classes (healthy and diseased), a confusion matrix was generated for each imaging modality. It displays the true labels versus predicted labels within a matrix structure, with diagonal entries indicating correct classifications and off-diagonal elements highlighting misclassifications.

#### 3.2.3 Classification Report

A detailed classification report was also generated, which includes precision, recall, F1-score, and support (number of true instances) for each class, as well as weighted-averaged and macro-averaged across all classes. This comprehensive report enables class-level performance analysis and helps evaluate how well the model generalizes.

**Precision** is the ratio of true positive predictions to all predicted positives for a given class. It is a measure of the model's ability to avoid false positives. The formula for precision is:

```
Precision = TP / (TP + FP)
```

**Recall (sensitivity)** measures the model's ability to correctly identify all actual positive instances. The formula for recall is:

```
Recall = TP / (TP + FN)
```

**F1-Score** is the harmonic mean of precision and recall. It is especially informative in datasets with class imbalance. The formula for F1-Score is:

```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

#### 3.2.4 Additional Clinical Metrics

For clinical validation, additional metrics specific to medical applications were computed:

*   **Specificity:** TN / (TN + FP) - Ability to correctly identify healthy cases
*   **AUC-ROC:** Area Under the Receiver Operating Characteristic Curve - Overall discriminative ability
*   **Feature Importance:** For ML approach, identification of most predictive radiomics features

### 3.3 Dataset

For this study, a comprehensive dataset comprising both MRI and ultrasound imaging modalities was used for muscle health classification. The dataset was collected from clinical sources and includes annotated medical images for both healthy and diseased muscle tissue conditions.

The dataset includes a wide range of imaging modalities:
*   **MRI:** 3D volumetric scans for comprehensive muscle tissue analysis
*   **Ultrasound:** 2D imaging for real-time muscle assessment

A set of sample images across the imaging modalities and health conditions can be seen in Figure 3.2.

```
[Figure 3.2: Sample Images from Different Imaging Modalities and Health Conditions]

(a) Healthy Muscle (MRI)    (b) Diseased Muscle (MRI)
(c) Healthy Muscle (Ultrasound)  (d) Diseased Muscle (Ultrasound)
```

#### 3.3.1 Dataset Structure

Table 3.1 presents a comprehensive breakdown of each imaging modality, its respective classes, and the total number of images.

**Table 3.1: Summary of the Dataset Structure**

| Imaging Modality | Classes | Total Images | Image Characteristics |
|------------------|---------|--------------|----------------------|
| MRI | Healthy, Diseased | Variable | 3D volumetric scans, multiple slice thicknesses |
| Ultrasound | Healthy, Diseased | Variable | 2D images, various resolutions and orientations |

#### 3.3.2 Data Splitting and Distribution

The data was divided into three subsets: training, validation, and testing, using a 70:15:15 ratio. The training set, which consists of 70% of the data, offers a sufficient number of examples for the models to learn from. This large training set supports the deep learning models to learn details about the patterns and distinctions of healthy versus diseased muscle tissue. The validation set, which consists of 15% of the data, is used to track the models' performance and fine-tune their learning. The remaining 15% of the data makes up the testing set, which serves as an independent baseline against which to evaluate the models' performance as well as their generalizing ability.

The number of training samples per imaging modality and class can be seen in Figure 3.3.

```
[Figure 3.3: Data Distribution Across Imaging Modalities and Classes]

MRI Healthy: [count]
MRI Diseased: [count]
Ultrasound Healthy: [count]
Ultrasound Diseased: [count]
```

#### 3.3.3 Data Processing and Augmentation Techniques

The pre-processing of images is crucial in the training process, as it improves the performance as well as the stability of deep-learning models. The dataset underwent comprehensive preprocessing and augmentation to ensure robust model training.

**MRI Data Processing:**
For MRI volumetric data, the following preprocessing steps were applied:
1. **Volume Normalization:** Intensity normalization across all slices
2. **Resampling:** Standardized voxel spacing for consistency
3. **Skull Stripping:** Removal of non-relevant tissue (if applicable)
4. **Registration:** Alignment to standard anatomical space
5. **Volume Cropping:** Focus on region of interest (muscle tissue)

**Ultrasound Data Processing:**
For ultrasound images, the following preprocessing steps were applied:
1. **Intensity Normalization:** Standardization of pixel intensity ranges
2. **Speckle Reduction:** Application of filters to reduce noise
3. **Contrast Enhancement:** Improvement of tissue visibility
4. **Resizing:** Standardization to 224×224 pixels for EfficientNet input

**Data Augmentation Techniques:**
To increase dataset diversity and prevent overfitting, the following augmentation techniques were applied to the training set:

1. **Rotation:** Rotation of the images by up to 15 degrees for ultrasound and 10 degrees for MRI slices
2. **Width and Height Shift:** Horizontal and vertical shifts of up to 10% of the image size
3. **Shearing and Zooming:** Shearing and zooming variations of up to 10%
4. **Horizontal Flipping:** Randomly flips the images for additional diversity (primarily for ultrasound)
5. **Brightness Adjustment:** Brightness adjustments range from 0.8 to 1.2 for varying imaging conditions
6. **Elastic Deformation:** Small elastic deformations to simulate tissue variations (MRI)

**Radiomics Feature Extraction:**
For the machine learning approach, radiomics features were extracted from both imaging modalities using PyRadiomics library. The extracted features include:

*   **First-order Statistics:** Intensity-based features (mean, median, skewness, kurtosis, etc.)
*   **Shape-based Features:** Geometric characteristics of regions of interest
*   **Texture Features:** Gray Level Co-occurrence Matrix (GLCM), Gray Level Run Length Matrix (GLRLM), Gray Level Size Zone Matrix (GLSZM)
*   **Higher-order Features:** Wavelet and Laplacian of Gaussian (LoG) filtered features

This comprehensive preprocessing and feature extraction pipeline ensured that both deep learning and machine learning approaches received high-quality, standardized input data for optimal performance in muscle health classification tasks.
