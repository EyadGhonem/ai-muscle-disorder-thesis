Muscle Disorder Detection — Thesis GUI
======================================

Data sources (only these two):
  1. FSHD ultrasound — data/ULTRASOUND_LABELD_1_FSHD/images
  2. MAT-labeled myopathy — data/images_extracted_from_mat_LABELED/
     (Normal, IBM, Dermatomyositis, Polymyositis)

ULTRASOUND_LABELD_2 tabular-only rows are NOT used in this GUI.

Train all GUI models on ALL labeled ultrasound images (~28k, no subsampling):
  python scripts/train_gui_on_real_ultrasound.py --epochs 15
  Log: output/training_logs/full_gui_train.log
  (--quick is smoke test only — not for thesis)

Writes output/gui_real_ultrasound_dataset.csv, trained_models.pkl,
output/dl_models/*_severity.keras, gui_demo/models/*_disease.keras

Setup
-----
  pip install -r gui_demo/requirements_gui.txt
  python gui_demo/prepare_demo_data.py
  python scripts/train_gui_disease_cnns.py    # optional; ~5 epochs for 4 DL models
  streamlit run gui_demo/app.py

Workflow
--------
  1. Choose demo image or upload ultrasound PNG
  2. Press Inspect — shows original, grayscale, threshold, ROI mask, processed ROI
  3. Choose Machine Learning or Deep Learning and a model (or Compare all)
  4. Press Predict — disease status, disease type, confidence, correct/wrong vs demo label

ML: 9 models from output/baseline_and_advanced_models/trained_models.pkl
    Features extracted live from image (Otsu mask + radiomics).

DL: ResNet50, DenseNet121, EfficientNetB0, MobileNetV2
    4-class disease on MAT images (gui_demo/models/*_disease.keras).
    If disease weights missing, severity weights from output/dl_models are used for FSHD.
