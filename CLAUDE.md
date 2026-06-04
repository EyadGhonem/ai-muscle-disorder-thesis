# Claude instructions — thesis repository

## Goal

Help complete the bachelor thesis **AI-Powered Radiomics for Assessment of Muscle Disorders** (Eyad Ghonem, GUC MET) using this codebase as the source of truth.

## Primary document

Edit and expand content from **`THESIS_DRAFT_FOR_CLAUDE.md`** into the official MET Word template.

- **Keep unchanged:** Chapters 1–2, bibliography, front matter (per file instructions).  
- **Rewrite:** Abstract (Section A in draft), Chapter 4 (results), Chapter 5 (discussion/conclusion).  
- **Update:** Chapter 3 for ultrasound-only final scope.

## Experimental scope (final)

- **Modality:** Ultrasound only (no MRI in final results).  
- **Diseases:** FSHD, Inclusion Body Myositis, Dermatomyositis, Polymyositis, Normal.  
- **No DMD/BMD** (data unavailable).  
- **Image data:** Two on-disk cohorts only (see `README.md`); not `ULTRASOUND_LABELD_2` images.

## Models in the GUI

- **ML:** SVM, Random Forest, Logistic Regression, Gradient Boosting, XGBoost, LightGBM, CatBoost, Extra Trees, Stacking — radiomics features, 5-class disease.  
- **DL:** ResNet50, DenseNet121, EfficientNetB0, MobileNetV2 — FSHD **severity** (binary) and MAT **disease** (4-class).

Training entry point: `scripts/train_gui_on_real_ultrasound.py`.

## Reporting metrics honestly

- Report **patient-level splits** and **macro F1**, not only accuracy (FSHD-heavy imbalance).  
- GUI may use presentation calibration for demo runs; distinguish **reported pipeline metrics** vs **demo UI behavior** in thesis text.  
- After training, read `gui_demo/models/gui_training_metrics.json` and `output/baseline_and_advanced_models/gui_ml_training_summary.csv` locally (not in git if under `output/`).

## Key code paths

| Path | Role |
|------|------|
| `gui_demo/app.py` | Streamlit demo |
| `gui_demo/image_pipeline.py` | ROI + radiomics for uploads |
| `scripts/train_gui_on_real_ultrasound.py` | Full ML+DL training on real images |
| `build_master_dataset.py` / `create_final_ultrasound_dataset.py` | Legacy master CSV pipeline |

## Tone

Formal third person, GUC MET style, tables comparing models, limitations section mandatory.
