# MRI Dataset Structure

Use this structure for a clean and scalable MRI workflow.

## Folder Layout

```text
data/
  mri/
    raw/
      images/           # Original MRI volumes (.nii/.nii.gz)
      masks/            # Original segmentation masks (.nii/.nii.gz)
    processed/
      images/           # Resampled/normalized volumes
      masks/            # Processed multi-class masks
      binary_masks/     # Muscle-vs-background masks
    label_maps/
      thigh_muscle_segmentation_labels.json
      calf_muscle_segmentation_labels.json
      whole_muscle_SAT_segmentation_labels.json
    metadata/
      combined_label_lookup.csv
      cases.csv         # Case-level metadata (optional)
    splits/
      train.csv
      val.csv
      test.csv
    reports/
      qc_notes.md
      stats.csv
```

## Naming Convention

- Keep paired names for image and mask:
  - `subject001_session1.nii.gz`
  - `subject001_session1_mask.nii.gz`
- One mask file per image.
- Mask voxel values must match label IDs in `data/mri/label_maps/*.json`.

## Minimal Rules

- `raw/` is immutable source data.
- All preprocessing outputs go to `processed/`.
- Do not overwrite raw files.
- Keep train/val/test split files in `data/mri/splits/`.

## Your Current Import Command

```bash
python integrate_mri_segmentation_labels.py --inputs "c:\Users\Lenovo\Downloads\mri\thigh_muscle_segmentation_labels.json" "c:\Users\Lenovo\Downloads\mri\whole_muscle_SAT_segmentation_labels.json" "c:\Users\Lenovo\Downloads\mri\calf_muscle_segmentation_labels.json"
```

