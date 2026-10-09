# Mpox Fusion Reproducibility Package — Reviewer 1 Comment 11

This package contains the exact artifacts needed to audit the numerical results reported in the revised manuscript.

## Core artifacts
- `generate_symptom_data_v2(1).py`: complete synthetic symptom generator; seed = 42.
- `class_conditional_probabilities.csv`: exact class-conditional probabilities used by the generator after blending.
- `synthetic_symptoms.csv`: generated 600-record symptom dataset (100 per class).
- `image_to_synthetic_pairing.csv`: reconstructed exact image-to-synthetic-record pairing using the training code logic and pairing RNG seed = 0.
- `train_val_test_indices.csv`: exact split assignment for all 755 paired samples.
- `per_sample_predictions.csv`: exact predictions for every test sample from image-only, symptom-only, and fusion models.
- `confusion_matrix_image_only.csv`, `confusion_matrix_symptom_only.csv`, `confusion_matrix_fusion.csv`: exact test confusion matrices.
- `Figure_9_Corrected_History.csv`: training history underlying Figure 9.
- `training_script_corrected.py`: corrected training script, including the explicit 12-feature order, seeds, pairing logic, split logic, prediction export, and confusion-matrix export.
- `reproduce_tables_V_VIII_and_Wilson.py`: calculates Tables V–VIII metrics and Wilson 95% confidence intervals from the sample-level predictions.
- `reproducibility_metadata.json`: seeds, dimensions, sample counts, and generation/pairing metadata.

## Exact symptom feature order
1. fever
2. lymphadenopathy
3. headache
4. muscle_ache
5. fatigue
6. cough
7. conjunctivitis
8. mouth_ulcers
9. travel_or_contact_history
10. animal_contact
11. rash_multistage
12. age_years

`symptom_count` is a derived field and is intentionally excluded from the model input.

## Seeds
- Symptom generation: 42
- Image-to-symptom pairing: 0
- Model training/splitting: 42

## Data split
The paired dataset contains 755 image records. The exact split assignments are in `train_val_test_indices.csv`.

## Verification
The sample-level predictions are the authoritative source for checking accuracy, per-class precision/recall/F1, confusion matrices, and Wilson intervals. The Wilson calculations use the exact integer number correct out of 151 test samples and the Wilson method.

## Important scope note
The symptom data are synthetic, and image records are paired with class-conditional synthetic symptom records rather than real patient-linked clinical observations. This package documents that procedure explicitly so the reported numerical results can be reproduced and audited.
