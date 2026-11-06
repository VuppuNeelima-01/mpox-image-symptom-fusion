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

### Statistical Reproducibility

The repository includes a script for reproducing the paired statistical analyses using the saved test-set predictions.

**Files**
- `reproduce_paired_statistics.py`: computes paired model comparisons and bootstrap confidence intervals.
- `paired_statistics_results.txt`: records the resulting statistics.
- `per_sample_predictions.csv`: contains the saved test-set predictions used for the analysis.

**Reproducibility note:** These analyses use saved predictions from a fixed test split. They do not independently reproduce model training or establish patient-level generalization. The symptoms are synthetically generated using class-conditioned distributions, and the results should be interpreted as a simulation-based feasibility study.

Run the script using Python with the dependencies specified in the repository.

## Statistical Reproducibility

The repository includes `reproduce_paired_statistics.py`, which verifies the saved predictions, confusion matrices, Wilson 95% confidence intervals, exact McNemar tests, and paired bootstrap confidence intervals.

### How to run

Install the required Python libraries:

```bash
pip install numpy pandas scipy statsmodels
```

Run from the repository root:

```bash
python reproduce_paired_statistics.py --predictions per_sample_predictions.csv --bootstrap 20000 --seed 42
```

The analysis uses saved predictions from a fixed test split. It does not retrain the models or establish patient-level generalization.

The image dataset and trained model weights are not included in this repository.

Figure note: Figure_9_Wilson_CI.png is the confidence-interval figure used in the revised manuscript. Figure_9_Corrected_History.csv is a separate legacy training-history file; it is not used to support the current Figure 9.


## Filename-derived candidate patient-key audit

The repository also includes:
- `patient_id_audit.py`: script to parse candidate patient keys and audit partition overlap.
- `patient_id_audit_per_image.csv`: per-image filename-derived key and partition-membership results.
- `patient_id_audit_per_class.csv`: class-level counts of candidate keys and partition overlap.
- `patient_id_audit_summary.csv`: overall audit counts.

Run from the repository root:

```bash
pip install pandas
python patient_id_audit.py --pairing image_to_synthetic_pairing.csv
```

The audit parses the dataset's published filename convention `DiseaseCode_PatientNumber_ImageNumber` from the 755 paths in `image_to_synthetic_pairing.csv`. It identifies 524 filename-derived candidate patient keys, of which 67 occur in more than one split. Of 151 test image rows, 61 have a candidate key that also appears in another partition (56 occur in training and 19 in validation; these categories overlap). The shared-key partition patterns are 34 train/test, 17 train/validation, 5 validation/test, and 11 in all three.

**Interpretation limitation:** These are filename-derived candidate keys, not independently verified patient identifiers. The 524 parsed keys do not reconcile with the 541 unique patients reported by the dataset source. Accordingly, this audit documents possible patient overlap in the existing image-level split; it does not establish patient-disjoint evaluation or patient-level generalization. A patient-disjoint retraining/evaluation was not performed.
