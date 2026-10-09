# Early Fusion of Skin-Lesion Images and Clinical Symptom Data for Improved Differential Diagnosis of Mpox

## Overview

This repository is intended to contain the code and reproducibility artifacts associated with the research paper on early fusion of skin-lesion images and synthetic clinical symptom data for six-class skin-lesion classification.

The study compares three models:

1. **Image-only model:** EfficientNetB0 with a frozen ImageNet-pretrained backbone.
2. **Symptom-only model:** A dense neural network using structured synthetic symptom features.
3. **Fusion model:** An early-fusion architecture combining image and symptom representations.

## Dataset

The image experiments use the Mpox Skin Lesion Dataset v2.0 (MSLD v2.0).

The symptom dataset is synthetic and contains 600 records, with 100 records generated for each of six classes. Each symptom vector consists of patient age and 11 binary symptom indicators, giving 12 input features.

**Important limitation:** Symptom records are generated using class-conditioned distributions and paired with images from the corresponding class. Therefore, the experiments should be interpreted as a simulation-based evaluation, not as evidence of validated clinical diagnostic performance.

## Experimental Results

The revised manuscript reports the following results on the fixed test set of 151 images:

| Model | Correct predictions | Accuracy |
|---|---:|---:|
| Image-only | 129/151 | 85.43% |
| Symptom-only | 138/151 | 91.39% |
| Early fusion | 143/151 | 94.70% |

These results are specific to the experimental protocol described in the manuscript. They do not establish superiority over contemporary state-of-the-art systems or demonstrate clinical benefit.

## Reproducibility

The repository should contain the actual project scripts and available artifacts for:

- Synthetic symptom generation and class-conditional probabilities.
- Feature order and input dimensionality.
- Random seeds and image-to-symptom pairing.
- Training, validation, and test split indices.
- Model definitions, training and evaluation scripts.
- Test predictions and confusion matrices.
- Training histories and confidence-interval calculations.

Only artifacts that have actually been uploaded and verified should be considered available.

## Limitations

- The symptom data are synthetic and class-conditioned.
- The image split is image-level, not confirmed patient-disjoint.
- Paired statistical tests and patient-disjoint validation have not been completed.
- The reported accuracy should not be interpreted as proof of clinical utility.

## Intended Use

This project is intended for research reproducibility and educational purposes. It is not a clinically validated diagnostic system.
