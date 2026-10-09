"""
generate_symptom_data_v2.py
------------------------------
HARDER, more realistic version of the synthetic symptom generator,
built in response to the reviewer's circularity concern.

This does NOT remove the fundamental circularity (that's only fixable
with real paired patient data -- see the paper's Limitations section).
What it DOES do is make the simulation less artificially easy and more
representative of real clinical data collection, along three axes the
reviewer explicitly named as necessary:

1. OVERLAPPING DISTRIBUTIONS: each class's symptom probabilities are
   blended toward the population-average profile (BLEND_FACTOR controls
   how much), so classes are no longer as cleanly separable by symptoms
   alone as before.

2. NOISE: after sampling from the (now-overlapping) class distribution,
   each binary symptom has a small independent chance of being flipped,
   simulating measurement/reporting error unrelated to the true class.

3. MISSINGNESS: after noise, each symptom has a chance of being
   recorded as absent even if actually present, simulating incomplete
   real-world symptom reporting (patients under-report, forget, or are
   not asked about every symptom).

Compare this file's SYMPTOM_PROFILES and sampling logic against the
original generate_symptom_data.py to see exactly what changed.
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)

CLASSES = ["Mpox", "Chickenpox", "Measles", "Cowpox", "HFMD", "Healthy"]

# ---- Same base clinical profiles as the original file ----
BASE_SYMPTOM_PROFILES = {
    "Mpox":        dict(fever=0.85, lymphadenopathy=0.80, headache=0.65,
                         muscle_ache=0.60, fatigue=0.70, cough=0.15,
                         conjunctivitis=0.10, mouth_ulcers=0.10,
                         travel_or_contact_history=0.55, animal_contact=0.10,
                         age_years_mean=32, rash_multistage=0.75),
    "Chickenpox":  dict(fever=0.65, lymphadenopathy=0.15, headache=0.30,
                         muscle_ache=0.25, fatigue=0.40, cough=0.10,
                         conjunctivitis=0.05, mouth_ulcers=0.10,
                         travel_or_contact_history=0.35, animal_contact=0.02,
                         age_years_mean=9,  rash_multistage=0.85),
    "Measles":     dict(fever=0.95, lymphadenopathy=0.20, headache=0.40,
                         muscle_ache=0.25, fatigue=0.55, cough=0.80,
                         conjunctivitis=0.75, mouth_ulcers=0.05,
                         travel_or_contact_history=0.30, animal_contact=0.02,
                         age_years_mean=12, rash_multistage=0.20),
    "Cowpox":      dict(fever=0.35, lymphadenopathy=0.20, headache=0.15,
                         muscle_ache=0.15, fatigue=0.20, cough=0.05,
                         conjunctivitis=0.03, mouth_ulcers=0.02,
                         travel_or_contact_history=0.10, animal_contact=0.80,
                         age_years_mean=28, rash_multistage=0.30),
    "HFMD":        dict(fever=0.55, lymphadenopathy=0.10, headache=0.15,
                         muscle_ache=0.10, fatigue=0.30, cough=0.10,
                         conjunctivitis=0.03, mouth_ulcers=0.70,
                         travel_or_contact_history=0.20, animal_contact=0.02,
                         age_years_mean=4,  rash_multistage=0.15),
    "Healthy":     dict(fever=0.02, lymphadenopathy=0.02, headache=0.05,
                         muscle_ache=0.03, fatigue=0.08, cough=0.05,
                         conjunctivitis=0.02, mouth_ulcers=0.01,
                         travel_or_contact_history=0.10, animal_contact=0.05,
                         age_years_mean=30, rash_multistage=0.0),
}

BINARY_SYMPTOMS = ["fever", "lymphadenopathy", "headache", "muscle_ache",
                    "fatigue", "cough", "conjunctivitis", "mouth_ulcers",
                    "travel_or_contact_history", "animal_contact",
                    "rash_multistage"]

# ---- NEW: difficulty knobs ----
BLEND_FACTOR = 0.35   # 0 = original (fully class-specific), 1 = fully population-average (uninformative)
NOISE_FLIP_PROB = 0.07   # chance any binary symptom gets randomly flipped after sampling
MISSING_PROB = 0.12      # chance a present symptom is recorded as absent (under-reporting)

# Compute population-average probability per symptom (mean across classes)
POPULATION_AVG = {
    s: np.mean([BASE_SYMPTOM_PROFILES[c][s] for c in CLASSES])
    for s in BINARY_SYMPTOMS
}

# Build the blended (overlapping) profiles actually used for sampling
SYMPTOM_PROFILES = {}
for c in CLASSES:
    blended = dict(BASE_SYMPTOM_PROFILES[c])  # copy (keeps age_years_mean etc.)
    for s in BINARY_SYMPTOMS:
        base_p = BASE_SYMPTOM_PROFILES[c][s]
        blended[s] = (1 - BLEND_FACTOR) * base_p + BLEND_FACTOR * POPULATION_AVG[s]
    SYMPTOM_PROFILES[c] = blended


def generate_patient_record(label, patient_id):
    profile = SYMPTOM_PROFILES[label]
    record = {"patient_id": patient_id, "label": label}
    for symptom in BINARY_SYMPTOMS:
        p = profile[symptom]
        true_value = int(RNG.random() < p)

        # NEW: independent noise -- flip regardless of class
        if RNG.random() < NOISE_FLIP_PROB:
            true_value = 1 - true_value

        # NEW: missingness -- a present symptom may go unreported
        if true_value == 1 and RNG.random() < MISSING_PROB:
            true_value = 0

        record[symptom] = true_value

    age = RNG.normal(profile["age_years_mean"], 8)
    record["age_years"] = int(np.clip(age, 1, 85))
    record["symptom_count"] = sum(record[s] for s in BINARY_SYMPTOMS)
    return record


def generate_dataset(n_per_class=100, out_path=None):
    rows = []
    pid = 0
    for label in CLASSES:
        for _ in range(n_per_class):
            rows.append(generate_patient_record(label, pid))
            pid += 1
    df = pd.DataFrame(rows)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} synthetic patient records to {out_path}")
    return df


if __name__ == "__main__":
    print(f"Difficulty settings: BLEND_FACTOR={BLEND_FACTOR}, "
          f"NOISE_FLIP_PROB={NOISE_FLIP_PROB}, MISSING_PROB={MISSING_PROB}\n")
    df = generate_dataset(n_per_class=100, out_path="data/symptoms/synthetic_symptoms.csv")
    print(df.head(10))
    print("\nClass distribution:")
    print(df["label"].value_counts())
