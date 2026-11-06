#!/usr/bin/env python3
"""
Filename-derived candidate patient-key audit for the Mpox fusion package.

This script audits the 755 image paths in image_to_synthetic_pairing.csv.
It parses the public filename convention:
    DiseaseCode_PatientNumber_ImageNumber
with a Train/Valid/Test prefix, e.g. Train_MKP_136_01.jpg.

IMPORTANT:
- The parsed values are candidate patient keys derived from filenames.
- They are not independently verified clinical patient identifiers.
- This is an audit of partition overlap, NOT a patient-disjoint evaluation.
- The source-reported patient total (541) does not reconcile with the
  524 parsed candidate keys; this discrepancy remains unresolved.

Run:
    python patient_id_audit.py --pairing image_to_synthetic_pairing.csv

Outputs:
    patient_id_audit_per_image.csv
    patient_id_audit_per_class.csv
    patient_id_audit_summary.csv
"""
import argparse
import re
from pathlib import Path
import pandas as pd

FILENAME_RE = re.compile(
    r"(?:Train|Valid|Test)_([A-Za-z]+)_(\d+)_(\d+)\.jpg$",
    re.IGNORECASE,
)
PARTITION_ORDER = ["train", "validation", "test"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairing", default="image_to_synthetic_pairing.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.pairing)
    required = {"path", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    parts = df["path"].astype(str).str.extract(FILENAME_RE)
    if parts.isna().any(axis=1).any():
        bad = df.loc[parts.isna().any(axis=1), "path"].tolist()
        raise ValueError(f"Could not parse {len(bad)} image paths; first examples: {bad[:5]}")

    df["candidate_patient_key"] = parts[0].str.upper() + "_" + parts[1]
    df["image_number"] = parts[2]
    groups = df.groupby("candidate_patient_key")["split"].agg(lambda s: sorted(set(s)))
    partitions = groups.apply(lambda xs: ";".join(p for p in PARTITION_ORDER if p in xs))
    shared = groups.apply(lambda xs: len(xs) > 1)
    train_test = groups.apply(lambda xs: "train" in xs and "test" in xs)
    train_val = groups.apply(lambda xs: "train" in xs and "validation" in xs)
    val_test = groups.apply(lambda xs: "validation" in xs and "test" in xs)

    image = df.copy()
    image["candidate_key_partitions"] = image["candidate_patient_key"].map(partitions)
    image["candidate_key_crosses_partitions"] = image["candidate_patient_key"].map(shared)
    image["candidate_key_train_test_overlap"] = image["candidate_patient_key"].map(train_test)
    image["candidate_key_train_validation_overlap"] = image["candidate_patient_key"].map(train_val)
    image["candidate_key_validation_test_overlap"] = image["candidate_patient_key"].map(val_test)
    image["is_test_image_with_candidate_key_in_other_partition"] = (
        (image["split"] == "test") & image["candidate_patient_key"].map(shared)
    )
    image.to_csv("patient_id_audit_per_image.csv", index=False)

    rows = []
    for label, g in df.groupby("label", sort=True):
        gp = g.groupby("candidate_patient_key")["split"].agg(lambda s: set(s))
        shared_keys = set(gp[gp.map(len) > 1].index)
        test_g = g[g["split"] == "test"]
        rows.append({
            "class": label,
            "image_rows": len(g),
            "unique_candidate_patient_keys": g["candidate_patient_key"].nunique(),
            "candidate_keys_crossing_partitions": len(shared_keys),
            "image_rows_with_candidate_key_crossing_partitions": int(g["candidate_patient_key"].isin(shared_keys).sum()),
            "test_image_rows": len(test_g),
            "test_images_with_candidate_key_in_other_partition": int(test_g["candidate_patient_key"].isin(shared_keys).sum()),
        })
    pd.DataFrame(rows).to_csv("patient_id_audit_per_class.csv", index=False)

    test = df[df["split"] == "test"]
    train_keys = set(df.loc[df["split"] == "train", "candidate_patient_key"])
    val_keys = set(df.loc[df["split"] == "validation", "candidate_patient_key"])
    summary_rows = [
        ("image_rows_audited", len(df), "Rows in image_to_synthetic_pairing.csv"),
        ("unparsed_image_paths", 0, "Expected zero under the filename parser"),
        ("unique_filename_derived_candidate_patient_keys", df["candidate_patient_key"].nunique(), "Candidate keys, not independently verified patient IDs"),
        ("candidate_keys_crossing_any_partitions", int(shared.sum()), "Keys appearing in more than one partition"),
        ("candidate_keys_train_and_test", int(train_test.sum()), "May also occur in validation"),
        ("candidate_keys_train_and_validation", int(train_val.sum()), "May also occur in test"),
        ("candidate_keys_validation_and_test", int(val_test.sum()), "May also occur in train"),
        ("candidate_keys_in_all_three_partitions", int((groups.map(len) == 3).sum()), "Keys appearing in all three partitions"),
        ("test_image_rows", len(test), "Test rows in pairing file"),
        ("test_images_with_candidate_key_in_other_partition", int(test["candidate_patient_key"].isin(set(groups[shared].index)).sum()), "Test images whose candidate key appears in at least one other partition"),
        ("test_images_with_candidate_key_in_train", int(test["candidate_patient_key"].isin(train_keys).sum()), "May overlap with validation category"),
        ("test_images_with_candidate_key_in_validation", int(test["candidate_patient_key"].isin(val_keys).sum()), "May overlap with train category"),
        ("unique_candidate_keys_in_test", int(test["candidate_patient_key"].nunique()), "Distinct candidate keys among test rows"),
    ]
    pd.DataFrame(summary_rows, columns=["metric", "value", "interpretation"]).to_csv(
        "patient_id_audit_summary.csv", index=False
    )
    print("Audit complete. Outputs written to current directory.")

if __name__ == "__main__":
    main()
