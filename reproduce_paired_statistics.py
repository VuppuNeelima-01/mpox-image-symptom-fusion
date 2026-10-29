#!/usr/bin/env python3
"""Reproduce paired statistics from the committed test predictions.

This script does not retrain any model. It audits the committed 151-row prediction
file, verifies correctness flags and confusion-matrix CSVs, computes exact Wilson
intervals and two-sided exact McNemar tests, and runs a paired, class-stratified
percentile bootstrap for accuracy and macro-F1 differences.

Default bootstrap settings match the manuscript: 20,000 resamples, seed 42.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint

MODELS = {
    "image_only": "image_only_pred",
    "symptom_only": "symptom_only_pred",
    "fusion": "fusion_pred",
}
CORRECT_FLAGS = {
    "image_only": "image_only_correct",
    "symptom_only": "symptom_only_correct",
    "fusion": "fusion_correct",
}
CONFUSION_FILES = {
    "image_only": "confusion_matrix_image_only.csv",
    "symptom_only": "confusion_matrix_symptom_only.csv",
    "fusion": "confusion_matrix_fusion.csv",
}


def confusion_counts(y: np.ndarray, pred: np.ndarray, labels: list[str]) -> np.ndarray:
    """Rows=true labels; columns=predicted labels."""
    pos = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, predicted in zip(y, pred):
        cm[pos[actual], pos[predicted]] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> dict[str, float]:
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    total = cm.sum()
    return {
        "correct": int(tp.sum()),
        "n": int(total),
        "accuracy": float(tp.sum() / total),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


def check_saved_artifacts(df: pd.DataFrame, base: Path, labels: list[str]) -> None:
    required = {"true_label", *MODELS.values(), *CORRECT_FLAGS.values()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")
    y = df["true_label"].astype(str).to_numpy()
    for model, pred_col in MODELS.items():
        pred = df[pred_col].astype(str).to_numpy()
        flag = df[CORRECT_FLAGS[model]].astype(int).to_numpy()
        actual_flag = (pred == y).astype(int)
        if not np.array_equal(flag, actual_flag):
            raise ValueError(f"Correctness flags disagree with {pred_col} predictions")
        path = base / CONFUSION_FILES[model]
        if path.exists():
            committed = pd.read_csv(path, index_col=0)
            cm = confusion_counts(y, pred, list(committed.index.astype(str)))
            if list(committed.index.astype(str)) != list(committed.columns.astype(str)):
                raise ValueError(f"Confusion matrix labels do not match in {path.name}")
            if not np.array_equal(cm, committed.to_numpy(dtype=np.int64)):
                raise ValueError(f"Saved confusion matrix does not match predictions: {path.name}")
        else:
            raise FileNotFoundError(f"Missing expected confusion matrix: {path}")


def paired_stratified_bootstrap(
    df: pd.DataFrame, labels: list[str], n_resamples: int, seed: int
) -> dict[str, dict[str, float]]:
    """Percentile CIs from matched, within-true-class resampling.

    For every bootstrap replicate, sample each true class independently with
    replacement while retaining its original support. The same sampled row IDs
    are used for all models, preserving the paired design. Macro-F1 is recomputed
    from the six-class confusion matrix for each replicate.
    """
    y = df["true_label"].astype(str).to_numpy()
    predictions = {name: df[col].astype(str).to_numpy() for name, col in MODELS.items()}
    class_indices = {label: np.flatnonzero(y == label) for label in labels}
    n, k = len(y), len(labels)
    rng = np.random.default_rng(seed)
    cm_boot = {name: np.zeros((n_resamples, k, k), dtype=np.int32) for name in MODELS}

    for true_i, label in enumerate(labels):
        idx = class_indices[label]
        # A single draw matrix per class is shared by all three models.
        sampled_rows = rng.choice(idx, size=(n_resamples, len(idx)), replace=True)
        for name, pred in predictions.items():
            draw_predictions = pred[sampled_rows]
            for pred_i, predicted_label in enumerate(labels):
                cm_boot[name][:, true_i, pred_i] = np.sum(
                    draw_predictions == predicted_label, axis=1
                )

    boot_metrics: dict[str, dict[str, np.ndarray]] = {}
    for name, matrices in cm_boot.items():
        tp = np.diagonal(matrices, axis1=1, axis2=2).astype(float)
        support = matrices.sum(axis=2).astype(float)
        predicted_count = matrices.sum(axis=1).astype(float)
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        precision = np.divide(tp, predicted_count, out=np.zeros_like(tp), where=predicted_count > 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(tp), where=(precision + recall) > 0)
        boot_metrics[name] = {
            "accuracy": tp.sum(axis=1) / n,
            "macro_f1": f1.mean(axis=1),
        }

    results: dict[str, dict[str, float]] = {}
    for comparator in ("image_only", "symptom_only"):
        for metric in ("accuracy", "macro_f1"):
            delta = boot_metrics["fusion"][metric] - boot_metrics[comparator][metric]
            key = f"fusion_minus_{comparator}_{metric}"
            results[key] = {
                "ci_low": float(np.quantile(delta, 0.025)),
                "ci_high": float(np.quantile(delta, 0.975)),
                "bootstrap_mean": float(delta.mean()),
            }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="per_sample_predictions.csv")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        parser.error("Use at least 1000 bootstrap resamples; manuscript default is 20000.")

    pred_path = Path(args.predictions)
    base = pred_path.resolve().parent
    df = pd.read_csv(pred_path)
    labels = sorted(df["true_label"].astype(str).unique().tolist())
    check_saved_artifacts(df, base, labels)

    print("Artifact checks: correctness flags and all three confusion matrices match the prediction CSV.")
    print(f"Audited file: {pred_path}")
    print(f"Rows: {len(df)}; labels: {', '.join(labels)}")
    print("\nOverall model metrics (macro metrics use the unweighted mean across labels):")
    observed = {}
    y = df["true_label"].astype(str).to_numpy()
    for model, pred_col in MODELS.items():
        cm = confusion_counts(y, df[pred_col].astype(str).to_numpy(), labels)
        m = metrics_from_cm(cm)
        observed[model] = m
        lo, hi = proportion_confint(m["correct"], m["n"], alpha=0.05, method="wilson")
        print(
            f"{model}: {m['correct']}/{m['n']}={m['accuracy']:.6%}; "
            f"macro-P={m['macro_precision']:.6f}; macro-R={m['macro_recall']:.6f}; "
            f"macro-F1={m['macro_f1']:.6f}; Wilson95=[{lo:.6%}, {hi:.6%}]"
        )

    print("\nExact two-sided McNemar tests on paired correctness indicators:")
    y_true = df["true_label"].astype(str).to_numpy()
    correct = {model: df[col].astype(str).to_numpy() == y_true for model, col in MODELS.items()}
    for comparator in ("image_only", "symptom_only"):
        fusion_only = int(np.sum(correct["fusion"] & ~correct[comparator]))
        comparator_only = int(np.sum(~correct["fusion"] & correct[comparator]))
        pvalue = float(binomtest(min(fusion_only, comparator_only),
                                 fusion_only + comparator_only, p=0.5,
                                 alternative="two-sided").pvalue)
        print(f"fusion vs {comparator}: fusion-only-correct={fusion_only}; "
              f"comparator-only-correct={comparator_only}; exact two-sided p={pvalue:.10f}")

    print(f"\nPaired class-stratified percentile bootstrap: B={args.bootstrap}; seed={args.seed}")
    print("Rows are resampled within each true class; identical sampled rows are used across models.")
    boot = paired_stratified_bootstrap(df, labels, args.bootstrap, args.seed)
    for key, values in boot.items():
        metric = "accuracy" if key.endswith("_accuracy") else "macro_f1"
        # Accuracy deltas reported as percentage points; macro-F1 deltas on 0-1 scale.
        scale = 100 if metric == "accuracy" else 1
        unit = "percentage points" if scale == 100 else "macro-F1 units"
        comparator = "image_only" if "image_only" in key else "symptom_only"
        comp = "image-only" if comparator == "image_only" else "symptom-only"
        observed_delta = observed["fusion"][metric] - observed[comparator][metric]
        print(f"fusion - {comp} {metric}: observed={observed_delta*scale:.6f} {unit}; "
              f"95% CI [{values['ci_low']*scale:.6f}, {values['ci_high']*scale:.6f}]; "
              f"bootstrap mean={values['bootstrap_mean']*scale:.6f}")

    print("\nInterpretation: all statistics are conditional on the saved fixed test split. They do not"
          " establish patient-level generalization, clinical utility, or independent multimodal benefit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
