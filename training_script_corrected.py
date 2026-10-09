"""
train_keras_full.py  (CORRECTED VERSION)
-------------------------------------------
Two fixes applied vs. the version you already ran:

1. DIMENSIONALITY BUG FIX (Reviewer 1): symptom_cols now explicitly
   excludes 'symptom_count', a redundant derived column (the sum of
   the 11 binary indicators). This brings the true symptom input down
   to R^12 (11 binary indicators + age), matching the corrected
   manuscript. Previously it was accidentally R^13.

2. AUTOMATIC PREDICTION EXPORT: at the end of main(), after all three
   models finish training, this script now saves a CSV with every
   test image's true label and each model's prediction. This is what
   we need to run McNemar's paired test and paired bootstrap CIs.

Run this exactly the same way you ran the original:
    !python src/train_keras_full.py

It will train all three models (same as before) AND automatically
save outputs/per_sample_predictions.csv when done — no extra steps,
no separate cell needed.
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

IMAGE_DIR = "data/images"
SYMPTOM_CSV = "data/symptoms/synthetic_symptoms.csv"
CLASSES = ["Mpox", "Chickenpox", "Measles", "Cowpox", "HFMD", "Healthy"]
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_paired_data():
    """Mirrors data_loader.build_paired_dataset() -- pairs images with
    symptom records of the same class."""
    import glob
    # Map the dataset folder name to the label used in the CSV/model.
    # The image folder is named "Monkeypox", while the class label is "Mpox".
    IMAGE_FOLDER_MAP = {
        "Mpox": "Monkeypox",
        "Chickenpox": "Chickenpox",
        "Measles": "Measles",
        "Cowpox": "Cowpox",
        "HFMD": "HFMD",
        "Healthy": "Healthy"
    }

    rows = []
    for label in CLASSES:
        folder_name = IMAGE_FOLDER_MAP[label]
        image_folder = os.path.join(IMAGE_DIR, folder_name)

        for path in sorted(glob.glob(os.path.join(image_folder, "*"))):
            rows.append({
                "path": path,
                "label": label
            })

    img_df = pd.DataFrame(rows)

    if img_df.empty:
        raise ValueError(
            f"No images found in {IMAGE_DIR}. Expected folders: "
            f"{', '.join(CLASSES)}"
        )

    missing_classes = [
        label for label in CLASSES
        if not (img_df["label"] == label).any()
    ]
    if missing_classes:
        raise ValueError(
            f"No images found for classes: {missing_classes}. "
            f"Check folder names inside {IMAGE_DIR}."
        )

    print("\nImages found per class:")
    print(img_df["label"].value_counts().reindex(CLASSES, fill_value=0))

    symptom_df = pd.read_csv(SYMPTOM_CSV)

    # --- FIX APPLIED HERE ---
    # Previously: [c for c in symptom_df.columns if c not in ("patient_id", "label")]
    # That accidentally included 'symptom_count' (sum of the 11 binary
    # indicators), giving 13 input features instead of the documented 12.
    symptom_cols = [
        c for c in symptom_df.columns
        if c not in ("patient_id", "label", "symptom_count")
    ]

    EXPECTED = [
        "fever", "lymphadenopathy", "headache", "muscle_ache", "fatigue",
        "cough", "conjunctivitis", "mouth_ulcers",
        "travel_or_contact_history", "animal_contact", "rash_multistage",
        "age_years",
    ]
    assert sorted(symptom_cols) == sorted(EXPECTED), (
        f"Symptom feature mismatch: got {sorted(symptom_cols)} "
        f"({len(symptom_cols)} features), expected {sorted(EXPECTED)} (12 features)."
    )
    print(f"Symptom input dimensionality: R^{len(symptom_cols)} (expected R^12)")

    rng = np.random.default_rng(0)
    paired_rows = []
    for label in CLASSES:
        imgs = img_df[img_df["label"] == label].reset_index(drop=True)
        symptoms = symptom_df[symptom_df["label"] == label].reset_index(drop=True)

        if symptoms.empty:
            raise ValueError(
                f"No symptom records found for class '{label}'."
            )

        # The current synthetic file may contain fewer symptom records
        # than images. Replacement is therefore required in that case.
        # This limitation must be disclosed in the research paper.
        replace_sampling = len(imgs) > len(symptoms)
        if replace_sampling:
            print(
                f"Warning: class '{label}' has {len(imgs)} images but "
                f"only {len(symptoms)} symptom records; sampling with "
                f"replacement will occur."
            )

        idx = rng.choice(
            len(symptoms),
            size=len(imgs),
            replace=replace_sampling
        )
        for i in range(len(imgs)):
            row = {"path": imgs.loc[i, "path"], "label": label}
            for c in symptom_cols:
                row[c] = symptoms.loc[idx[i], c]
            paired_rows.append(row)
    return pd.DataFrame(paired_rows), symptom_cols


def load_image_array(path):
    img = tf.keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = tf.keras.utils.img_to_array(img)
    return preprocess_input(arr)


def build_fusion_model(num_symptom_features, num_classes):
    img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")
    base_cnn = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=img_input)
    base_cnn.trainable = False
    x_img = layers.GlobalAveragePooling2D()(base_cnn.output)
    x_img = layers.Dense(128, activation="relu")(x_img)
    x_img = layers.Dropout(0.3)(x_img)

    sym_input = layers.Input(shape=(num_symptom_features,), name="symptom_input")
    x_sym = layers.Dense(32, activation="relu")(sym_input)
    x_sym = layers.Dense(16, activation="relu")(x_sym)

    fused = layers.Concatenate()([x_img, x_sym])
    fused = layers.Dense(64, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)
    output = layers.Dense(num_classes, activation="softmax")(fused)

    model = Model(inputs=[img_input, sym_input], outputs=output, name="fusion_model")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return model


def build_image_only_model(num_classes):
    img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base_cnn = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=img_input)
    base_cnn.trainable = False
    x = layers.GlobalAveragePooling2D()(base_cnn.output)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=img_input, outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return model


def build_symptom_only_model(num_symptom_features, num_classes):
    sym_input = layers.Input(shape=(num_symptom_features,))
    x = layers.Dense(32, activation="relu")(sym_input)
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=sym_input, outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return model


def main():
    paired_df, symptom_cols = load_paired_data()
    print(f"Loaded {len(paired_df)} paired records across {len(CLASSES)} classes.")

    le = LabelEncoder()
    y = le.fit_transform(paired_df["label"])

    print("Loading and preprocessing images (this may take a while)...")
    X_img = np.stack([load_image_array(p) for p in paired_df["path"]])

    X_sym = paired_df[symptom_cols].values.astype(float)

    # Create explicit, reproducible train/validation/test indices.
    all_indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        all_indices, test_size=0.2, random_state=SEED, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.15, random_state=SEED,
        stratify=y[train_val_idx]
    )

    X_img_train, X_img_val, X_img_test = X_img[train_idx], X_img[val_idx], X_img[test_idx]
    X_sym_train, X_sym_val, X_sym_test = X_sym[train_idx], X_sym[val_idx], X_sym[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    scaler = StandardScaler()
    X_sym_train = scaler.fit_transform(X_sym_train)
    X_sym_val = scaler.transform(X_sym_val)
    X_sym_test = scaler.transform(X_sym_test)

    os.makedirs("outputs", exist_ok=True)
    split_df = pd.DataFrame({
        "paired_index": np.arange(len(y)),
        "label": le.inverse_transform(y),
        "split": "unused"
    })
    split_df.loc[train_idx, "split"] = "train"
    split_df.loc[val_idx, "split"] = "validation"
    split_df.loc[test_idx, "split"] = "test"
    split_df.to_csv("outputs/train_val_test_indices.csv", index=False)

    num_classes = len(CLASSES)
    num_sym_feats = X_sym_train.shape[1]

    print("\n--- Training image-only model ---")
    img_model = build_image_only_model(num_classes)
    img_history = img_model.fit(
        X_img_train,
        y_train,
        validation_data=(X_img_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )
    img_pred = np.argmax(img_model.predict(X_img_test), axis=1)
    print(classification_report(y_test, img_pred, target_names=le.classes_))

    print("\n--- Training symptom-only model ---")
    sym_model = build_symptom_only_model(num_sym_feats, num_classes)
    sym_history = sym_model.fit(
        X_sym_train,
        y_train,
        validation_data=(X_sym_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )
    sym_pred = np.argmax(sym_model.predict(X_sym_test), axis=1)
    print(classification_report(y_test, sym_pred, target_names=le.classes_))

    print("\n--- Training fusion model ---")
    fusion_model = build_fusion_model(num_sym_feats, num_classes)
    fusion_history = fusion_model.fit(
        [X_img_train, X_sym_train],
        y_train,
        validation_data=([X_img_val, X_sym_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )
    fusion_pred = np.argmax(fusion_model.predict([X_img_test, X_sym_test]), axis=1)
    print(classification_report(y_test, fusion_pred, target_names=le.classes_))

    # Save training histories and accuracy/loss plots
    import json
    import matplotlib.pyplot as plt

    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/histories", exist_ok=True)

    histories = {
        "image_only": img_history,
        "symptom_only": sym_history,
        "fusion": fusion_history
    }

    def save_history_and_plots(history, model_name):
        history_path = f"outputs/histories/{model_name}_history.json"

        with open(history_path, "w") as f:
            json.dump(
                {
                    key: [float(value) for value in values]
                    for key, values in history.history.items()
                },
                f,
                indent=2
            )

        plt.figure(figsize=(8, 5))
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{model_name.replace('_', ' ').title()} - Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"outputs/plots/{model_name}_accuracy.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{model_name.replace('_', ' ').title()} - Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"outputs/plots/{model_name}_loss.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        print(f"Saved history and plots for {model_name}")

    for model_name, history in histories.items():
        save_history_and_plots(history, model_name)

    print("\nTraining histories saved to outputs/histories/")
    print("Training plots saved to outputs/plots/")

    # Save models
    os.makedirs("outputs/models", exist_ok=True)
    img_model.save("outputs/models/image_only.keras")
    sym_model.save("outputs/models/symptom_only.keras")
    fusion_model.save("outputs/models/fusion.keras")
    print("\nModels saved to outputs/models/")

    # --- NEW: automatic per-sample prediction export ---
    # This is what's needed for McNemar's paired test and paired
    # bootstrap confidence intervals comparing the three models.
    results_df = pd.DataFrame({
        "paired_index": test_idx,
        "true_label": le.inverse_transform(y_test),
        "image_only_pred": le.inverse_transform(img_pred),
        "symptom_only_pred": le.inverse_transform(sym_pred),
        "fusion_pred": le.inverse_transform(fusion_pred),
        "image_only_correct": (img_pred == y_test).astype(int),
        "symptom_only_correct": (sym_pred == y_test).astype(int),
        "fusion_correct": (fusion_pred == y_test).astype(int),
    })
    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/per_sample_predictions.csv", index=False)

    # Save confusion matrices for all three models.
    predictions = {
        "image_only": img_pred,
        "symptom_only": sym_pred,
        "fusion": fusion_pred
    }
    for model_name, predictions_for_model in predictions.items():
        cm = confusion_matrix(y_test, predictions_for_model,
                              labels=np.arange(num_classes))
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        cm_df.to_csv(f"outputs/{model_name}_confusion_matrix.csv")
        print(f"Saved confusion matrix: outputs/{model_name}_confusion_matrix.csv")

    print("\nPer-sample predictions saved to outputs/per_sample_predictions.csv")
    print(results_df.head())


if __name__ == "__main__":
    main()
