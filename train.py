# =============================================================================
# ECG Arrhythmia Detection — Full Retraining Pipeline
# Dataset : MIT-BIH Arrhythmia Database (via wfdb)
# Model   : CNN-LSTM with proper class balancing
# Classes : N (Normal), S (Supraventricular), V (Ventricular),
#           F (Fusion), Q (Unknown/Unclassifiable)
# =============================================================================
# Install dependencies first:
#   pip install wfdb tensorflow scikit-learn imbalanced-learn matplotlib seaborn
# =============================================================================

import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb
from collections import Counter

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, TensorBoard)

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# CONFIGURATION  — edit these paths / hyperparameters as needed
# =============================================================================
CFG = {
    # Where wfdb will download / read MIT-BIH records
    "data_dir"    : "C:\\Users\\Arjun\\OneDrive\\Desktop\\ECG\\ecg-app\\mitbih_data",

    # Where to save model artefacts
    "model_dir"   : "./models",

    # MIT-BIH record numbers (all 48 records)
    "records"     : [
        100, 101, 102, 103, 104, 105, 106, 107,
        108, 109, 111, 112, 113, 114, 115, 116,
        117, 118, 119, 121, 122, 123, 124, 200,
        201, 202, 203, 205, 207, 208, 209, 210,
        212, 213, 214, 215, 217, 219, 220, 221,
        222, 223, 228, 230, 231, 232, 233, 234,
    ],

    # Sampling frequency of MIT-BIH
    "fs"          : 360,

    # Beat window: 90 samples before R-peak, 90 after → 180 total
    "beat_before" : 90,
    "beat_after"  : 90,

    # AAMI standard label mapping
    # N  → Normal + all non-ectopic variations
    # S  → Supraventricular ectopic (PAC, etc.)
    # V  → Ventricular ectopic (PVC, etc.)
    # F  → Fusion beat
    # Q  → Unknown / paced / unclassifiable
    "label_map"   : {
        # Normal / non-ectopic
        "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
        # Supraventricular ectopic
        "A": "S", "a": "S", "J": "S", "S": "S",
        # Ventricular ectopic
        "V": "V", "E": "V",
        # Fusion
        "F": "F",
        # Unknown / paced
        "Q": "Q", "/": "Q", "f": "Q", "!": "Q",
    },

    # Training hyperparameters
    "test_size"   : 0.20,
    "val_size"    : 0.10,   # fraction of training set used for validation
    "batch_size"  : 128,
    "epochs"      : 60,
    "learning_rate": 1e-3,

    # SMOTE: oversample minority classes to this minimum count
    # Set to None to skip SMOTE and rely only on class_weight
    "smote_min_count": None,

    # Dropout rate
    "dropout"     : 0.40,
}

os.makedirs(CFG["data_dir"], exist_ok=True)
os.makedirs(CFG["model_dir"], exist_ok=True)

# =============================================================================
# STEP 1 — DATA EXTRACTION
# =============================================================================

def load_mitbih_beats(cfg):
    """
    Download (if needed) and extract labelled beat segments from all
    MIT-BIH records.  Returns X (n_beats, 180) and y (n_beats,) string labels.
    """
    print("\n" + "="*60)
    print("STEP 1 — Extracting beats from MIT-BIH database")
    print("="*60)

    X_all, y_all = [], []
    before = cfg["beat_before"]
    after  = cfg["beat_after"]
    win    = before + after

    for rec_num in cfg["records"]:
        rec_path = os.path.join(cfg["data_dir"], str(rec_num))
        try:
            # Download from PhysioNet if not already present
            if not os.path.exists(rec_path + ".hea"):
                print(f"  Downloading record {rec_num}...")
                wfdb.dl_database("mitdb", dl_dir=cfg["data_dir"],
                                 records=[str(rec_num)])

            record = wfdb.rdrecord(rec_path)
            ann    = wfdb.rdann(rec_path, "atr")

            # Use channel 0 (MLII) — most records have it as the primary lead
            signal = record.p_signal[:, 0].astype(np.float32)

            # Normalise per-record (zero-mean, unit-variance)
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

            r_peaks  = ann.sample
            sym_list = ann.symbol

            beats_in_record = 0
            for peak, sym in zip(r_peaks, sym_list):
                aami_label = cfg["label_map"].get(sym, None)
                if aami_label is None or aami_label == "Q":
                    continue   # skip symbols not in our mapping
                if peak < before or peak + after > len(signal):
                    continue   # skip beats too close to record edges

                beat = signal[peak - before: peak + after]
                if len(beat) != win:
                    continue

                X_all.append(beat)
                y_all.append(aami_label)
                beats_in_record += 1

            print(f"  Record {rec_num}: {beats_in_record} beats extracted")

        except Exception as exc:
            print(f"  ⚠️  Record {rec_num} failed: {exc}")

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all)

    print(f"\n✅ Total beats extracted: {len(X)}")
    print(f"   Class distribution (raw):")
    for cls, cnt in sorted(Counter(y).items()):
        pct = 100 * cnt / len(y)
        print(f"     {cls}: {cnt:6d}  ({pct:.1f}%)")

    return X, y


# =============================================================================
# STEP 2 — PREPROCESSING & BALANCING
# =============================================================================

def preprocess_and_balance(X, y, cfg):
    """
    1. Fit StandardScaler on training data only (no data leakage).
    2. Encode labels.
    3. Split into train / val / test.
    4. Apply SMOTE to training set only.
    5. Compute class weights for the loss function.
    """
    print("\n" + "="*60)
    print("STEP 2 — Preprocessing, splitting, balancing")
    print("="*60)

    # --- Label encoding ---
    le = LabelEncoder()
    le.fit(sorted(set(y)))   # sorted → deterministic alphabetical order
    y_enc = le.transform(y)
    print(f"\nLabel encoder order: {list(le.classes_)}")
    print("⚠️  IMPORTANT: Save this order — it must match inference code exactly.")

    # --- Train / test split (stratified to preserve class ratios) ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc,
        test_size=cfg["test_size"],
        random_state=42,
        stratify=y_enc
    )

    # --- Train / val split ---
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr,
        test_size=cfg["val_size"],
        random_state=42,
        stratify=y_tr
    )

    # --- Fit scaler on training data ONLY ---
    scaler = StandardScaler()
    X_tr_scaled  = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_te_scaled  = scaler.transform(X_te)

    print(f"\nSplit sizes:")
    print(f"  Train : {len(X_tr_scaled)}")
    print(f"  Val   : {len(X_val_scaled)}")
    print(f"  Test  : {len(X_te_scaled)}")

    # --- SMOTE oversampling on training set only ---
    if cfg["smote_min_count"] is not None:
        print(f"\nApplying SMOTE (min samples per class → {cfg['smote_min_count']})...")
        counts = Counter(y_tr)
        strategy = {
            cls: max(cnt, cfg["smote_min_count"])
            for cls, cnt in counts.items()
        }
        # k_neighbors must be < smallest class count
        min_count = min(counts.values())
        k = min(5, min_count - 1) if min_count > 1 else 1
        sm = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=42)
        X_tr_scaled, y_tr = sm.fit_resample(X_tr_scaled, y_tr)
        print(f"  After SMOTE — class distribution:")
        for cls_idx, cnt in sorted(Counter(y_tr).items()):
            lbl = le.inverse_transform([cls_idx])[0]
            print(f"    {lbl}: {cnt}")

    # --- Class weights (applied to loss; complements SMOTE) ---
    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    print(f"\nClass weights for loss function:")
    for cls_idx, w in class_weight_dict.items():
        lbl = le.inverse_transform([cls_idx])[0]
        print(f"  {lbl}: {w:.3f}")

    # Reshape for CNN-LSTM input: (n_samples, timesteps, channels)
    X_tr_3d  = X_tr_scaled.reshape(-1, 180, 1)
    X_val_3d = X_val_scaled.reshape(-1, 180, 1)
    X_te_3d  = X_te_scaled.reshape(-1, 180, 1)

    n_classes = len(le.classes_)

    return (X_tr_3d, y_tr,
            X_val_3d, y_val,
            X_te_3d, y_te,
            scaler, le,
            class_weight_dict, n_classes)


# =============================================================================
# STEP 3 — MODEL DEFINITION
# =============================================================================

def build_model(n_classes, cfg):
    """
    CNN-LSTM with:
    - Two CNN blocks (feature extraction from raw beat waveform)
    - BatchNorm + Dropout for regularisation
    - Bidirectional LSTM (captures temporal context in both directions)
    - Dense head with softmax output
    """
    inp = keras.Input(shape=(180, 1), name="ecg_input")

    # --- CNN Block 1 ---
    x = layers.Conv1D(64, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(64, kernel_size=5, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(cfg["dropout"] * 0.5)(x)

    # --- CNN Block 2 ---
    x = layers.Conv1D(128, kernel_size=3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(128, kernel_size=3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(cfg["dropout"] * 0.5)(x)

    # --- Bidirectional LSTM ---
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True,
                    kernel_regularizer=regularizers.l2(1e-4))
    )(x)
    x = layers.Dropout(cfg["dropout"])(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False,
                    kernel_regularizer=regularizers.l2(1e-4))
    )(x)
    x = layers.Dropout(cfg["dropout"])(x)

    # --- Dense head ---
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(cfg["dropout"])(x)
    x = layers.Dense(64, activation="relu")(x)

    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


# =============================================================================
# STEP 4 — TRAINING
# =============================================================================

def train_model(model, X_tr, y_tr, X_val, y_val, class_weight_dict, cfg):
    print("\n" + "="*60)
    print("STEP 4 — Training")
    print("="*60)

    ckpt_path = os.path.join(cfg["model_dir"], "best_model.h5")

    callbacks = [
        # Stop if val_loss doesn't improve for 10 epochs
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),

        # Halve LR if val_loss plateaus for 5 epochs
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1),

        # Save best checkpoint
        ModelCheckpoint(ckpt_path, monitor="val_loss",
                        save_best_only=True, verbose=1),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        class_weight=class_weight_dict,   # ← key: weighted loss function
        callbacks=callbacks,
        verbose=1,
    )

    return history


# =============================================================================
# STEP 5 — EVALUATION & PLOTS
# =============================================================================

def evaluate_and_plot(model, X_te, y_te, le, history, cfg):
    print("\n" + "="*60)
    print("STEP 5 — Evaluation on held-out test set")
    print("="*60)

    y_pred_probs = model.predict(X_te, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    class_names = list(le.classes_)

    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=class_names))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_te, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix (counts)")

    # Row-normalised (recall per class)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title("Confusion Matrix (row-normalised recall)")
    axes[1].set_ylabel("True label")
    axes[1].set_xlabel("Predicted label")

    plt.tight_layout()
    plt.savefig(os.path.join(cfg["model_dir"], "confusion_matrix.png"), dpi=150)
    plt.close()

    # --- Training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, metric, title in zip(
        axes,
        [("loss", "val_loss"), ("accuracy", "val_accuracy")],
        ["Loss", "Accuracy"]
    ):
        ax.plot(history.history[metric[0]],  label="Train")
        ax.plot(history.history[metric[1]],  label="Val")
        ax.set_title(f"Training {title}")
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["model_dir"], "training_curves.png"), dpi=150)
    plt.close()

    print(f"\nPlots saved to: {cfg['model_dir']}/")

    # Per-class F score summary (quick sanity check for F-class)
    from sklearn.metrics import f1_score
    f1s = f1_score(y_te, y_pred, average=None, labels=list(range(len(class_names))))
    print("\nPer-class F1 score:")
    for cls, score in zip(class_names, f1s):
        bar = "█" * int(score * 30)
        print(f"  {cls}: {score:.3f}  {bar}")


# =============================================================================
# STEP 6 — SAVE ARTEFACTS
# =============================================================================

def save_artefacts(model, scaler, le, cfg):
    print("\n" + "="*60)
    print("STEP 6 — Saving model artefacts")
    print("="*60)

    # Save final model (best weights already restored by EarlyStopping)
    final_model_path = os.path.join(cfg["model_dir"], "cnn_lstm_model.h5")
    model.save(final_model_path)

    scaler_path = os.path.join(cfg["model_dir"], "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    le_path = os.path.join(cfg["model_dir"], "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    # Save label order as plain text for easy inspection
    order_path = os.path.join(cfg["model_dir"], "class_order.txt")
    with open(order_path, "w") as f:
        for i, cls in enumerate(le.classes_):
            f.write(f"{i}: {cls}\n")

    print(f"\n✅ Saved:")
    print(f"   Model   → {final_model_path}")
    print(f"   Scaler  → {scaler_path}")
    print(f"   Encoder → {le_path}")
    print(f"   Classes → {order_path}")
    print(f"\n   Label order (copy this into app.py if needed):")
    for i, cls in enumerate(le.classes_):
        print(f"     index {i} → class '{cls}'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # 1. Extract beats
    X, y = load_mitbih_beats(CFG)

    # 2. Preprocess & balance
    (X_tr, y_tr,
     X_val, y_val,
     X_te, y_te,
     scaler, le,
     class_weight_dict, n_classes) = preprocess_and_balance(X, y, CFG)

    # 3. Build model
    print("\n" + "="*60)
    print("STEP 3 — Building model")
    print("="*60)
    model = build_model(n_classes, CFG)

    # 4. Train
    history = train_model(model, X_tr, y_tr, X_val, y_val,
                          class_weight_dict, CFG)

    # 5. Evaluate
    evaluate_and_plot(model, X_te, y_te, le, history, CFG)

    # 6. Save
    save_artefacts(model, scaler, le, CFG)

    print("\n🎉 Done. Copy the ./models/ folder next to your app.py and restart Flask.")