from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import pickle
import json
from tensorflow import keras
from scipy.signal import butter, filtfilt, find_peaks
from collections import Counter

app = app = Flask(__name__, 
            static_folder='static', 
            static_url_path='/static',
            template_folder='templates')

MODEL_PATH = "models"

# ================================
# LOAD MODEL ARTEFACTS
# ================================
print("🔄 Loading model...")

def build_model():
    from tensorflow.keras import layers, regularizers

    weights_path = os.path.join(MODEL_PATH, "ecg_weights.weights.h5")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    reg = regularizers.L2(1e-4)

    inputs = keras.Input(shape=(180, 1), name="ecg_input")

    # Block 1
    x = layers.Conv1D(64, 5, padding='same', kernel_regularizer=reg, name='conv1d')(inputs)
    x = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name='batch_normalization')(x)
    x = layers.Activation('relu', name='activation')(x)
    x = layers.Conv1D(64, 5, padding='same', kernel_regularizer=reg, name='conv1d_1')(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name='batch_normalization_1')(x)
    x = layers.Activation('relu', name='activation_1')(x)
    x = layers.MaxPooling1D(2, name='max_pooling1d')(x)
    x = layers.Dropout(0.2, name='dropout')(x)

    # Block 2
    x = layers.Conv1D(128, 3, padding='same', kernel_regularizer=reg, name='conv1d_2')(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name='batch_normalization_2')(x)
    x = layers.Activation('relu', name='activation_2')(x)
    x = layers.Conv1D(128, 3, padding='same', kernel_regularizer=reg, name='conv1d_3')(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name='batch_normalization_3')(x)
    x = layers.Activation('relu', name='activation_3')(x)
    x = layers.MaxPooling1D(2, name='max_pooling1d_1')(x)
    x = layers.Dropout(0.2, name='dropout_1')(x)

    # BiLSTM Block
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, kernel_regularizer=reg),
        name='bidirectional'
    )(x)
    x = layers.Dropout(0.4, name='dropout_2')(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, kernel_regularizer=reg),
        name='bidirectional_1'
    )(x)
    x = layers.Dropout(0.4, name='dropout_3')(x)

    # Dense Block
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg, name='dense')(x)
    x = layers.Dropout(0.4, name='dropout_4')(x)
    x = layers.Dense(64, activation='relu', name='dense_1')(x)
    outputs = layers.Dense(4, activation='softmax', name='output')(x)

    m = keras.Model(inputs, outputs)
    m.load_weights(weights_path)
    print("✅ Model built from code + weights loaded successfully")
    return m

model = build_model()

with open(os.path.join(MODEL_PATH, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_PATH, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

CLASS_LABELS = list(label_encoder.classes_)

print(f"✅ Model loaded")
print(f"📋 Class order: {CLASS_LABELS}")

# ================================
# SIGNAL PROCESSING
# ================================

def bandpass_filter(signal, fs=360):
    nyq  = 0.5 * fs
    low  = 0.5 / nyq
    high = 40.0 / nyq
    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, signal)


def extract_beats(signal, fs=360):
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    signal = bandpass_filter(signal, fs=fs)

    peaks, _ = find_peaks(
        signal,
        distance=int(0.35 * fs),
        prominence=0.3,
        height=0.0,
    )

    beats = []
    valid_peaks = []

    for p in peaks:
        if 90 <= p < len(signal) - 90:
            beat = signal[p - 90: p + 90]
            if len(beat) == 180:
                beats.append(beat)
                valid_peaks.append(p)

    return beats, valid_peaks


# ================================
# PREDICTION
# ================================

def predict_beat(beat):
    beat_scaled = scaler.transform(beat.reshape(1, -1))
    X = beat_scaled.reshape(1, 180, 1)

    probs = model.predict(X, verbose=0)[0]

    sorted_idx = np.argsort(probs)[::-1]

    top_idx    = sorted_idx[0]
    second_idx = sorted_idx[1]

    top_label  = CLASS_LABELS[top_idx]
    top_conf   = float(probs[top_idx])
    second_conf = float(probs[second_idx])

    # Confidence + margin check
    if top_conf < 0.65 or (top_conf - second_conf) < 0.15:
        return "Q", top_conf

    return top_label, top_conf


def aggregate_predictions(results):
    counts = Counter(r["prediction"] for r in results)
    total  = len(results)

    v_count = counts.get("V", 0)
    s_count = counts.get("S", 0)
    f_count = counts.get("F", 0)
    q_count = counts.get("Q", 0)

    v_burden = v_count / total if total else 0
    s_burden = s_count / total if total else 0

    if v_burden >= 0.10:
        status = "Abnormal — Ventricular ectopy (PVC burden ≥ 10%)"
    elif v_count >= 2:
        status = "Borderline — Isolated PVCs detected"
    elif s_burden >= 0.10:
        status = "Abnormal — Supraventricular ectopy"
    elif s_count >= 3:
        status = "Borderline — Isolated SVEs detected"
    elif f_count >= 3:
        status = "Borderline — Possible Fusion beats"
    elif total > 0 and q_count / total > 0.30:
        status = "Inconclusive — Poor signal quality"
    else:
        status = "Normal Sinus Rhythm"

    non_q    = [r["confidence"] for r in results if r["prediction"] != "Q"]
    avg_conf = float(np.mean(non_q)) if non_q else 0.0

    return {
        "total_beats":    total,
        "class_counts":   dict(counts),
        "status":         status,
        "avg_confidence": round(avg_conf, 3),
        "signal_length":  total,
        "note":           "Automated screening only. Confirm with a cardiologist.",
    }


# ================================
# ROUTES
# ================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "classes": CLASS_LABELS
    })

# Replace predict_beat function with this batch version
# Instead of calling model.predict() 30 times (very slow/heavy),
# run ONE batch prediction

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = "/tmp/temp_ecg.csv"
    file.save(filepath)

    try:
        print("📂 Loading signal...")

        # Try column 1 first, fall back to column 0
        try:
            signal = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(1))
        except Exception:
            signal = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(0))

        print(f"📊 Signal: {len(signal)} samples")

        if len(signal) < 360:
            return jsonify({"error": "ECG signal too short"}), 400

        beats, peaks = extract_beats(signal[:7200])
        print(f"💓 Beats found: {len(beats)}")

        if len(beats) == 0:
            return jsonify({"error": "No valid beats detected"}), 400

        beats_to_use = beats[:30]
        peaks_to_use = peaks[:30]

        # ── BATCH predict (1 model call instead of 30) ──────
        import gc
        beat_array = np.array(beats_to_use)                      # (N, 180)
        beat_scaled = scaler.transform(beat_array)                # (N, 180)
        X = beat_scaled.reshape(len(beat_scaled), 180, 1)        # (N, 180, 1)

        print("🧠 Running batch prediction...")
        all_probs = model.predict(X, verbose=0, batch_size=8)    # (N, 4)
        print("✅ Prediction done")
        gc.collect()

        results = []
        for i, probs in enumerate(all_probs):
            sorted_idx   = np.argsort(probs)[::-1]
            top_idx      = sorted_idx[0]
            second_idx   = sorted_idx[1]
            top_conf     = float(probs[top_idx])
            second_conf  = float(probs[second_idx])
            top_label    = CLASS_LABELS[top_idx]

            if top_conf < 0.65 or (top_conf - second_conf) < 0.15:
                label = "Q"
            else:
                label = top_label

            results.append({
                "prediction":    label,
                "confidence":    round(top_conf, 4),
                "sample_index":  int(peaks_to_use[i]),
                "probabilities": {
                    CLASS_LABELS[j]: round(float(probs[j]), 4)
                    for j in range(len(CLASS_LABELS))
                }
            })

        summary = aggregate_predictions(results)
        print("📤 Sending response")

        return jsonify({
            "summary":     summary,
            "predictions": results,
            "signal":      [float(x) for x in signal[:7200]],
            "peaks":       [int(p) for p in peaks_to_use],
            "beats":       [b.tolist() for b in beats_to_use]
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("❌ ERROR:\n", tb)
        return jsonify({"error": str(e), "traceback": tb}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/debug/classes", methods=["GET"])
def debug_classes():
    return jsonify({"label_encoder_classes": CLASS_LABELS})



# ================================
# RUN
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
