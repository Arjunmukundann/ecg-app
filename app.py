from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import pickle
import json
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from scipy.signal import butter, filtfilt, find_peaks
from collections import Counter

app = Flask(__name__, template_folder='templates')

MODEL_PATH = "models"

# ── Lazy globals ─────────────────────────────────────────────
_model        = None
_scaler       = None
_label_encoder = None
CLASS_LABELS  = None

def get_model():
    global _model, _scaler, _label_encoder, CLASS_LABELS
    if _model is not None:
        return _model, _scaler, CLASS_LABELS

    print("🔄 Lazy-loading model...")

    reg = regularizers.L2(1e-4)
    inputs = keras.Input(shape=(180, 1), name="ecg_input")
    x = layers.Conv1D(64, 5, padding='same', kernel_regularizer=reg, name='conv1d')(inputs)
    x = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name='batch_normalization')(x)
    x = layers.Activation('relu', name='activation')(x)
    x = layers.Conv1D(64, 5, padding='same', kernel_regularizer=reg, name='conv1d_1')(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name='batch_normalization_1')(x)
    x = layers.Activation('relu', name='activation_1')(x)
    x = layers.MaxPooling1D(2, name='max_pooling1d')(x)
    x = layers.Dropout(0.2, name='dropout')(x)
    x = layers.Conv1D(128, 3, padding='same', kernel_regularizer=reg, name='conv1d_2')(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name='batch_normalization_2')(x)
    x = layers.Activation('relu', name='activation_2')(x)
    x = layers.Conv1D(128, 3, padding='same', kernel_regularizer=reg, name='conv1d_3')(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name='batch_normalization_3')(x)
    x = layers.Activation('relu', name='activation_3')(x)
    x = layers.MaxPooling1D(2, name='max_pooling1d_1')(x)
    x = layers.Dropout(0.2, name='dropout_1')(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, kernel_regularizer=reg), name='bidirectional')(x)
    x = layers.Dropout(0.4, name='dropout_2')(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, kernel_regularizer=reg), name='bidirectional_1')(x)
    x = layers.Dropout(0.4, name='dropout_3')(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg, name='dense')(x)
    x = layers.Dropout(0.4, name='dropout_4')(x)
    x = layers.Dense(64, activation='relu', name='dense_1')(x)
    outputs = layers.Dense(4, activation='softmax', name='output')(x)
    _model = keras.Model(inputs, outputs)
    _model.load_weights(os.path.join(MODEL_PATH, "ecg_weights.weights.h5"))

    with open(os.path.join(MODEL_PATH, "scaler.pkl"), "rb") as f:
        _scaler = pickle.load(f)
    with open(os.path.join(MODEL_PATH, "label_encoder.pkl"), "rb") as f:
        _label_encoder = pickle.load(f)

    CLASS_LABELS = list(_label_encoder.classes_)
    print(f"✅ Model loaded. Classes: {CLASS_LABELS}")
    return _model, _scaler, CLASS_LABELS


# ── Signal processing ─────────────────────────────────────────
def bandpass_filter(signal, fs=360):
    nyq  = 0.5 * fs
    b, a = butter(5, [0.5/nyq, 40.0/nyq], btype='band')
    return filtfilt(b, a, signal)

def extract_beats(signal, fs=360):
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    signal = bandpass_filter(signal, fs)
    peaks, _ = find_peaks(signal, distance=int(0.35*fs), prominence=0.3, height=0.0)
    beats, valid_peaks = [], []
    for p in peaks:
        if 90 <= p < len(signal) - 90:
            beat = signal[p-90:p+90]
            if len(beat) == 180:
                beats.append(beat)
                valid_peaks.append(p)
    return beats, valid_peaks

def aggregate_predictions(results):
    from collections import Counter
    counts = Counter(r["prediction"] for r in results)
    total  = len(results)
    v_burden = counts.get("V", 0) / total if total else 0
    s_burden = counts.get("S", 0) / total if total else 0
    if v_burden >= 0.10:       status = "Abnormal — Ventricular ectopy (PVC burden ≥ 10%)"
    elif counts.get("V",0)>=2: status = "Borderline — Isolated PVCs detected"
    elif s_burden >= 0.10:     status = "Abnormal — Supraventricular ectopy"
    elif counts.get("S",0)>=3: status = "Borderline — Isolated SVEs detected"
    elif counts.get("F",0)>=3: status = "Borderline — Possible Fusion beats"
    elif total > 0 and counts.get("Q",0)/total > 0.30: status = "Inconclusive — Poor signal quality"
    else:                      status = "Normal Sinus Rhythm"
    non_q    = [r["confidence"] for r in results if r["prediction"] != "Q"]
    avg_conf = float(np.mean(non_q)) if non_q else 0.0
    return {"total_beats": total, "class_counts": dict(counts),
            "status": status, "avg_confidence": round(avg_conf, 3),
            "signal_length": total, "note": "Automated screening only. Confirm with a cardiologist."}


# ── Routes ────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    model, scaler, class_labels = get_model()  # lazy load here

    file     = request.files["file"]
    filepath = "/tmp/temp_ecg.csv"
    file.save(filepath)

    try:
        print("📂 Step 1: File saved")
        try:
            signal = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(1))
        except Exception:
            try:
                signal = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(0))
            except Exception:
                import pandas as pd
                df = pd.read_csv(filepath)
                signal = df.iloc[:, 1].values if df.shape[1] > 1 else df.iloc[:, 0].values

        signal = np.array(signal, dtype=np.float32)
        print(f"📊 Step 2: {len(signal)} samples loaded")

        if len(signal) < 360:
            return jsonify({"error": f"Signal too short: {len(signal)} samples"}), 400

        signal_trimmed = signal[:7200]
        beats, peaks   = extract_beats(signal_trimmed)
        print(f"💓 Step 3: {len(beats)} beats found")

        if len(beats) == 0:
            return jsonify({"error": "No beats detected — check signal format"}), 400

        beats_to_use = beats[:20]
        peaks_to_use = peaks[:20]

        beat_array  = np.array(beats_to_use, dtype=np.float32)
        beat_scaled = scaler.transform(beat_array)
        X           = beat_scaled.reshape(-1, 180, 1)

        print("🧠 Step 4: Predicting...")
        all_probs = model.predict(X, verbose=0, batch_size=4)
        print("✅ Step 5: Done")

        import gc; gc.collect()

        results = []
        for i, probs in enumerate(all_probs):
            sorted_idx  = np.argsort(probs)[::-1]
            top_idx     = int(sorted_idx[0])
            top_conf    = float(probs[top_idx])
            second_conf = float(probs[int(sorted_idx[1])])
            top_label   = class_labels[top_idx]
            label = "Q" if (top_conf < 0.65 or top_conf - second_conf < 0.15) else top_label
            results.append({
                "prediction":    label,
                "confidence":    round(top_conf, 4),
                "sample_index":  int(peaks_to_use[i]),
                "probabilities": {class_labels[j]: round(float(probs[j]), 4) for j in range(len(class_labels))}
            })

        summary = aggregate_predictions(results)
        print("📤 Step 6: Sending response")

        return jsonify({
            "summary":     summary,
            "predictions": results,
            "signal":      [float(x) for x in signal_trimmed],
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
