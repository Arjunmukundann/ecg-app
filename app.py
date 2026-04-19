from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import pickle
import json
from tensorflow import keras
from scipy.signal import butter, filtfilt, find_peaks
from collections import Counter

app = Flask(__name__)

MODEL_PATH = "models"

# ================================
# LOAD MODEL ARTEFACTS
# ================================
print("🔄 Loading model...")

def build_model():
    config_path  = os.path.join(MODEL_PATH, "model_config.json")
    weights_path = os.path.join(MODEL_PATH, "ecg_weights.weights.h5")
    h5_path      = os.path.join(MODEL_PATH, "cnn_lstm_model.h5")

    # Option 1: config + weights (most compatible)
    if os.path.exists(config_path) and os.path.exists(weights_path):
        print("📦 Loading from config + weights...")
        with open(config_path, "r") as f:
            config = json.load(f)
        m = keras.Model.from_config(config)
        m.load_weights(weights_path)
        return m

    # Option 2: fallback to .h5
    if os.path.exists(h5_path):
        print("📦 Loading from .h5...")
        return keras.models.load_model(h5_path, compile=False)

    raise FileNotFoundError("No model file found in models/ directory!")

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


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file     = request.files["file"]
    filepath = "temp_ecg.csv"
    file.save(filepath)

    try:
        signal = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=(1))

        if len(signal) < 360:
            return jsonify({"error": "ECG signal too short"}), 400

        beats, peaks = extract_beats(signal[:7200])

        if len(beats) == 0:
            return jsonify({"error": "No valid beats detected"}), 400

        results = []
        for i, beat in enumerate(beats[:30]):
            label, conf = predict_beat(beat)
            results.append({
                "prediction":   label,
                "confidence":   round(conf, 4),
                "sample_index": int(peaks[i])
            })

        summary = aggregate_predictions(results)

        return jsonify({
            "summary":     summary,
            "predictions": results,
            "signal":      [float(x) for x in signal],
            "peaks":       [int(p) for p in peaks[:30]],
            "beats":       [b.tolist() for b in beats[:30]]
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error":     str(e),
            "traceback": traceback.format_exc()
        }), 500

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
