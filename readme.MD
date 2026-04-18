# CardioLens — ECG Dashboard Integration Guide

## Project Structure

```
your_flask_project/
├── app.py                        ← Flask backend (merge with your existing app)
├── templates/
│   └── index.html                ← Dashboard HTML
└── static/
    ├── css/
    │   └── dashboard.css         ← Styles
    └── js/
        └── dashboard.js          ← Frontend logic
```

---

## 1. Copy Files

```bash
cp templates/index.html   your_project/templates/
cp static/css/dashboard.css   your_project/static/css/
cp static/js/dashboard.js    your_project/static/js/
```

---

## 2. Merge Flask Routes

Add these two routes to your existing `app.py`:

```python
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # ... see app.py for full implementation
```

---

## 3. Plug In Your Model

In `app.py`, inside `classify_beats()`, replace the stub section:

```python
# REPLACE THIS:
raw = np.random.dirichlet(...)   # stub

# WITH YOUR MODEL:
x = np.expand_dims(beat, axis=(0, -1))   # shape (1, 180, 1)
probs_arr = model.predict(x, verbose=0)[0]   # shape (5,) → [N, S, V, F, Q]
```

---

## 4. JSON Response Format

The frontend expects exactly this shape from `POST /predict`:

```json
{
  "signal":      [0.12, 0.13, ...],
  "peaks":       [150, 450, 750, ...],
  "beats":       [[0.1, 0.2, ...], ...],
  "predictions": [
    {
      "prediction":   "N",
      "confidence":   0.97,
      "sample_index": 150,
      "probabilities": {"N": 0.97, "S": 0.01, "V": 0.01, "F": 0.005, "Q": 0.005}
    }
  ],
  "summary": {
    "total_beats":    42,
    "class_counts":   {"N": 38, "V": 2, "S": 2, "F": 0, "Q": 0},
    "avg_confidence": 0.91,
    "signal_length":  10800
  }
}
```

---

## 5. Test Without Backend (UI Dev Mode)

In `dashboard.js`, uncomment the mock block at the bottom:

```js
(function mockAPIForDev() { ... })();
```

This injects a fake response so you can develop/test the UI without Flask running.

---

## 6. Install Dependencies

```bash
pip install flask numpy scipy
```

---

## 7. Run

```bash
python app.py
# Open http://localhost:5000
```

---

## Features Implemented

| Feature | Status |
|---|---|
| Drag-and-drop CSV upload | ✅ |
| Full ECG waveform (Chart.js) | ✅ |
| R-peak markers, color-coded by class | ✅ |
| Beat zoom inspector | ✅ |
| All-class probability bars | ✅ |
| Clinical explanation text per beat | ✅ |
| Beat-by-beat navigation (prev/next) | ✅ |
| Class filter on beat table | ✅ |
| Summary dashboard + pie chart | ✅ |
| Status badge (Normal/Abnormal/etc.) | ✅ |
| Dark mode toggle | ✅ |
| PDF report export | ✅ |
| Medical disclaimer banner | ✅ |
| Responsive (mobile + desktop) | ✅ |
| Toast notifications | ✅ |

---

## Notes

- **Sample rate**: defaults to 360 Hz (MIT-BIH). Change `SAMPLE_RATE` in both `app.py` and `dashboard.js` if your data differs.
- **CSV format**: the parser expects MIT-BIH export CSVs where signal is in column index 1. Adjust `extract_beats()` for your format.
- **Confidence threshold**: `CONFIDENCE_THRESHOLD = 0.65` in both files controls when a beat is marked "Q".