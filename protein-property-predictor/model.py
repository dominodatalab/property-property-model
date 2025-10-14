# model.py — inference only (used by app.py)
#
# Loads model.joblib (trained via train.py) and provides predict() for Streamlit app.
# No training logic here.

import os
import joblib
import numpy as np
import pandas as pd

# ---------- Feature functions (same as in train.py) ----------
HYDRO = set("AILMFWVY")

def _clean(s):
    return (s or "").upper().replace(" ", "").replace("\n", "").replace("\r", "")

def _hyd_frac(s):
    s = _clean(s)
    return sum(1 for c in s if c in HYDRO) / max(len(s), 1)

def _nterm_hyd_frac(s, w=20):
    s = _clean(s)
    return _hyd_frac(s[:w])

def featurize(seq):
    seq = _clean(seq)
    return np.array([_hyd_frac(seq), _nterm_hyd_frac(seq), float(len(seq))]).reshape(1, -1)


# ---------- Resolve model path ----------
DATASET_DIR = os.getenv("DATASET_DIR", "").strip()
HERE = os.path.dirname(__file__)

if DATASET_DIR:
    MODEL_PATH = os.path.join(DATASET_DIR, "models", "latest", "model.joblib")
else:
    MODEL_PATH = os.path.join(HERE, "models", "latest", "model.joblib")

# ---------- Load trained model ----------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Model file not found at {MODEL_PATH}. "
        "Please run train.py first to generate model.joblib."
    )

MODEL = joblib.load(MODEL_PATH)


# ---------- Prediction ----------
def predict(seq: str, mode: str = "auto") -> dict:
    """
    Predicts protein property from amino acid sequence.

    Args:
        seq (str): Protein sequence string
        mode (str): One of {"auto", "ml", "rule"}

    Returns:
        dict: {"prediction": ..., "probability": ..., "mode": ...}
    """
    seq = _clean(seq)
    if not seq:
        return {"error": "Empty sequence."}

    # Rule-based fallback mode
    if mode == "rule":
        hyd = _hyd_frac(seq)
        label = 1 if hyd > 0.45 else 0
        return {"prediction": label, "probability": hyd, "mode": "rule-based"}

    # ML-based prediction
    X = featurize(seq)
    try:
        prob = MODEL.predict_proba(X)[0, 1]
        pred = int(prob >= 0.5)
    except Exception as e:
        return {"error": f"Model prediction failed: {str(e)}"}

    return {"prediction": pred, "probability": float(prob), "mode": "ml" if mode == "ml" else "auto"}
