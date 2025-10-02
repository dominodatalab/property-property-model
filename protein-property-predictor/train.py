# train.py — minimal logistic regression trainer
# Reads data/train.csv (id,sequence,label), featurizes, trains, saves models/latest/model.joblib
# Usage:
#   python train.py
#   # Domino with Dataset mounted:
#   # export DATASET_DIR=/domino/datasets/<plane>/protein-property-predictor[-na]
#   # python train.py

import os, json, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

# Prefer Domino Dataset if provided; else use local repo folders
DATASET_DIR = os.getenv("DATASET_DIR", "").strip()
if DATASET_DIR:
    DATA_CSV   = os.path.join(DATASET_DIR, "data", "train.csv")
    MODELS_DIR = os.path.join(DATASET_DIR, "models")
else:
    HERE       = os.path.dirname(__file__)
    DATA_CSV   = os.path.join(HERE, "data", "train.csv")
    MODELS_DIR = os.path.join(HERE, "models")

LATEST_DIR = os.path.join(MODELS_DIR, "latest")
os.makedirs(LATEST_DIR, exist_ok=True)
MODEL_PATH = os.path.join(LATEST_DIR, "model.joblib")


# Simple features (match model.py)
HYDRO = set("AILMFWVY")

def clean(s: str) -> str:
    return (s or "").upper().replace(" ", "").replace("\n", "").replace("\r", "")

def hyd_frac(s: str) -> float:
    s = clean(s)
    return sum(1 for c in s if c in HYDRO) / max(len(s), 1)

def nterm_hyd_frac(s: str, w: int = 20) -> float:
    s = clean(s)
    head = s[:w] if len(s) >= w else s
    return hyd_frac(head)

def featurize(s: str) -> np.ndarray:
    s = clean(s)
    return np.array([hyd_frac(s), nterm_hyd_frac(s), float(len(s))], dtype=float)

def main():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Training CSV not found at: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV).dropna(subset=["sequence", "label"])
    if df.empty:
        raise ValueError("Training CSV has no rows with both 'sequence' and 'label'.")

    X = np.vstack([featurize(seq) for seq in df["sequence"].astype(str)])
    y = df["label"].astype(int).values

    clf = LogisticRegression(max_iter=500, solver="liblinear",
                             class_weight="balanced", random_state=0)
    clf.fit(X, y)

    joblib.dump(clf, MODEL_PATH)

    print(json.dumps({
        "status": "ok",
        "samples": int(X.shape[0]),
        "features_dim": int(X.shape[1]),
        "model_path": MODEL_PATH
    }, indent=2))

if __name__ == "__main__":
    main()
