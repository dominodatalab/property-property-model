# train.py — logistic regression + MLflow + Artifacts
# Reads data/train.csv (id,sequence,label), featurizes, trains, saves:
#   - Dataset:   models/latest/model.joblib
#   - MLflow:    runs under ${DATASET_DIR or HERE}/mlruns
#   - Artifacts: artifacts/train_summary.json (+ optional plot)
#
# Usage:
#   export DATASET_DIR=/domino/datasets/<plane>/protein-property-predictor   # if using a Domino Dataset
#   python3 train.py

import os, json, time
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any

# --- optional plotting (safe if matplotlib not present) ---
try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

# --- prefer Domino Dataset if provided; else use local repo folders ---
DATASET_DIR = os.getenv("DATASET_DIR", "").strip()
HERE        = os.path.dirname(__file__)

if DATASET_DIR:
    DATA_CSV    = os.path.join(DATASET_DIR, "data", "train.csv")
    MODELS_DIR  = os.path.join(DATASET_DIR, "models")
    MLFLOW_DIR  = os.path.join(DATASET_DIR, "mlruns")
else:
    DATA_CSV    = os.path.join(HERE, "data", "train.csv")
    MODELS_DIR  = os.path.join(HERE, "models")
    MLFLOW_DIR  = os.path.join(HERE, "mlruns")

LATEST_DIR  = os.path.join(MODELS_DIR, "latest")
MODEL_PATH  = os.path.join(LATEST_DIR, "model.joblib")

# Artifacts we want Domino to publish from the run (keep these local so Jobs can “Publish output files”)
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
os.makedirs(LATEST_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MLFLOW_DIR, exist_ok=True)

# --- simple features (same as model.py) ---
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

    # Save model for the app/endpoint to load
    joblib.dump(clf, MODEL_PATH)

    # --- MLflow logging (file store under MLFLOW_DIR) ---
    import mlflow, mlflow.sklearn
    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("protein-property-predictor")

    with mlflow.start_run(run_name=time.strftime("train-%Y%m%d-%H%M%S")) as run:
        mlflow.log_params({
            "solver": "liblinear",
            "max_iter": 500,
            "class_weight": "balanced",
            "feature_dim": int(X.shape[1]),
            "n_samples": int(X.shape[0]),
        })
        # quick metric on the tiny toy set
        mlflow.log_metric("train_accuracy", float(clf.score(X, y)))

        # log the raw training CSV and the sklearn model
        if os.path.exists(DATA_CSV):
            mlflow.log_artifact(DATA_CSV, artifact_path="data")
        mlflow.sklearn.log_model(clf, artifact_path="model")

        run_id = run.info.run_id

    # --- Write a small JSON summary as an Artifact (local) ---
    summary: Dict[str, Any] = {
        "status": "ok",
        "samples": int(X.shape[0]),
        "features_dim": int(X.shape[1]),
        "model_path": MODEL_PATH,
        "mlflow_tracking": MLFLOW_DIR,
        "mlflow_last_run_id": run_id,
        "timestamp": int(time.time()),
    }
    with open(os.path.join(ARTIFACTS_DIR, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- Optional: quick diagnostic histogram (hydrophobicity) ---
    if _HAS_PLT:
        try:
            vals = [hyd_frac(s) for s in df["sequence"].astype(str)]
            plt.figure()
            plt.hist(vals, bins=10)
            plt.xlabel("overall hydrophobic fraction")
            plt.ylabel("count")
            plt.title("Toy training set — hydrophobicity")
            fig_path = os.path.join(ARTIFACTS_DIR, "hydrophobicity_hist.png")
            plt.savefig(fig_path, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()