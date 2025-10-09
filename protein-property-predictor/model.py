# train.py — logistic regression + MLflow + Domino Artifacts integration
#
# Reads data/train.csv (id, sequence, label)
# Featurizes, trains, logs metrics to MLflow, and saves artifacts.
#
# Artifacts:
#   - models/latest/model.joblib
#   - artifacts/train_summary.json
#   - /mnt/artifacts/* (for Domino Artifacts tab)
#
# Usage:
#   export DATASET_DIR=/domino/datasets/<workspace>/protein-property-predictor
#   python3 train.py

import os, json, time, shutil
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path

# --- MLflow integration ---
import mlflow
import mlflow.sklearn

# --- Safe optional plotting ---
try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False


# ---------- Resolve paths ----------
DATASET_DIR = os.getenv("DATASET_DIR", "").strip()
HERE = os.path.dirname(__file__)
if DATASET_DIR:
    DATA_PATH = os.path.join(DATASET_DIR, "data", "train.csv")
    MODEL_DIR = os.path.join(DATASET_DIR, "models", "latest")
    ARTIFACTS_DIR = os.path.join(DATASET_DIR, "artifacts")
else:
    DATA_PATH = os.path.join(HERE, "data", "train.csv")
    MODEL_DIR = os.path.join(HERE, "models", "latest")
    ARTIFACTS_DIR = os.path.join(HERE, "artifacts")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs("/mnt/artifacts", exist_ok=True)  # Domino artifact sync


# ---------- Feature functions ----------
HYDRO = set("AILMFWVY")

def _clean(s): return (s or "").upper().replace(" ", "").replace("\n", "").replace("\r", "")
def _hyd_frac(s): s=_clean(s); return sum(1 for c in s if c in HYDRO)/max(len(s),1)
def _nterm_hyd_frac(s, w=20): s=_clean(s); return _hyd_frac(s[:w])
def featurize(seq): seq=_clean(seq); return [_hyd_frac(seq), _nterm_hyd_frac(seq), float(len(seq))]


# ---------- Load dataset ----------
df = pd.read_csv(DATA_PATH)
X = np.array([featurize(s) for s in df["sequence"]])
y = df["label"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ---------- MLflow setup ----------
mlflow.set_experiment("Protein_Property_Predictor")

with mlflow.start_run(run_name=f"train_run_{int(time.time())}"):

    # Parameters
    params = dict(model_type="LogisticRegression", solver="lbfgs", max_iter=1000)
    mlflow.log_params(params)

    # Train
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # Predict & Metrics
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    mlflow.log_metrics({"accuracy": acc, "roc_auc": auc})
    print(f"Accuracy={acc:.3f}, AUC={auc:.3f}")

    # ---------- Save model ----------
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    joblib.dump(model, model_path)
    mlflow.sklearn.log_model(model, artifact_path="model")

    # ---------- Save summary JSON ----------
    summary = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": acc,
        "roc_auc": auc,
        "timestamp": time.asctime(),
    }
    summary_path = os.path.join(ARTIFACTS_DIR, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    mlflow.log_artifact(summary_path)

    # ---------- Optional visualization ----------
    if _HAS_PLT:
        plt.figure()
        plt.hist(y_prob, bins=20, color="gray")
        plt.title("Predicted membrane probabilities")
        plt.xlabel("P(membrane)")
        plt.ylabel("Count")
        plt.tight_layout()
        plot_path = os.path.join(ARTIFACTS_DIR, "prob_hist.png")
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

    # ---------- Copy artifacts to Domino mount ----------
    for f in [model_path, summary_path]:
        shutil.copy(f, "/mnt/artifacts/")

    if _HAS_PLT:
        shutil.copy(plot_path, "/mnt/artifacts/")

    print("Training complete. Artifacts saved and logged to MLflow.")

