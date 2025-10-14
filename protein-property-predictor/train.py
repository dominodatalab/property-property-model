import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import mlflow

# ========== 1. Configure MLflow for Domino ==========
tracking_uri = os.getenv("DOMINO_MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)

# Optional: create or use an experiment name (will show up in Domino)
mlflow.set_experiment("Protein Property Predictor")

# ========== 2. Start a run ==========
with mlflow.start_run(run_name="Training_Run"):

    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("solver", "liblinear")

    # ========== 3. Load Data ==========
    dataset_dir = os.getenv("DATASET_DIR", "./data")
    df = pd.read_csv(os.path.join(dataset_dir, "train.csv"))

    X = []
    y = []
    for _, row in df.iterrows():
        seq = row["sequence"]
        # Example features
        hydrophobic_fraction = sum(aa in "AILMFWYV" for aa in seq) / len(seq)
        nterm_fraction = sum(aa in "AILMFWYV" for aa in seq[:10]) / 10
        length = len(seq)
        X.append([hydrophobic_fraction, nterm_fraction, length])
        y.append(row["label"])
    X = np.array(X)
    y = np.array(y)

    # ========== 4. Train Model ==========
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X, y)

    # ========== 5. Evaluate ==========
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    # Log metrics
    mlflow.log_metric("train_accuracy", acc)
    mlflow.log_metric("train_auc", auc)

    # ========== 6. Save Model ==========
    model_dir = os.path.join(dataset_dir, "models/latest")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)

    # Log model as artifact
    mlflow.log_artifact(model_path, artifact_path="model")

    # Log summary
    mlflow.log_text(json.dumps({
        "train_accuracy": acc,
        "train_auc": auc,
        "model_path": model_path
    }, indent=2), "run_summary.json")

    print(json.dumps({
        "train_accuracy": acc,
        "train_auc": auc,
        "model_path": model_path
    }, indent=2))
