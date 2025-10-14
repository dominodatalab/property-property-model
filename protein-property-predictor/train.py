import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import mlflow

# ==============================================================
# 1️⃣ Configure MLflow for Domino
# ==============================================================

# Domino exposes its internal MLflow tracking server via this env var
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)
print(f"✅ MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

# Create or use an experiment name (will show up in Domino Experiments tab)
mlflow.set_experiment("Protein Property Predictor")

# ==============================================================
# 2️⃣ Start MLflow run
# ==============================================================

with mlflow.start_run(run_name="Training_Run") as run:

    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("solver", "liblinear")

    # ==========================================================
    # 3️⃣ Load Data
    # ==========================================================
    dataset_dir = os.getenv("DATASET_DIR", "./data")
    data_path = os.path.join(dataset_dir, "train.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Training data not found at: {data_path}")

    df = pd.read_csv(data_path)

    X, y = [], []
    for _, row in df.iterrows():
        seq = row["sequence"]
        hydrophobic_fraction = sum(aa in "AILMFWYV" for aa in seq) / len(seq)
        nterm_fraction = sum(aa in "AILMFWYV" for aa in seq[:10]) / 10
        length = len(seq)
        X.append([hydrophobic_fraction, nterm_fraction, length])
        y.append(row["label"])

    X = np.array(X)
    y = np.array(y)

    # ==========================================================
    # 4️⃣ Train Model
    # ==========================================================
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X, y)

    # ==========================================================
    # 5️⃣ Evaluate Model
    # ==========================================================
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    mlflow.log_metric("train_accuracy", acc)
    mlflow.log_metric("train_auc", auc)

    # ==========================================================
    # 6️⃣ Save Model + Log Artifacts
    # ==========================================================
    model_dir = os.path.join(dataset_dir, "models/latest")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)

    # Log model file
    mlflow.log_artifact(model_path, artifact_path="model")

    # Log summary file
    summary = {
        "train_accuracy": acc,
        "train_auc": auc,
        "model_path": model_path
    }
    mlflow.log_text(json.dumps(summary, indent=2), "run_summary.json")

    print(json.dumps(summary, indent=2))
    print(f"✅ Run logged to MLflow with run_id={run.info.run_id}")
