Protein Property Predictor (Toy) — Domino-first Demo

Classifies protein sequences as soluble or membrane-bound.
Built to showcase the full Domino flow: Workspace → Dataset → Training Job → App (and later Endpoint).

📌 What’s here

Two predictors

Rule-based (no training): simple hydrophobicity heuristics

ML (logistic regression): trained on tiny toy data

Domino-friendly storage: reads/writes under a mounted Dataset via DATASET_DIR

Simple UI: Streamlit app you can publish as a Domino App

🧠 Biology in 30 seconds

Proteins are chains of 20 amino acids (letters A, C, D, …, Y).

Inside/outside cells is watery; membranes are oily.

Hydrophobic (“oily”) letters like A I L M F W V Y tend to favor membranes.

Many membrane/secreted proteins start with a hydrophobic N-terminus (first ~20 aa).

We compute:

overall hydrophobic fraction

N-terminal hydrophobic fraction

sequence length
…and predict membrane-bound vs soluble.

📁 Repo layout
property-property-model/
├─ app.sh                                 # Domino App entry (MUST be in repo root)
├─ README.md
└─ protein-property-predictor/
   ├─ app.py                               # Streamlit UI
   ├─ model.py                             # Prediction (rule + ML loader)
   ├─ train.py                             # Training (logistic regression)
   ├─ data/
   │  └─ train.csv                         # Tiny toy set: id,sequence,label
   ├─ env/
   │  └─ requirements.txt                  # numpy, pandas, scikit-learn, joblib, streamlit, requests
   └─ models/
      └─ latest/                           # Trained model will be saved here (or in Dataset)

✅ Requirements

Domino project connected to this Git repo

A Domino Compute Environment with Python 3 + pip

(Optional) A Dataset mounted in the project (NetApp-backed recommended)

Python packages (already listed in env/requirements.txt):

numpy
pandas
scikit-learn
joblib
streamlit
requests

🚀 Quickstart on Domino
1) Connect code & Dataset

Project → Code → use this GitHub repo (prefer “use latest from external repo”).

Data → Datasets → mount a Dataset.
Note the mount path (example):
/domino/datasets/local/protein-property-predictor

2) Workspace: set env + install deps

Open a terminal in the repo root:

# Tell code where to read/write data & models
export DATASET_DIR=/domino/datasets/local/protein-property-predictor   # ← use your actual mount

# Install deps at user level
pip3 install --user -r protein-property-predictor/env/requirements.txt
export PATH="$HOME/.local/bin:$PATH"

3) (Optional) copy toy training data into the Dataset
mkdir -p "$DATASET_DIR/data" "$DATASET_DIR/models/latest"
cp protein-property-predictor/data/train.csv "$DATASET_DIR/data/"

4) Train (writes a model.joblib)
cd protein-property-predictor
python3 train.py
# → prints JSON with model_path, samples, etc.

5) Predict (CLI)
# Force ML (uses model saved in ${DATASET_DIR} if set, else ./models/latest/)
python3 model.py --seq "MKKLLLLLLLLLALALALAAAGAGA" --mode ml

# Force rule (no training needed)
python3 model.py --seq "MSTNPKPQRKTKRNTNRRPQDVK" --mode rule

# Auto: try ML, fallback to rule
python3 model.py --seq ">p\nMAALALLLGVVVVALAAA" --mode auto

▶️ Publish the Streamlit App (Domino Apps)

Files used:

app.sh (repo root, Domino App entry)

protein-property-predictor/app.py (UI)

Steps:

Deploy → Apps → New App

Name: e.g., ppp-app
Entry script: app.sh (must be exactly at repo root)
Compute environment: any Python env with pip
Datasets: mount the same Dataset you trained to
Environment variables: add
DATASET_DIR=/domino/datasets/local/protein-property-predictor (adjust to your mount)

Publish / Launch → open the URL.

🧪 Quickstart locally (optional)
cd property-property-model/protein-property-predictor
pip install -r env/requirements.txt

# Train (saves to ./models/latest/model.joblib)
python train.py

# Predict
python model.py --seq "MKKLLLLLLLLLALALALAAAGAGA" --mode ml

# Run the app
streamlit run app.py
# open http://localhost:8501

🛠 How it works (very briefly)

train.py

Reads data/train.csv (or ${DATASET_DIR}/data/train.csv if set)

Featurizes each sequence (overall hydrophobicity, N-terminal hydrophobicity, length)

Trains a LogisticRegression model

Saves to ${DATASET_DIR or ./}/models/latest/model.joblib

model.py

Rule mode: threshold on hydrophobicity → label + confidence

ML mode: loads model.joblib, outputs probability + label

Auto: tries ML; if model missing, falls back to rule

app.py (Streamlit)

Text area to paste sequence / tiny FASTA

Dropdown for mode (auto / ml / rule)

Calls predict() and renders JSON

🧭 Roadmap (next)

Endpoint: wrap predict() as a Domino Endpoint with request/response logging

Jobs: schedule train.py retraining, snapshot the Dataset

MLflow: log params/metrics/artifacts for each training run

Tests: unit tests for parsing/featurization & golden predictions

Data: expand training set; add metrics (confusion matrix, ROC/PR)

🧩 Troubleshooting

App error “entry script './app.sh' not found”

Ensure app.sh is at the repo root (visible in Project → Code top level)

In Apps → Edit, set Entry script to app.sh (no ./, no subfolder)

ML mode says model not found

Run train.py first

Verify the path printed in training output exists (under ${DATASET_DIR}/models/latest/ or ./models/latest/)

Blank Streamlit page

Check App logs

Ensure streamlit is installed and on PATH (the app.sh does both)

In a Workspace, sanity check:

export DOMINO_APP_PORT=8501
streamlit run protein-property-predictor/app.py --server.port $DOMINO_APP_PORT --server.address 0.0.0.0 &
sleep 3
curl -s http://127.0.0.1:$DOMINO_APP_PORT/_stcore/health


Git push from Workspace fails

git config --global user.name "YOUR_GH_USERNAME"

git config --global user.email "you@company.com"

Use a GitHub PAT; complete SAML SSO if prompted
