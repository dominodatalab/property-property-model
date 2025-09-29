# model.py — single-file predictor (rule + ML). No training here.
# Usage examples:
#   python model.py --seq "MKKLLLLLLLLLALALALAAAGAGA" --mode auto
#   python model.py --seq ">p\nMSTNPKPQRKTKRNTNRRPQDVK" --mode ml
#   python model.py --seq "MAALALLLGVVVVALAAA" --mode rule

import os, json, argparse
import joblib  # only needed for ML mode

# Where the trained model lives:
# - If running in Domino with a Dataset mounted, set DATASET_DIR to that mount path.
# - Otherwise it uses ./models/latest/model.joblib in the repo.
DATASET_DIR = os.getenv("DATASET_DIR", "").strip()
if DATASET_DIR:
    MODEL_PATH = os.path.join(DATASET_DIR, "models", "latest", "model.joblib")
else:
    HERE = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(HERE, "models", "latest", "model.joblib")

# Biology-lite helpers
HYDRO = set("AILMFWVY")                       # hydrophobic amino acids
VALID = set("ACDEFGHIKLMNPQRSTVWY")           # 20 standard amino acids

def _clean(s: str) -> str:
    return (s or "").upper().replace(" ", "").replace("\n", "").replace("\r", "")

def _parse_maybe_fasta(text: str) -> str:
    """Accept a raw sequence or a tiny FASTA string and return just the sequence."""
    if not text:
        return ""
    t = text.lstrip()
    if t.startswith(">"):
        lines = []
        for line in text.splitlines():
            x = line.strip()
            if not x or x.startswith(">"):
                continue
            lines.append(x)
        return _clean("".join(lines))
    return _clean(text)

def _hyd_frac(s: str) -> float:
    s = _clean(s)
    return sum(1 for c in s if c in HYDRO) / max(len(s), 1)

def _nterm_hyd_frac(s: str, w: int = 20) -> float:
    s = _clean(s)
    head = s[:w] if len(s) >= w else s
    return _hyd_frac(head)

def _feats(s: str):
    s = _clean(s)
    return [[_hyd_frac(s), _nterm_hyd_frac(s), float(len(s))]]

# ---------- Rule-based prediction ----------
def predict_rule(input_text: str):
    seq = _parse_maybe_fasta(input_text)
    if not seq:
        return {"status": "error", "error": "Empty or invalid sequence/FASTA."}
    if any(c not in VALID for c in seq):
        return {"status": "error", "error": "Use one-letter amino acids (A..Y)."}

    overall = _hyd_frac(seq)
    nterm   = _nterm_hyd_frac(seq)
    label = "membrane-bound" if (nterm >= 0.45 or overall >= 0.50) else "soluble"
    confidence = min(0.99, round(0.6 + 0.4 * max(nterm, overall), 3))

    return {
        "status": "ok",
        "prediction": label,
        "confidence": confidence,
        "features": {
            "length": len(seq),
            "hydrophobic_fraction": round(overall, 3),
            "nterm_hydrophobic_fraction": round(nterm, 3),
        },
        "mode": "rule",
        "version": "0.0.2",
    }

# ---------- ML prediction (loads joblib model) ----------
_model_cache = None
def _load_model():
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run train.py first.")
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache

def predict_ml(input_text: str):
    seq = _parse_maybe_fasta(input_text)
    if not seq:
        return {"status": "error", "error": "Empty or invalid sequence/FASTA."}
    if any(c not in VALID for c in seq):
        return {"status": "error", "error": "Use one-letter amino acids (A..Y)."}

    model = _load_model()
    p = float(model.predict_proba(_feats(seq))[0][1])   # P(membrane)
    label = "membrane-bound" if p >= 0.5 else "soluble"
    conf  = round(p if label == "membrane-bound" else 1 - p, 3)

    return {
        "status": "ok",
        "prediction": label,
        "confidence": conf,
        "features": {
            "length": len(seq),
            "hydrophobic_fraction": round(_hyd_frac(seq), 3),
            "nterm_hydrophobic_fraction": round(_nterm_hyd_frac(seq), 3),
        },
        "mode": "ml",
        "model_path": MODEL_PATH,
        "version": "0.1.0-ml",
    }

# ---------- Unified entry ----------
def predict(input_text: str, mode: str = "auto"):
    """
    mode: 'ml' → force ML, 'rule' → force rule, 'auto' → try ML then fallback to rule.
    """
    if mode == "rule":
        return predict_rule(input_text)
    if mode == "ml":
        return predict_ml(input_text)
    # auto
    try:
        return predict_ml(input_text)
    except Exception:
        return predict_rule(input_text)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Protein Property Predictor (rule + ML)")
    ap.add_argument("--seq", required=True, help="Raw AA string or tiny FASTA text")
    ap.add_argument("--mode", choices=["auto","ml","rule"], default="auto")
    args = ap.parse_args()
    print(json.dumps(predict(args.seq, mode=args.mode), indent=2))
