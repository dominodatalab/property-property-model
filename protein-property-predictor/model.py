# Protein Property Predictor — demo stub (no dependencies)
import json, argparse

_HYDRO = set("AILMFWVY")                 # hydrophobic letters
_VALID = set("ACDEFGHIKLMNPQRSTVWY")     # valid amino acids

def _clean(s: str) -> str:
    return (s or "").upper().replace(" ", "").replace("\r", "").replace("\n", "")

def _parse_maybe_fasta(text: str) -> str:
    if not text: return ""
    t = text.lstrip()
    if t.startswith(">"):
        lines = []
        for line in text.splitlines():
            x = line.strip()
            if not x or x.startswith(">"): continue
            lines.append(x)
        return _clean("".join(lines))
    return _clean(text)

def _hyd_frac(seq: str) -> float:
    return sum(1 for c in seq if c in _HYDRO) / max(len(seq), 1)

def _nterm_hyd_frac(seq: str, w: int = 20) -> float:
    head = seq[:w] if len(seq) >= w else seq
    return _hyd_frac(head)

def predict(input_text: str):
    seq = _parse_maybe_fasta(input_text)
    if not seq:
        return {"status":"error","error":"Empty or invalid sequence/FASTA."}
    if any(c not in _VALID for c in seq):
        return {"status":"error","error":"Use one-letter amino acids (A,C,D,...,Y)."}

    overall = _hyd_frac(seq)
    nterm   = _nterm_hyd_frac(seq)
    label = "membrane-bound" if (nterm >= 0.45 or overall >= 0.50) else "soluble"
    confidence = min(0.99, round(0.6 + 0.4 * max(nterm, overall), 3))

    return {
        "status":"ok",
        "prediction":label,
        "confidence":confidence,
        "features":{
            "length":len(seq),
            "hydrophobic_fraction":round(overall,3),
            "nterm_hydrophobic_fraction":round(nterm,3),
        },
        "note":"Rule-based demo stub. Next: swap for logistic regression.",
        "version":"0.0.1-stub",
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Protein Property Predictor (demo stub)")
    ap.add_argument("--seq", required=True, help="Raw AA string or tiny FASTA text")
    args = ap.parse_args()
    print(json.dumps(predict(args.seq), indent=2))
