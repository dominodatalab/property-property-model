# predict.py — Domino endpoint entrypoint
import sys, os, json
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.append(HERE)

from model import predict  # your model.py function

def main(args):
    try:
        seq = args.get("sequence", "")
        mode = args.get("mode", "auto")

        result = predict(seq, mode)

        # Ensure result is JSON serializable (dict)
        return {"result": result}

    except Exception as e:
        # Proper error handling so Domino won’t crash
        return {"error": str(e)}
