# predict.py — Domino endpoint entrypoint
import sys, os, json
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.append(HERE)

from model import predict  # your model.py function

def main(args):
    try:
        # Handle Domino's nested JSON ("data" or "parameters")
        if "data" in args:
            args = args["data"]
        elif "parameters" in args and isinstance(args["parameters"], list) and len(args["parameters"]) > 0:
            args = args["parameters"][0]

        seq = args.get("sequence", "")
        mode = args.get("mode", "auto")

        result = predict(seq, mode)

        # Return properly wrapped JSON
        return {"result": result}

    except Exception as e:
        return {"error": str(e)}
