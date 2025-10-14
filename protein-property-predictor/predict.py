# predict.py — Domino Model API entrypoint
import os
import json
from model import predict  # reuse your inference function

def main(args):
    """
    Domino calls this function for each REST request.
    Args:
        args (dict): JSON payload sent to the endpoint.
                     Example: {"sequence": "MKKLLLLLALALALAAAGAGA", "mode": "auto"}
    Returns:
        dict: JSON-safe output (predictions)
    """
    seq = args.get("sequence", "")
    mode = args.get("mode", "auto")
    result = predict(seq, mode)
    return result
