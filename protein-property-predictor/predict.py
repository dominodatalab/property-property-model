# predict.py — Domino endpoint entrypoint

import sys, os

# Add this folder (protein-property-predictor) to sys.path dynamically
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.append(HERE)

from model import predict  # now works because model.py is in the same folder


def main(args):
    seq = args.get("sequence", "")
    mode = args.get("mode", "auto")
    return predict(seq, mode)
