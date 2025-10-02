#!/bin/bash
set -euo pipefail
pip3 install --user -r protein-property-predictor/env/requirements.txt streamlit
export PATH="$HOME/.local/bin:$PATH"
: "${DOMINO_APP_PORT:=8501}"
streamlit run protein-property-predictor/app.py \
  --server.port "$DOMINO_APP_PORT" \
  --server.address 0.0.0.0
