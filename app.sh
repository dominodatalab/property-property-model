#!/bin/bash
set -euo pipefail
# Keep startup lightweight; install user-level if needed
pip3 install --user --no-warn-script-location -r protein-property-predictor/env/requirements.txt streamlit
export PATH="$HOME/.local/bin:$PATH"
: "${DOMINO_APP_PORT:=8501}"
exec streamlit run protein-property-predictor/app.py \
  --server.port "$DOMINO_APP_PORT" \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
