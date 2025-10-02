cat > app.sh <<'SH'
#!/bin/bash
set -euo pipefail

# Install deps at user level (no root)
pip3 install --user --no-warn-script-location -r protein-property-predictor/env/requirements.txt streamlit
export PATH="$HOME/.local/bin:$PATH"

# Debug: show where we are and what's here
echo "Repo root: $(pwd)"
ls -la

# Domino injects this port; default to 8501 if not set
: "${DOMINO_APP_PORT:=8501}"

# Run Streamlit app
exec streamlit run protein-property-predictor/app.py \
  --server.port "$DOMINO_APP_PORT" \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
SH

# make it executable and committed
chmod +x app.sh

