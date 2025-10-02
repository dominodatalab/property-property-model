cat > app.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

# Install deps to user site (keeps startup lightweight)
pip3 install --user --no-warn-script-location -r protein-property-predictor/env/requirements.txt streamlit

# Ensure user-installed binaries are on PATH
export PATH="$HOME/.local/bin:$PATH"

# Domino exposes a port env var when running as an App; default to 8501 if missing
: "${DOMINO_APP_PORT:=8501}"

# Launch Streamlit
exec streamlit run protein-property-predictor/app.py \
  --server.port "$DOMINO_APP_PORT" \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
SH

chmod +x app.sh
