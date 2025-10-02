#!/bin/bash
set -euo pipefail

# Ensure user-site bin (where pip --user installs streamlit) is on PATH
export PATH="$HOME/.local/bin:$PATH"

# Streamlit must bind to Domino's required host/port
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml <<'EOF'
[browser]
gatherUsageStats = true

[server]
port = 8888
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false
EOF

# (Optional) Install deps at startup if not baked into the environment.
# Comment these 2 lines out if you've baked them into the Compute Environment.
#pip3 install --user -r protein-property-predictor/env/requirements.txt || true
#hash -r

# Run the Streamlit app from the correct directory
#cd protein-property-predictor
#exec streamlit run app.py
