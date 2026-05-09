#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
MODEL_ID="${SIGNCAN_MODEL_ID:-google/gemma-4-E4B-it}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"

echo "[setup] HF_ENDPOINT=$HF_ENDPOINT"
echo "[setup] PIP_INDEX=$PIP_INDEX"
echo "[setup] MODEL_ID=$MODEL_ID"
echo "[setup] HF_HOME=$HF_HOME"

mkdir -p "$HF_HOME"
export HF_HOME

python -m pip install --upgrade pip -i "$PIP_INDEX"
python -m pip install -r requirements.txt -i "$PIP_INDEX"

if ! python -c "import huggingface_hub" >/dev/null 2>&1; then
    python -m pip install huggingface_hub -i "$PIP_INDEX"
fi

echo "[setup] Pre-downloading model weights to $HF_HOME ..."
MODEL_ID="$MODEL_ID" HF_HOME="$HF_HOME" HF_ENDPOINT="$HF_ENDPOINT" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    cache_dir=os.environ["HF_HOME"],
    resume_download=True,
)
PY

echo "[setup] Done. Run: python app.py"
