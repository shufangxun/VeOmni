#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

export PATH="${ROOT_DIR}/.venv/bin:${PATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/multimodal/openpangu_vl/30a3b/sft.yaml}"

if [[ "${WANDB_MODE:-}" != "disabled" && "${WANDB_DISABLED:-}" != "true" ]] \
  && grep -qE '^[[:space:]]+enable:[[:space:]]+true[[:space:]]*$' "${CONFIG}" \
  && [[ -z "${WANDB_API_KEY:-}" ]]; then
  if ! wandb status 2>/dev/null | grep -q "api_key"; then
    echo "W&B is enabled in ${CONFIG}, but WANDB_API_KEY is not set and wandb is not logged in." >&2
    echo "Run: PATH=${ROOT_DIR}/.venv/bin:\$PATH wandb login" >&2
    exit 1
  fi
fi

bash train.sh tasks/train_vlm.py "${CONFIG}" "$@"
