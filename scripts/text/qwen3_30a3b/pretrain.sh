#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PATH="${ROOT_DIR}/.venv/bin:${PATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_API_KEY=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

CONFIG="${CONFIG:-configs/text/qwen3/30b-a3b/pt/fineweb_finewiki_scratch_ep1.yaml}"

if [[ "${WANDB_MODE:-}" != "disabled" && "${WANDB_DISABLED:-}" != "true" ]] \
  && grep -qE '^[[:space:]]+enable:[[:space:]]+true[[:space:]]*$' "${CONFIG}" \
  && [[ -z "${WANDB_API_KEY:-}" ]]; then
  if ! wandb status 2>/dev/null | grep -q "api_key"; then
    echo "W&B is enabled in ${CONFIG}, but WANDB_API_KEY is not set and wandb is not logged in." >&2
    echo "Run: PATH=${ROOT_DIR}/.venv/bin:\\$PATH wandb login" >&2
    exit 1
  fi
fi

bash train.sh tasks/train_text.py "${CONFIG}" "$@"
