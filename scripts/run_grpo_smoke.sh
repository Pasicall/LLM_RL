#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

MODEL_PATH="${MODEL_PATH:-$HOME/models/OLMo-2-0425-1B}"
RUN_DIR="${RUN_DIR:-runs/olmo-offload-smoke-$(date +%Y%m%d-%H%M%S)}"

python -m scripts.train_grpo \
  --mode pretrained \
  --model-name-or-path "$MODEL_PATH" \
  --local-files-only \
  --device cuda:0 \
  --precision bf16 \
  --gradient-checkpointing \
  --optimizer-state-device cpu \
  --steps 2 \
  --rollout-batch-size 2 \
  --group-size 2 \
  --microbatch-size 1 \
  --generation-batch-size 1 \
  --max-new-tokens 64 \
  --val-limit 4 \
  --eval-every 1 \
  --output-dir "$RUN_DIR"
