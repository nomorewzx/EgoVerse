#!/usr/bin/env bash
set -euo pipefail

source /home/zxwang/miniconda3/bin/activate "${EGO_ENV:-lerobot_py312}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 NAME=CHECKPOINT [NAME=CHECKPOINT ...]" >&2
  exit 2
fi

tag="${TAG:-so100_close_ablation_fixed_suite_$(date +%Y-%m-%d_%H-%M-%S)}"
log_jsonl="${SO100_CLOSE_SUITE_LOG:-logs/so100_hpt/rollout_logs/so100_hpt_rollout_2026-05-31_12-53-24.jsonl}"
video="${SO100_CLOSE_SUITE_VIDEO:-logs/so100_hpt/rollout_videos/so100_hpt_rollout_2026-05-31_12-53-24.mp4}"
device="${DEVICE:-cuda}"

model_args=()
for spec in "$@"; do
  model_args+=(--model "${spec}")
done

python egomimic/scripts/diagnostics/probe_so100_rollout_counterfactual.py \
  --log-jsonl "${log_jsonl}" \
  --video "${video}" \
  --device "${device}" \
  --seeds 0 1 2 \
  --resampled-action-len 45 \
  --execute-steps 30 \
  --target-max-distance-px 140 \
  --tag "${tag}" \
  --query-step 1080 \
  --query-step 1110 \
  --query-step 1140 \
  --query-step 1170 \
  --query-step 1200 \
  --query-step 1230 \
  --query-step 1260 \
  --query-step 1290 \
  --query-step 1320 \
  --query-step 1350 \
  --query-step 2520 \
  --query-step 2550 \
  --query-step 2580 \
  --query-step 2610 \
  --query-step 2640 \
  "${model_args[@]}"
