#!/usr/bin/env bash
set -euo pipefail

source /home/zxwang/miniconda3/bin/activate "${EGO_ENV:-lerobot_py312}"

run="${1:-first2}"

declare -A configs=(
  [a0]=train_zarr_so100_close_ablation_headonly_a0
  [a1]=train_zarr_so100_close_ablation_headonly_a1_gripper5x
  [a2]=train_zarr_so100_close_ablation_headonly_a2_sampling
  [a3]=train_zarr_so100_close_ablation_headonly_a3_sampling_gripper5x
)

case "${run}" in
  first2)
    runs=(a0 a2)
    ;;
  all)
    runs=(a0 a1 a2 a3)
    ;;
  a0|a1|a2|a3)
    runs=("${run}")
    ;;
  *)
    echo "Usage: $0 [first2|all|a0|a1|a2|a3]" >&2
    exit 2
    ;;
esac

for key in "${runs[@]}"; do
  config="${configs[$key]}"
  echo "[train_so100_close_ablation] starting ${key}: ${config}"
  python egomimic/trainHydra.py --config-name "${config}"
done
