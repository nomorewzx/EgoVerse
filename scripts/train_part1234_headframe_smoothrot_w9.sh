#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /home/zxwang/miniconda3/bin/activate emimic

export APRICOT_HUMAN_RIGHT_ARM_PART1234_ZARR_ROOT="${APRICOT_HUMAN_RIGHT_ARM_PART1234_ZARR_ROOT:-/home/zxwang/repos/apricot_human_in_domain/egoverse_zarr_apricot_in_domain_merged_part1_part2_part3_part4_min90_ego_view_right_arm}"

python egomimic/trainHydra.py \
  --config-name train_zarr_so100_apricot_right_arm_cotrain_human_indomain_part1234_headframe_smoothrot_w9 \
  "$@"
