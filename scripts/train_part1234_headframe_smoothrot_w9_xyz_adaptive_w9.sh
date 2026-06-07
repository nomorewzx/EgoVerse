#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /home/zxwang/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_py312

export SO100_HPT_ZARR_ROOT="${SO100_HPT_ZARR_ROOT:-/home/zxwang/so100-ee-cam-egoverse-zarr}"
export APRICOT_HUMAN_RIGHT_ARM_PART1234_ZARR_ROOT="${APRICOT_HUMAN_RIGHT_ARM_PART1234_ZARR_ROOT:-/home/zxwang/repos/apricot_human_in_domain/egoverse_zarr_apricot_in_domain_merged_part1_part2_part3_part4_min90_ego_view_right_arm}"

STEPS="${STEPS:-80000}"
DESC="${DESC:-right_arm_human_indomain_part1234_headframe_smoothrot_w9_xyzadaptive_w9_steps${STEPS}}"

python egomimic/trainHydra.py \
  --config-name train_zarr_so100_apricot_right_arm_cotrain_human_indomain_part1234_headframe_smoothrot_w9_xyz_adaptive_w9 \
  trainer.max_steps="${STEPS}" \
  model.scheduler.T_max="${STEPS}" \
  description="${DESC}" \
  ckpt_path=null \
  "$@"
