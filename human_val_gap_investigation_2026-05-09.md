# Human Val Gap Investigation

Date: 2026-05-09

## Problem statement

We want to understand why the human domain validation loss is much higher than both:

- human train loss
- SO100 train / valid loss

The main reference checkpoint so far is:

- `logs/so100_apricot_cotrain/right_arm_human_nogripper_weighted0p5_scratch_merged_40k_probe_2026-05-07_21-54-38/checkpoints/last.ckpt`

Key baseline numbers from `diagnostics_cotrain_checkpoint_w0p5_40k.json`:

- `human train normalized smooth_l1_mean = 0.052551`
- `human valid normalized smooth_l1_mean = 0.133658`
- `so100 train normalized smooth_l1_mean = 0.021139`
- `so100 valid normalized smooth_l1_mean = 0.068294`
