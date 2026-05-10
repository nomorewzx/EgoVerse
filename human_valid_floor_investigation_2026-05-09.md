# Human Valid Floor Investigation

Date: 2026-05-09

## Reframed question

The earlier investigation focused on the human train-to-valid gap.
That framing is incomplete.

The more precise question is:

1. is the human validation loss floor abnormally high?
2. does a large train-to-valid gap necessarily indicate a real problem?
3. if there is a problem, is it likely to have one clean explanation or one clean fix?

This document tracks the revised interpretation.

## Baseline reference

Main reference checkpoint:

- `logs/so100_apricot_cotrain/right_arm_human_nogripper_weighted0p5_scratch_merged_40k_probe_2026-05-07_21-54-38/checkpoints/last.ckpt`

Key normalized losses from `diagnostics_cotrain_checkpoint_w0p5_40k.json`:

- human train: `0.052551`
- human valid: `0.133658`
- SO100 train: `0.021139`
- SO100 valid: `0.068294`

Within-domain gap summary at the same checkpoint:

- human normalized absolute gap: `0.081108`
- human normalized gap ratio: `2.543x`
- SO100 normalized absolute gap: `0.047155`
- SO100 normalized gap ratio: `3.231x`

Within-domain raw-unit MAE summary at the same checkpoint:

- human train: `0.025366`
- human valid: `0.054725`
- human raw gap ratio: `2.157x`
- SO100 train: `0.199170`
- SO100 valid: `0.524990`
- SO100 raw gap ratio: `2.636x`

Important caveat:

- human and SO100 losses are not directly comparable on one shared physical scale
- each embodiment uses its own normalization statistics
- human is `6D` without gripper
- SO100 is `7D` with gripper

So the useful comparisons are:

- train vs valid within the same embodiment
- final valid floor across different training procedures on the same embodiment

## Updated core observation

The strongest observation is no longer:

- "human train-to-valid gap is unusually large"

The stronger observation is:

- human valid appears to settle around a relatively high floor, about `0.134`
- this happens under both cotrain and human-only scratch
- SO100 also shows a large train-to-valid gap ratio, so gap ratio alone is not enough to call the human behavior pathological

Put differently:

- a large gap can be acceptable if the validation floor is already good enough for downstream behavior
- the human question is whether `~0.134` is an acceptable floor for the human task, not whether the ratio alone looks large

## What the evidence now supports

### 1. Large gap by itself is not a sufficient failure signal

Evidence:

- SO100 has `train 0.021139 -> valid 0.068294`, ratio `3.231x`
- this ratio is larger than the human ratio at the same checkpoint
- if SO100 rollout behavior is already acceptable, then a large offline gap can still be operationally acceptable

Interpretation:

- gap ratio is highly sensitive to very low train loss
- once train gets small, even a moderate valid floor can produce a large ratio
- therefore the ratio should not be the primary decision criterion

### 2. Human valid floor looks more important than human gap ratio

Evidence:

- cotrain `40k`: human valid `0.133658`
- scratch `40k`: human valid `0.135944`
- cotrain `40k` and scratch `40k` are very close on human valid despite different train losses

Interpretation:

- the human domain seems to have a stable validation floor near `0.134`
- cotrain mostly improves train fit more than valid fit
- the main unresolved question is why this floor is so much higher than the SO100 valid floor

### 3. The high human valid floor is probably not caused by one single dirty episode

Evidence:

- removing the top 5 suspicious human valid episodes only improved human valid from `0.133658` to `0.125755`
- per-dimension human valid loss is elevated across multiple action dimensions
- dataset diagnostics showed train-vs-valid differences in several action dimensions, not just one wrap bug

Interpretation:

- dirty episodes contribute
- but they are not the whole story
- there is likely a broader floor caused by distribution shift, target noise, or task difficulty

### 4. The high human valid floor is probably not explained mainly by cotrain weighting

Evidence:

- cotrain `10k` already beats scratch `10k` on both human train and human valid
- human-only finetune from cotrain `40k` for `5k` steps made human valid worse, not better
- scratch trained out to `40k` still ends near the same human valid floor as cotrain

Interpretation:

- "human weight is only `0.5`, so the model never really learned human" is too simple
- the problem persists even when the model is trained on human only

### 5. A mixed explanation is more plausible than a single-cause explanation

Most plausible contributors at this point:

- residual train-vs-valid distribution shift in the human split
- higher intrinsic ambiguity or noise in human targets
- human action representation being harder than SO100 control targets
- offline validation loss being an imperfect proxy for downstream rollout quality
- possible ceiling from the current observation/action representation rather than from cotrain itself

## What the evidence does not yet support

Not supported yet:

- "the human valid split is definitely representative"
- "the human valid split is definitely broken"
- "one bug in the export pipeline explains everything"
- "there is one obvious training fix that will collapse human valid to SO100 levels"
- "more human data will automatically solve the problem"

## Is there a reasonable explanation?

Yes, but it is probably not a clean single explanation.

The most reasonable current explanation is:

- large train-to-valid gaps can happen normally once train loss becomes very small
- the human domain additionally has a higher validation floor than SO100
- that higher floor is likely caused by a combination of target noise, split mismatch, and task difficulty

So I do think there is a reasonable explanation.
I do not think the explanation is likely to collapse to one neat root cause.

## Is there a reasonable solution?

Possibly, but probably not one silver bullet.

The likely outcome is:

- no single tweak fully fixes the human valid floor
- improvements, if they come, will probably be incremental and come from several directions at once

Most realistic solution buckets are:

1. improve the metric

- judge human quality with rollout behavior or task success, not only offline smooth L1
- track per-episode and per-dimension valid metrics, not just one scalar

2. improve the validation split

- stratify by episode type instead of pure random episode shuffle
- separate "clean in-distribution valid" from "stress-test valid"

3. improve data quality and coverage

- add data that matches the hard valid modes
- reduce target noise, calibration drift, and inconsistent human demonstration styles

4. improve training adaptation

- smaller-LR human finetune
- partial freezing
- representation changes if current human target is intrinsically noisy

So the honest answer is:

- there may be no single elegant fix
- that does not mean the issue is unexplained or unsalvageable
- it more likely means the current scalar human valid loss is sitting on top of several overlapping effects

## Decision guidance

At this point, the key practical question should be:

- is the current human valid floor actually too high for downstream behavior?

If human rollout quality is bad, then the high floor is a real product problem.
If human rollout quality is acceptable, then the floor may be more of an evaluation artifact or a limitation of the current metric.

That makes rollout evaluation the next anchor.

## Recommended next steps

1. measure human downstream behavior with the current best checkpoint before chasing more offline loss
2. split human validation into a clean slice and a stress-test slice
3. compute per-episode human valid loss for cotrain `40k` and scratch `40k`
4. if collecting more data, target the hard valid modes instead of simply increasing volume
5. only treat offline loss reduction as meaningful if it correlates with better human rollout behavior

## Key artifacts

- prior broad investigation:
  `human_val_gap_investigation_2026-05-09.md`
- cotrain `40k` checkpoint diagnostics:
  `logs/so100_apricot_cotrain/right_arm_human_nogripper_weighted0p5_scratch_merged_40k_probe_2026-05-07_21-54-38/diagnostics_cotrain_checkpoint_w0p5_40k.json`
- human dataset stats:
  `logs/so100_apricot_cotrain/right_arm_human_nogripper_weighted0p5_scratch_merged_40k_probe_2026-05-07_21-54-38/diagnostics_human_dataset_stats_full.json`
- cotrain-vs-scratch timeline figure:
  `logs/so100_apricot_human_scratch/right_arm_human_nogripper_scratch40k_resume_from10k_2026-05-09_2026-05-09_12-44-39/human_gap_timeline_cotrain_vs_scratch.png`
- cotrain-vs-scratch timeline CSV:
  `logs/so100_apricot_human_scratch/right_arm_human_nogripper_scratch40k_resume_from10k_2026-05-09_2026-05-09_12-44-39/human_gap_timeline_cotrain_vs_scratch.csv`
