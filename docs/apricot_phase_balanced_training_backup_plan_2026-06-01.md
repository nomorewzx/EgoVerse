# Apricot Phase-Balanced Training Backup Plan

Date: 2026-06-01

This is a backup plan for the apricot pick-place task. The goal is to handle the natural within-episode imbalance:

- Approach and place occupy many frames.
- Pick/contact/lift occupy fewer frames.
- Pick/contact/lift are more complex and more failure-sensitive than approach/place.

The main idea is not traditional image augmentation. Instead, use phase annotation, event-centered sampling, phase-weighted loss, and only geometry-safe augmentation.

## Current Hypothesis

The current failures are not only caused by gripper close classification. They are more likely caused by the joint timing of:

- end-effector XYZ
- wrist rotation
- gripper close
- contact moment
- lift transition

Therefore, simply oversampling gripper-close frames or increasing gripper loss can be insufficient. The model needs more training pressure on the full pick/contact/lift transition window.

## Episode Annotation

Each successful SO100 episode should be annotated into task phases.

Recommended phases:

| Phase | Meaning | Main supervision value |
| --- | --- | --- |
| `approach` | Moving toward apricot before precise pre-grasp alignment | visual servoing, coarse XYZ |
| `pre_contact` | Gripper is near apricot, before closing/contact | precise XYZ, wrist alignment, close timing |
| `close_contact` | Gripper starts closing and contacts apricot | gripper timing, contact geometry |
| `lift` | Apricot is held and starts moving upward | grasp stability, post-contact motion |
| `place` | Moving apricot to target and releasing | transport, release timing |

### Automatic Signals

Use deterministic heuristics first. Manual correction can be added later if needed.

Candidate signals:

- `gripper_close_onset`: first frame where gripper command begins closing meaningfully.
- `gripper_closed`: frame range where gripper is near closed or closure command is sustained.
- `lift_onset`: first frame after close where EE Z increases consistently.
- `place_release_onset`: first frame where gripper begins opening after transport.
- `episode_end`: final valid frame before reset/hold.

### Suggested Window Rules

Use frame rate 30 Hz unless otherwise specified.

| Event window | Frame rule |
| --- | --- |
| `pre_contact` | `gripper_close_onset - 30` to `gripper_close_onset - 1` |
| `close_contact` | `gripper_close_onset - 10` to `gripper_close_onset + 20` |
| `lift` | `lift_onset - 10` to `lift_onset + 30` |
| `place` | `place_release_onset - 45` to `place_release_onset + 30` |
| `approach` | valid frames before `pre_contact`, excluding reset/idle frames |

If the detected windows overlap, keep the more specific phase:

`close_contact` > `lift` > `pre_contact` > `place` > `approach`

### Episode End Handling

Do not train heavily on final hold/reset frames.

Recommended rules:

- Drop the final 0.5-1.0 seconds if the robot is holding still after rollout completion.
- Drop frames after object release if they are only reset motion.
- Do not treat end-of-episode Z spikes as useful high-frequency events.

## Sampler Plan

Replace uniform timestep sampling with phase-aware event-centered sampling.

### Recommended Initial Mixture

For SO100 successful episodes:

| Source | Sampling probability |
| --- | ---: |
| `pre_contact` chunks | 20% |
| `close_contact` chunks | 25% |
| `lift` chunks | 15% |
| `approach` chunks | 20% |
| `place` chunks | 10% |
| uniform random chunks | 10% |

This gives 60% of SO100 chunks to the pick-critical region while keeping approach/place coverage.

### Chunk Sampling

For each sampled phase:

1. Pick an episode with that phase available.
2. Pick an event anchor inside the phase window.
3. Sample a training chunk that contains the event anchor.
4. Randomly jitter the anchor by a small amount, for example +/- 5 frames.

For action-chunk policies, the event should appear inside the predicted action horizon, not only inside the observation history.

### Human Data Sampling

Human data should be used mainly for visual and approach generalization.

Recommended human sampling:

- Keep Human OOD/EV strong for visual diversity and approach behavior.
- Keep Human ID as a bridge between SO100 ID and Human OOD/EV.
- Avoid making human pick/contact dominate SO100 robot pick/contact, because human gripper/contact labels are not robot-equivalent.

Initial cotrain ratio suggestion:

| Data source | Suggested batch share |
| --- | ---: |
| SO100 ID, phase-balanced | 50-60% |
| Human ID | 20-25% |
| Human OOD/EV | 20-25% |

If OOD approach improves but close/pick remains weak, increase SO100 phase-balanced share rather than increasing human data.

## Loss Weighting Plan

Use phase-weighted action loss. Do not only increase gripper loss.

### Initial Phase Weights

| Phase | Loss weight |
| --- | ---: |
| `approach` | 1.0 |
| `place` | 1.0 |
| uniform random | 1.0 |
| `pre_contact` | 2.0 |
| `close_contact` | 4.0 |
| `lift` | 2.0 |

### Dimension Weights

During `pre_contact`, `close_contact`, and `lift`, increase the joint supervision for:

- XYZ position
- rotation, if training rotation
- gripper

Suggested starting weights:

| Dimension group | Normal phase | Pick-critical phase |
| --- | ---: | ---: |
| XYZ | 1.0 | 2.0 |
| rotation | 1.0 | 1.5 |
| gripper | 1.0 | 3.0 |

Avoid using only `gripper=5x` without also increasing XYZ/rotation supervision. Pick success depends on spatial timing, not only close command.

### Guardrails

- Keep global loss scale normalized so optimization does not become unstable.
- Track per-source loss: SO100 train, Human train, SO100 val, Human val.
- Track per-phase loss if possible: approach, pre_contact, close_contact, lift, place.
- Compare rollout success, not only validation loss.

## Safer Augmentation

Use only augmentations that do not break the geometry between image observations and action labels.

### Safe To Try First

Photometric augmentation:

- brightness jitter
- contrast jitter
- saturation jitter
- hue jitter, very small
- exposure / gamma jitter
- white balance jitter

Sensor-like augmentation:

- mild Gaussian noise
- mild motion blur
- mild defocus blur
- JPEG/compression noise

Observation noise:

- very small proprioception noise on observation inputs
- very small EE pose noise on observation inputs
- no change to action labels

### Use With Caution

Small crop/resize can be used only if the model and camera preprocessing already tolerate it.

Rules:

- Keep crop small.
- Do not crop out the gripper or apricot.
- Avoid changing the apparent object location too much.
- Validate visually on several rollouts before training.

### Avoid For Now

These are risky because they change geometry without changing action labels:

- horizontal flip / mirror
- image rotation
- affine transform
- perspective transform
- large random crop
- random object translation in image
- cut-and-paste object augmentation

These can create incorrect supervision unless camera intrinsics, EE pose, action trajectory, and object frame are transformed consistently.

## Smoothing Policy

Do not use training-time XYZ smoothing for this backup plan.

Reason:

- Pick/contact windows naturally contain high-frequency but task-relevant motion.
- Internal-ratio based smoothing can hit exactly the critical frames.
- Smoothing contact-adjacent XYZ can shift the close/contact/lift timing.

Allowed:

- Rotation smoothing if it has already shown clear rollout benefit.
- Very mild denoising only outside pick-critical phases.
- Runtime safety filters that clip unsafe deltas without changing training labels.

## Evaluation Plan

Report both full success and subtask scores.

Recommended subtask metrics:

| Metric | Meaning |
| --- | --- |
| `approach` | gripper reaches apricot neighborhood |
| `close` | gripper closes at plausible contact timing |
| `pick_lift` | apricot is lifted from surface |
| `place` | apricot is moved to target and released |
| `full_success` | complete pick-place succeeds |

Also track:

- ID success rate
- OOD board approach rate
- OOD board close/contact rate
- OOD board pick/lift rate
- OOD board full success rate

This matters because human cotrain may improve approach while still failing close/pick.

## Recommended Ablation Order

1. SO100-only with phase-balanced sampler and phase-weighted loss.
2. SO100 + Human XYZ-only with phase-balanced SO100 sampler.
3. SO100 + Human XYZ-only + rotation smooth if rotation is included.
4. Same as above with safe photometric augmentation.
5. Only after these: test any small crop/noise augmentation.

Do not combine many changes in the first run.

## Success Criteria

This backup plan is useful if it improves at least one of:

- ID pick/lift success over the current cotrain variants.
- OOD board close/contact after approach.
- OOD board pick/lift after approach.
- Smoothness without hurting contact timing.

If it only improves validation loss but not rollout pick/lift, treat it as unsuccessful.
