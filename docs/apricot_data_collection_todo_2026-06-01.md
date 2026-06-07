# Apricot Data Collection Todo

Date: 2026-06-01

## Goal

Improve SO100 apricot pick-place performance while keeping two experiment tracks separate:

1. **Strict human-transfer track**
   - Train robot data only on SO100 ID.
   - Use Human ID and Human OOD/EV to test whether human data improves OOD robot generalization.
   - Do **not** include SO100 board/OOD robot data in this track.

2. **Practical adaptation track**
   - Add a small amount of SO100 board/OOD robot data.
   - This is not a strict human-transfer experiment.
   - Use it as an engineering upper bound for OOD board success rate.

## Current Data Baseline

Approximate current data:

| Dataset | Episodes | Frames | Time |
|---|---:|---:|---:|
| SO100 ID | 63 | 86,003 | 0.80 h |
| Human ID P1234 | 195 | 114,146 | 1.06 h |
| Human ID + OOD P12345 | 261 | 142,977 | 1.32 h |
| Human OOD extra | 66 | 28,831 | 0.26 h |

Notes:

- In the EgoVerse paper, `EV` means diverse EgoVerse-A human data, not strictly "OOD".
- For this apricot project, Human OOD/EV means task-matched human apricot demos with different scene/object/background distribution, such as the board setup.
- Current rollout evidence: Human cotrain improves OOD approach, but gripper/contact/pick remains weaker than SO100-only.

## Priority Data To Collect

### P0: SO100 ID Robot Data

Target:

- Expand SO100 ID from `0.80 h` to `1.5-2.0 h`.
- Add roughly `50-100` more successful SO100 ID episodes, depending on episode length.

Purpose:

- Strengthen close, grasp, lift, and place timing.
- Preserve SO100 action/contact distribution when cotraining with human data.
- Reduce reliance on human trajectories for contact-sensitive behavior.

Must include:

- Successful pick and lift.
- Successful place.
- Varied apricot initial positions.
- Varied approach angles.
- Varied apricot orientations.
- Near-boundary but successful grasps.
- Both easy center placements and harder workspace-edge placements.

Avoid:

- Long hesitation before closing.
- Repeated corrective motions unless they are intentional and successful.
- Failed grasps unless they are explicitly labeled as failed and excluded from training.

### P1: Human OOD / EV Data

Target:

- Expand Human OOD/EV from `0.26 h` to at least `1.0 h`.
- Preferred target: `2.0 h`.

Purpose:

- Improve OOD visual and geometric generalization.
- Cover apricot on board / different table / different background / different lighting.
- Help model approach OOD apricot reliably.

Must include:

- Board setup used in OOD evaluation.
- Apricot on different locations of the board.
- Different hand approach directions.
- Different camera/head viewpoints.
- Lighting/background variation.
- Clean decisive reach, grasp, move, and place motions.

Important:

- Human OOD/EV does not provide robot gripper labels.
- It should be treated as perception/approach diversity, not as sufficient contact supervision.

### P2: Human ID Data

Target:

- Expand Human ID from `1.06 h` to around `1.5-2.0 h`.

Purpose:

- Provide domain-aligned human anchor, matching the SO100 ID task setup.
- Help bridge Human OOD/EV to robot task semantics.

Must include:

- Same apricot ID table setup as SO100.
- Same pick-place task definition.
- Similar object placement range as robot demos.
- Multiple demonstrators if available.

Risk:

- Too much Human ID without stronger SO100 contact supervision can still dilute SO100 gripper/contact behavior.
- Do not prioritize this over SO100 ID or Human OOD/EV.

## Optional Engineering-Only Data

### SO100 Board / OOD Robot Data

This data should **not** be included in the strict human-transfer experiment.

Target:

- Start with `20-40` successful SO100 board/OOD episodes.
- If useful, expand to `0.5 h`.

Purpose:

- Directly supervise SO100 gripper/contact on the board setup.
- Estimate an engineering upper bound for OOD board success.
- Diagnose whether the remaining failure is contact/gripper or visual localization.

Use only for:

- Practical adaptation track.
- Upper-bound comparison.
- Debugging close/pick failure after OOD approach succeeds.

Do not claim:

- Pure human-to-robot OOD transfer.
- Strict EgoVerse-style OOD generalization from human data only.

## Recommended Experiment Matrix

### Strict Human-Transfer Track

Use:

- SO100 ID expanded to `1.5-2.0 h`
- Human ID expanded to `1.5-2.0 h`
- Human OOD/EV expanded to `1.0-2.0 h`
- No SO100 board/OOD robot data

Train/evaluate:

1. SO100-only ID baseline.
2. SO100 ID + Human ID.
3. SO100 ID + Human ID + Human OOD/EV.
4. Human XYZ-only cotrain variants.
5. Smooth rotation only if using human rotation.

Do not use:

- Training-time XYZ smoothing.
- Chunk-level XYZ adaptive smoothing.

### Practical Adaptation Track

Use:

- Strict track data.
- Plus `20-40` SO100 board/OOD robot episodes.

Train/evaluate:

1. Add SO100 board/OOD to training.
2. Compare against strict-transfer best model.
3. Measure final pick success rate on board.

## Evaluation Checklist

Report both binary success and subtask scores:

| Metric | Meaning |
|---|---|
| Approach | End-effector reaches near apricot |
| Close | Gripper closes at the correct phase |
| Pick/Lift | Apricot is grasped and lifted |
| Place | Apricot is placed at target |
| Full success | Pick-place completed |

Reason:

- Current evidence suggests Human OOD helps approach but hurts or fails close/pick.
- Binary full success alone hides where human cotrain helps.

## Current Working Hypotheses

1. SO100-only is best in-domain because robot contact/gripper labels match deployment.
2. Human XYZ-only cotrain helps OOD approach because it adds scene/object/background diversity.
3. Human cotrain does not solve gripper close because human data has no robot gripper label.
4. Training-time XYZ smoothing can damage contact geometry and should be avoided.
5. Rotation smoothing helps reduce harmful human rotation noise, but human rotation still has embodiment gap.
6. The most likely path to better OOD board success is:

   ```text
   more SO100 ID contact data
   + more Human OOD/EV visual diversity
   + SO100-dominant or SO100-finetuned training
   ```

## Concrete Next Collection Plan

Minimum next batch:

| Priority | Data | Target Addition |
|---|---|---:|
| P0 | SO100 ID | +0.7 h to +1.2 h |
| P1 | Human OOD/EV board | +0.75 h to +1.75 h |
| P2 | Human ID | +0.5 h to +1.0 h |
| Optional | SO100 board/OOD | +20 to +40 successful episodes |

Recommended order:

1. Collect SO100 ID successful pick-place demos.
2. Collect Human OOD/EV board demos.
3. Collect Human ID if time remains.
4. Separately collect a small SO100 board/OOD set only for practical adaptation.
