from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.scripts.diagnostics.probe_so100_val_action_quality import (
    ACTION_KEY,
    EMBODIMENT,
    build_valid_dataset,
    candidate_metadata,
    expected_action_len,
    load_sample,
    parse_model_spec,
    predict_samples,
    resample_chunk,
    spread_select,
)


PHASES = ("open_hold_approach", "closing_grasp")
HORIZONS = ("model_raw_full", "resample45_full", "rollout_resample45_exec30")
GRIPPER_CLOSE_THRESHOLD = 30.0
GRIPPER_DROP_THRESHOLD = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose SO100 validation grasp timing. Reports first close, close "
            "pose delta, reopen timing, and predicted-vs-GT phase offsets for "
            "open approach and closing grasp states."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model checkpoint as NAME=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/zxwang/so100-ee-cam-egoverse-zarr",
        help="SO100 zarr root used by the train config.",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples-per-phase", type=int, default=154)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--output-dir",
        default="logs/so100_hpt/val_grasp_timing_probe",
    )
    parser.add_argument("--tag", default="so100_mlp_vs_fm_grasp_timing_2026-05-24")
    return parser.parse_args()


def select_phase_samples(candidates: list[dict], max_samples_per_phase: int) -> list[dict]:
    by_phase: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        if item["phase"] in PHASES:
            by_phase[item["phase"]].append(item)

    selected: list[dict] = []
    for phase in PHASES:
        selected.extend(spread_select(by_phase.get(phase, []), max_samples_per_phase))
    selected.sort(key=lambda item: item["global_idx"])
    return selected


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def finite_or_blank(value: float | int | None) -> float | int | str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def first_ge(values: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(values >= threshold)
    return int(hits[0]) if hits.size else None


def first_lt_after(values: np.ndarray, start_idx: int | None, threshold: float) -> int | None:
    if start_idx is None or start_idx + 1 >= len(values):
        return None
    hits = np.flatnonzero(values[start_idx + 1 :] < threshold)
    return int(start_idx + 1 + hits[0]) if hits.size else None


def first_drop_after(
    values: np.ndarray,
    start_idx: int | None,
    *,
    drop_threshold: float,
) -> int | None:
    if start_idx is None or start_idx + 1 >= len(values):
        return None
    close_value = float(values[start_idx])
    hits = np.flatnonzero(values[start_idx + 1 :] <= close_value - drop_threshold)
    return int(start_idx + 1 + hits[0]) if hits.size else None


def prepare_horizon(chunk: np.ndarray, horizon: str) -> np.ndarray:
    if horizon == "model_raw_full":
        return np.asarray(chunk, dtype=np.float64)
    if horizon == "resample45_full":
        return resample_chunk(chunk, 45)
    if horizon == "rollout_resample45_exec30":
        return resample_chunk(chunk, 45)[:30]
    raise ValueError(f"Unknown horizon: {horizon}")


def sequence_timing_metrics(
    chunk: np.ndarray,
    state: np.ndarray,
    *,
    close_threshold: float = GRIPPER_CLOSE_THRESHOLD,
) -> dict:
    arr = np.asarray(chunk, dtype=np.float64)
    gripper = arr[:, 6]
    state_g = float(state[6])
    first_close = first_ge(gripper, close_threshold)
    first_effective_close = first_ge(gripper, max(close_threshold, state_g + GRIPPER_DROP_THRESHOLD))
    reopen_abs = first_lt_after(gripper, first_close, close_threshold)
    reopen_drop = first_drop_after(
        gripper,
        first_close,
        drop_threshold=GRIPPER_DROP_THRESHOLD,
    )

    metrics: dict[str, float | int | None | bool] = {
        "has_close": first_close is not None,
        "first_close_idx": first_close,
        "first_effective_close_idx": first_effective_close,
        "first_reopen_below30_idx": reopen_abs,
        "first_reopen_drop8_idx": reopen_drop,
        "close_to_reopen_below30_steps": (
            None if first_close is None or reopen_abs is None else reopen_abs - first_close
        ),
        "close_to_reopen_drop8_steps": (
            None if first_close is None or reopen_drop is None else reopen_drop - first_close
        ),
        "gripper_first": float(gripper[0]),
        "gripper_last": float(gripper[-1]),
        "gripper_min": float(gripper.min()),
        "gripper_max": float(gripper.max()),
        "gripper_range": float(gripper.max() - gripper.min()),
    }
    if first_close is None:
        metrics.update(
            {
                "close_gripper": None,
                "close_pos_delta_m": None,
                "close_xy_delta_m": None,
                "close_z_delta_m": None,
                "close_rot_delta_rad": None,
                "close_yaw_delta_rad": None,
                "close_pitch_delta_rad": None,
                "close_roll_delta_rad": None,
            }
        )
        return metrics

    close_action = arr[first_close]
    xyz_delta = close_action[:3] - state[:3]
    rot_delta = np.asarray([wrap_angle(v) for v in close_action[3:6] - state[3:6]])
    metrics.update(
        {
            "close_gripper": float(close_action[6]),
            "close_pos_delta_m": float(np.linalg.norm(xyz_delta)),
            "close_xy_delta_m": float(np.linalg.norm(xyz_delta[:2])),
            "close_z_delta_m": float(xyz_delta[2]),
            "close_rot_delta_rad": float(np.linalg.norm(rot_delta)),
            "close_yaw_delta_rad": float(rot_delta[0]),
            "close_pitch_delta_rad": float(rot_delta[1]),
            "close_roll_delta_rad": float(rot_delta[2]),
        }
    )
    return metrics


def motion_alignment_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    state: np.ndarray,
    *,
    gt_close_idx: int | None,
) -> dict:
    pred_arr = np.asarray(pred, dtype=np.float64)
    gt_arr = np.asarray(gt, dtype=np.float64)
    if pred_arr.ndim != 2 or gt_arr.ndim != 2:
        return {}
    base_len = min(len(pred_arr), len(gt_arr))
    if base_len <= 0:
        return {}
    end = base_len if gt_close_idx is None else min(base_len, int(gt_close_idx) + 1)
    if end <= 0:
        return {}

    pred_sel = pred_arr[:end]
    gt_sel = gt_arr[:end]
    pred_delta = pred_sel[:, :3] - state[:3]
    gt_delta = gt_sel[:, :3] - state[:3]
    pred_norm = np.linalg.norm(pred_delta, axis=1)
    gt_norm = np.linalg.norm(gt_delta, axis=1)
    denom = np.maximum(gt_norm, 1e-8)
    ratio = pred_norm / denom

    dot = np.sum(pred_delta * gt_delta, axis=1)
    cos_denom = np.maximum(pred_norm * gt_norm, 1e-8)
    cos = dot / cos_denom
    valid_cos = (pred_norm > 1e-6) & (gt_norm > 1e-6)

    pred_path = (
        float(np.linalg.norm(np.diff(pred_sel[:, :3], axis=0), axis=1).sum())
        if end > 1
        else 0.0
    )
    gt_path = (
        float(np.linalg.norm(np.diff(gt_sel[:, :3], axis=0), axis=1).sum())
        if end > 1
        else 0.0
    )

    return {
        "motion_window_len": int(end),
        "pred_preclose_pos_delta_mean": float(pred_norm.mean()),
        "gt_preclose_pos_delta_mean": float(gt_norm.mean()),
        "preclose_pos_delta_ratio_mean": float(ratio.mean()),
        "preclose_pos_delta_ratio_final": float(ratio[-1]),
        "preclose_pos_delta_cos_mean": (
            float(cos[valid_cos].mean()) if np.any(valid_cos) else None
        ),
        "preclose_pos_delta_cos_final": (
            float(cos[-1]) if valid_cos[-1] else None
        ),
        "pred_preclose_path_len_m": pred_path,
        "gt_preclose_path_len_m": gt_path,
        "preclose_path_ratio": float(pred_path / max(gt_path, 1e-8)),
    }


def prefixed(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{key}": finite_or_blank(value) for key, value in values.items()}


def row_for_sample(
    *,
    model_name: str,
    checkpoint: Path,
    head_class: str,
    diffusion: bool,
    expected_t: int,
    seed_label: str,
    sample: dict,
    pred: np.ndarray,
    gt: np.ndarray,
    horizon: str,
) -> dict:
    state = sample["state"].detach().cpu().numpy()
    pred_h = prepare_horizon(pred, horizon)
    gt_h = prepare_horizon(gt, horizon)
    pred_metrics = sequence_timing_metrics(pred_h, state)
    gt_metrics = sequence_timing_metrics(gt_h, state)
    motion_metrics = motion_alignment_metrics(
        pred_h,
        gt_h,
        state,
        gt_close_idx=gt_metrics["first_close_idx"],
    )

    pred_close = pred_metrics["first_close_idx"]
    gt_close = gt_metrics["first_close_idx"]
    both_close = pred_close is not None and gt_close is not None
    close_offset = int(pred_close - gt_close) if both_close else None
    pred_reopen = pred_metrics["first_reopen_below30_idx"]
    gt_reopen = gt_metrics["first_reopen_below30_idx"]
    both_reopen = pred_reopen is not None and gt_reopen is not None
    reopen_offset = int(pred_reopen - gt_reopen) if both_reopen else None
    imminent_closing = bool(sample["phase"] == "closing_grasp" and gt_close is not None)
    if imminent_closing:
        diagnostic_phase = "closing_grasp_imminent"
    elif sample["phase"] == "closing_grasp":
        diagnostic_phase = "closing_grasp_late_or_tail"
    else:
        diagnostic_phase = sample["phase"]

    motion_ratio = motion_metrics.get("preclose_pos_delta_ratio_mean")
    motion_cos = motion_metrics.get("preclose_pos_delta_cos_mean")
    low_motion = (
        bool(imminent_closing and motion_ratio is not None and float(motion_ratio) < 0.5)
    )
    wrong_direction = (
        bool(imminent_closing and motion_cos is not None and float(motion_cos) < 0.25)
    )

    row = {
        "model": model_name,
        "checkpoint": str(checkpoint),
        "head_class": head_class,
        "diffusion": diffusion,
        "seed": seed_label,
        "global_idx": sample["global_idx"],
        "episode": sample["episode"],
        "local_idx": sample["local_idx"],
        "phase": sample["phase"],
        "diagnostic_phase": diagnostic_phase,
        "horizon": horizon,
        "expected_action_len": expected_t,
        "state_gripper": float(state[6]),
        "imminent_closing_grasp": imminent_closing,
        "pred_has_close": pred_metrics["has_close"],
        "gt_has_close": gt_metrics["has_close"],
        "pred_extra_close": bool(pred_metrics["has_close"] and not gt_metrics["has_close"]),
        "pred_missed_close": bool(gt_metrics["has_close"] and not pred_metrics["has_close"]),
        "both_close": bool(both_close),
        "close_idx_offset_pred_minus_gt": finite_or_blank(close_offset),
        "pred_early_close": bool(both_close and close_offset < 0),
        "pred_late_close": bool(both_close and close_offset > 0),
        "pred_close_ge3_early": bool(both_close and close_offset <= -3),
        "pred_close_ge3_late": bool(both_close and close_offset >= 3),
        "both_reopen_below30": bool(both_reopen),
        "reopen_below30_idx_offset_pred_minus_gt": finite_or_blank(reopen_offset),
        "pred_low_motion_vs_gt": low_motion,
        "pred_wrong_direction_vs_gt": wrong_direction,
        "pred_missed_close_and_low_motion": bool(
            imminent_closing and pred_metrics["has_close"] is False and low_motion
        ),
        "pred_missed_close_and_wrong_direction": bool(
            imminent_closing and pred_metrics["has_close"] is False and wrong_direction
        ),
        "pred_missed_close_and_bad_motion": bool(
            imminent_closing
            and pred_metrics["has_close"] is False
            and (low_motion or wrong_direction)
        ),
    }
    row.update(prefixed("pred", pred_metrics))
    row.update(prefixed("gt", gt_metrics))
    row.update({key: finite_or_blank(value) for key, value in motion_metrics.items()})
    return row


def numeric_values(rows: list[dict], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key)
        if value in ("", None):
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            values.append(value_f)
    return np.asarray(values, dtype=np.float64)


def summarize_group(rows: list[dict]) -> dict:
    summary: dict[str, float | int | None] = {
        "n_rows": len(rows),
        "unique_samples": len({row["global_idx"] for row in rows}),
        "pred_close_rate": sum(bool(row["pred_has_close"]) for row in rows) / len(rows),
        "gt_close_rate": sum(bool(row["gt_has_close"]) for row in rows) / len(rows),
        "pred_extra_close_rate": sum(bool(row["pred_extra_close"]) for row in rows)
        / len(rows),
        "pred_missed_close_rate": sum(bool(row["pred_missed_close"]) for row in rows)
        / len(rows),
        "both_close_rate": sum(bool(row["both_close"]) for row in rows) / len(rows),
        "pred_early_close_rate": sum(bool(row["pred_early_close"]) for row in rows)
        / len(rows),
        "pred_late_close_rate": sum(bool(row["pred_late_close"]) for row in rows)
        / len(rows),
        "pred_close_ge3_early_rate": sum(bool(row["pred_close_ge3_early"]) for row in rows)
        / len(rows),
        "pred_close_ge3_late_rate": sum(bool(row["pred_close_ge3_late"]) for row in rows)
        / len(rows),
        "imminent_closing_grasp_rate": sum(
            bool(row["imminent_closing_grasp"]) for row in rows
        )
        / len(rows),
        "pred_low_motion_vs_gt_rate": sum(
            bool(row["pred_low_motion_vs_gt"]) for row in rows
        )
        / len(rows),
        "pred_wrong_direction_vs_gt_rate": sum(
            bool(row["pred_wrong_direction_vs_gt"]) for row in rows
        )
        / len(rows),
        "pred_missed_close_and_low_motion_rate": sum(
            bool(row["pred_missed_close_and_low_motion"]) for row in rows
        )
        / len(rows),
        "pred_missed_close_and_wrong_direction_rate": sum(
            bool(row["pred_missed_close_and_wrong_direction"]) for row in rows
        )
        / len(rows),
        "pred_missed_close_and_bad_motion_rate": sum(
            bool(row["pred_missed_close_and_bad_motion"]) for row in rows
        )
        / len(rows),
    }
    for key in (
        "motion_window_len",
        "close_idx_offset_pred_minus_gt",
        "reopen_below30_idx_offset_pred_minus_gt",
        "pred_first_close_idx",
        "gt_first_close_idx",
        "pred_first_reopen_below30_idx",
        "gt_first_reopen_below30_idx",
        "pred_close_to_reopen_below30_steps",
        "gt_close_to_reopen_below30_steps",
        "pred_close_pos_delta_m",
        "gt_close_pos_delta_m",
        "pred_close_xy_delta_m",
        "gt_close_xy_delta_m",
        "pred_close_z_delta_m",
        "gt_close_z_delta_m",
        "pred_close_rot_delta_rad",
        "gt_close_rot_delta_rad",
        "pred_gripper_max",
        "gt_gripper_max",
        "pred_preclose_pos_delta_mean",
        "gt_preclose_pos_delta_mean",
        "preclose_pos_delta_ratio_mean",
        "preclose_pos_delta_ratio_final",
        "preclose_pos_delta_cos_mean",
        "preclose_pos_delta_cos_final",
        "pred_preclose_path_len_m",
        "gt_preclose_path_len_m",
        "preclose_path_ratio",
    ):
        values = numeric_values(rows, key)
        if values.size == 0:
            summary[f"{key}_mean"] = None
            summary[f"{key}_median"] = None
            summary[f"{key}_p25"] = None
            summary[f"{key}_p75"] = None
            summary[f"{key}_p95"] = None
            continue
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_median"] = float(np.median(values))
        summary[f"{key}_p25"] = float(np.quantile(values, 0.25))
        summary[f"{key}_p75"] = float(np.quantile(values, 0.75))
        summary[f"{key}_p95"] = float(np.quantile(values, 0.95))
    return summary


def summarize_rows(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    summaries = []
    for group, group_rows in sorted(grouped.items()):
        item = {key: value for key, value in zip(group_keys, group, strict=True)}
        item.update(summarize_group(group_rows))
        summaries.append(item)
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(
    *,
    model_name: str,
    checkpoint: Path,
    samples: list[dict],
    seeds_arg: list[int],
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict], dict]:
    wrapper = ModelWrapper.load_from_checkpoint(
        str(checkpoint), weights_only=False, map_location="cpu"
    )
    wrapper = wrapper.to(device)
    wrapper.eval()
    head = wrapper.model.nets["policy"].heads[EMBODIMENT]
    diffusion = bool(getattr(wrapper.model, "diffusion", False))
    seeds: list[int | None] = list(seeds_arg) if diffusion else [None]
    seed_labels = ["det" if seed is None else str(seed) for seed in seeds]
    expected_t = expected_action_len(wrapper)

    predictions = predict_samples(
        wrapper,
        samples,
        seeds=seeds,
        expected_t=expected_t,
        batch_size=batch_size,
        device=device,
    )

    rows: list[dict] = []
    for sample in samples:
        # Use the full dataset GT chunk for timing diagnostics. MLP checkpoints may
        # predict fewer tokens than FM checkpoints, but GT close/reopen timing
        # should not change across models.
        gt = sample["actions"].detach().cpu().numpy()
        for seed_label in seed_labels:
            pred = predictions[(seed_label, sample["global_idx"])]
            for horizon in HORIZONS:
                rows.append(
                    row_for_sample(
                        model_name=model_name,
                        checkpoint=checkpoint,
                        head_class=head.__class__.__name__,
                        diffusion=diffusion,
                        expected_t=expected_t,
                        seed_label=seed_label,
                        sample=sample,
                        pred=pred,
                        gt=gt,
                        horizon=horizon,
                    )
                )

    metadata = {
        "model": model_name,
        "checkpoint": str(checkpoint),
        "head_class": head.__class__.__name__,
        "diffusion": diffusion,
        "expected_action_len": expected_t,
        "seeds": seed_labels,
    }
    del wrapper
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, metadata


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    model_specs = [parse_model_spec(spec) for spec in args.model]
    for _, checkpoint in model_specs:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    dataset = build_valid_dataset(dataset_root, args.valid_ratio)
    candidates = candidate_metadata(dataset)
    selected_meta = select_phase_samples(candidates, args.max_samples_per_phase)
    selected_samples = [load_sample(dataset, meta) for meta in selected_meta]

    all_rows: list[dict] = []
    model_metadata: list[dict] = []
    for model_name, checkpoint in model_specs:
        rows, metadata = evaluate_model(
            model_name=model_name,
            checkpoint=checkpoint,
            samples=selected_samples,
            seeds_arg=args.seeds,
            batch_size=args.batch_size,
            device=device,
        )
        all_rows.extend(rows)
        model_metadata.append(metadata)

    phase_summary = summarize_rows(all_rows, ("model", "phase", "horizon"))
    diagnostic_phase_summary = summarize_rows(
        all_rows, ("model", "diagnostic_phase", "horizon")
    )
    horizon_summary = summarize_rows(all_rows, ("model", "horizon"))

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = output_dir / f"{args.tag}_rows.csv"
    phase_summary_csv = output_dir / f"{args.tag}_phase_summary.csv"
    diagnostic_phase_summary_csv = (
        output_dir / f"{args.tag}_diagnostic_phase_summary.csv"
    )
    horizon_summary_csv = output_dir / f"{args.tag}_horizon_summary.csv"
    summary_json = output_dir / f"{args.tag}_summary.json"

    write_csv(rows_csv, all_rows)
    write_csv(phase_summary_csv, phase_summary)
    write_csv(diagnostic_phase_summary_csv, diagnostic_phase_summary)
    write_csv(horizon_summary_csv, horizon_summary)

    summary = {
        "dataset_root": str(dataset_root),
        "valid_ratio": args.valid_ratio,
        "valid_len": len(dataset),
        "candidate_phase_counts": dict(Counter(item["phase"] for item in candidates)),
        "selected_count": len(selected_samples),
        "selected_phase_counts": dict(Counter(item["phase"] for item in selected_meta)),
        "models": model_metadata,
        "horizons": HORIZONS,
        "outputs": {
            "rows": str(rows_csv),
            "phase_summary": str(phase_summary_csv),
            "diagnostic_phase_summary": str(diagnostic_phase_summary_csv),
            "horizon_summary": str(horizon_summary_csv),
        },
        "phase_summary": phase_summary,
        "diagnostic_phase_summary": diagnostic_phase_summary,
        "horizon_summary": horizon_summary,
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"OUT_JSON {summary_json}")


if __name__ == "__main__":
    main()
