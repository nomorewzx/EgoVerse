from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.so100 import So100SingleArm
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
from egomimic.utils.egomimicUtils import interpolate_arr_euler


ACTION_KEY = "so100_singlearm_actions_cartesian"
EMBODIMENT = "so100_singlearm"
EMBODIMENT_ID = get_embodiment_id(EMBODIMENT)
ACTION_DIMS = ("x", "y", "z", "yaw", "pitch", "roll", "gripper")
ROLLOUT_RESAMPLED_ACTION_LEN = 45
ROLLOUT_QUERY_FREQUENCY = 30
LEGACY_ROLLOUT_START_INDEX = 1
LEGACY_ROLLOUT_STRIDE = 3


def parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Model spec must be NAME=PATH, got: {spec}")
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Model spec has empty name: {spec}")
    return name, Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SO100 validation action quality across checkpoints. "
            "Reports per-dim error, phase error, horizon bucket error, FM seed "
            "variance, and receding-horizon consistency."
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
    parser.add_argument("--max-samples", type=int, default=768)
    parser.add_argument("--max-consistency-pairs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--output-dir",
        default="logs/so100_hpt/val_action_quality_probe",
    )
    parser.add_argument(
        "--tag",
        default="so100_action_quality",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def build_valid_dataset(dataset_root: Path, valid_ratio: float) -> MultiDataset:
    resolver = LocalEpisodeResolver(
        dataset_root,
        key_map=So100SingleArm.get_keymap(mode="camera_frame_ypr"),
        transform_list=So100SingleArm.get_transform_list(
            mode="camera_frame_ypr", chunk_length=100
        ),
    )
    filters = DatasetFilter(
        filter_lambdas=["lambda row: row.get('embodiment') == 'so100_singlearm'"]
    )
    return MultiDataset._from_resolver(
        resolver, filters=filters, mode="valid", valid_ratio=valid_ratio
    )


def expected_action_len(wrapper: ModelWrapper) -> int:
    stats = wrapper.model.data_schematic.norm_stats[EMBODIMENT_ID]["actions_cartesian"]
    for key in ("quantile_1", "mean", "min"):
        if key in stats:
            arr = np.asarray(stats[key])
            if arr.ndim >= 2:
                return int(arr.shape[0])
    return So100SingleArm.ACTION_CHUNK_LENGTH


def classify_phase(state: np.ndarray, actions: np.ndarray) -> str:
    state_g = float(state[6])
    gt_g = actions[:, 6].astype(float)
    gt_min = float(gt_g.min())
    gt_max = float(gt_g.max())
    gt_last = float(gt_g[-1])

    if (
        state_g >= 30.0
        and gt_min >= state_g - 3.0
        and gt_last >= state_g - 2.0
    ):
        return "closed_hold"
    if state_g >= 30.0 and gt_min <= state_g - 8.0:
        return "release_place"
    if state_g < 30.0 and gt_max >= max(30.0, state_g + 8.0):
        return "closing_grasp"
    if state_g < 20.0 and gt_max <= state_g + 3.0:
        return "open_hold_approach"
    return "transition"


def candidate_metadata(dataset: MultiDataset) -> list[dict]:
    candidates: list[dict] = []
    for global_idx in range(len(dataset)):
        sample = dataset[global_idx]
        state = sample["observations.state.ee_pose"].detach().cpu().numpy()
        actions = sample["actions_cartesian"].detach().cpu().numpy()
        episode, local_idx = dataset.index_map[global_idx]
        candidates.append(
            {
                "global_idx": int(global_idx),
                "episode": episode,
                "local_idx": int(local_idx),
                "phase": classify_phase(state, actions),
            }
        )
    return candidates


def spread_select(items: list[dict], count: int) -> list[dict]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    positions = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return [items[int(pos)] for pos in positions]


def select_stratified(candidates: list[dict], max_samples: int) -> list[dict]:
    if len(candidates) <= max_samples:
        return list(candidates)

    by_phase: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        by_phase[item["phase"]].append(item)

    phases = sorted(by_phase)
    target = max(1, max_samples // max(1, len(phases)))
    selected: list[dict] = []
    selected_ids: set[int] = set()
    for phase in phases:
        for item in spread_select(by_phase[phase], target):
            if item["global_idx"] not in selected_ids:
                selected.append(item)
                selected_ids.add(item["global_idx"])

    if len(selected) < max_samples:
        remaining = [
            item for item in candidates if item["global_idx"] not in selected_ids
        ]
        for item in spread_select(remaining, max_samples - len(selected)):
            if item["global_idx"] not in selected_ids:
                selected.append(item)
                selected_ids.add(item["global_idx"])

    selected.sort(key=lambda item: item["global_idx"])
    return selected[:max_samples]


def select_consistency_pairs(
    candidates: list[dict],
    *,
    max_pairs: int,
) -> list[tuple[int, int]]:
    by_key = {
        (item["episode"], item["local_idx"]): item["global_idx"] for item in candidates
    }
    pairs: list[tuple[int, int]] = []
    for item in candidates:
        next_idx = by_key.get((item["episode"], item["local_idx"] + 1))
        if next_idx is not None:
            pairs.append((item["global_idx"], int(next_idx)))
    selected = spread_select([{"pair": pair} for pair in pairs], max_pairs)
    return [(int(item["pair"][0]), int(item["pair"][1])) for item in selected]


def load_sample(dataset: MultiDataset, meta: dict) -> dict:
    sample = dataset[meta["global_idx"]]
    return {
        "global_idx": meta["global_idx"],
        "episode": meta["episode"],
        "local_idx": meta["local_idx"],
        "phase": meta["phase"],
        "front": sample["observations.images.front_img_1"].detach().cpu(),
        "state": sample["observations.state.ee_pose"].detach().cpu(),
        "actions": sample["actions_cartesian"].detach().cpu(),
    }


def batched_tensor(key: str, batch_samples: list[dict], expected_t: int | None = None):
    tensor = torch.stack([sample[key] for sample in batch_samples], dim=0)
    if key == "actions" and expected_t is not None:
        tensor = tensor[:, :expected_t, :]
    return tensor


def predict_samples(
    wrapper: ModelWrapper,
    samples: list[dict],
    *,
    seeds: list[int | None],
    expected_t: int,
    batch_size: int,
    device: torch.device,
) -> dict[tuple[str, int], np.ndarray]:
    predictions: dict[tuple[str, int], np.ndarray] = {}
    wrapper.eval()
    wrapper.model.device = device

    for seed in seeds:
        seed_label = "det" if seed is None else str(seed)
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            if seed is not None:
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
            raw_batch = {
                EMBODIMENT: {
                    "observations.images.front_img_1": batched_tensor(
                        "front", batch_samples
                    ),
                    "observations.state.ee_pose": batched_tensor(
                        "state", batch_samples
                    ),
                    "actions_cartesian": batched_tensor(
                        "actions", batch_samples, expected_t
                    ),
                }
            }
            with torch.inference_mode():
                processed = wrapper.model.process_batch_for_training(raw_batch)
                pred = (
                    wrapper.model.forward_eval(processed)[ACTION_KEY]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
            for i, sample in enumerate(batch_samples):
                predictions[(seed_label, sample["global_idx"])] = pred[i]
    return predictions


def horizon_indices(name: str, pred_len: int) -> list[int]:
    if name == "all":
        return list(range(pred_len))
    if name == "early33":
        return [idx for idx in range(1, 34) if idx < pred_len]
    if name == "rollout_stride3":
        return [idx for idx in (1 + 3 * k for k in range(99)) if idx < pred_len]
    if name == "rollout_stride3_exec":
        return list(range(pred_len))
    if name == "rollout_resample45_exec30":
        return list(range(min(ROLLOUT_QUERY_FREQUENCY, pred_len)))
    if name == "common_stride3_64":
        return [idx for idx in (1 + 3 * k for k in range(21)) if idx < pred_len]
    if name == "tail_third":
        start = max(0, pred_len - max(1, int(np.ceil(pred_len / 3))))
        return list(range(start, pred_len))
    if name == "abs_tail66_99":
        return [idx for idx in range(66, min(pred_len, 100))]
    raise ValueError(f"Unknown horizon: {name}")


def resample_chunk(chunk: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(chunk, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"Expected action chunk shape (T, 7), got {arr.shape}")
    if arr.shape[0] == target_len:
        return arr
    return interpolate_arr_euler(arr[None, ...], target_len)[0]


def legacy_stride3_execution_chunk(chunk: np.ndarray) -> np.ndarray:
    arr = np.asarray(chunk, dtype=np.float64)
    indices = [
        idx
        for idx in (
            LEGACY_ROLLOUT_START_INDEX + LEGACY_ROLLOUT_STRIDE * k
            for k in range(99)
        )
        if idx < len(arr)
    ]
    return arr[np.asarray(indices, dtype=int)]


def prepare_action_horizon(
    horizon: str,
    pred: np.ndarray,
    gt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if horizon == "rollout_stride3_exec":
        pred_exec = legacy_stride3_execution_chunk(pred)
        gt_exec = legacy_stride3_execution_chunk(gt)
        exec_len = min(len(pred_exec), len(gt_exec))
        return pred_exec[:exec_len], gt_exec[:exec_len], list(range(exec_len))
    if horizon == "rollout_resample45_exec30":
        pred_exec = resample_chunk(pred, ROLLOUT_RESAMPLED_ACTION_LEN)
        gt_exec = resample_chunk(gt, ROLLOUT_RESAMPLED_ACTION_LEN)
        exec_len = min(ROLLOUT_QUERY_FREQUENCY, len(pred_exec), len(gt_exec))
        return pred_exec, gt_exec, list(range(exec_len))
    indices = horizon_indices(horizon, min(len(pred), len(gt)))
    return pred, gt, indices


def prepare_consistency_horizon(
    horizon: str,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if horizon == "rollout_stride3_exec":
        exec_a = legacy_stride3_execution_chunk(pred_a)
        exec_b = legacy_stride3_execution_chunk(pred_b)
        exec_len = min(len(exec_a), len(exec_b))
        return exec_a[:exec_len], exec_b[:exec_len], list(range(max(0, exec_len - 1)))
    if horizon == "rollout_resample45_exec30":
        exec_a = resample_chunk(pred_a, ROLLOUT_RESAMPLED_ACTION_LEN)
        exec_b = resample_chunk(pred_b, ROLLOUT_RESAMPLED_ACTION_LEN)
        exec_len = min(ROLLOUT_QUERY_FREQUENCY, len(exec_a), len(exec_b))
        return exec_a, exec_b, list(range(max(0, exec_len - 1)))
    indices = horizon_indices(horizon, min(len(pred_a), len(pred_b)))
    return pred_a, pred_b, indices


def prepare_prediction_horizon(horizon: str, pred: np.ndarray) -> tuple[np.ndarray, list[int]]:
    if horizon == "rollout_stride3_exec":
        exec_pred = legacy_stride3_execution_chunk(pred)
        return exec_pred, list(range(len(exec_pred)))
    if horizon == "rollout_resample45_exec30":
        exec_pred = resample_chunk(pred, ROLLOUT_RESAMPLED_ACTION_LEN)
        return exec_pred, list(range(min(ROLLOUT_QUERY_FREQUENCY, len(exec_pred))))
    indices = horizon_indices(horizon, len(pred))
    return pred, indices


def action_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    state: np.ndarray,
    indices: list[int],
) -> dict:
    if not indices:
        return {}
    idx = np.asarray(indices, dtype=int)
    pred_sel = pred[idx]
    gt_sel = gt[idx]
    err = pred_sel - gt_sel
    abs_err = np.abs(err)

    pos_err = np.linalg.norm(err[:, :3], axis=1)
    rot_err = np.linalg.norm(err[:, 3:6], axis=1)
    pred_pos_delta = pred_sel[:, :3] - state[:3]
    gt_pos_delta = gt_sel[:, :3] - state[:3]
    pred_pos_delta_norm = np.linalg.norm(pred_pos_delta, axis=1)
    gt_pos_delta_norm = np.linalg.norm(gt_pos_delta, axis=1)
    denom = np.maximum(gt_pos_delta_norm, 1e-8)
    delta_ratio = pred_pos_delta_norm / denom

    dot = np.sum(pred_pos_delta * gt_pos_delta, axis=1)
    cos_denom = np.maximum(pred_pos_delta_norm * gt_pos_delta_norm, 1e-8)
    cos = dot / cos_denom
    valid_cos = gt_pos_delta_norm > 1e-6

    metrics = {
        "selected_len": int(len(idx)),
        "pos_err_mean": float(pos_err.mean()),
        "pos_err_p95": float(np.quantile(pos_err, 0.95)),
        "pos_err_max": float(pos_err.max()),
        "xy_err_mean": float(np.linalg.norm(err[:, :2], axis=1).mean()),
        "z_abs_err_mean": float(abs_err[:, 2].mean()),
        "rot_err_mean": float(rot_err.mean()),
        "gripper_abs_err_mean": float(abs_err[:, 6].mean()),
        "gripper_bias_mean": float(err[:, 6].mean()),
        "pred_pos_delta_mean": float(pred_pos_delta_norm.mean()),
        "gt_pos_delta_mean": float(gt_pos_delta_norm.mean()),
        "pos_delta_ratio_mean": float(delta_ratio.mean()),
        "pos_delta_ratio_median": float(np.median(delta_ratio)),
        "pred_gripper_delta_mean": float(np.abs(pred_sel[:, 6] - state[6]).mean()),
        "gt_gripper_delta_mean": float(np.abs(gt_sel[:, 6] - state[6]).mean()),
        "dim_x_mae": float(abs_err[:, 0].mean()),
        "dim_y_mae": float(abs_err[:, 1].mean()),
        "dim_z_mae": float(abs_err[:, 2].mean()),
        "dim_yaw_mae": float(abs_err[:, 3].mean()),
        "dim_pitch_mae": float(abs_err[:, 4].mean()),
        "dim_roll_mae": float(abs_err[:, 5].mean()),
        "dim_gripper_mae": float(abs_err[:, 6].mean()),
    }
    metrics["pos_delta_cos_mean"] = (
        float(cos[valid_cos].mean()) if np.any(valid_cos) else float("nan")
    )
    return metrics


def add_prefixed(row: dict, prefix: str, values: dict) -> None:
    for key, value in values.items():
        row[f"{prefix}_{key}"] = value


def summarize_metric(rows: list[dict], key: str) -> dict:
    values = np.asarray(
        [float(row[key]) for row in rows if row.get(key) not in ("", None)],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def summarize_rows(
    rows: list[dict],
    group_keys: tuple[str, ...],
    metric_keys: tuple[str, ...],
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    summary: list[dict] = []
    for group, group_rows in sorted(grouped.items()):
        item = {key: value for key, value in zip(group_keys, group, strict=True)}
        item["n_rows"] = len(group_rows)
        item["unique_samples"] = len({row["global_idx"] for row in group_rows})
        for metric_key in metric_keys:
            stats = summarize_metric(group_rows, metric_key)
            item[f"{metric_key}_mean"] = stats["mean"]
            item[f"{metric_key}_p95"] = stats["p95"]
            item[f"{metric_key}_max"] = stats["max"]
        summary.append(item)
    return summary


def seed_variance_rows(
    model_name: str,
    samples: list[dict],
    predictions: dict[tuple[str, int], np.ndarray],
    seed_labels: list[str],
    expected_t: int,
) -> list[dict]:
    if len(seed_labels) <= 1:
        return []
    rows: list[dict] = []
    for sample in samples:
        preds = [predictions[(seed, sample["global_idx"])] for seed in seed_labels]
        for horizon in HORIZONS:
            prepared_pairs = [prepare_prediction_horizon(horizon, pred) for pred in preds]
            prepared = [pair[0] for pair in prepared_pairs]
            indices = prepared_pairs[0][1]
            if not indices:
                continue
            stack = np.stack(prepared, axis=0)
            std = stack.std(axis=0)
            sel = std[np.asarray(indices, dtype=int)]
            rows.append(
                {
                    "model": model_name,
                    "global_idx": sample["global_idx"],
                    "episode": sample["episode"],
                    "local_idx": sample["local_idx"],
                    "phase": sample["phase"],
                    "horizon": horizon,
                    "pos_std_mean": float(
                        np.linalg.norm(sel[:, :3], axis=1).mean()
                    ),
                    "rot_std_mean": float(
                        np.linalg.norm(sel[:, 3:6], axis=1).mean()
                    ),
                    "gripper_std_mean": float(sel[:, 6].mean()),
                    "z_std_mean": float(sel[:, 2].mean()),
                }
            )
    return rows


def consistency_metrics(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    indices: list[int],
) -> dict:
    overlap = [idx for idx in indices if idx + 1 < len(pred_a) and idx < len(pred_b)]
    if not overlap:
        return {}
    a = pred_a[np.asarray(overlap, dtype=int) + 1]
    b = pred_b[np.asarray(overlap, dtype=int)]
    err = a - b
    return {
        "selected_len": int(len(overlap)),
        "pos_consistency_mean": float(np.linalg.norm(err[:, :3], axis=1).mean()),
        "rot_consistency_mean": float(np.linalg.norm(err[:, 3:6], axis=1).mean()),
        "gripper_consistency_mean": float(np.abs(err[:, 6]).mean()),
        "z_consistency_mean": float(np.abs(err[:, 2]).mean()),
    }


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


HORIZONS = (
    "all",
    "early33",
    "common_stride3_64",
    "rollout_stride3",
    "rollout_stride3_exec",
    "rollout_resample45_exec30",
    "tail_third",
    "abs_tail66_99",
)


def evaluate_model(
    *,
    model_name: str,
    checkpoint: Path,
    samples: list[dict],
    consistency_samples: list[dict],
    consistency_pairs: list[tuple[int, int]],
    seeds_arg: list[int],
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict], dict]:
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

    combined_by_idx = {sample["global_idx"]: sample for sample in samples}
    for sample in consistency_samples:
        combined_by_idx.setdefault(sample["global_idx"], sample)
    combined_samples = list(combined_by_idx.values())

    predictions = predict_samples(
        wrapper,
        combined_samples,
        seeds=seeds,
        expected_t=expected_t,
        batch_size=batch_size,
        device=device,
    )

    action_rows: list[dict] = []
    for sample in samples:
        gt = sample["actions"].detach().cpu().numpy()[:expected_t]
        state = sample["state"].detach().cpu().numpy()
        for seed_label in seed_labels:
            pred = predictions[(seed_label, sample["global_idx"])]
            for horizon in HORIZONS:
                pred_h, gt_h, indices = prepare_action_horizon(horizon, pred, gt)
                metrics = action_metrics(pred_h, gt_h, state, indices)
                if not metrics:
                    continue
                row = {
                    "model": model_name,
                    "checkpoint": str(checkpoint),
                    "head_class": head.__class__.__name__,
                    "diffusion": diffusion,
                    "seed": seed_label,
                    "global_idx": sample["global_idx"],
                    "episode": sample["episode"],
                    "local_idx": sample["local_idx"],
                    "phase": sample["phase"],
                    "horizon": horizon,
                    "expected_action_len": expected_t,
                }
                row.update(metrics)
                action_rows.append(row)

    variance_rows = seed_variance_rows(
        model_name,
        samples,
        predictions,
        seed_labels,
        expected_t,
    )

    sample_by_idx = {sample["global_idx"]: sample for sample in consistency_samples}
    consistency_rows: list[dict] = []
    for seed_label in seed_labels:
        for idx_a, idx_b in consistency_pairs:
            if idx_a not in sample_by_idx or idx_b not in sample_by_idx:
                continue
            pred_a = predictions[(seed_label, idx_a)]
            pred_b = predictions[(seed_label, idx_b)]
            sample_a = sample_by_idx[idx_a]
            for horizon in HORIZONS:
                pred_a_h, pred_b_h, indices = prepare_consistency_horizon(
                    horizon, pred_a, pred_b
                )
                metrics = consistency_metrics(pred_a_h, pred_b_h, indices)
                if not metrics:
                    continue
                row = {
                    "model": model_name,
                    "checkpoint": str(checkpoint),
                    "head_class": head.__class__.__name__,
                    "diffusion": diffusion,
                    "seed": seed_label,
                    "global_idx": idx_a,
                    "next_global_idx": idx_b,
                    "episode": sample_a["episode"],
                    "local_idx": sample_a["local_idx"],
                    "phase": sample_a["phase"],
                    "horizon": horizon,
                    "expected_action_len": expected_t,
                }
                row.update(metrics)
                consistency_rows.append(row)

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
    return action_rows, variance_rows, consistency_rows, metadata


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    model_specs = [parse_model_spec(spec) for spec in args.model]
    for _, path in model_specs:
        if not path.exists():
            raise FileNotFoundError(path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset = build_valid_dataset(dataset_root, args.valid_ratio)
    candidates = candidate_metadata(dataset)
    selected_meta = select_stratified(candidates, args.max_samples)
    selected_samples = [load_sample(dataset, meta) for meta in selected_meta]

    consistency_pairs = select_consistency_pairs(
        candidates,
        max_pairs=args.max_consistency_pairs,
    )
    consistency_indices = sorted({idx for pair in consistency_pairs for idx in pair})
    meta_by_idx = {item["global_idx"]: item for item in candidates}
    consistency_samples = [
        load_sample(dataset, meta_by_idx[idx]) for idx in consistency_indices
    ]

    all_action_rows: list[dict] = []
    all_variance_rows: list[dict] = []
    all_consistency_rows: list[dict] = []
    model_metadata: list[dict] = []
    for model_name, checkpoint in model_specs:
        action_rows, variance_rows, consistency_rows, metadata = evaluate_model(
            model_name=model_name,
            checkpoint=checkpoint,
            samples=selected_samples,
            consistency_samples=consistency_samples,
            consistency_pairs=consistency_pairs,
            seeds_arg=args.seeds,
            batch_size=args.batch_size,
            device=device,
        )
        all_action_rows.extend(action_rows)
        all_variance_rows.extend(variance_rows)
        all_consistency_rows.extend(consistency_rows)
        model_metadata.append(metadata)

    action_metric_keys = (
        "pos_err_mean",
        "pos_err_p95",
        "z_abs_err_mean",
        "rot_err_mean",
        "gripper_abs_err_mean",
        "gripper_bias_mean",
        "pred_pos_delta_mean",
        "gt_pos_delta_mean",
        "pos_delta_ratio_mean",
        "pos_delta_cos_mean",
        "pred_gripper_delta_mean",
        "gt_gripper_delta_mean",
    )
    consistency_metric_keys = (
        "pos_consistency_mean",
        "rot_consistency_mean",
        "gripper_consistency_mean",
        "z_consistency_mean",
    )
    variance_metric_keys = (
        "pos_std_mean",
        "rot_std_mean",
        "gripper_std_mean",
        "z_std_mean",
    )
    action_summary = summarize_rows(
        all_action_rows,
        ("model", "horizon"),
        action_metric_keys,
    )
    phase_summary = summarize_rows(
        all_action_rows,
        ("model", "phase", "horizon"),
        action_metric_keys,
    )
    consistency_summary = summarize_rows(
        all_consistency_rows,
        ("model", "horizon"),
        consistency_metric_keys,
    )
    variance_summary = summarize_rows(
        all_variance_rows,
        ("model", "horizon"),
        variance_metric_keys,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    action_csv = output_dir / f"{args.tag}_action_rows.csv"
    variance_csv = output_dir / f"{args.tag}_seed_variance_rows.csv"
    consistency_csv = output_dir / f"{args.tag}_consistency_rows.csv"
    summary_json = output_dir / f"{args.tag}_summary.json"
    action_summary_csv = output_dir / f"{args.tag}_action_summary.csv"
    phase_summary_csv = output_dir / f"{args.tag}_phase_summary.csv"
    consistency_summary_csv = output_dir / f"{args.tag}_consistency_summary.csv"
    variance_summary_csv = output_dir / f"{args.tag}_seed_variance_summary.csv"

    write_csv(action_csv, all_action_rows)
    write_csv(variance_csv, all_variance_rows)
    write_csv(consistency_csv, all_consistency_rows)
    write_csv(action_summary_csv, action_summary)
    write_csv(phase_summary_csv, phase_summary)
    write_csv(consistency_summary_csv, consistency_summary)
    write_csv(variance_summary_csv, variance_summary)

    summary = {
        "dataset_root": str(dataset_root),
        "valid_ratio": args.valid_ratio,
        "valid_len": len(dataset),
        "candidate_phase_counts": dict(Counter(item["phase"] for item in candidates)),
        "selected_count": len(selected_samples),
        "selected_phase_counts": dict(Counter(s["phase"] for s in selected_samples)),
        "consistency_pair_count": len(consistency_pairs),
        "models": model_metadata,
        "horizons": HORIZONS,
        "outputs": {
            "action_rows": str(action_csv),
            "seed_variance_rows": str(variance_csv),
            "consistency_rows": str(consistency_csv),
            "action_summary": str(action_summary_csv),
            "phase_summary": str(phase_summary_csv),
            "consistency_summary": str(consistency_summary_csv),
            "seed_variance_summary": str(variance_summary_csv),
        },
        "action_summary": action_summary,
        "phase_summary": phase_summary,
        "consistency_summary": consistency_summary,
        "seed_variance_summary": variance_summary,
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"OUT_JSON {summary_json}")


if __name__ == "__main__":
    main()
