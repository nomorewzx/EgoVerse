from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
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


ACTION_KEY = "so100_singlearm_actions_cartesian"
EMBODIMENT = "so100_singlearm"
EMBODIMENT_ID = get_embodiment_id(EMBODIMENT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe SO100 validation closed-hold gripper states for release-like "
            "predictions before real robot rollout."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument(
        "--dataset-root",
        default="/home/zxwang/so100-ee-cam-egoverse-zarr",
        help="SO100 zarr root used by the train config.",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--close-threshold", type=float, default=30.0)
    parser.add_argument("--hold-tolerance", type=float, default=3.0)
    parser.add_argument("--last-tolerance", type=float, default=2.0)
    parser.add_argument("--release-threshold", type=float, default=8.0)
    parser.add_argument(
        "--output-dir",
        default="logs/so100_hpt/val_gripper_probe",
        help="Directory for JSON/CSV outputs.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional output tag. Defaults to checkpoint parent directory name.",
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


def sample_gripper(sample: dict) -> tuple[float, np.ndarray]:
    state_g = float(sample["observations.state.ee_pose"][6])
    gt_gripper = sample["actions_cartesian"][:, 6].detach().cpu().numpy().astype(float)
    return state_g, gt_gripper


def select_closed_hold_samples(
    dataset: MultiDataset,
    *,
    max_samples: int,
    close_threshold: float,
    hold_tolerance: float,
    last_tolerance: float,
) -> list[dict]:
    candidates: list[dict] = []
    for global_idx in range(len(dataset)):
        sample = dataset[global_idx]
        state_g, gt_gripper = sample_gripper(sample)
        if (
            state_g >= close_threshold
            and float(gt_gripper.min()) >= state_g - hold_tolerance
            and float(gt_gripper[-1]) >= state_g - last_tolerance
        ):
            episode, local_idx = dataset.index_map[global_idx]
            candidates.append(
                {
                    "global_idx": global_idx,
                    "episode": episode,
                    "local_idx": local_idx,
                    "front": sample["observations.images.front_img_1"].detach().cpu(),
                    "state": sample["observations.state.ee_pose"].detach().cpu(),
                    "actions": sample["actions_cartesian"].detach().cpu(),
                    "state_g": state_g,
                    "gt_min": float(gt_gripper.min()),
                    "gt_last": float(gt_gripper[-1]),
                }
            )

    if len(candidates) <= max_samples:
        return candidates

    # Deterministic spread over the full candidate list while preserving valid split order.
    positions = np.linspace(0, len(candidates) - 1, num=max_samples, dtype=int)
    return [candidates[int(i)] for i in positions]


def expected_action_len(wrapper: ModelWrapper) -> int:
    stats = wrapper.model.data_schematic.norm_stats[EMBODIMENT_ID]["actions_cartesian"]
    for key in ("quantile_1", "mean", "min"):
        if key in stats:
            arr = np.asarray(stats[key])
            if arr.ndim >= 2:
                return int(arr.shape[0])
    return So100SingleArm.ACTION_CHUNK_LENGTH


def batched_tensor(key: str, batch_samples: list[dict], expected_t: int | None = None):
    tensor = torch.stack([sample[key] for sample in batch_samples], dim=0)
    if key == "actions" and expected_t is not None:
        tensor = tensor[:, :expected_t, :]
    return tensor


def max_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def sequence_release_metrics(
    gripper: np.ndarray,
    state_g: float,
    indices: list[int],
    release_threshold: float,
) -> dict:
    selected_indices = np.asarray(indices, dtype=int)
    selected = gripper[selected_indices].astype(float)
    drop = state_g - selected
    excess = np.maximum(drop - release_threshold, 0.0)
    release_mask = drop > release_threshold
    release_positions = np.flatnonzero(release_mask)
    tail_start = max(0, len(selected_indices) - max(1, int(np.ceil(len(selected_indices) / 3))))
    tail_drop = drop[tail_start:]
    tail_excess = excess[tail_start:]
    tail_release_mask = release_mask[tail_start:]

    first_release_rank = int(release_positions[0]) if len(release_positions) else -1
    first_release_index = (
        int(selected_indices[first_release_rank]) if first_release_rank >= 0 else -1
    )
    return {
        "selected_len": int(len(selected_indices)),
        "first_release_index": first_release_index,
        "first_release_rank": first_release_rank,
        "release_count": int(release_mask.sum()),
        "release_frac": float(release_mask.mean()),
        "release_max_run": max_true_run(release_mask),
        "release_area": float(excess.sum()),
        "release_area_norm": float(excess.mean()),
        "tail_drop_mean": float(tail_drop.mean()),
        "tail_excess_mean": float(tail_excess.mean()),
        "tail_release_frac": float(tail_release_mask.mean()),
    }


def add_prefixed(row: dict, prefix: str, metrics: dict) -> None:
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def value_stats(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def unique_count_for_mask(rows: list[dict], mask: np.ndarray) -> int:
    return int(
        len({row["global_idx"] for row, keep in zip(rows, mask, strict=True) if keep})
    )


def first_release_stats(rows: list[dict], prefix: str) -> dict:
    indices = np.asarray([row[f"{prefix}_first_release_index"] for row in rows], dtype=int)
    valid = indices >= 0
    if not np.any(valid):
        return {
            "count": 0,
            "unique_samples": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    values = indices[valid].astype(float)
    return {
        "count": int(valid.sum()),
        "unique_samples": unique_count_for_mask(rows, valid),
        "min": int(values.min()),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "max": int(values.max()),
    }


def summarize_rows(rows: list[dict], release_threshold: float) -> dict:
    summary = {
        "n": len(rows),
        "unique_samples": len({r["global_idx"] for r in rows}),
    }
    for key in ("drop_all", "drop_qf33", "drop_qf99_stride3"):
        values = np.asarray([r[key] for r in rows], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()),
            "count_drop_gt5": int(np.sum(values > 5.0)),
            "count_drop_gt_release_threshold": int(np.sum(values > release_threshold)),
            "count_drop_gt15": int(np.sum(values > 15.0)),
            "unique_drop_gt_release_threshold": int(
                len(
                    {
                        r["global_idx"]
                        for r, value in zip(rows, values, strict=True)
                        if value > release_threshold
                    }
                )
            ),
        }
    for prefix in ("all", "qf33", "qf99_stride3"):
        release_count = np.asarray(
            [r[f"{prefix}_release_count"] for r in rows], dtype=float
        )
        max_run = np.asarray([r[f"{prefix}_release_max_run"] for r in rows], dtype=float)
        release_area = np.asarray(
            [r[f"{prefix}_release_area"] for r in rows], dtype=float
        )
        release_area_norm = np.asarray(
            [r[f"{prefix}_release_area_norm"] for r in rows], dtype=float
        )
        tail_drop_mean = np.asarray(
            [r[f"{prefix}_tail_drop_mean"] for r in rows], dtype=float
        )
        tail_excess_mean = np.asarray(
            [r[f"{prefix}_tail_excess_mean"] for r in rows], dtype=float
        )
        tail_release_frac = np.asarray(
            [r[f"{prefix}_tail_release_frac"] for r in rows], dtype=float
        )
        sustained_mask = max_run >= 3
        tail_collapse_mask = tail_drop_mean > release_threshold
        any_release_mask = release_count > 0
        summary[f"{prefix}_sequence"] = {
            "any_release_count": int(any_release_mask.sum()),
            "any_release_unique": unique_count_for_mask(rows, any_release_mask),
            "sustained_run_ge3_count": int(sustained_mask.sum()),
            "sustained_run_ge3_unique": unique_count_for_mask(rows, sustained_mask),
            "tail_mean_drop_gt_release_threshold_count": int(
                tail_collapse_mask.sum()
            ),
            "tail_mean_drop_gt_release_threshold_unique": unique_count_for_mask(
                rows, tail_collapse_mask
            ),
            "first_release_index": first_release_stats(rows, prefix),
            "release_max_run": value_stats(max_run),
            "release_area": value_stats(release_area),
            "release_area_norm": value_stats(release_area_norm),
            "tail_drop_mean": value_stats(tail_drop_mean),
            "tail_excess_mean": value_stats(tail_excess_mean),
            "tail_release_frac": value_stats(tail_release_frac),
        }
    return summary


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    dataset_root = Path(args.dataset_root).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if requested_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset = build_valid_dataset(dataset_root, args.valid_ratio)
    samples = select_closed_hold_samples(
        dataset,
        max_samples=args.max_samples,
        close_threshold=args.close_threshold,
        hold_tolerance=args.hold_tolerance,
        last_tolerance=args.last_tolerance,
    )
    if not samples:
        raise RuntimeError("No closed-hold validation samples matched the probe criteria.")

    wrapper = ModelWrapper.load_from_checkpoint(
        str(checkpoint), weights_only=False, map_location="cpu"
    )
    wrapper = wrapper.to(requested_device)
    wrapper.eval()
    wrapper.model.device = requested_device
    head = wrapper.model.nets["policy"].heads[EMBODIMENT]
    diffusion = bool(getattr(wrapper.model, "diffusion", False))
    seeds = args.seeds if diffusion else [None]
    expected_t = expected_action_len(wrapper)

    rows: list[dict] = []
    for seed in seeds:
        seed_label = "det" if seed is None else str(seed)
        for start in range(0, len(samples), args.batch_size):
            batch_samples = samples[start : start + args.batch_size]
            if seed is not None:
                torch.manual_seed(seed)
                if requested_device.type == "cuda":
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
                gripper = pred[i, :, 6].astype(float)
                state_g = float(sample["state_g"])
                qf33_indices = [idx for idx in range(1, 34) if idx < len(gripper)]
                qf99_stride3_indices = [
                    idx for idx in (1 + 3 * k for k in range(99)) if idx < len(gripper)
                ]
                row = {
                    "global_idx": sample["global_idx"],
                    "episode": sample["episode"],
                    "local_idx": sample["local_idx"],
                    "seed": seed_label,
                    "state_g": state_g,
                    "gt_min": sample["gt_min"],
                    "gt_last": sample["gt_last"],
                    "pred_len": int(len(gripper)),
                    "pred_first": float(gripper[0]),
                    "pred_last": float(gripper[-1]),
                    "pred_min_all": float(gripper.min()),
                    "pred_min_qf33": float(gripper[qf33_indices].min()),
                    "pred_min_qf99_stride3": float(
                        gripper[qf99_stride3_indices].min()
                    ),
                    "drop_all": float(state_g - gripper.min()),
                    "drop_qf33": float(state_g - gripper[qf33_indices].min()),
                    "drop_qf99_stride3": float(
                        state_g - gripper[qf99_stride3_indices].min()
                    ),
                }
                add_prefixed(
                    row,
                    "all",
                    sequence_release_metrics(
                        gripper,
                        state_g,
                        list(range(len(gripper))),
                        args.release_threshold,
                    ),
                )
                add_prefixed(
                    row,
                    "qf33",
                    sequence_release_metrics(
                        gripper,
                        state_g,
                        qf33_indices,
                        args.release_threshold,
                    ),
                )
                add_prefixed(
                    row,
                    "qf99_stride3",
                    sequence_release_metrics(
                        gripper,
                        state_g,
                        qf99_stride3_indices,
                        args.release_threshold,
                    ),
                )
                rows.append(row)

    tag = args.tag or checkpoint.parent.parent.name
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"{tag}_so100_val_gripper_release_probe.json"
    out_csv = output_dir / f"{tag}_so100_val_gripper_release_probe.csv"

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "valid_ratio": args.valid_ratio,
        "valid_episodes": sorted(dataset.datasets.keys()),
        "selected_sample_count": len(samples),
        "selected_episode_counts": dict(Counter(s["episode"] for s in samples)),
        "head_class": head.__class__.__name__,
        "diffusion": diffusion,
        "expected_action_len": expected_t,
        "seeds": seeds,
        "closed_hold_criteria": {
            "close_threshold": args.close_threshold,
            "hold_tolerance": args.hold_tolerance,
            "last_tolerance": args.last_tolerance,
        },
        "release_threshold": args.release_threshold,
        "summary": summarize_rows(rows, args.release_threshold),
        "output_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"OUT_JSON {out_json}")
    print(f"OUT_CSV {out_csv}")


if __name__ == "__main__":
    main()
