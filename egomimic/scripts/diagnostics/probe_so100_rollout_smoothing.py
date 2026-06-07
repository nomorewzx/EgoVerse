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


def parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Model spec must be NAME=PATH, got: {spec}")
    name, path = spec.split("=", 1)
    return name.strip(), Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline validation probe for rollout-time smoothing strategies. "
            "Compares direct re-query, overlap blending, best-of-K sampling, "
            "mean-of-K sampling, and a simple temporal ensemble."
        )
    )
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument(
        "--dataset-root",
        default="/home/zxwang/so100-ee-cam-egoverse-zarr",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--query-gap", type=int, default=30)
    parser.add_argument("--resampled-action-len", type=int, default=45)
    parser.add_argument("--execute-len", type=int, default=30)
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--output-dir",
        default="logs/so100_hpt/val_rollout_smoothing_probe",
    )
    parser.add_argument("--tag", default="so100_rollout_smoothing")
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


def candidate_metadata(dataset: MultiDataset) -> list[dict]:
    candidates = []
    for global_idx in range(len(dataset)):
        episode, local_idx = dataset.index_map[global_idx]
        sample = dataset[global_idx]
        state = sample["observations.state.ee_pose"].detach().cpu().numpy()
        actions = sample["actions_cartesian"].detach().cpu().numpy()
        candidates.append(
            {
                "global_idx": int(global_idx),
                "episode": episode,
                "local_idx": int(local_idx),
                "phase": classify_phase(state, actions),
            }
        )
    return candidates


def classify_phase(state: np.ndarray, actions: np.ndarray) -> str:
    state_g = float(state[6])
    gt_g = actions[:, 6].astype(float)
    gt_min = float(gt_g.min())
    gt_max = float(gt_g.max())
    gt_last = float(gt_g[-1])

    if state_g >= 30.0 and gt_min >= state_g - 3.0 and gt_last >= state_g - 2.0:
        return "closed_hold"
    if state_g >= 30.0 and gt_min <= state_g - 8.0:
        return "release_place"
    if state_g < 30.0 and gt_max >= max(30.0, state_g + 8.0):
        return "closing_grasp"
    if state_g < 20.0 and gt_max <= state_g + 3.0:
        return "open_hold_approach"
    return "transition"


def spread_select(items: list, count: int) -> list:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    positions = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return [items[int(pos)] for pos in positions]


def select_query_gap_pairs(candidates: list[dict], query_gap: int, max_pairs: int):
    by_key = {
        (item["episode"], item["local_idx"]): item["global_idx"] for item in candidates
    }
    pairs = []
    for item in candidates:
        next_idx = by_key.get((item["episode"], item["local_idx"] + query_gap))
        if next_idx is not None:
            pairs.append((item["global_idx"], int(next_idx)))
    return spread_select(pairs, max_pairs)


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


def resample_chunk(chunk: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(chunk, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"Expected action chunk shape (T, 7), got {arr.shape}")
    if arr.shape[0] == target_len:
        return arr
    return interpolate_arr_euler(arr[None, ...], target_len)[0]


def apply_blend(
    old_chunk: np.ndarray,
    new_chunk: np.ndarray,
    *,
    execute_len: int,
    blend_steps: int,
    pose_only: bool = False,
) -> np.ndarray:
    out = new_chunk.copy()
    old_tail = old_chunk[execute_len:]
    n = min(blend_steps, len(old_tail), len(out))
    if n <= 0:
        return out
    dims = slice(0, 6) if pose_only else slice(None)
    for i in range(n):
        alpha = float(i + 1) / float(n)
        out[i, dims] = alpha * new_chunk[i, dims] + (1.0 - alpha) * old_tail[i, dims]
    return out


def apply_temporal_ensemble_equal(
    old_chunk: np.ndarray,
    new_chunk: np.ndarray,
    *,
    execute_len: int,
    pose_only: bool = False,
) -> np.ndarray:
    out = new_chunk.copy()
    old_tail = old_chunk[execute_len:]
    n = min(len(old_tail), len(out))
    if n <= 0:
        return out
    dims = slice(0, 6) if pose_only else slice(None)
    out[:n, dims] = 0.5 * old_tail[:n, dims] + 0.5 * new_chunk[:n, dims]
    return out


def overlap_score(
    old_chunk: np.ndarray,
    new_chunk: np.ndarray,
    *,
    execute_len: int,
) -> float:
    old_tail = old_chunk[execute_len:]
    n = min(len(old_tail), len(new_chunk), 15)
    if n <= 0:
        return float("inf")
    err = new_chunk[:n] - old_tail[:n]
    pos = np.linalg.norm(err[:, :3], axis=1) / 0.01
    rot = np.linalg.norm(err[:, 3:6], axis=1) / 0.1
    grip = np.abs(err[:, 6]) / 5.0
    return float(np.mean(pos + rot + grip))


def choose_best_candidate(
    old_chunk: np.ndarray,
    candidates: list[np.ndarray],
    *,
    execute_len: int,
) -> tuple[np.ndarray, int]:
    scores = [
        overlap_score(old_chunk, candidate, execute_len=execute_len)
        for candidate in candidates
    ]
    best = int(np.argmin(scores))
    return candidates[best], best


def command_metrics(
    *,
    old_chunk: np.ndarray,
    new_exec: np.ndarray,
    gt_exec: np.ndarray,
    state: np.ndarray,
    execute_len: int,
) -> dict:
    n = min(execute_len, len(new_exec), len(gt_exec))
    new_exec = new_exec[:n]
    gt_exec = gt_exec[:n]
    old_last = old_chunk[execute_len - 1]
    old_next = old_chunk[execute_len]
    old_tail = old_chunk[execute_len : execute_len + n]
    overlap_n = min(len(old_tail), len(new_exec))

    boundary_command = new_exec[0] - old_last
    boundary_replan = new_exec[0] - old_next
    overlap_err = new_exec[:overlap_n] - old_tail[:overlap_n]
    gt_err = new_exec - gt_exec
    step_delta = np.diff(np.vstack([old_last[None, :], new_exec]), axis=0)

    state_g = float(state[6])
    gt_g = gt_exec[:, 6]
    closed_hold = bool(state_g >= 30.0 and float(np.min(gt_g)) >= state_g - 3.0)
    release_drop = state_g - new_exec[:, 6]
    release_mask = release_drop > 8.0

    return {
        "boundary_pos_m": float(np.linalg.norm(boundary_command[:3])),
        "boundary_rot": float(np.linalg.norm(boundary_command[3:6])),
        "boundary_z_m": float(abs(boundary_command[2])),
        "boundary_gripper": float(abs(boundary_command[6])),
        "replan_pos_m": float(np.linalg.norm(boundary_replan[:3])),
        "replan_rot": float(np.linalg.norm(boundary_replan[3:6])),
        "replan_z_m": float(abs(boundary_replan[2])),
        "replan_gripper": float(abs(boundary_replan[6])),
        "overlap_pos_m": float(np.linalg.norm(overlap_err[:, :3], axis=1).mean()),
        "overlap_rot": float(np.linalg.norm(overlap_err[:, 3:6], axis=1).mean()),
        "overlap_z_m": float(np.abs(overlap_err[:, 2]).mean()),
        "overlap_gripper": float(np.abs(overlap_err[:, 6]).mean()),
        "gt_pos_m": float(np.linalg.norm(gt_err[:, :3], axis=1).mean()),
        "gt_rot": float(np.linalg.norm(gt_err[:, 3:6], axis=1).mean()),
        "gt_z_m": float(np.abs(gt_err[:, 2]).mean()),
        "gt_gripper": float(np.abs(gt_err[:, 6]).mean()),
        "step_pos_m_p95": float(
            np.quantile(np.linalg.norm(step_delta[:, :3], axis=1), 0.95)
        ),
        "step_rot_p95": float(
            np.quantile(np.linalg.norm(step_delta[:, 3:6], axis=1), 0.95)
        ),
        "closed_hold": closed_hold,
        "closed_hold_any_release": bool(closed_hold and np.any(release_mask)),
        "closed_hold_release_frac": (
            float(np.mean(release_mask)) if closed_hold else float("nan")
        ),
        "closed_hold_release_max_drop": (
            float(np.max(release_drop)) if closed_hold else float("nan")
        ),
    }


def strategy_chunks(
    *,
    old_chunk: np.ndarray,
    new_chunks: list[np.ndarray],
    execute_len: int,
) -> list[tuple[str, np.ndarray, str]]:
    direct = new_chunks[0]
    strategies: list[tuple[str, np.ndarray, str]] = [
        ("direct_seed0", direct, "seed0"),
        ("blend5_linear", apply_blend(old_chunk, direct, execute_len=execute_len, blend_steps=5), "seed0"),
        ("blend10_linear", apply_blend(old_chunk, direct, execute_len=execute_len, blend_steps=10), "seed0"),
        (
            "blend5_pose_only",
            apply_blend(
                old_chunk,
                direct,
                execute_len=execute_len,
                blend_steps=5,
                pose_only=True,
            ),
            "seed0",
        ),
        (
            "blend10_pose_only",
            apply_blend(
                old_chunk,
                direct,
                execute_len=execute_len,
                blend_steps=10,
                pose_only=True,
            ),
            "seed0",
        ),
        (
            "temporal_ensemble_equal15",
            apply_temporal_ensemble_equal(old_chunk, direct, execute_len=execute_len),
            "seed0",
        ),
        (
            "temporal_ensemble_pose_only15",
            apply_temporal_ensemble_equal(
                old_chunk, direct, execute_len=execute_len, pose_only=True
            ),
            "seed0",
        ),
    ]
    if len(new_chunks) > 1:
        mean_chunk = np.mean(np.stack(new_chunks, axis=0), axis=0)
        best_chunk, best_idx = choose_best_candidate(
            old_chunk, new_chunks, execute_len=execute_len
        )
        strategies.extend(
            [
                ("sample_mean_k", mean_chunk, "all"),
                ("best_of_k_pose", best_chunk, str(best_idx)),
            ]
        )
    return strategies


def value_stats(values: list[float], scale: float = 1.0) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)] * scale
    if arr.size == 0:
        return {"mean": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.5)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["strategy"])].append(row)
    metric_specs = [
        ("boundary_pos_m", 1000.0),
        ("boundary_rot", 1.0),
        ("boundary_z_m", 1000.0),
        ("boundary_gripper", 1.0),
        ("replan_pos_m", 1000.0),
        ("replan_rot", 1.0),
        ("overlap_pos_m", 1000.0),
        ("overlap_rot", 1.0),
        ("gt_pos_m", 1000.0),
        ("gt_rot", 1.0),
        ("gt_gripper", 1.0),
        ("step_pos_m_p95", 1000.0),
        ("step_rot_p95", 1.0),
    ]
    out = []
    for (model, strategy), group_rows in sorted(grouped.items()):
        item = {
            "model": model,
            "strategy": strategy,
            "n_rows": len(group_rows),
            "closed_hold_rows": int(sum(bool(r["closed_hold"]) for r in group_rows)),
            "closed_hold_any_release_count": int(
                sum(bool(r["closed_hold_any_release"]) for r in group_rows)
            ),
        }
        for metric, scale in metric_specs:
            stats = value_stats([float(r[metric]) for r in group_rows], scale=scale)
            for key, value in stats.items():
                item[f"{metric}_{key}"] = value
        out.append(item)
    return out


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
    pairs: list[tuple[int, int]],
    seeds_arg: list[int],
    batch_size: int,
    device: torch.device,
    resampled_action_len: int,
    execute_len: int,
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

    sample_by_idx = {sample["global_idx"]: sample for sample in samples}
    rows = []
    base_seed = seed_labels[0]
    for idx_old, idx_new in pairs:
        sample_old = sample_by_idx[idx_old]
        sample_new = sample_by_idx[idx_new]
        old_pred = predictions[(base_seed, idx_old)]
        old_chunk = resample_chunk(old_pred, resampled_action_len)
        new_chunks = [
            resample_chunk(predictions[(seed_label, idx_new)], resampled_action_len)
            for seed_label in seed_labels
        ]
        gt_new = (
            sample_new["actions"].detach().cpu().numpy()[:expected_t].astype(np.float64)
        )
        gt_chunk = resample_chunk(gt_new, resampled_action_len)
        state_new = sample_new["state"].detach().cpu().numpy()

        for strategy, new_chunk, selected_seed in strategy_chunks(
            old_chunk=old_chunk,
            new_chunks=new_chunks,
            execute_len=execute_len,
        ):
            metrics = command_metrics(
                old_chunk=old_chunk,
                new_exec=new_chunk[:execute_len],
                gt_exec=gt_chunk[:execute_len],
                state=state_new,
                execute_len=execute_len,
            )
            row = {
                "model": model_name,
                "checkpoint": str(checkpoint),
                "head_class": head.__class__.__name__,
                "diffusion": diffusion,
                "expected_action_len": expected_t,
                "strategy": strategy,
                "selected_seed": selected_seed,
                "global_idx_old": idx_old,
                "global_idx_new": idx_new,
                "episode": sample_new["episode"],
                "local_idx_old": sample_old["local_idx"],
                "local_idx_new": sample_new["local_idx"],
                "phase_new": sample_new["phase"],
            }
            row.update(metrics)
            rows.append(row)

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
    model_specs = [parse_model_spec(spec) for spec in args.model]
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    for _, checkpoint in model_specs:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
    if args.query_gap <= 0:
        raise ValueError("--query-gap must be positive")
    if args.execute_len <= 0 or args.execute_len >= args.resampled_action_len:
        raise ValueError("--execute-len must be positive and less than --resampled-action-len")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset = build_valid_dataset(dataset_root, args.valid_ratio)
    candidates = candidate_metadata(dataset)
    pairs = select_query_gap_pairs(candidates, args.query_gap, args.max_pairs)
    if not pairs:
        raise RuntimeError(f"No valid pairs found for query_gap={args.query_gap}")
    pair_indices = sorted({idx for pair in pairs for idx in pair})
    meta_by_idx = {item["global_idx"]: item for item in candidates}
    samples = [load_sample(dataset, meta_by_idx[idx]) for idx in pair_indices]

    all_rows = []
    model_metadata = []
    for model_name, checkpoint in model_specs:
        rows, metadata = evaluate_model(
            model_name=model_name,
            checkpoint=checkpoint,
            samples=samples,
            pairs=pairs,
            seeds_arg=args.seeds,
            batch_size=args.batch_size,
            device=device,
            resampled_action_len=args.resampled_action_len,
            execute_len=args.execute_len,
        )
        all_rows.extend(rows)
        model_metadata.append(metadata)

    summary_rows = summarize(all_rows)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = output_dir / f"{args.tag}_rows.csv"
    summary_csv = output_dir / f"{args.tag}_summary.csv"
    summary_json = output_dir / f"{args.tag}_summary.json"
    write_csv(rows_csv, all_rows)
    write_csv(summary_csv, summary_rows)
    summary = {
        "dataset_root": str(dataset_root),
        "valid_ratio": args.valid_ratio,
        "valid_len": len(dataset),
        "query_gap": args.query_gap,
        "resampled_action_len": args.resampled_action_len,
        "execute_len": args.execute_len,
        "pair_count": len(pairs),
        "phase_counts_new": dict(
            Counter(meta_by_idx[idx_new]["phase"] for _, idx_new in pairs)
        ),
        "models": model_metadata,
        "strategies": sorted({row["strategy"] for row in all_rows}),
        "outputs": {
            "rows": str(rows_csv),
            "summary": str(summary_csv),
            "summary_json": str(summary_json),
        },
        "summary": summary_rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"OUT_JSON {summary_json}")


if __name__ == "__main__":
    main()
