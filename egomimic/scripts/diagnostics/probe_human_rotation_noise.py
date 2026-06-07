from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.utils import DataSchematic, set_global_seed


DOMAINS = ("so100_singlearm", "ego_view_right_arm")
SPLITS = ("train", "valid")
ACTION_KEY = "actions_cartesian"
OBS_KEY = "observations.state.ee_pose"
ACTION_KEY_NAME = "actions_cartesian"
OBS_KEY_NAME = "ee_pose"
DIM_NAMES = ("x", "y", "z", "yaw", "pitch", "roll")
EPS = 1e-12


def _load_cfg(args: argparse.Namespace):
    if args.run_dir is not None:
        config_path = Path(args.run_dir) / ".hydra" / "config.yaml"
    elif args.config is not None:
        config_path = Path(args.config)
    else:
        raise ValueError("Provide either --run-dir or --config")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    if OmegaConf.select(cfg, "norm_stats.precomputed_norm_path", default=None) is None:
        if args.run_dir is not None:
            cache_path = Path(args.run_dir) / "norm_stats" / "norm_stats.json"
            if cache_path.is_file():
                cfg.norm_stats.precomputed_norm_path = str(cache_path)
    return cfg, config_path


def _dataset_cfg(cfg, split: str, domain: str):
    split_cfg = cfg.data.train_datasets if split == "train" else cfg.data.valid_datasets
    if domain not in split_cfg:
        raise KeyError(f"Domain '{domain}' not present in cfg.data.{split}_datasets")

    instantiate_copy = copy.deepcopy(split_cfg[domain])
    keymap_cfg = instantiate_copy.resolver.key_map
    km = OmegaConf.to_container(keymap_cfg, resolve=False)
    # Removes camera / annotation keys. The transforms still produce canonical
    # action and state tensors, but zarr image decode is skipped.
    km["norm_mode"] = True
    instantiate_copy.resolver.key_map = km
    return instantiate_copy


def _instantiate_dataset(cfg, split: str, domain: str):
    return hydra.utils.instantiate(_dataset_cfg(cfg, split, domain))


def _infer_schematic(cfg, train_datasets: dict[str, Any]) -> DataSchematic:
    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)
    for dataset in train_datasets.values():
        data_schematic.infer_shapes_from_batch(dataset[0])

    for domain, dataset in train_datasets.items():
        data_schematic.infer_norm_from_dataset(
            dataset,
            domain,
            sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
            num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
            precomputed_norm_path=OmegaConf.select(
                cfg, "norm_stats.precomputed_norm_path", default=None
            ),
        )
    return data_schematic


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _angle_diff(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _series_stats(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    values = values * float(scale)
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _moving_average_same(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.shape[0] < 3:
        return x.copy()
    window = min(int(window), x.shape[0])
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return x.copy()
    pad = window // 2
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    out = np.empty_like(x, dtype=np.float64)
    for dim in range(x.shape[1]):
        out[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return out


def _high_frequency_ratio(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3:
        return np.full(x.shape[-1] if x.ndim == 2 else 1, np.nan, dtype=np.float64)
    smooth = _moving_average_same(x, window)
    residual_energy = np.mean((x - smooth) ** 2, axis=0)
    centered_energy = np.mean((x - np.mean(x, axis=0, keepdims=True)) ** 2, axis=0)
    return residual_energy / np.maximum(centered_energy, EPS)


def _norm_spans(data_schematic: DataSchematic, embodiment_id: int, key_name: str, dims: int):
    stats = data_schematic.norm_stats[embodiment_id][key_name]
    if "quantile_1" in stats and "quantile_99" in stats:
        low = np.asarray(stats["quantile_1"], dtype=np.float64)
        high = np.asarray(stats["quantile_99"], dtype=np.float64)
    elif "min" in stats and "max" in stats:
        low = np.asarray(stats["min"], dtype=np.float64)
        high = np.asarray(stats["max"], dtype=np.float64)
    else:
        low = np.asarray(stats["mean"], dtype=np.float64) - np.asarray(
            stats["std"], dtype=np.float64
        )
        high = np.asarray(stats["mean"], dtype=np.float64) + np.asarray(
            stats["std"], dtype=np.float64
        )
    if low.ndim > 1:
        low = low.reshape(-1, low.shape[-1])[0]
        high = high.reshape(-1, high.shape[-1])[0]
    return np.maximum((high - low)[:dims], EPS)


def _sequence_metrics(
    seq: np.ndarray,
    spans: np.ndarray,
    *,
    high_freq_window: int,
) -> dict[str, Any]:
    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 2 or seq.shape[-1] < 6:
        raise ValueError(f"Expected sequence shape (T, >=6), got {seq.shape}")
    dims = min(seq.shape[-1], len(spans))
    seq6 = seq[:, :6]
    spans6 = spans[:6]

    if seq6.shape[0] < 2:
        return {
            "num_steps": 0,
            "xyz_step_m": _series_stats([]),
            "rot_step_rad": _series_stats([]),
            "xyz_accel_m": _series_stats([]),
            "rot_ypr_accel_rad": _series_stats([]),
            "wrap_rate_per_1k": 0.0,
            "normalized_step": {name: _series_stats([]) for name in DIM_NAMES},
            "high_freq_ratio": {name: float("nan") for name in DIM_NAMES},
            "xyz_norm_step_mean_p95": float("nan"),
            "rot_norm_step_mean_p95": float("nan"),
            "rot_over_xyz_norm_step_p95": float("nan"),
        }

    diff = np.diff(seq6, axis=0)
    xyz_step = np.linalg.norm(diff[:, :3], axis=1)
    raw_rot_diff = diff[:, 3:6]
    wrapped_rot_diff = _angle_diff(raw_rot_diff)
    wrap_count = int(np.sum(np.abs(raw_rot_diff) > np.pi))
    wrap_rate_per_1k = 1000.0 * wrap_count / max(1, raw_rot_diff.size)

    rots = R.from_euler("ZYX", seq6[:, 3:6], degrees=False)
    rot_step = (rots[1:] * rots[:-1].inv()).magnitude()

    if seq6.shape[0] >= 3:
        xyz_accel = np.linalg.norm(np.diff(seq6[:, :3], n=2, axis=0), axis=1)
        ypr_unwrapped = np.unwrap(seq6[:, 3:6], axis=0)
        rot_accel = np.linalg.norm(np.diff(ypr_unwrapped, n=2, axis=0), axis=1)
    else:
        xyz_accel = np.array([], dtype=np.float64)
        rot_accel = np.array([], dtype=np.float64)

    norm_diff = np.empty_like(diff[:, :dims])
    norm_diff[:, :3] = np.abs(diff[:, :3]) / spans6[:3]
    norm_diff[:, 3:6] = np.abs(wrapped_rot_diff) / spans6[3:6]

    ypr_unwrapped = np.unwrap(seq6[:, 3:6], axis=0)
    smooth_input = np.concatenate([seq6[:, :3], ypr_unwrapped], axis=1)
    hf = _high_frequency_ratio(smooth_input, high_freq_window)
    normalized_step = {
        name: _series_stats(norm_diff[:, idx])
        for idx, name in enumerate(DIM_NAMES[:dims])
    }
    xyz_norm_step_mean_p95 = float(
        np.mean([normalized_step[name]["p95"] for name in DIM_NAMES[:3]])
    )
    rot_norm_step_mean_p95 = float(
        np.mean([normalized_step[name]["p95"] for name in DIM_NAMES[3:6]])
    )

    return {
        "num_steps": int(seq6.shape[0] - 1),
        "xyz_step_m": _series_stats(xyz_step),
        "rot_step_rad": _series_stats(rot_step),
        "rot_step_deg": _series_stats(rot_step, scale=180.0 / np.pi),
        "xyz_accel_m": _series_stats(xyz_accel),
        "rot_ypr_accel_rad": _series_stats(rot_accel),
        "rot_ypr_accel_deg": _series_stats(rot_accel, scale=180.0 / np.pi),
        "wrap_count": wrap_count,
        "wrap_rate_per_1k": float(wrap_rate_per_1k),
        "normalized_step": normalized_step,
        "high_freq_ratio": {
            name: float(hf[idx]) for idx, name in enumerate(DIM_NAMES[: len(hf)])
        },
        "xyz_norm_step_mean_p95": xyz_norm_step_mean_p95,
        "rot_norm_step_mean_p95": rot_norm_step_mean_p95,
        "rot_over_xyz_norm_step_p95": float(
            rot_norm_step_mean_p95 / max(xyz_norm_step_mean_p95, EPS)
        ),
    }


def _merge_metric_lists(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    merged: dict[str, list[float]] = defaultdict(list)
    counts = {"num_episodes": len(rows), "num_steps": 0, "wrap_count": 0}
    for row in rows:
        counts["num_steps"] += int(row["num_steps"])
        counts["wrap_count"] += int(row.get("wrap_count", 0))
        for family in (
            "xyz_step_m",
            "rot_step_rad",
            "rot_step_deg",
            "xyz_accel_m",
            "rot_ypr_accel_rad",
            "rot_ypr_accel_deg",
        ):
            for stat_name, value in row[family].items():
                merged[f"{family}.{stat_name}"].append(float(value))
        for name, stats in row["normalized_step"].items():
            for stat_name, value in stats.items():
                merged[f"normalized_step.{name}.{stat_name}"].append(float(value))
        for name, value in row["high_freq_ratio"].items():
            merged[f"high_freq_ratio.{name}"].append(float(value))
        for key in (
            "xyz_norm_step_mean_p95",
            "rot_norm_step_mean_p95",
            "rot_over_xyz_norm_step_p95",
            "wrap_rate_per_1k",
        ):
            merged[key].append(float(row[key]))

    summary = dict(counts)
    for key, values in merged.items():
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        summary[key] = {
            "episode_mean": float(np.mean(arr)) if arr.size else float("nan"),
            "episode_p50": float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
            "episode_p90": float(np.quantile(arr, 0.90)) if arr.size else float("nan"),
            "episode_max": float(np.max(arr)) if arr.size else float("nan"),
        }
    summary["wrap_rate_per_1k_global"] = float(
        1000.0 * counts["wrap_count"] / max(1, counts["num_steps"] * 3)
    )
    return summary


def _episode_iter(dataset, max_episodes: int | None):
    items = list(dataset.datasets.items())
    items.sort(key=lambda item: item[0])
    if max_episodes is not None:
        items = items[:max_episodes]
    return items


def _collect_domain_split(
    dataset,
    data_schematic: DataSchematic,
    domain: str,
    *,
    max_episodes: int | None,
    max_frames_per_episode: int | None,
    frame_stride: int,
    high_freq_window: int,
) -> dict[str, Any]:
    embodiment_id = get_embodiment_id(domain)
    obs_spans = _norm_spans(data_schematic, embodiment_id, OBS_KEY_NAME, 6)
    action_spans = _norm_spans(data_schematic, embodiment_id, ACTION_KEY_NAME, 6)
    episode_rows = []

    items = _episode_iter(dataset, max_episodes=max_episodes)
    for episode_name, leaf in tqdm(items, desc=f"{domain}", total=len(items)):
        frame_count = len(leaf)
        if max_frames_per_episode is not None:
            frame_count = min(frame_count, max_frames_per_episode)
        indices = range(0, frame_count, max(1, frame_stride))
        obs_seq = []
        action0_seq = []
        for local_idx in indices:
            sample = leaf[local_idx]
            obs = _to_numpy(sample[OBS_KEY]).astype(np.float64, copy=False)
            action = _to_numpy(sample[ACTION_KEY]).astype(np.float64, copy=False)
            obs_seq.append(obs[:6])
            action0_seq.append(action[0, :6])

        if len(obs_seq) < 2:
            continue

        obs_metrics = _sequence_metrics(
            np.asarray(obs_seq),
            obs_spans,
            high_freq_window=high_freq_window,
        )
        action0_metrics = _sequence_metrics(
            np.asarray(action0_seq),
            action_spans,
            high_freq_window=high_freq_window,
        )
        episode_rows.append(
            {
                "episode_name": str(episode_name),
                "num_frames": int(len(obs_seq)),
                "obs": obs_metrics,
                "action0": action0_metrics,
            }
        )

    obs_rows = [row["obs"] for row in episode_rows]
    action_rows = [row["action0"] for row in episode_rows]
    top_rot_episodes = sorted(
        episode_rows,
        key=lambda row: (
            row["action0"]["rot_step_deg"]["p99"],
            row["action0"]["wrap_rate_per_1k"],
        ),
        reverse=True,
    )[:10]

    return {
        "num_episodes_scanned": int(len(episode_rows)),
        "num_frames_scanned": int(sum(row["num_frames"] for row in episode_rows)),
        "obs_summary": _merge_metric_lists(obs_rows),
        "action0_summary": _merge_metric_lists(action_rows),
        "top_action0_rot_episodes": [
            {
                "episode_name": row["episode_name"],
                "num_frames": row["num_frames"],
                "rot_step_deg_p99": row["action0"]["rot_step_deg"]["p99"],
                "rot_ypr_accel_deg_p99": row["action0"]["rot_ypr_accel_deg"]["p99"],
                "wrap_rate_per_1k": row["action0"]["wrap_rate_per_1k"],
                "rot_over_xyz_norm_step_p95": row["action0"][
                    "rot_over_xyz_norm_step_p95"
                ],
            }
            for row in top_rot_episodes
        ],
    }


def _summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split, split_payload in payload["splits"].items():
        for domain, domain_payload in split_payload.items():
            for signal in ("obs", "action0"):
                summary = domain_payload[f"{signal}_summary"]
                row = {
                    "split": split,
                    "domain": domain,
                    "signal": signal,
                    "num_episodes": summary.get("num_episodes", 0),
                    "num_steps": summary.get("num_steps", 0),
                    "wrap_rate_per_1k_global": summary.get(
                        "wrap_rate_per_1k_global", float("nan")
                    ),
                }
                flat_keys = {
                    "xyz_step_m_p99_ep_mean": "xyz_step_m.p99",
                    "rot_step_deg_p99_ep_mean": "rot_step_deg.p99",
                    "xyz_accel_m_p99_ep_mean": "xyz_accel_m.p99",
                    "rot_accel_deg_p99_ep_mean": "rot_ypr_accel_deg.p99",
                    "xyz_norm_step_mean_p95_ep_mean": "xyz_norm_step_mean_p95",
                    "rot_norm_step_mean_p95_ep_mean": "rot_norm_step_mean_p95",
                    "rot_over_xyz_norm_step_p95_ep_mean": "rot_over_xyz_norm_step_p95",
                    "hf_xyz_mean": None,
                    "hf_rot_mean": None,
                }
                for out_key, nested_key in flat_keys.items():
                    if nested_key is None:
                        continue
                    row[out_key] = summary.get(nested_key, {}).get(
                        "episode_mean", float("nan")
                    )
                row["hf_xyz_mean"] = float(
                    np.mean(
                        [
                            summary.get(f"high_freq_ratio.{name}", {}).get(
                                "episode_mean", float("nan")
                            )
                            for name in DIM_NAMES[:3]
                        ]
                    )
                )
                row["hf_rot_mean"] = float(
                    np.mean(
                        [
                            summary.get(f"high_freq_ratio.{name}", {}).get(
                                "episode_mean", float("nan")
                            )
                            for name in DIM_NAMES[3:6]
                        ]
                    )
                )
                rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare human and SO100 action/state rotation temporal noise using "
            "geodesic rotation steps, Euler wrap rate, normalized step size, and "
            "high-frequency energy."
        )
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=DOMAINS)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--high-freq-window", type=int, default=9)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    cfg, config_path = _load_cfg(args)
    train_datasets = {
        domain: _instantiate_dataset(cfg, "train", domain) for domain in args.domains
    }
    data_schematic = _infer_schematic(cfg, train_datasets)

    datasets: dict[str, dict[str, Any]] = {"train": train_datasets}
    for split in args.splits:
        if split == "train":
            continue
        datasets[split] = {
            domain: _instantiate_dataset(cfg, split, domain) for domain in args.domains
        }

    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "run_dir": args.run_dir,
        "norm_mode": data_schematic.norm_mode,
        "args": vars(args),
        "splits": {},
    }
    for split in args.splits:
        payload["splits"][split] = {}
        for domain in args.domains:
            payload["splits"][split][domain] = _collect_domain_split(
                datasets[split][domain],
                data_schematic,
                domain,
                max_episodes=args.max_episodes,
                max_frames_per_episode=args.max_frames_per_episode,
                frame_stride=args.frame_stride,
                high_freq_window=args.high_freq_window,
            )

    rows = _summary_rows(payload)

    output_json = (
        Path(args.output_json)
        if args.output_json
        else Path("logs/so100_hpt/rotation_noise_probe")
        / f"human_rotation_noise_{datetime.now():%Y-%m-%d_%H-%M-%S}.json"
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else output_json.with_suffix(".summary.csv")
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(rows, output_csv)

    print(json.dumps({"output_json": str(output_json), "output_csv": str(output_csv)}, indent=2))
    print("summary_csv:")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
