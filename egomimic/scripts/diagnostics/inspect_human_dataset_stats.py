from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.utils import DataSchematic, set_global_seed


DOMAIN = "ego_view_right_arm"
ACTION_KEY = "actions_cartesian"
OBS_KEY = "observations.state.ee_pose"
ACTION_KEY_NAME = "actions_cartesian"
OBS_KEY_NAME = "ee_pose"
DIM_NAMES = ["x", "y", "z", "yaw", "pitch", "roll"]
HIST_BINS = 100
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


def _instantiate_no_image_dataset(cfg, split_name: str):
    split_cfg = cfg.data.train_datasets if split_name == "train" else cfg.data.valid_datasets
    instantiate_copy = copy.deepcopy(split_cfg[DOMAIN])
    keymap_cfg = instantiate_copy.resolver.key_map
    km = OmegaConf.to_container(keymap_cfg, resolve=False)
    km["norm_mode"] = True
    instantiate_copy.resolver.key_map = km
    return hydra.utils.instantiate(instantiate_copy)


def _infer_schematic(cfg, train_dataset):
    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)
    data_schematic.infer_shapes_from_batch(train_dataset[0])
    data_schematic.infer_norm_from_dataset(
        train_dataset,
        DOMAIN,
        sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
        num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
        precomputed_norm_path=OmegaConf.select(
            cfg, "norm_stats.precomputed_norm_path", default=None
        ),
    )
    return data_schematic


def _stack_stats(arr: np.ndarray, dim_names: list[str]) -> dict:
    stats = {}
    for idx, name in enumerate(dim_names):
        col = arr[:, idx]
        stats[name] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "p01": float(np.quantile(col, 0.01)),
            "p50": float(np.quantile(col, 0.50)),
            "p99": float(np.quantile(col, 0.99)),
            "max": float(col.max()),
        }
    return stats


def _ks_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.sort(lhs.astype(np.float64, copy=False))
    rhs = np.sort(rhs.astype(np.float64, copy=False))
    if lhs.size == 0 or rhs.size == 0:
        return 0.0
    values = np.concatenate([lhs, rhs])
    lhs_cdf = np.searchsorted(lhs, values, side="right") / lhs.size
    rhs_cdf = np.searchsorted(rhs, values, side="right") / rhs.size
    return float(np.max(np.abs(lhs_cdf - rhs_cdf)))


def _histogram_probs(lhs: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = float(min(lhs.min(), rhs.min()))
    high = float(max(lhs.max(), rhs.max()))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return np.array([1.0], dtype=np.float64), np.array([1.0], dtype=np.float64)

    lhs_counts, edges = np.histogram(lhs, bins=HIST_BINS, range=(low, high))
    rhs_counts, _ = np.histogram(rhs, bins=edges)
    lhs_probs = lhs_counts.astype(np.float64) + EPS
    rhs_probs = rhs_counts.astype(np.float64) + EPS
    lhs_probs /= lhs_probs.sum()
    rhs_probs /= rhs_probs.sum()
    return lhs_probs, rhs_probs


def _distribution_distances(
    train_arr: np.ndarray,
    valid_arr: np.ndarray,
    dim_names: list[str],
) -> dict:
    distances = {}
    for idx, name in enumerate(dim_names):
        train_col = train_arr[:, idx]
        valid_col = valid_arr[:, idx]
        p_train, p_valid = _histogram_probs(train_col, valid_col)
        midpoint = 0.5 * (p_train + p_valid)
        distances[name] = {
            "mean_delta_valid_minus_train": float(valid_col.mean() - train_col.mean()),
            "std_ratio_valid_over_train": float(
                valid_col.std() / (train_col.std() + EPS)
            ),
            "ks_distance": _ks_distance(train_col, valid_col),
            "kl_train_to_valid": float(np.sum(p_train * np.log(p_train / p_valid))),
            "kl_valid_to_train": float(np.sum(p_valid * np.log(p_valid / p_train))),
            "js_divergence": float(
                0.5 * np.sum(p_train * np.log(p_train / midpoint))
                + 0.5 * np.sum(p_valid * np.log(p_valid / midpoint))
            ),
        }
    return distances


def _episode_stats(
    dataset,
    data_schematic: DataSchematic,
    embodiment_id: int,
    max_episodes: int | None = None,
):
    episode_rows = []
    items = list(dataset.datasets.items())
    if max_episodes is not None:
        items = items[:max_episodes]
    for episode_name, leaf in tqdm(items, desc="episode_scan", total=len(items)):
        prev_obs = None
        prev_action0 = None
        max_obs_xyz_jump = 0.0
        max_action0_xyz_jump = 0.0
        max_obs_rot_jump = 0.0
        max_action0_rot_jump = 0.0
        yaw_wrap_like = 0
        norm_out_of_range = 0
        norm_total = 0

        for local_idx in range(len(leaf)):
            sample = leaf[local_idx]
            obs = sample[OBS_KEY].detach().cpu().numpy().astype(np.float32)
            action = sample[ACTION_KEY].detach().cpu().numpy().astype(np.float32)
            action0 = action[0]
            norm_sample = data_schematic.normalize_data(
                {
                    OBS_KEY_NAME: sample[OBS_KEY].clone(),
                    ACTION_KEY_NAME: sample[ACTION_KEY].clone(),
                },
                embodiment_id,
            )
            norm_action = (
                norm_sample[ACTION_KEY_NAME].detach().cpu().numpy().astype(np.float32)
            )
            norm_out_of_range += int((np.abs(norm_action) > 1.0).sum())
            norm_total += int(norm_action.size)

            if prev_obs is not None:
                max_obs_xyz_jump = max(
                    max_obs_xyz_jump, float(np.linalg.norm(obs[:3] - prev_obs[:3]))
                )
                max_action0_xyz_jump = max(
                    max_action0_xyz_jump,
                    float(np.linalg.norm(action0[:3] - prev_action0[:3])),
                )
                max_obs_rot_jump = max(
                    max_obs_rot_jump,
                    float(np.max(np.abs(obs[3:6] - prev_obs[3:6]))),
                )
                max_action0_rot_jump = max(
                    max_action0_rot_jump,
                    float(np.max(np.abs(action0[3:6] - prev_action0[3:6]))),
                )
                yaw_wrap_like += int(abs(float(obs[3] - prev_obs[3])) > np.pi)
                yaw_wrap_like += int(abs(float(action0[3] - prev_action0[3])) > np.pi)

            prev_obs = obs
            prev_action0 = action0

        episode_rows.append(
            {
                "episode_name": episode_name,
                "num_samples": int(len(leaf)),
                "max_obs_xyz_jump_m": max_obs_xyz_jump,
                "max_action0_xyz_jump_m": max_action0_xyz_jump,
                "max_obs_rot_jump_rad": max_obs_rot_jump,
                "max_action0_rot_jump_rad": max_action0_rot_jump,
                "yaw_wrap_like_count": yaw_wrap_like,
                "norm_action_out_of_range_frac": (
                    float(norm_out_of_range / norm_total) if norm_total else 0.0
                ),
            }
        )

    episode_rows.sort(
        key=lambda row: (
            row["norm_action_out_of_range_frac"],
            row["max_obs_xyz_jump_m"],
            row["max_action0_xyz_jump_m"],
        ),
        reverse=True,
    )
    return episode_rows


def _collect_split_stats(
    dataset,
    data_schematic: DataSchematic,
    embodiment_id: int,
    max_samples: int | None = None,
):
    action_chunks = []
    action_chunks_norm = []
    obs_rows = []
    obs_rows_norm = []
    action_t0 = []
    action_t0_norm = []

    total = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    for idx in tqdm(range(total), desc="split_scan"):
        sample = dataset[idx]
        action = sample[ACTION_KEY].detach().cpu().numpy().astype(np.float32)
        obs = sample[OBS_KEY].detach().cpu().numpy().astype(np.float32)
        norm_sample = data_schematic.normalize_data(
            {
                OBS_KEY_NAME: sample[OBS_KEY].clone(),
                ACTION_KEY_NAME: sample[ACTION_KEY].clone(),
            },
            embodiment_id,
        )
        action_norm = (
            norm_sample[ACTION_KEY_NAME].detach().cpu().numpy().astype(np.float32)
        )
        obs_norm = norm_sample[OBS_KEY_NAME].detach().cpu().numpy().astype(np.float32)

        action_chunks.append(action.reshape(-1, action.shape[-1]))
        action_chunks_norm.append(action_norm.reshape(-1, action_norm.shape[-1]))
        obs_rows.append(obs.reshape(1, -1))
        obs_rows_norm.append(obs_norm.reshape(1, -1))
        action_t0.append(action[0:1])
        action_t0_norm.append(action_norm[0:1])

    action_all = np.concatenate(action_chunks, axis=0)
    action_norm_all = np.concatenate(action_chunks_norm, axis=0)
    obs_all = np.concatenate(obs_rows, axis=0)
    obs_norm_all = np.concatenate(obs_rows_norm, axis=0)
    action_t0_all = np.concatenate(action_t0, axis=0)
    action_t0_norm_all = np.concatenate(action_t0_norm, axis=0)

    num_samples = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    arrays = {
        "action_raw": action_all,
        "action_norm": action_norm_all,
        "action_t0_raw": action_t0_all,
        "action_t0_norm": action_t0_norm_all,
        "obs_raw": obs_all,
        "obs_norm": obs_norm_all,
    }
    stats = {
        "num_samples": int(num_samples),
        "num_episodes": int(len(getattr(dataset, "datasets", {}))),
        "num_action_vectors": int(action_all.shape[0]),
        "action_raw": _stack_stats(action_all, DIM_NAMES),
        "action_norm": _stack_stats(action_norm_all, DIM_NAMES),
        "action_t0_raw": _stack_stats(action_t0_all, DIM_NAMES),
        "action_t0_norm": _stack_stats(action_t0_norm_all, DIM_NAMES),
        "obs_raw": _stack_stats(obs_all, DIM_NAMES),
        "obs_norm": _stack_stats(obs_norm_all, DIM_NAMES),
        "action_norm_abs_gt_1_frac": {
            name: float((np.abs(action_norm_all[:, idx]) > 1.0).mean())
            for idx, name in enumerate(DIM_NAMES)
        },
        "action_t0_norm_abs_gt_1_frac": {
            name: float((np.abs(action_t0_norm_all[:, idx]) > 1.0).mean())
            for idx, name in enumerate(DIM_NAMES)
        },
    }
    return stats, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect human dataset split statistics and episode-level anomalies."
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    set_global_seed(args.seed)

    cfg, config_path = _load_cfg(args)
    train_dataset = _instantiate_no_image_dataset(cfg, "train")
    valid_dataset = _instantiate_no_image_dataset(cfg, "valid")
    data_schematic = _infer_schematic(cfg, train_dataset)
    embodiment_id = get_embodiment_id(DOMAIN)

    train_stats, train_arrays = _collect_split_stats(
        train_dataset,
        data_schematic,
        embodiment_id,
        max_samples=args.max_samples,
    )
    valid_stats, valid_arrays = _collect_split_stats(
        valid_dataset,
        data_schematic,
        embodiment_id,
        max_samples=args.max_samples,
    )
    train_episodes = _episode_stats(
        train_dataset,
        data_schematic,
        embodiment_id,
        max_episodes=args.max_episodes,
    )
    valid_episodes = _episode_stats(
        valid_dataset,
        data_schematic,
        embodiment_id,
        max_episodes=args.max_episodes,
    )

    payload = {
        "config_path": str(config_path),
        "domain": DOMAIN,
        "action_key": ACTION_KEY,
        "obs_key": OBS_KEY,
        "train": train_stats,
        "valid": valid_stats,
        "train_valid_distances": {
            key: _distribution_distances(train_arrays[key], valid_arrays[key], DIM_NAMES)
            for key in ("action_norm", "action_t0_norm", "obs_norm")
        },
        "top_train_episodes": train_episodes[:10],
        "top_valid_episodes": valid_episodes[:10],
    }

    text = json.dumps(payload, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
