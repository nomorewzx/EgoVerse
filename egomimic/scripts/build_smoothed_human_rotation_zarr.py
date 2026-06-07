from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


DEFAULT_SRC_ROOT = (
    "/home/zxwang/repos/apricot_human_in_domain/"
    "egoverse_zarr_apricot_in_domain_merged_part1_part2_part3_part4_min90_ego_view_right_arm"
)
DEFAULT_DST_ROOT = (
    "/home/zxwang/repos/apricot_human_in_domain/"
    "egoverse_zarr_apricot_in_domain_merged_part1_part2_part3_part4_min90_ego_view_right_arm_smoothrot_w9"
)
DEFAULT_POSE_KEY = "right.obs_ee_pose"
EPS = 1e-12


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {key: float("nan") for key in ("mean", "p50", "p90", "p95", "p99", "max")}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def _quat_wxyz_to_rotation(quat_wxyz: np.ndarray) -> R:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    norms = np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
    if np.any(norms < EPS):
        raise ValueError("Encountered zero-norm quaternion")
    quat_wxyz = quat_wxyz / norms
    return R.from_quat(quat_wxyz[:, [1, 2, 3, 0]])


def _rotation_to_quat_wxyz(rot: R) -> np.ndarray:
    quat_xyzw = rot.as_quat()
    return quat_xyzw[:, [3, 0, 1, 2]]


def _gaussian_weights(window: int, sigma: float) -> np.ndarray:
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / max(float(sigma), EPS)) ** 2)
    return weights / np.sum(weights)


def _boxcar_weights(window: int) -> np.ndarray:
    weights = np.ones(int(window), dtype=np.float64)
    return weights / np.sum(weights)


def _smooth_rotations(
    rotations: R,
    *,
    window: int,
    weights: np.ndarray,
) -> R:
    if window < 1 or window % 2 != 1:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    count = len(rotations)
    if count <= 1 or window == 1:
        return rotations
    half = window // 2
    smoothed = []
    for idx in range(count):
        lo = max(0, idx - half)
        hi = min(count, idx + half + 1)
        weight_lo = half - (idx - lo)
        weight_hi = weight_lo + (hi - lo)
        smoothed.append(rotations[lo:hi].mean(weights=weights[weight_lo:weight_hi]))
    return R.concatenate(smoothed)


def _smooth_pose_array(
    poses: np.ndarray,
    *,
    window: int,
    weights: np.ndarray,
) -> np.ndarray:
    poses = np.asarray(poses)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"Expected pose array shape (T, 7), got {poses.shape}")
    out = poses.astype(np.float64, copy=True)
    rotations = _quat_wxyz_to_rotation(out[:, 3:7])
    smoothed = _smooth_rotations(rotations, window=window, weights=weights)
    out[:, 3:7] = _rotation_to_quat_wxyz(smoothed)
    return out.astype(poses.dtype, copy=False)


def _rotation_metrics(poses: np.ndarray) -> dict[str, Any]:
    rotations = _quat_wxyz_to_rotation(np.asarray(poses)[:, 3:7])
    if len(rotations) < 2:
        return {"rot_step_deg": _stats([]), "rot_accel_ypr_deg": _stats([])}
    step_deg = (rotations[1:] * rotations[:-1].inv()).magnitude() * 180.0 / np.pi
    ypr = rotations.as_euler("ZYX", degrees=False)
    ypr_unwrapped = np.unwrap(ypr, axis=0)
    accel = (
        np.linalg.norm(np.diff(ypr_unwrapped, n=2, axis=0), axis=1)
        * 180.0
        / np.pi
        if len(rotations) >= 3
        else np.asarray([], dtype=np.float64)
    )
    return {
        "rot_step_deg": _stats(step_deg),
        "rot_accel_ypr_deg": _stats(accel),
    }


def _copy_episode_hardlinked(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f"{dst.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp, copy_function=os.link)
    tmp.rename(dst)


def _replace_array(
    *,
    dst_episode: Path,
    key: str,
    data: np.ndarray,
    chunks: tuple[int, ...],
    compressor: Any,
    attrs: dict[str, Any],
) -> None:
    group = zarr.open_group(str(dst_episode), mode="a")
    if key in group:
        del group[key]
    arr = group.create_dataset(
        key,
        shape=data.shape,
        data=data,
        chunks=chunks,
        dtype=data.dtype,
        compressor=compressor,
        overwrite=True,
    )
    arr.attrs.update(attrs)


def _process_episode(
    *,
    src_episode: Path,
    dst_episode: Path,
    pose_key: str,
    window: int,
    weights: np.ndarray,
    overwrite: bool,
) -> dict[str, Any]:
    if dst_episode.exists():
        if not overwrite:
            return {
                "episode": src_episode.name,
                "status": "skipped_existing",
                "dst_episode": str(dst_episode),
            }
        shutil.rmtree(dst_episode)

    src_group = zarr.open_group(str(src_episode), mode="r")
    if pose_key not in src_group:
        raise KeyError(f"{pose_key} not found in {src_episode}")
    src_arr = src_group[pose_key]
    poses = np.asarray(src_arr[:])
    smoothed = _smooth_pose_array(poses, window=window, weights=weights)

    _copy_episode_hardlinked(src_episode, dst_episode)
    _replace_array(
        dst_episode=dst_episode,
        key=pose_key,
        data=smoothed,
        chunks=src_arr.chunks,
        compressor=src_arr.compressor,
        attrs=dict(src_arr.attrs),
    )
    dst_group = zarr.open_group(str(dst_episode), mode="a")
    dst_group.attrs["rotation_smoothing"] = {
        "pose_key": pose_key,
        "window": int(window),
        "created_by": "egomimic/scripts/build_smoothed_human_rotation_zarr.py",
    }

    before = _rotation_metrics(poses)
    after = _rotation_metrics(smoothed)
    return {
        "episode": src_episode.name,
        "status": "written",
        "dst_episode": str(dst_episode),
        "num_frames": int(poses.shape[0]),
        "before": before,
        "after": after,
    }


def _aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    written = [row for row in reports if row.get("status") == "written"]
    if not written:
        return {"num_written": 0}
    keys = [
        ("rot_step_deg", "p99"),
        ("rot_accel_ypr_deg", "p99"),
        ("rot_step_deg", "mean"),
        ("rot_accel_ypr_deg", "mean"),
    ]
    out: dict[str, Any] = {"num_written": len(written)}
    for family, stat in keys:
        before = np.asarray([row["before"][family][stat] for row in written], dtype=np.float64)
        after = np.asarray([row["after"][family][stat] for row in written], dtype=np.float64)
        out[f"{family}.{stat}"] = {
            "before_episode_mean": float(np.nanmean(before)),
            "after_episode_mean": float(np.nanmean(after)),
            "ratio_after_over_before": float(np.nanmean(after) / max(np.nanmean(before), EPS)),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a hardlinked copy of a human zarr root with SO(3)-smoothed "
            "right-hand rotations. XYZ, images, gripper, head pose, and all other "
            "arrays are preserved."
        )
    )
    parser.add_argument("--src-root", default=DEFAULT_SRC_ROOT)
    parser.add_argument("--dst-root", default=DEFAULT_DST_ROOT)
    parser.add_argument("--pose-key", default=DEFAULT_POSE_KEY)
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--weight-mode", choices=("gaussian", "boxcar"), default="gaussian")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    if not src_root.is_dir():
        raise FileNotFoundError(f"src root not found: {src_root}")
    if args.window < 1 or args.window % 2 != 1:
        raise ValueError(f"--window must be a positive odd integer, got {args.window}")

    episodes = sorted(src_root.glob("*.zarr"))
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise RuntimeError(f"No .zarr episodes found under {src_root}")

    if args.weight_mode == "gaussian":
        weights = _gaussian_weights(args.window, args.sigma)
    else:
        weights = _boxcar_weights(args.window)

    dst_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for src_episode in tqdm(episodes, desc="smooth human rotations"):
        reports.append(
            _process_episode(
                src_episode=src_episode,
                dst_episode=dst_root / src_episode.name,
                pose_key=args.pose_key,
                window=args.window,
                weights=weights,
                overwrite=args.overwrite,
            )
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "pose_key": args.pose_key,
        "window": int(args.window),
        "weight_mode": args.weight_mode,
        "sigma": float(args.sigma),
        "weights": [float(x) for x in weights],
        "num_input_episodes": len(episodes),
        "aggregate": _aggregate_reports(reports),
        "episodes": reports,
    }
    report_json = Path(args.report_json) if args.report_json else dst_root / "rotation_smoothing_manifest.json"
    report_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"dst_root": str(dst_root), "report_json": str(report_json), "aggregate": manifest["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
