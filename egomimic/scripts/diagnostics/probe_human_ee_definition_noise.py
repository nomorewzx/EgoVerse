from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R


DIM_NAMES = ("yaw", "pitch", "roll")
THRESHOLDS_DEG = (1.0, 2.0, 3.0, 5.0)
EPS = 1e-12


def _normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    return vec / norm


def _old_middle_ring_pose_cam(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if joints.shape[0] < 18:
        return None
    origin = (joints[5] + joints[9] + joints[13] + joints[17]) / 4.0
    x_axis = _normalize(joints[9] - origin)
    y_seed = _normalize(joints[13] - origin)
    if x_axis is None or y_seed is None:
        return None
    z_axis = _normalize(np.cross(x_axis, y_seed))
    if z_axis is None:
        return None
    y_axis = _normalize(np.cross(z_axis, x_axis))
    if y_axis is None:
        return None
    return origin, np.stack([x_axis, y_axis, z_axis], axis=1)


def _wrist_mcp_spread_pose_cam(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if joints.shape[0] < 18:
        return None
    wrist = joints[0]
    index_mcp = joints[5]
    middle_mcp = joints[9]
    ring_mcp = joints[13]
    pinky_mcp = joints[17]
    origin = np.mean(np.stack([index_mcp, middle_mcp, ring_mcp, pinky_mcp], axis=0), axis=0)

    x_axis = _normalize(origin - wrist)
    y_seed = _normalize(pinky_mcp - index_mcp)
    if x_axis is None or y_seed is None:
        return None
    y_axis = y_seed - float(np.dot(y_seed, x_axis)) * x_axis
    y_axis = _normalize(y_axis)
    if y_axis is None:
        return None
    z_axis = _normalize(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None
    y_axis = _normalize(np.cross(z_axis, x_axis))
    if y_axis is None:
        return None
    return origin, np.stack([x_axis, y_axis, z_axis], axis=1)


def _xyzwxyz_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    mats = np.broadcast_to(np.eye(4, dtype=np.float64), (pose.shape[0], 4, 4)).copy()
    mats[:, :3, :3] = R.from_quat(pose[:, [4, 5, 6, 3]]).as_matrix()
    mats[:, :3, 3] = pose[:, :3]
    return mats


def _matrix_to_ypr(mats: np.ndarray) -> np.ndarray:
    xyz = mats[:, :3, 3]
    ypr = R.from_matrix(mats[:, :3, :3]).as_euler("ZYX", degrees=False)
    return np.concatenate([xyz, ypr], axis=-1)


def _pose_cam_to_matrix(origin: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rotation
    mat[:3, 3] = origin
    return mat


def _load_jsonl_by_frame(path: Path) -> dict[int, dict[str, Any]]:
    records = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records[int(record["frame_index"])] = record
    return records


def _select_hand(record: dict[str, Any], side: str) -> dict[str, Any] | None:
    hands = record.get("hands") if isinstance(record.get("hands"), list) else []
    track = (record.get("gripper_tracks") or {}).get(side) or {}
    hand_index = track.get("source_hand_index")
    if hand_index is not None:
        try:
            hand_index = int(hand_index)
            if 0 <= hand_index < len(hands):
                hand = hands[hand_index]
                if isinstance(hand, dict):
                    return hand
        except (TypeError, ValueError):
            pass
    expected = 1.0 if side == "right" else 0.0
    for hand in hands:
        try:
            if float(hand.get("is_right", -1.0)) == expected:
                return hand
        except (TypeError, ValueError):
            continue
    return None


def _raw_frame_start_from_episode_name(episode_name: str) -> int:
    matches = re.findall(r"_(\d+)-(\d+)", episode_name)
    if not matches:
        return 0
    return int(matches[-1][0])


def _series_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in ("mean", "p50", "p90", "p95", "p99", "max")}
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


def _high_freq_ratio(ypr_rad: np.ndarray, window: int) -> dict[str, float]:
    ypr_unwrapped = np.unwrap(np.asarray(ypr_rad, dtype=np.float64), axis=0)
    smooth = _moving_average_same(ypr_unwrapped, window)
    residual_energy = np.mean((ypr_unwrapped - smooth) ** 2, axis=0)
    centered_energy = np.mean((ypr_unwrapped - np.mean(ypr_unwrapped, axis=0, keepdims=True)) ** 2, axis=0)
    ratio = residual_energy / np.maximum(centered_energy, EPS)
    return {name: float(value) for name, value in zip(DIM_NAMES, ratio)}


def _alternation_counts(ypr_rad: np.ndarray, threshold_deg: float) -> dict[str, Any]:
    steps_deg = np.diff(np.unwrap(ypr_rad, axis=0), axis=0) * 180.0 / np.pi
    signs = np.sign(steps_deg)
    abs_steps = np.abs(steps_deg)
    flips = (
        (signs[:-1] * signs[1:] < 0)
        & (abs_steps[:-1] >= threshold_deg)
        & (abs_steps[1:] >= threshold_deg)
    )
    triples = (
        (signs[:-2] * signs[1:-1] < 0)
        & (signs[1:-1] * signs[2:] < 0)
        & (abs_steps[:-2] >= threshold_deg)
        & (abs_steps[1:-1] >= threshold_deg)
        & (abs_steps[2:] >= threshold_deg)
    )
    return {
        "flips_by_dim": {name: int(value) for name, value in zip(DIM_NAMES, flips.sum(axis=0))},
        "triples_by_dim": {name: int(value) for name, value in zip(DIM_NAMES, triples.sum(axis=0))},
        "flip_den": int(max(0, steps_deg.shape[0] - 1)),
        "triple_den": int(max(0, steps_deg.shape[0] - 2)),
    }


def _top_alternating_triplets(ypr_rad: np.ndarray, threshold_deg: float, top_k: int) -> list[dict[str, Any]]:
    ypr_unwrapped_deg = np.unwrap(ypr_rad, axis=0) * 180.0 / np.pi
    steps_deg = np.diff(ypr_unwrapped_deg, axis=0)
    signs = np.sign(steps_deg)
    abs_steps = np.abs(steps_deg)
    triples = (
        (signs[:-2] * signs[1:-1] < 0)
        & (signs[1:-1] * signs[2:] < 0)
        & (abs_steps[:-2] >= threshold_deg)
        & (abs_steps[1:-1] >= threshold_deg)
        & (abs_steps[2:] >= threshold_deg)
    )
    out = []
    for step_idx, dim_idx in np.argwhere(triples):
        step_triplet = steps_deg[step_idx : step_idx + 3, dim_idx]
        value_window = ypr_unwrapped_deg[step_idx : step_idx + 4, dim_idx]
        out.append(
            {
                "score": float(np.mean(np.abs(step_triplet))),
                "first_step_index": int(step_idx),
                "center_frame_index": int(step_idx + 1),
                "dim": DIM_NAMES[int(dim_idx)],
                "steps_deg": [float(x) for x in step_triplet],
                "values_deg": [float(x) for x in value_window],
            }
        )
    out.sort(key=lambda row: row["score"], reverse=True)
    return out[:top_k]


def _metrics(mats: np.ndarray, *, high_freq_window: int, top_k: int) -> dict[str, Any]:
    mats = np.asarray(mats, dtype=np.float64)
    rots = R.from_matrix(mats[:, :3, :3])
    ypr_rad = rots.as_euler("ZYX", degrees=False)
    rot_step_deg = (rots[1:] * rots[:-1].inv()).magnitude() * 180.0 / np.pi
    rot_accel_deg = (
        np.linalg.norm(np.diff(np.unwrap(ypr_rad, axis=0), n=2, axis=0), axis=1)
        * 180.0
        / np.pi
    )
    return {
        "num_frames": int(mats.shape[0]),
        "rot_step_deg": _series_stats(rot_step_deg),
        "rot_accel_deg": _series_stats(rot_accel_deg),
        "high_freq_ratio": _high_freq_ratio(ypr_rad, high_freq_window),
        "alternation_counts": {
            f"{threshold:g}deg": _alternation_counts(ypr_rad, threshold)
            for threshold in THRESHOLDS_DEG
        },
        "top_3deg_alternating_triplets": _top_alternating_triplets(ypr_rad, 3.0, top_k),
    }


def _window_payload(mats: np.ndarray, start: int, end: int) -> dict[str, Any]:
    start = max(0, int(start))
    end = min(int(end), mats.shape[0])
    if end <= start:
        return {}
    rots = R.from_matrix(mats[start:end, :3, :3])
    ypr_deg = rots.as_euler("ZYX", degrees=True)
    geo_step_deg = (rots[1:] * rots[:-1].inv()).magnitude() * 180.0 / np.pi
    euler_step_deg = np.diff(np.unwrap(ypr_deg * np.pi / 180.0, axis=0), axis=0) * 180.0 / np.pi
    return {
        "frame_indices": list(range(start, end)),
        "ypr_deg": np.round(ypr_deg, 4).tolist(),
        "euler_step_deg": np.round(euler_step_deg, 4).tolist(),
        "geodesic_step_deg": np.round(geo_step_deg, 4).tolist(),
    }


def _collect_pose_mats(
    raw_records: dict[int, dict[str, Any]],
    *,
    side: str,
    raw_frame_start: int,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    old_mats = []
    new_mats = []
    valid_local_indices = []
    for local_idx in range(num_frames):
        record = raw_records.get(raw_frame_start + local_idx)
        if record is None:
            continue
        hand = _select_hand(record, side)
        if hand is None or "joints_3d_cam" not in hand:
            continue
        joints = np.asarray(hand["joints_3d_cam"], dtype=np.float64)
        old_pose = _old_middle_ring_pose_cam(joints)
        new_pose = _wrist_mcp_spread_pose_cam(joints)
        if old_pose is None or new_pose is None:
            continue
        old_mats.append(_pose_cam_to_matrix(*old_pose))
        new_mats.append(_pose_cam_to_matrix(*new_pose))
        valid_local_indices.append(local_idx)
    return np.asarray(old_mats), np.asarray(new_mats), valid_local_indices


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    zarr_path = Path(args.zarr)
    raw_path = Path(args.raw)
    store = zarr.open(str(zarr_path), mode="r")
    raw_records = _load_jsonl_by_frame(raw_path)
    episode_name = zarr_path.name.removesuffix(".zarr")
    raw_frame_start = args.raw_frame_start
    if raw_frame_start is None:
        raw_frame_start = _raw_frame_start_from_episode_name(episode_name)

    num_frames = int(store[f"{args.side}.obs_ee_pose"].shape[0])
    old_raw_mats, new_mats, valid_local_indices = _collect_pose_mats(
        raw_records,
        side=args.side,
        raw_frame_start=raw_frame_start,
        num_frames=num_frames,
    )
    if len(valid_local_indices) < 4:
        raise RuntimeError("Not enough valid frames to compare EE definitions")

    hand_world = np.asarray(store[f"{args.side}.obs_ee_pose"][valid_local_indices], dtype=np.float64)
    head_world = np.asarray(store["obs_head_pose"][valid_local_indices], dtype=np.float64)
    zarr_rel_mats = np.linalg.inv(_xyzwxyz_to_matrix(head_world)) @ _xyzwxyz_to_matrix(hand_world)
    old_raw_delta_deg = (
        R.from_matrix(old_raw_mats[:, :3, :3]) * R.from_matrix(zarr_rel_mats[:, :3, :3]).inv()
    ).magnitude() * 180.0 / np.pi

    report = {
        "episode_name": episode_name,
        "zarr": str(zarr_path),
        "raw": str(raw_path),
        "side": args.side,
        "raw_frame_start": int(raw_frame_start),
        "num_zarr_frames": num_frames,
        "num_valid_frames": int(len(valid_local_indices)),
        "valid_local_index_first": int(valid_local_indices[0]),
        "valid_local_index_last": int(valid_local_indices[-1]),
        "definitions": {
            "old_middle_ring": "origin=mean(index/middle/ring/pinky MCP); x=origin->middle_mcp; y_seed=origin->ring_mcp; z=cross(x,y)",
            "wrist_mcp_spread": "origin=mean(index/middle/ring/pinky MCP); x=wrist->origin; y_seed=index_mcp->pinky_mcp; z=cross(x,y)",
        },
        "zarr_vs_raw_old_geodesic_deg": _series_stats(old_raw_delta_deg),
        "old_middle_ring": _metrics(old_raw_mats, high_freq_window=args.high_freq_window, top_k=args.top_k),
        "wrist_mcp_spread": _metrics(new_mats, high_freq_window=args.high_freq_window, top_k=args.top_k),
        "focused_window": {
            "start": int(args.window_start),
            "end": int(args.window_end),
            "old_middle_ring": _window_payload(old_raw_mats, args.window_start, args.window_end),
            "wrist_mcp_spread": _window_payload(new_mats, args.window_start, args.window_end),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--side", default="right", choices=("left", "right"))
    parser.add_argument("--raw-frame-start", type=int, default=None)
    parser.add_argument("--high-freq-window", type=int, default=9)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--window-start", type=int, default=66)
    parser.add_argument("--window-end", type=int, default=74)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    report = build_report(args)
    output_json = Path(args.output_json) if args.output_json else None
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = {
        "output_json": str(output_json) if output_json is not None else None,
        "episode_name": report["episode_name"],
        "num_valid_frames": report["num_valid_frames"],
        "zarr_vs_raw_old_geodesic_deg_p99": report["zarr_vs_raw_old_geodesic_deg"]["p99"],
        "old_rot_step_p99": report["old_middle_ring"]["rot_step_deg"]["p99"],
        "new_rot_step_p99": report["wrist_mcp_spread"]["rot_step_deg"]["p99"],
        "old_rot_accel_p99": report["old_middle_ring"]["rot_accel_deg"]["p99"],
        "new_rot_accel_p99": report["wrist_mcp_spread"]["rot_accel_deg"]["p99"],
        "old_hf_rot_mean": float(np.mean(list(report["old_middle_ring"]["high_freq_ratio"].values()))),
        "new_hf_rot_mean": float(np.mean(list(report["wrist_mcp_spread"]["high_freq_ratio"].values()))),
        "old_3deg_triples": report["old_middle_ring"]["alternation_counts"]["3deg"]["triples_by_dim"],
        "new_3deg_triples": report["wrist_mcp_spread"]["alternation_counts"]["3deg"]["triples_by_dim"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
