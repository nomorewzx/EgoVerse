from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egomimic.scripts.diagnostics.probe_human_ee_definition_noise import (
    _old_middle_ring_pose_cam,
    _raw_frame_start_from_episode_name,
    _select_hand,
    _wrist_mcp_spread_pose_cam,
)


DIM_NAMES = ("yaw", "pitch", "roll")
PALM_INDICES = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
MCP_INDICES = np.asarray([5, 9, 13, 17], dtype=np.int64)
BONE_PAIRS = np.asarray(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ],
    dtype=np.int64,
)
EPS = 1e-12


def _load_jsonl_by_frame(path: Path) -> dict[int, dict[str, Any]]:
    records = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records[int(record["frame_index"])] = record
    return records


def _rotation_step_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    return float((R.from_matrix(rot_b) * R.from_matrix(rot_a).inv()).magnitude() * 180.0 / np.pi)


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    cov = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    trans = target_centroid - rot @ source_centroid
    aligned = (rot @ source.T).T + trans
    residual = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return rot, trans, residual


def _kabsch_metrics(source: np.ndarray, target: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    rot, _, residual = _kabsch(source[indices], target[indices])
    hand_scale = _hand_scale(source, target)
    return {
        "rot_deg": float(R.from_matrix(rot).magnitude() * 180.0 / np.pi),
        "residual_m": float(residual),
        "residual_norm": float(residual / max(hand_scale, EPS)),
    }


def _hand_scale(source: np.ndarray, target: np.ndarray) -> float:
    def one(joints: np.ndarray) -> float:
        mcp_center = np.mean(joints[MCP_INDICES], axis=0)
        palm_width = np.linalg.norm(joints[17] - joints[5])
        palm_length = np.linalg.norm(mcp_center - joints[0])
        return float(0.5 * (palm_width + palm_length))

    return float(0.5 * (one(source) + one(target)))


def _bone_lengths(joints: np.ndarray) -> np.ndarray:
    return np.linalg.norm(joints[BONE_PAIRS[:, 1]] - joints[BONE_PAIRS[:, 0]], axis=1)


def _bone_delta_metrics(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    src = _bone_lengths(source)
    tgt = _bone_lengths(target)
    diff = tgt - src
    rel = diff / np.maximum(0.5 * (src + tgt), EPS)
    return {
        "rms_m": float(np.sqrt(np.mean(diff**2))),
        "max_abs_m": float(np.max(np.abs(diff))),
        "rms_rel": float(np.sqrt(np.mean(rel**2))),
        "max_abs_rel": float(np.max(np.abs(rel))),
    }


def _centered_rms_delta(source: np.ndarray, target: np.ndarray, indices: np.ndarray) -> float:
    src = source[indices] - np.mean(source[indices], axis=0, keepdims=True)
    tgt = target[indices] - np.mean(target[indices], axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum((tgt - src) ** 2, axis=1))))


def _keypoints2d_metrics(hand_a: dict[str, Any], hand_b: dict[str, Any]) -> dict[str, float]:
    if "keypoints_2d" not in hand_a or "keypoints_2d" not in hand_b:
        return {
            "mean_px": float("nan"),
            "palm_mean_px": float("nan"),
            "centered_mean_px": float("nan"),
        }
    src = np.asarray(hand_a["keypoints_2d"], dtype=np.float64)
    tgt = np.asarray(hand_b["keypoints_2d"], dtype=np.float64)
    if src.shape[0] < 21 or tgt.shape[0] < 21:
        return {
            "mean_px": float("nan"),
            "palm_mean_px": float("nan"),
            "centered_mean_px": float("nan"),
        }
    direct = np.linalg.norm(tgt - src, axis=1)
    centered_src = src - np.mean(src[MCP_INDICES], axis=0, keepdims=True)
    centered_tgt = tgt - np.mean(tgt[MCP_INDICES], axis=0, keepdims=True)
    centered = np.linalg.norm(centered_tgt - centered_src, axis=1)
    return {
        "mean_px": float(np.mean(direct)),
        "palm_mean_px": float(np.mean(direct[PALM_INDICES])),
        "centered_mean_px": float(np.mean(centered)),
    }


def _series_stats(values: list[float] | np.ndarray) -> dict[str, float]:
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


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _collect_frame_rows(
    raw_path: Path,
    *,
    side: str,
    raw_frame_start: int,
    max_frames: int | None,
) -> list[dict[str, Any]]:
    records = _load_jsonl_by_frame(raw_path)
    if max_frames is None:
        frame_indices = sorted(records)
    else:
        frame_indices = [raw_frame_start + idx for idx in range(max_frames)]

    rows = []
    for raw_idx in frame_indices:
        record = records.get(raw_idx)
        if record is None:
            continue
        local_idx = raw_idx - raw_frame_start
        if local_idx < 0:
            continue
        hand = _select_hand(record, side)
        if hand is None or "joints_3d_cam" not in hand:
            continue
        joints = np.asarray(hand["joints_3d_cam"], dtype=np.float64)
        if joints.shape[0] < 21:
            continue
        old_pose = _old_middle_ring_pose_cam(joints)
        new_pose = _wrist_mcp_spread_pose_cam(joints)
        if old_pose is None or new_pose is None:
            continue
        track = (record.get("gripper_tracks") or {}).get(side) or {}
        rows.append(
            {
                "raw_frame_index": int(raw_idx),
                "local_index": int(local_idx),
                "joints": joints,
                "hand": hand,
                "old_rot": old_pose[1],
                "new_rot": new_pose[1],
                "track_state": str(track.get("track_state", "")),
                "source_hand_index": track.get("source_hand_index"),
                "det_score": _safe_float(hand.get("det_score", track.get("det_score"))),
            }
        )
    rows.sort(key=lambda row: row["local_index"])
    return rows


def _pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for lhs, rhs in zip(rows[:-1], rows[1:]):
        if int(rhs["local_index"]) != int(lhs["local_index"]) + 1:
            continue
        joints_a = lhs["joints"]
        joints_b = rhs["joints"]
        all21 = _kabsch_metrics(joints_a, joints_b, np.arange(21, dtype=np.int64))
        palm5 = _kabsch_metrics(joints_a, joints_b, PALM_INDICES)
        mcp4 = _kabsch_metrics(joints_a, joints_b, MCP_INDICES)
        bone = _bone_delta_metrics(joints_a, joints_b)
        key2d = _keypoints2d_metrics(lhs["hand"], rhs["hand"])
        source_idx_changed = lhs["source_hand_index"] != rhs["source_hand_index"]
        pairs.append(
            {
                "local_index": int(lhs["local_index"]),
                "raw_frame_index": int(lhs["raw_frame_index"]),
                "old_ee_step_deg": _rotation_step_deg(lhs["old_rot"], rhs["old_rot"]),
                "new_ee_step_deg": _rotation_step_deg(lhs["new_rot"], rhs["new_rot"]),
                "kabsch_all21_rot_deg": all21["rot_deg"],
                "kabsch_all21_residual_m": all21["residual_m"],
                "kabsch_all21_residual_norm": all21["residual_norm"],
                "kabsch_palm5_rot_deg": palm5["rot_deg"],
                "kabsch_palm5_residual_m": palm5["residual_m"],
                "kabsch_palm5_residual_norm": palm5["residual_norm"],
                "kabsch_mcp4_rot_deg": mcp4["rot_deg"],
                "kabsch_mcp4_residual_m": mcp4["residual_m"],
                "kabsch_mcp4_residual_norm": mcp4["residual_norm"],
                "bone_rms_delta_m": bone["rms_m"],
                "bone_max_abs_delta_m": bone["max_abs_m"],
                "bone_rms_delta_rel": bone["rms_rel"],
                "bone_max_abs_delta_rel": bone["max_abs_rel"],
                "centered_all21_rms_delta_m": _centered_rms_delta(
                    joints_a, joints_b, np.arange(21, dtype=np.int64)
                ),
                "centered_palm5_rms_delta_m": _centered_rms_delta(joints_a, joints_b, PALM_INDICES),
                "keypoints2d_mean_px": key2d["mean_px"],
                "keypoints2d_palm_mean_px": key2d["palm_mean_px"],
                "keypoints2d_centered_mean_px": key2d["centered_mean_px"],
                "det_score_min": float(np.nanmin([lhs["det_score"], rhs["det_score"]])),
                "source_hand_index_changed": bool(source_idx_changed),
                "track_state_pair": f"{lhs['track_state']}->{rhs['track_state']}",
            }
        )
    return pairs


def _top_rows(rows: list[dict[str, Any]], key: str, top_k: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: float(row.get(key, float("nan"))), reverse=True)
    return [_jsonable_row(row) for row in ranked[:top_k]]


def _jsonable_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        elif isinstance(value, np.ndarray):
            continue
        else:
            out[key] = value
    return out


def _subset_summary(rows: list[dict[str, Any]], mask: np.ndarray) -> dict[str, Any]:
    if not rows or not np.any(mask):
        return {"num_pairs": 0}
    selected = [rows[idx] for idx in np.flatnonzero(mask)]
    keys = [
        "old_ee_step_deg",
        "new_ee_step_deg",
        "kabsch_all21_rot_deg",
        "kabsch_all21_residual_norm",
        "kabsch_palm5_rot_deg",
        "kabsch_palm5_residual_norm",
        "bone_rms_delta_rel",
        "keypoints2d_centered_mean_px",
        "det_score_min",
    ]
    summary = {"num_pairs": len(selected)}
    for key in keys:
        summary[key] = _series_stats([row[key] for row in selected])
    summary["source_hand_index_changed_count"] = int(
        sum(bool(row["source_hand_index_changed"]) for row in selected)
    )
    return summary


def build_episode_report(
    *,
    episode_name: str,
    raw_path: Path,
    raw_frame_start: int,
    side: str,
    max_frames: int | None,
    spike_quantile: float,
    top_k: int,
) -> dict[str, Any]:
    frames = _collect_frame_rows(
        raw_path,
        side=side,
        raw_frame_start=raw_frame_start,
        max_frames=max_frames,
    )
    pairs = _pair_rows(frames)
    if not pairs:
        raise RuntimeError(f"No valid consecutive frame pairs found for {episode_name}")

    old_steps = np.asarray([row["old_ee_step_deg"] for row in pairs], dtype=np.float64)
    spike_threshold = float(np.quantile(old_steps, spike_quantile))
    spike_mask = old_steps >= spike_threshold

    report = {
        "episode_name": episode_name,
        "raw": str(raw_path),
        "side": side,
        "raw_frame_start": int(raw_frame_start),
        "num_valid_frames": int(len(frames)),
        "num_valid_pairs": int(len(pairs)),
        "spike_quantile": float(spike_quantile),
        "spike_threshold_old_ee_step_deg": spike_threshold,
        "summary_all_pairs": _subset_summary(pairs, np.ones(len(pairs), dtype=bool)),
        "summary_spike_pairs": _subset_summary(pairs, spike_mask),
        "correlations": {
            "old_ee_vs_kabsch_all21_rot": _corr(
                old_steps, np.asarray([row["kabsch_all21_rot_deg"] for row in pairs])
            ),
            "old_ee_vs_kabsch_palm5_rot": _corr(
                old_steps, np.asarray([row["kabsch_palm5_rot_deg"] for row in pairs])
            ),
            "old_ee_vs_all21_residual_norm": _corr(
                old_steps, np.asarray([row["kabsch_all21_residual_norm"] for row in pairs])
            ),
            "old_ee_vs_bone_rms_delta_rel": _corr(
                old_steps, np.asarray([row["bone_rms_delta_rel"] for row in pairs])
            ),
            "old_ee_vs_keypoints2d_centered_mean_px": _corr(
                old_steps, np.asarray([row["keypoints2d_centered_mean_px"] for row in pairs])
            ),
        },
        "top_old_ee_step_pairs": _top_rows(pairs, "old_ee_step_deg", top_k),
        "top_all21_residual_pairs": _top_rows(pairs, "kabsch_all21_residual_norm", top_k),
    }
    return report


def _episode_name_from_zarr(path: Path) -> str:
    return path.name.removesuffix(".zarr")


def _num_frames_from_zarr(path: Path, side: str) -> int:
    store = zarr.open(str(path), mode="r")
    return int(store[f"{side}.obs_ee_pose"].shape[0])


def _recording_id_from_episode_name(episode_name: str) -> str:
    match = re.match(r"^(recording_[^_]+)", episode_name)
    if match is None:
        raise ValueError(f"Cannot parse recording id from {episode_name}")
    return match.group(1)


def _find_raw(raw_root: Path, episode_name: str) -> Path:
    recording_id = _recording_id_from_episode_name(episode_name)
    candidates = sorted(
        raw_root.glob(
            f"**/{recording_id}/wilor_out_pipeline_egoverse*/selected_pose_stream/data/hand_poses_gripper.jsonl"
        )
    )
    if not candidates:
        raise FileNotFoundError(f"No hand_poses_gripper.jsonl found for {recording_id} under {raw_root}")
    landscape = [path for path in candidates if "landscape" in str(path)]
    return landscape[0] if landscape else candidates[0]


def _flatten_summary(report: dict[str, Any]) -> dict[str, Any]:
    all_pairs = report["summary_all_pairs"]
    spike = report["summary_spike_pairs"]
    corr = report["correlations"]
    return {
        "episode_name": report["episode_name"],
        "num_valid_pairs": report["num_valid_pairs"],
        "spike_threshold_old_ee_step_deg": report["spike_threshold_old_ee_step_deg"],
        "old_step_p99": all_pairs["old_ee_step_deg"]["p99"],
        "kabsch_all21_rot_p99": all_pairs["kabsch_all21_rot_deg"]["p99"],
        "kabsch_palm5_rot_p99": all_pairs["kabsch_palm5_rot_deg"]["p99"],
        "all21_residual_norm_p99": all_pairs["kabsch_all21_residual_norm"]["p99"],
        "palm5_residual_norm_p99": all_pairs["kabsch_palm5_residual_norm"]["p99"],
        "bone_rms_delta_rel_p99": all_pairs["bone_rms_delta_rel"]["p99"],
        "keypoints2d_centered_mean_px_p99": all_pairs["keypoints2d_centered_mean_px"]["p99"],
        "spike_old_step_mean": spike["old_ee_step_deg"]["mean"],
        "spike_kabsch_all21_rot_mean": spike["kabsch_all21_rot_deg"]["mean"],
        "spike_all21_residual_norm_mean": spike["kabsch_all21_residual_norm"]["mean"],
        "spike_bone_rms_delta_rel_mean": spike["bone_rms_delta_rel"]["mean"],
        "spike_keypoints2d_centered_mean_px_mean": spike["keypoints2d_centered_mean_px"]["mean"],
        "spike_source_hand_index_changed_count": spike["source_hand_index_changed_count"],
        "corr_old_ee_all21_rot": corr["old_ee_vs_kabsch_all21_rot"],
        "corr_old_ee_palm5_rot": corr["old_ee_vs_kabsch_palm5_rot"],
        "corr_old_ee_all21_residual": corr["old_ee_vs_all21_residual_norm"],
        "corr_old_ee_bone_delta": corr["old_ee_vs_bone_rms_delta_rel"],
        "corr_old_ee_2d_centered": corr["old_ee_vs_keypoints2d_centered_mean_px"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", nargs="*", default=None)
    parser.add_argument("--raw", default=None)
    parser.add_argument("--raw-root", default="/home/zxwang/repos/apricot_human_in_domain")
    parser.add_argument("--episode-name", default=None)
    parser.add_argument("--side", default="right", choices=("left", "right"))
    parser.add_argument("--raw-frame-start", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--spike-quantile", type=float, default=0.99)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    jobs = []
    if args.zarr:
        for zarr_arg in args.zarr:
            zarr_path = Path(zarr_arg)
            episode_name = _episode_name_from_zarr(zarr_path)
            raw_path = Path(args.raw) if args.raw else _find_raw(Path(args.raw_root), episode_name)
            raw_frame_start = args.raw_frame_start
            if raw_frame_start is None:
                raw_frame_start = _raw_frame_start_from_episode_name(episode_name)
            max_frames = args.max_frames
            if max_frames is None:
                max_frames = _num_frames_from_zarr(zarr_path, args.side)
            jobs.append((episode_name, raw_path, raw_frame_start, max_frames))
    else:
        if args.raw is None:
            raise ValueError("Provide --zarr or --raw")
        episode_name = args.episode_name or Path(args.raw).parent.parent.parent.parent.name
        raw_frame_start = args.raw_frame_start
        if raw_frame_start is None:
            raw_frame_start = _raw_frame_start_from_episode_name(episode_name)
        jobs.append((episode_name, Path(args.raw), raw_frame_start, args.max_frames))

    reports = [
        build_episode_report(
            episode_name=episode_name,
            raw_path=raw_path,
            raw_frame_start=raw_frame_start,
            side=args.side,
            max_frames=max_frames,
            spike_quantile=args.spike_quantile,
            top_k=args.top_k,
        )
        for episode_name, raw_path, raw_frame_start, max_frames in jobs
    ]
    rows = [_flatten_summary(report) for report in reports]

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"reports": reports, "summary_rows": rows}, indent=2), encoding="utf-8")

    output_csv = Path(args.output_csv) if args.output_csv else output_json.with_suffix(".summary.csv")
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"output_json": str(output_json), "output_csv": str(output_csv), "summary_rows": rows}, indent=2))


if __name__ == "__main__":
    main()
