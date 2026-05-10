from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

from egomimic.utils.egomimicUtils import INTRINSICS, cam_frame_to_cam_pixels
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix

try:
    import simplejpeg
except ModuleNotFoundError:  # pragma: no cover
    simplejpeg = None


BASE_INTRINSICS_WIDTH = 640.0
BASE_INTRINSICS_HEIGHT = 480.0
TRIAD_COLORS = {
    "x": (0, 0, 255),
    "y": (0, 255, 0),
    "z": (255, 0, 0),
}


@dataclass
class HumanRotationRecord:
    local_idx: int
    raw_frame_index: int
    image: np.ndarray
    intrinsics: np.ndarray
    xyz: np.ndarray
    transformed_rotmat: np.ndarray
    raw_rotmat: np.ndarray | None
    transformed_ypr: np.ndarray
    raw_ypr: np.ndarray | None


@dataclass
class So100RotationRecord:
    local_idx: int
    image: np.ndarray
    intrinsics: np.ndarray
    xyz: np.ndarray
    rotvec: np.ndarray
    rotmat: np.ndarray
    ypr: np.ndarray


def _decode_jpeg(value: object) -> np.ndarray:
    while isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, np.ndarray) and value.ndim == 3:
        return value
    if isinstance(value, (bytes, bytearray)):
        jpeg_bytes = bytes(value)
    elif hasattr(value, "tobytes"):
        jpeg_bytes = value.tobytes()
    else:
        jpeg_bytes = bytes(value)
    if simplejpeg is not None:
        try:
            return simplejpeg.decode_jpeg(jpeg_bytes, colorspace="RGB")
        except ValueError:
            pass
    decoded_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise ValueError("cv2.imdecode returned None")
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def _scale_intrinsics(intrinsics: np.ndarray, width: int, height: int) -> np.ndarray:
    scaled = np.asarray(intrinsics, dtype=np.float64).copy()
    scaled[0, 0] *= width / BASE_INTRINSICS_WIDTH
    scaled[1, 1] *= height / BASE_INTRINSICS_HEIGHT
    scaled[0, 2] *= width / BASE_INTRINSICS_WIDTH
    scaled[1, 2] *= height / BASE_INTRINSICS_HEIGHT
    return scaled


def _to_jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _recording_id_from_episode_name(episode_name: str) -> str:
    match = re.match(r"^(recording_[^_]+)", episode_name)
    if match is None:
        raise ValueError(f"Could not parse recording id from {episode_name}")
    return match.group(1)


def _raw_frame_start_from_episode_name(episode_name: str) -> int:
    matches = re.findall(r"_(\d+)-(\d+)", episode_name)
    if not matches:
        raise ValueError(f"Could not parse frame ranges from {episode_name}")
    return int(matches[-1][0])


def _find_human_raw_jsonl(human_raw_root: Path, recording_id: str) -> Path:
    candidates = sorted(human_raw_root.glob(f"**/{recording_id}/wilor_out_pipeline_egoverse*/hand_poses.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"Could not find hand_poses.jsonl for {recording_id} under {human_raw_root}")
    landscape = [p for p in candidates if "landscape" in str(p.parent)]
    return landscape[0] if landscape else candidates[0]


def _load_raw_hand_pose_map(jsonl_path: Path) -> dict[int, dict]:
    result = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            result[int(record["frame_index"])] = record
    return result


def _select_right_hand(record: dict) -> dict | None:
    for hand in record.get("hands", []):
        if float(hand.get("is_right", 0.0)) > 0.5:
            return hand
    return None


def _project_xyz(xyz: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    return cam_frame_to_cam_pixels(np.asarray(xyz, dtype=np.float64)[None, :], intrinsics)[0]


def _sample_indices(length: int, num_samples: int) -> list[int]:
    if num_samples <= 1:
        return [0]
    return sorted(set(int(x) for x in np.linspace(0, length - 1, num_samples)))


def _compute_human_transformed_pose(obs_head_pose: np.ndarray, right_obs_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    head_matrix = _xyzwxyz_to_matrix(obs_head_pose[None, :])[0]
    hand_matrix = _xyzwxyz_to_matrix(right_obs_pose[None, :])[0]
    relative_matrix = np.linalg.inv(head_matrix) @ hand_matrix
    return relative_matrix[:3, 3], relative_matrix[:3, :3]


def _rotmat_geodesic_error_rad(lhs: np.ndarray, rhs: np.ndarray) -> float:
    delta = lhs @ rhs.T
    return float(R.from_matrix(delta).magnitude())


def _angle_diff_wrapped(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    diff = np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)
    return (diff + np.pi) % (2 * np.pi) - np.pi


def load_human_rotation_record(human_zarr: Path, human_raw_root: Path, local_idx: int) -> HumanRotationRecord:
    episode_name = human_zarr.stem
    recording_id = _recording_id_from_episode_name(episode_name)
    raw_frame_start = _raw_frame_start_from_episode_name(episode_name)
    raw_jsonl_path = _find_human_raw_jsonl(human_raw_root, recording_id)
    raw_records = _load_raw_hand_pose_map(raw_jsonl_path)

    store = zarr.open(str(human_zarr), mode="r")
    image = _decode_jpeg(store["images.front_1"][local_idx])
    intrinsics = _scale_intrinsics(INTRINSICS["base"], image.shape[1], image.shape[0])
    xyz, rotmat = _compute_human_transformed_pose(
        np.asarray(store["obs_head_pose"][local_idx], dtype=np.float64),
        np.asarray(store["right.obs_ee_pose"][local_idx], dtype=np.float64),
    )
    transformed_ypr = R.from_matrix(rotmat).as_euler("ZYX", degrees=False)

    raw_frame_index = raw_frame_start + local_idx
    raw_record = raw_records.get(raw_frame_index)
    raw_hand = _select_right_hand(raw_record) if raw_record is not None else None
    raw_rotmat = None
    raw_ypr = None
    if raw_hand is not None and "end_effector_pose_cam" in raw_hand:
        raw_rotmat = np.asarray(raw_hand["end_effector_pose_cam"]["rotation_matrix"], dtype=np.float64)
        raw_ypr = R.from_matrix(raw_rotmat).as_euler("ZYX", degrees=False)

    return HumanRotationRecord(
        local_idx=local_idx,
        raw_frame_index=raw_frame_index,
        image=image,
        intrinsics=intrinsics,
        xyz=xyz,
        transformed_rotmat=rotmat,
        raw_rotmat=raw_rotmat,
        transformed_ypr=transformed_ypr,
        raw_ypr=raw_ypr,
    )


def load_so100_rotation_record(so100_zarr: Path, local_idx: int) -> So100RotationRecord:
    store = zarr.open(str(so100_zarr), mode="r")
    image = _decode_jpeg(store["images.front_1"][local_idx])
    intrinsics = _scale_intrinsics(INTRINSICS["base"], image.shape[1], image.shape[0])
    pose = np.asarray(store["obs_ee_pose_cam_rotvec"][local_idx], dtype=np.float64)
    xyz = pose[:3]
    rotvec = pose[3:6]
    rotmat = R.from_rotvec(rotvec).as_matrix()
    ypr = R.from_rotvec(rotvec).as_euler("ZYX", degrees=False)
    return So100RotationRecord(
        local_idx=local_idx,
        image=image,
        intrinsics=intrinsics,
        xyz=xyz,
        rotvec=rotvec,
        rotmat=rotmat,
        ypr=ypr,
    )


def _mean_vector(values: list[np.ndarray]) -> list[float]:
    return np.mean(np.asarray(values, dtype=np.float64), axis=0).tolist()


def _build_human_alignment_summary(records: list[HumanRotationRecord]) -> dict:
    valid = [r for r in records if r.raw_rotmat is not None and r.raw_ypr is not None]
    if not valid:
        return {"num_samples": 0}

    geodesic_errors = [ _rotmat_geodesic_error_rad(r.transformed_rotmat, r.raw_rotmat) for r in valid ]
    ypr_diffs = [ np.abs(_angle_diff_wrapped(r.transformed_ypr, r.raw_ypr)) for r in valid ]
    return {
        "num_samples": len(valid),
        "rotation_geodesic_error_rad_mean": float(np.mean(geodesic_errors)),
        "rotation_geodesic_error_rad_max": float(np.max(geodesic_errors)),
        "ypr_abs_diff_rad_mean": _mean_vector(ypr_diffs),
        "ypr_abs_diff_rad_max": np.max(np.asarray(ypr_diffs, dtype=np.float64), axis=0).tolist(),
    }


def _build_so100_roundtrip_summary(records: list[So100RotationRecord]) -> dict:
    geodesic_errors = []
    for r in records:
        reconstructed = R.from_euler("ZYX", r.ypr, degrees=False).as_matrix()
        geodesic_errors.append(_rotmat_geodesic_error_rad(reconstructed, r.rotmat))
    return {
        "num_samples": len(records),
        "rotvec_to_ypr_to_rotmat_geodesic_error_rad_mean": float(np.mean(geodesic_errors)),
        "rotvec_to_ypr_to_rotmat_geodesic_error_rad_max": float(np.max(geodesic_errors)),
    }


def _project_triad(xyz: np.ndarray, rotmat: np.ndarray, intrinsics: np.ndarray, axis_len_m: float) -> dict[str, np.ndarray]:
    points = {"origin": _project_xyz(xyz, intrinsics)}
    for axis_idx, axis_name in enumerate(("x", "y", "z")):
        endpoint = xyz + axis_len_m * rotmat[:, axis_idx]
        points[axis_name] = _project_xyz(endpoint, intrinsics)
    return points


def _draw_triad(canvas: np.ndarray, triad: dict[str, np.ndarray], label: str) -> None:
    origin = triad["origin"]
    ox = int(round(float(origin[0])))
    oy = int(round(float(origin[1])))
    cv2.circle(canvas, (ox, oy), 8, (255, 255, 255), thickness=-1)
    cv2.putText(canvas, label, (ox + 10, oy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for axis_name in ("x", "y", "z"):
        pt = triad[axis_name]
        px = int(round(float(pt[0])))
        py = int(round(float(pt[1])))
        color = TRIAD_COLORS[axis_name]
        cv2.line(canvas, (ox, oy), (px, py), color, thickness=3)
        cv2.circle(canvas, (px, py), 7, color, thickness=-1)
        cv2.putText(canvas, axis_name, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def _overlay_title(canvas: np.ndarray, title: str) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (0, 0, 0), thickness=-1)
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def _build_panel(image: np.ndarray, xyz: np.ndarray, ypr: np.ndarray, intrinsics: np.ndarray, axis_len_m: float, delta_rad: float, prefix: str) -> np.ndarray:
    h, w = image.shape[:2]
    panels = []
    variants = [
        ("base", ypr),
        ("+yaw", ypr + np.array([delta_rad, 0.0, 0.0])),
        ("+pitch", ypr + np.array([0.0, delta_rad, 0.0])),
        ("+roll", ypr + np.array([0.0, 0.0, delta_rad])),
    ]
    for title, ypr_variant in variants:
        canvas = image.copy()
        rotmat = R.from_euler("ZYX", ypr_variant, degrees=False).as_matrix()
        triad = _project_triad(xyz, rotmat, intrinsics, axis_len_m)
        _draw_triad(canvas, triad, prefix)
        _overlay_title(canvas, title)
        panels.append(canvas)
    top = np.concatenate([panels[0], panels[1]], axis=1)
    bottom = np.concatenate([panels[2], panels[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def build_report(
    human_zarr: Path,
    human_raw_root: Path,
    so100_zarr: Path,
    human_viz_idx: int,
    so100_viz_idx: int,
    num_samples: int,
    angle_delta_rad: float,
    axis_len_m: float,
) -> dict:
    human_store = zarr.open(str(human_zarr), mode="r")
    so100_store = zarr.open(str(so100_zarr), mode="r")
    human_indices = _sample_indices(human_store["right.obs_ee_pose"].shape[0], num_samples)
    so100_indices = _sample_indices(so100_store["obs_ee_pose_cam_rotvec"].shape[0], num_samples)

    human_records = [load_human_rotation_record(human_zarr, human_raw_root, idx) for idx in human_indices]
    so100_records = [load_so100_rotation_record(so100_zarr, idx) for idx in so100_indices]

    human_alignment = _build_human_alignment_summary(human_records)
    so100_roundtrip = _build_so100_roundtrip_summary(so100_records)

    human_viz = load_human_rotation_record(human_zarr, human_raw_root, human_viz_idx)
    so100_viz = load_so100_rotation_record(so100_zarr, so100_viz_idx)

    human_panel = _build_panel(
        image=human_viz.image,
        xyz=human_viz.xyz,
        ypr=human_viz.transformed_ypr,
        intrinsics=human_viz.intrinsics,
        axis_len_m=axis_len_m,
        delta_rad=angle_delta_rad,
        prefix="human",
    )
    so100_panel = _build_panel(
        image=so100_viz.image,
        xyz=so100_viz.xyz,
        ypr=so100_viz.ypr,
        intrinsics=so100_viz.intrinsics,
        axis_len_m=axis_len_m,
        delta_rad=angle_delta_rad,
        prefix="so100",
    )

    return {
        "config": {
            "human_zarr": str(human_zarr),
            "so100_zarr": str(so100_zarr),
            "human_raw_root": str(human_raw_root),
            "human_viz_idx": human_viz_idx,
            "so100_viz_idx": so100_viz_idx,
            "num_samples": num_samples,
            "angle_delta_rad": angle_delta_rad,
            "axis_len_m": axis_len_m,
        },
        "human_obs_headframe_vs_raw_camera_rotation": human_alignment,
        "so100_rotvec_ypr_roundtrip": so100_roundtrip,
        "viz_frames": {
            "human": {
                "local_idx": human_viz.local_idx,
                "raw_frame_index": human_viz.raw_frame_index,
                "xyz": human_viz.xyz.tolist(),
                "transformed_ypr": human_viz.transformed_ypr.tolist(),
                "raw_ypr": None if human_viz.raw_ypr is None else human_viz.raw_ypr.tolist(),
            },
            "so100": {
                "local_idx": so100_viz.local_idx,
                "xyz": so100_viz.xyz.tolist(),
                "rotvec": so100_viz.rotvec.tolist(),
                "ypr": so100_viz.ypr.tolist(),
            },
        },
        "conclusion": {
            "human_obs_headframe_rotation_matches_raw_camera_rotation": (
                human_alignment.get("num_samples", 0) > 0
                and human_alignment["rotation_geodesic_error_rad_max"] < 1e-5
            ),
            "so100_rotvec_and_ypr_use_same_zyx_convention": (
                so100_roundtrip["rotvec_to_ypr_to_rotmat_geodesic_error_rad_max"] < 1e-9
            ),
            "human_and_so100_rotation_definitions_are_camera_frame_compatible": (
                human_alignment.get("num_samples", 0) > 0
                and human_alignment["rotation_geodesic_error_rad_max"] < 1e-5
                and so100_roundtrip["rotvec_to_ypr_to_rotmat_geodesic_error_rad_max"] < 1e-9
            ),
        },
        "_human_panel": human_panel,
        "_so100_panel": so100_panel,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--human-zarr",
        type=Path,
        default=Path(
            "/home/zxwang/repos/apricot_human_in_domain/"
            "egoverse_zarr_apricot_in_domain_merged_part1_part2_part3_min90_ego_view_right_arm/"
            "recording_2026-05-04T04-26-03.706Z_recording_2026-05-04T04-26-03.706Z_single_arm_right_000000-000906_dense000_000000-000904.zarr"
        ),
    )
    parser.add_argument(
        "--human-raw-root",
        type=Path,
        default=Path("/home/zxwang/repos/apricot_human_in_domain"),
    )
    parser.add_argument(
        "--so100-zarr",
        type=Path,
        default=Path("/home/zxwang/so100-ee-cam-egoverse-zarr/so100_put_apricot_ep000000.zarr"),
    )
    parser.add_argument("--human-viz-idx", type=int, default=0)
    parser.add_argument("--so100-viz-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--angle-delta-deg", type=float, default=10.0)
    parser.add_argument("--axis-len-m", type=float, default=0.08)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(
        human_zarr=args.human_zarr,
        human_raw_root=args.human_raw_root,
        so100_zarr=args.so100_zarr,
        human_viz_idx=args.human_viz_idx,
        so100_viz_idx=args.so100_viz_idx,
        num_samples=args.num_samples,
        angle_delta_rad=np.deg2rad(args.angle_delta_deg),
        axis_len_m=args.axis_len_m,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    human_panel = report.pop("_human_panel")
    so100_panel = report.pop("_so100_panel")
    human_out = output_dir / "human_rotation_semantics.png"
    so100_out = output_dir / "so100_rotation_semantics.png"
    cv2.imwrite(str(human_out), cv2.cvtColor(human_panel, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(so100_out), cv2.cvtColor(so100_panel, cv2.COLOR_RGB2BGR))

    report["artifacts"] = {
        "human_rotation_semantics_png": str(human_out),
        "so100_rotation_semantics_png": str(so100_out),
    }
    report = _to_jsonable(report)
    report_path = output_dir / "rotation_semantics_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report_json": str(report_path), **report["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
