from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import zarr

from egomimic.utils.egomimicUtils import INTRINSICS, cam_frame_to_cam_pixels
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix

try:
    import simplejpeg
except ModuleNotFoundError:  # pragma: no cover
    simplejpeg = None


BASE_INTRINSICS_WIDTH = 640.0
BASE_INTRINSICS_HEIGHT = 480.0
AXIS_COLORS = {
    "base": (255, 255, 255),
    "raw_base": (0, 255, 255),
    "x": (0, 0, 255),
    "y": (0, 255, 0),
    "z": (255, 0, 0),
}


@dataclass
class HumanFrameRecord:
    local_idx: int
    raw_frame_index: int
    image: np.ndarray
    intrinsics: np.ndarray
    transformed_xyz: np.ndarray
    transformed_px: np.ndarray
    raw_cam_t: np.ndarray | None
    raw_px: np.ndarray | None


@dataclass
class So100FrameRecord:
    local_idx: int
    image: np.ndarray
    intrinsics: np.ndarray
    camera_xyz: np.ndarray
    camera_px: np.ndarray


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


def _project_xyz(xyz: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    return cam_frame_to_cam_pixels(np.asarray(xyz, dtype=np.float64)[None, :], intrinsics)[0]


def _perturb_xyz(xyz: np.ndarray, axis: int, delta_m: float) -> np.ndarray:
    out = np.asarray(xyz, dtype=np.float64).copy()
    out[axis] += delta_m
    return out


def _axis_delta_pixels(xyz: np.ndarray, intrinsics: np.ndarray, delta_m: float) -> dict[str, list[float]]:
    base_px = _project_xyz(xyz, intrinsics)[:2]
    deltas = {}
    for axis_idx, axis_name in enumerate(("x", "y", "z")):
        pert_px = _project_xyz(_perturb_xyz(xyz, axis_idx, delta_m), intrinsics)[:2]
        deltas[axis_name] = (pert_px - base_px).tolist()
    return deltas


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
        if float(hand.get("is_right", 0.0)) > 0.5 and "cam_t" in hand:
            return hand
    return None


def _compute_human_transformed_xyz(obs_head_pose: np.ndarray, right_obs_pose: np.ndarray) -> np.ndarray:
    head_matrix = _xyzwxyz_to_matrix(obs_head_pose[None, :])[0]
    hand_matrix = _xyzwxyz_to_matrix(right_obs_pose[None, :])[0]
    relative_matrix = np.linalg.inv(head_matrix) @ hand_matrix
    return relative_matrix[:3, 3]


def load_human_frame_record(
    human_zarr: Path,
    human_raw_root: Path,
    local_idx: int,
) -> HumanFrameRecord:
    episode_name = human_zarr.stem
    recording_id = _recording_id_from_episode_name(episode_name)
    raw_frame_start = _raw_frame_start_from_episode_name(episode_name)
    raw_jsonl_path = _find_human_raw_jsonl(human_raw_root, recording_id)
    raw_records = _load_raw_hand_pose_map(raw_jsonl_path)

    store = zarr.open(str(human_zarr), mode="r")
    image = _decode_jpeg(store["images.front_1"][local_idx])
    intrinsics = _scale_intrinsics(INTRINSICS["base"], image.shape[1], image.shape[0])
    transformed_xyz = _compute_human_transformed_xyz(
        np.asarray(store["obs_head_pose"][local_idx], dtype=np.float64),
        np.asarray(store["right.obs_ee_pose"][local_idx], dtype=np.float64),
    )
    transformed_px = _project_xyz(transformed_xyz, intrinsics)

    raw_frame_index = raw_frame_start + local_idx
    raw_record = raw_records.get(raw_frame_index)
    raw_hand = _select_right_hand(raw_record) if raw_record is not None else None
    raw_cam_t = None if raw_hand is None else np.asarray(raw_hand["cam_t"], dtype=np.float64)
    raw_px = None if raw_cam_t is None else _project_xyz(raw_cam_t, intrinsics)

    return HumanFrameRecord(
        local_idx=local_idx,
        raw_frame_index=raw_frame_index,
        image=image,
        intrinsics=intrinsics,
        transformed_xyz=transformed_xyz,
        transformed_px=transformed_px,
        raw_cam_t=raw_cam_t,
        raw_px=raw_px,
    )


def load_so100_frame_record(so100_zarr: Path, local_idx: int) -> So100FrameRecord:
    store = zarr.open(str(so100_zarr), mode="r")
    image = _decode_jpeg(store["images.front_1"][local_idx])
    intrinsics = _scale_intrinsics(INTRINSICS["base"], image.shape[1], image.shape[0])
    camera_xyz = np.asarray(store["obs_ee_pose_cam_rotvec"][local_idx, :3], dtype=np.float64)
    camera_px = _project_xyz(camera_xyz, intrinsics)
    return So100FrameRecord(
        local_idx=local_idx,
        image=image,
        intrinsics=intrinsics,
        camera_xyz=camera_xyz,
        camera_px=camera_px,
    )


def _sample_indices(length: int, num_samples: int) -> list[int]:
    if num_samples <= 1:
        return [0]
    return sorted(set(int(x) for x in np.linspace(0, length - 1, num_samples)))


def _mean_vector(values: list[list[float]]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    return arr.mean(axis=0).tolist()


def _axis_summary(
    xyz_points: list[np.ndarray],
    base_pixels: list[np.ndarray],
    intrinsics: np.ndarray,
    delta_m: float,
) -> dict:
    x_deltas = []
    y_deltas = []
    z_deltas = []
    z_toward_center = []
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    for xyz, base_px in zip(xyz_points, base_pixels):
        deltas = _axis_delta_pixels(xyz, intrinsics, delta_m)
        x_deltas.append(deltas["x"])
        y_deltas.append(deltas["y"])
        z_deltas.append(deltas["z"])
        toward_center_u = np.sign(deltas["z"][0]) == -np.sign(base_px[0] - cx) or abs(base_px[0] - cx) < 1e-6
        toward_center_v = np.sign(deltas["z"][1]) == -np.sign(base_px[1] - cy) or abs(base_px[1] - cy) < 1e-6
        z_toward_center.append(bool(toward_center_u and toward_center_v))
    return {
        "num_samples": len(xyz_points),
        "x_delta_px_mean": _mean_vector(x_deltas),
        "y_delta_px_mean": _mean_vector(y_deltas),
        "z_delta_px_mean": _mean_vector(z_deltas),
        "x_positive_u_frac": float(np.mean([d[0] > 0 for d in x_deltas])),
        "y_positive_v_frac": float(np.mean([d[1] > 0 for d in y_deltas])),
        "z_moves_toward_principal_point_frac": float(np.mean(z_toward_center)),
    }


def _draw_point(image: np.ndarray, px: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    x = int(round(float(px[0])))
    y = int(round(float(px[1])))
    cv2.circle(image, (x, y), 10, color, thickness=-1)
    cv2.putText(image, label, (x + 12, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def _visualize_axis_perturbations(
    image: np.ndarray,
    xyz: np.ndarray,
    intrinsics: np.ndarray,
    delta_m: float,
    include_raw_base: np.ndarray | None = None,
) -> np.ndarray:
    canvas = image.copy()
    base_px = _project_xyz(xyz, intrinsics)
    _draw_point(canvas, base_px, AXIS_COLORS["base"], "base")
    if include_raw_base is not None:
        _draw_point(canvas, include_raw_base, AXIS_COLORS["raw_base"], "raw")
    for axis_idx, axis_name in enumerate(("x", "y", "z")):
        pert_px = _project_xyz(_perturb_xyz(xyz, axis_idx, delta_m), intrinsics)
        _draw_point(canvas, pert_px, AXIS_COLORS[axis_name], f"+{axis_name}")
        cv2.line(
            canvas,
            (int(round(base_px[0])), int(round(base_px[1]))),
            (int(round(pert_px[0])), int(round(pert_px[1]))),
            AXIS_COLORS[axis_name],
            thickness=3,
        )
    return canvas


def build_report(
    human_zarr: Path,
    human_raw_root: Path,
    so100_zarr: Path,
    human_viz_idx: int,
    so100_viz_idx: int,
    num_samples: int,
    delta_m: float,
) -> dict:
    human_store = zarr.open(str(human_zarr), mode="r")
    so100_store = zarr.open(str(so100_zarr), mode="r")
    human_indices = _sample_indices(human_store["right.obs_ee_pose"].shape[0], num_samples)
    so100_indices = _sample_indices(so100_store["obs_ee_pose_cam_rotvec"].shape[0], num_samples)

    human_records = [load_human_frame_record(human_zarr, human_raw_root, idx) for idx in human_indices]
    so100_records = [load_so100_frame_record(so100_zarr, idx) for idx in so100_indices]

    human_intrinsics = human_records[0].intrinsics
    so100_intrinsics = so100_records[0].intrinsics

    human_head_summary = _axis_summary(
        [r.transformed_xyz for r in human_records],
        [r.transformed_px for r in human_records],
        human_intrinsics,
        delta_m,
    )
    human_raw_records = [r for r in human_records if r.raw_cam_t is not None and r.raw_px is not None]
    human_raw_summary = None
    transformed_raw_alignment = None
    if human_raw_records:
        human_raw_summary = _axis_summary(
            [r.raw_cam_t for r in human_raw_records if r.raw_cam_t is not None],
            [r.raw_px for r in human_raw_records if r.raw_px is not None],
            human_intrinsics,
            delta_m,
        )
        transformed_raw_alignment = {
            "num_samples": len(human_raw_records),
            "xyz_offset_mean_m": (
                np.mean([r.transformed_xyz - r.raw_cam_t for r in human_raw_records], axis=0).tolist()
            ),
            "pixel_offset_mean_px": (
                np.mean([r.transformed_px[:2] - r.raw_px[:2] for r in human_raw_records], axis=0).tolist()
            ),
            "x_delta_sign_match_frac": float(
                np.mean(
                    [
                        np.sign(_axis_delta_pixels(r.transformed_xyz, human_intrinsics, delta_m)["x"][0])
                        == np.sign(_axis_delta_pixels(r.raw_cam_t, human_intrinsics, delta_m)["x"][0])
                        for r in human_raw_records
                    ]
                )
            ),
            "y_delta_sign_match_frac": float(
                np.mean(
                    [
                        np.sign(_axis_delta_pixels(r.transformed_xyz, human_intrinsics, delta_m)["y"][1])
                        == np.sign(_axis_delta_pixels(r.raw_cam_t, human_intrinsics, delta_m)["y"][1])
                        for r in human_raw_records
                    ]
                )
            ),
        }

    so100_summary = _axis_summary(
        [r.camera_xyz for r in so100_records],
        [r.camera_px for r in so100_records],
        so100_intrinsics,
        delta_m,
    )

    human_viz = load_human_frame_record(human_zarr, human_raw_root, human_viz_idx)
    so100_viz = load_so100_frame_record(so100_zarr, so100_viz_idx)

    return {
        "config": {
            "human_zarr": str(human_zarr),
            "so100_zarr": str(so100_zarr),
            "human_raw_root": str(human_raw_root),
            "human_viz_idx": human_viz_idx,
            "so100_viz_idx": so100_viz_idx,
            "num_samples": num_samples,
            "delta_m": delta_m,
        },
        "human_headframe": human_head_summary,
        "human_raw_camera_frame": human_raw_summary,
        "human_transformed_vs_raw_alignment": transformed_raw_alignment,
        "so100_camera_frame": so100_summary,
        "viz_frames": {
            "human": {
                "local_idx": human_viz.local_idx,
                "raw_frame_index": human_viz.raw_frame_index,
                "transformed_xyz": human_viz.transformed_xyz.tolist(),
                "transformed_px": human_viz.transformed_px.tolist(),
                "raw_cam_t": None if human_viz.raw_cam_t is None else human_viz.raw_cam_t.tolist(),
                "raw_px": None if human_viz.raw_px is None else human_viz.raw_px.tolist(),
            },
            "so100": {
                "local_idx": so100_viz.local_idx,
                "camera_xyz": so100_viz.camera_xyz.tolist(),
                "camera_px": so100_viz.camera_px.tolist(),
            },
        },
        "conclusion": {
            "human_headframe_x_matches_camera_right": human_head_summary["x_positive_u_frac"] == 1.0,
            "human_headframe_y_matches_camera_down": human_head_summary["y_positive_v_frac"] == 1.0,
            "human_raw_and_transformed_signs_match": (
                transformed_raw_alignment is not None
                and transformed_raw_alignment["x_delta_sign_match_frac"] == 1.0
                and transformed_raw_alignment["y_delta_sign_match_frac"] == 1.0
            ),
            "so100_x_matches_camera_right": so100_summary["x_positive_u_frac"] == 1.0,
            "so100_y_matches_camera_down": so100_summary["y_positive_v_frac"] == 1.0,
        },
        "_human_viz_image": human_viz.image,
        "_human_viz_intrinsics": human_viz.intrinsics,
        "_human_viz_xyz": human_viz.transformed_xyz,
        "_human_viz_raw_px": human_viz.raw_px,
        "_so100_viz_image": so100_viz.image,
        "_so100_viz_intrinsics": so100_viz.intrinsics,
        "_so100_viz_xyz": so100_viz.camera_xyz,
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
    parser.add_argument("--delta-m", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(
        human_zarr=args.human_zarr,
        human_raw_root=args.human_raw_root,
        so100_zarr=args.so100_zarr,
        human_viz_idx=args.human_viz_idx,
        so100_viz_idx=args.so100_viz_idx,
        num_samples=args.num_samples,
        delta_m=args.delta_m,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    human_image = _visualize_axis_perturbations(
        report.pop("_human_viz_image"),
        report.pop("_human_viz_xyz"),
        report.pop("_human_viz_intrinsics"),
        args.delta_m,
        include_raw_base=report["viz_frames"]["human"]["raw_px"],
    )
    so100_image = _visualize_axis_perturbations(
        report.pop("_so100_viz_image"),
        report.pop("_so100_viz_xyz"),
        report.pop("_so100_viz_intrinsics"),
        args.delta_m,
    )

    human_out = output_dir / "human_axis_semantics.png"
    so100_out = output_dir / "so100_axis_semantics.png"
    cv2.imwrite(str(human_out), cv2.cvtColor(human_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(so100_out), cv2.cvtColor(so100_image, cv2.COLOR_RGB2BGR))

    report["artifacts"] = {
        "human_axis_semantics_png": str(human_out),
        "so100_axis_semantics_png": str(so100_out),
    }
    report_path = output_dir / "xyz_axis_semantics_report.json"
    report = _to_jsonable(report)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report_json": str(report_path), **report["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
