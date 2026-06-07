from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egomimic.pl_utils.pl_model import ModelWrapper  # noqa: E402
from egomimic.rldb.embodiment.embodiment import get_embodiment_id  # noqa: E402
from egomimic.rldb.embodiment.so100 import So100SingleArm  # noqa: E402
from egomimic.scripts.diagnostics.probe_so100_val_action_quality import (  # noqa: E402
    ACTION_KEY,
    EMBODIMENT,
    parse_model_spec,
    resample_chunk,
)

EMBODIMENT_ID = get_embodiment_id(EMBODIMENT)
DEFAULT_CALIBRATION = "/home/zxwang/so100_calib/selected_calibration_drop0007_5x7_29_21.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Counterfactually run multiple SO100 checkpoints on real rollout video "
            "query frames/states, project predicted EE trajectories back to image "
            "pixels, and compare motion toward the nearest apricot."
        )
    )
    parser.add_argument("--log-jsonl", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model checkpoint as NAME=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--query-step",
        type=int,
        action="append",
        required=True,
        help="Rollout query step/frame to evaluate. Can be repeated.",
    )
    parser.add_argument("--calibration-json", default=DEFAULT_CALIBRATION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--resampled-action-len", type=int, default=45)
    parser.add_argument("--execute-steps", type=int, default=30)
    parser.add_argument("--target-max-distance-px", type=float, default=95.0)
    parser.add_argument(
        "--output-dir",
        default="logs/so100_hpt/rollout_counterfactual_probe",
    )
    parser.add_argument("--tag", default="so100_rollout_counterfactual_2026-05-24")
    return parser.parse_args()


def expected_action_len(wrapper: ModelWrapper) -> int:
    stats = wrapper.model.data_schematic.norm_stats[EMBODIMENT_ID]["actions_cartesian"]
    for key in ("quantile_1", "mean", "min"):
        if key in stats:
            arr = np.asarray(stats[key])
            if arr.ndim >= 2:
                return int(arr.shape[0])
    return So100SingleArm.ACTION_CHUNK_LENGTH


def load_query_records(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            step = int(row.get("step", -1))
            if step >= 0 and row.get("query"):
                records[step] = row
    return records


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    intr = data["camera_intrinsics"]
    camera_matrix = np.asarray(intr["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(intr.get("dist_coeffs", []), dtype=np.float64)
    return camera_matrix, dist_coeffs


def project_points(points_xyz: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    z = np.maximum(pts[:, 2], 1e-6)
    u = camera_matrix[0, 0] * pts[:, 0] / z + camera_matrix[0, 2]
    v = camera_matrix[1, 1] * pts[:, 1] / z + camera_matrix[1, 2]
    return np.stack([u, v], axis=1)


def read_video_frame(video: Path, step: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(step))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {step} from {video}")
    return frame_bgr


def detect_apricots(frame_bgr: np.ndarray) -> list[dict]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([5, 60, 80]), np.array([35, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[dict] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 80.0 or area > 5000.0:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        x = float(moments["m10"] / moments["m00"])
        y = float(moments["m01"] / moments["m00"])
        detections.append({"x": x, "y": y, "area": area})
    detections.sort(key=lambda item: (item["x"], item["y"]))
    return detections


def choose_target(
    detections: list[dict],
    ee_px: np.ndarray,
    max_distance_px: float,
) -> dict | None:
    if not detections:
        return None
    best = None
    best_dist = float("inf")
    for item in detections:
        center = np.asarray([item["x"], item["y"]], dtype=np.float64)
        dist = float(np.linalg.norm(center - ee_px))
        if dist < best_dist:
            best = item
            best_dist = dist
    if best is None or best_dist > max_distance_px:
        return None
    out = dict(best)
    out["distance_px"] = best_dist
    return out


def image_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    front = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    return front


def predict_chunk(
    wrapper: ModelWrapper,
    frame_bgr: np.ndarray,
    ee_camera_ypr: np.ndarray,
    expected_t: int,
    *,
    seed: int | None,
    device: torch.device,
) -> np.ndarray:
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    front = image_to_tensor(frame_bgr).to(device)
    ee = torch.as_tensor(np.asarray(ee_camera_ypr, dtype=np.float32), device=device)
    dummy_actions = ee.view(1, 7).repeat(expected_t, 1)
    raw_batch = {
        EMBODIMENT: {
            "observations.images.front_img_1": front.unsqueeze(0),
            "observations.state.ee_pose": ee.unsqueeze(0),
            "actions_cartesian": dummy_actions.unsqueeze(0),
        }
    }
    with torch.inference_mode():
        processed = wrapper.model.process_batch_for_training(raw_batch)
        pred = wrapper.model.forward_eval(processed)[ACTION_KEY]
    return pred.detach().float().cpu().numpy().squeeze(0)


def trajectory_metrics(
    traj_px: np.ndarray,
    ee_px: np.ndarray,
    target_px: np.ndarray,
) -> dict:
    if len(traj_px) == 0:
        return {}
    target_vec = target_px - ee_px
    target_dist = float(np.linalg.norm(target_vec))
    if target_dist < 1e-6:
        target_unit = np.zeros(2, dtype=np.float64)
    else:
        target_unit = target_vec / target_dist
    disp = traj_px - ee_px[None, :]
    disp_norm = np.linalg.norm(disp, axis=1)
    cos = np.full(len(traj_px), np.nan, dtype=np.float64)
    valid = disp_norm > 1e-6
    cos[valid] = np.sum(disp[valid] * target_unit[None, :], axis=1) / disp_norm[valid]
    dist = np.linalg.norm(traj_px - target_px[None, :], axis=1)
    return {
        "target_distance_px": target_dist,
        "first_step_cos_to_target": float(cos[0]) if np.isfinite(cos[0]) else "",
        "first5_cos_to_target_mean": float(np.nanmean(cos[: min(5, len(cos))])),
        "first10_cos_to_target_mean": float(np.nanmean(cos[: min(10, len(cos))])),
        "exec_cos_to_target_mean": float(np.nanmean(cos)),
        "first10_progress_px": float(target_dist - dist[min(9, len(dist) - 1)]),
        "exec_min_distance_px": float(dist.min()),
        "exec_final_distance_px": float(dist[-1]),
        "exec_best_progress_px": float(target_dist - dist.min()),
        "exec_final_progress_px": float(target_dist - dist[-1]),
        "exec_displacement_px": float(disp_norm[-1]),
    }


def gripper_metrics(gripper: np.ndarray) -> dict:
    hits = np.flatnonzero(gripper >= 30.0)
    hits38 = np.flatnonzero(gripper >= 38.0)
    return {
        "pred_gripper_first": float(gripper[0]),
        "pred_gripper_final": float(gripper[-1]),
        "pred_gripper_max": float(gripper.max()),
        "pred_gripper_median": float(np.median(gripper)),
        "pred_gripper_min": float(gripper.min()),
        "pred_n_steps_ge38": int(hits38.size),
        "pred_frac_steps_ge38": float(hits38.size / max(1, len(gripper))),
        "pred_first_ge38_idx": "" if hits38.size == 0 else int(hits38[0]),
        "pred_first_close_idx": "" if hits.size == 0 else int(hits[0]),
        "pred_has_close": bool(hits.size > 0),
        "pred_has_ge38": bool(hits38.size > 0),
    }


def draw_overlay(
    frame_bgr: np.ndarray,
    *,
    step: int,
    ee_px: np.ndarray,
    target: dict | None,
    detections: list[dict],
    trajectories: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    draw = frame_bgr.copy()
    for det in detections:
        center = (int(round(det["x"])), int(round(det["y"])))
        cv2.circle(draw, center, 8, (0, 180, 255), 2)
    cv2.circle(draw, tuple(np.round(ee_px).astype(int)), 7, (255, 255, 255), -1)
    cv2.circle(draw, tuple(np.round(ee_px).astype(int)), 9, (0, 0, 0), 2)
    if target is not None:
        target_px = (int(round(target["x"])), int(round(target["y"])))
        cv2.circle(draw, target_px, 13, (0, 255, 0), 3)
    colors = {
        "mlp_80k": (0, 255, 0),
        "fm_baseline_60k": (255, 255, 0),
        "fm_gripper5x_60k": (0, 0, 255),
        "fm_gripper5x_80k": (255, 0, 255),
    }
    for name, traj in trajectories.items():
        color = colors.get(name, (255, 255, 255))
        pts = np.round(traj).astype(int)
        for i in range(1, len(pts)):
            cv2.line(draw, tuple(pts[i - 1]), tuple(pts[i]), color, 2)
        if len(pts):
            cv2.circle(draw, tuple(pts[min(9, len(pts) - 1)]), 5, color, -1)
            cv2.putText(
                draw,
                name,
                tuple(pts[min(9, len(pts) - 1)] + np.array([4, -4])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.rectangle(draw, (0, 0), (640, 34), (0, 0, 0), -1)
    cv2.putText(draw, f"step {step}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(out_path), draw)


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


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    out = []
    keys = (
        "first10_cos_to_target_mean",
        "exec_cos_to_target_mean",
        "exec_best_progress_px",
        "exec_final_progress_px",
        "exec_min_distance_px",
        "pred_gripper_max",
        "pred_n_steps_ge38",
        "pred_gripper_final",
        "pred_gripper_median",
    )
    for model, group in sorted(grouped.items()):
        item = {"model": model, "n_rows": len(group)}
        for key in keys:
            vals = []
            for row in group:
                value = row.get(key, "")
                if value == "":
                    continue
                vals.append(float(value))
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                item[f"{key}_mean"] = float(arr.mean())
                item[f"{key}_median"] = float(np.median(arr))
            else:
                item[f"{key}_mean"] = ""
                item[f"{key}_median"] = ""
        item["pred_close_rate"] = sum(bool(row["pred_has_close"]) for row in group) / len(group)
        item["pred_ge38_rate"] = sum(bool(row["pred_has_ge38"]) for row in group) / len(group)
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_jsonl).expanduser()
    video_path = Path(args.video).expanduser()
    model_specs = [parse_model_spec(spec) for spec in args.model]
    for _, checkpoint in model_specs:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    camera_matrix, _ = load_calibration(Path(args.calibration_json).expanduser())
    query_records = load_query_records(log_path)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    wrappers = {}
    model_meta = []
    for model_name, checkpoint in model_specs:
        wrapper = ModelWrapper.load_from_checkpoint(
            str(checkpoint), weights_only=False, map_location="cpu"
        )
        wrapper = wrapper.to(device)
        wrapper.eval()
        wrapper.model.device = device
        diffusion = bool(getattr(wrapper.model, "diffusion", False))
        expected_t = expected_action_len(wrapper)
        wrappers[model_name] = (wrapper, diffusion, expected_t, str(checkpoint))
        model_meta.append(
            {
                "model": model_name,
                "checkpoint": str(checkpoint),
                "diffusion": diffusion,
                "expected_action_len": expected_t,
            }
        )

    rows: list[dict] = []
    frame_rows: list[dict] = []
    for step in args.query_step:
        if step not in query_records:
            raise KeyError(f"Step {step} is not a query step in {log_path}")
        record = query_records[step]
        frame_bgr = read_video_frame(video_path, step)
        state = np.asarray(record["ee_camera_ypr"], dtype=np.float64)
        ee_px = project_points(state[:3][None, :], camera_matrix)[0]
        detections = detect_apricots(frame_bgr)
        target = choose_target(detections, ee_px, args.target_max_distance_px)
        if target is None:
            target_px = None
        else:
            target_px = np.asarray([target["x"], target["y"]], dtype=np.float64)
        frame_rows.append(
            {
                "step": step,
                "time_s": float(record["time"] - query_records[min(query_records)]["time"]),
                "state_gripper": float(state[6]),
                "ee_px_x": float(ee_px[0]),
                "ee_px_y": float(ee_px[1]),
                "target_x": "" if target is None else float(target["x"]),
                "target_y": "" if target is None else float(target["y"]),
                "target_distance_px": "" if target is None else float(target["distance_px"]),
                "n_apricot_detections": len(detections),
            }
        )
        if target is None:
            continue

        overlay_trajectories: dict[str, np.ndarray] = {}
        for model_name, (wrapper, diffusion, expected_t, checkpoint) in wrappers.items():
            seeds: list[int | None] = list(args.seeds) if diffusion else [None]
            seed_trajs = []
            for seed in seeds:
                chunk = predict_chunk(
                    wrapper,
                    frame_bgr,
                    state,
                    expected_t,
                    seed=seed,
                    device=device,
                )
                exec_chunk = resample_chunk(chunk, args.resampled_action_len)[
                    : args.execute_steps
                ]
                traj_px = project_points(exec_chunk[:, :3], camera_matrix)
                seed_trajs.append(traj_px)
                row = {
                    "step": step,
                    "model": model_name,
                    "seed": "det" if seed is None else int(seed),
                    "checkpoint": checkpoint,
                    "diffusion": diffusion,
                    "expected_action_len": expected_t,
                    "state_gripper": float(state[6]),
                    "ee_px_x": float(ee_px[0]),
                    "ee_px_y": float(ee_px[1]),
                    "target_x": float(target_px[0]),
                    "target_y": float(target_px[1]),
                }
                row.update(trajectory_metrics(traj_px, ee_px, target_px))
                row.update(gripper_metrics(exec_chunk[:, 6]))
                rows.append(row)
            overlay_trajectories[model_name] = np.mean(np.stack(seed_trajs, axis=0), axis=0)
        draw_overlay(
            frame_bgr,
            step=step,
            ee_px=ee_px,
            target=target,
            detections=detections,
            trajectories=overlay_trajectories,
            out_path=output_dir / f"{args.tag}_step{step}_overlay.png",
        )

    rows_csv = output_dir / f"{args.tag}_rows.csv"
    frame_csv = output_dir / f"{args.tag}_frames.csv"
    summary_csv = output_dir / f"{args.tag}_summary.csv"
    summary_json = output_dir / f"{args.tag}_summary.json"
    summary_rows = summarize(rows)
    write_csv(rows_csv, rows)
    write_csv(frame_csv, frame_rows)
    write_csv(summary_csv, summary_rows)
    summary = {
        "log_jsonl": str(log_path),
        "video": str(video_path),
        "calibration_json": str(Path(args.calibration_json).expanduser()),
        "query_steps": args.query_step,
        "model_meta": model_meta,
        "outputs": {
            "rows": str(rows_csv),
            "frames": str(frame_csv),
            "summary": str(summary_csv),
        },
        "summary_rows": summary_rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"OUT_JSON {summary_json}")


if __name__ == "__main__":
    main()
