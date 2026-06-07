from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egomimic.scripts.diagnostics.probe_human_keypoint_cloud_instability import (
    PALM_INDICES,
    _collect_frame_rows,
    _pair_rows,
)


HAND_CONNECTIONS = (
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
)


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_report(payload: dict[str, Any], episode_substr: str | None) -> dict[str, Any]:
    reports = payload.get("reports") or []
    if not reports:
        raise ValueError("Diagnostic JSON has no reports")
    if episode_substr is None:
        return reports[0]
    matches = [report for report in reports if episode_substr in report["episode_name"]]
    if not matches:
        raise ValueError(f"No report matched episode substring: {episode_substr}")
    if len(matches) > 1:
        names = "\n".join(report["episode_name"] for report in matches[:10])
        raise ValueError(f"Episode substring matched {len(matches)} reports; narrow it down:\n{names}")
    return matches[0]


def _recording_dir_from_raw(raw_path: Path) -> Path:
    for parent in raw_path.parents:
        if parent.name.startswith("recording_"):
            return parent
    raise ValueError(f"Cannot infer recording directory from raw path: {raw_path}")


def _find_video(raw_path: Path) -> Path:
    recording_dir = _recording_dir_from_raw(raw_path)
    preferred = recording_dir / "video.mp4"
    if preferred.is_file():
        return preferred
    candidates = sorted(recording_dir.glob("*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"No mp4 video found in {recording_dir}")
    return candidates[0]


def _safe_keypoints2d(row: dict[str, Any] | None) -> np.ndarray | None:
    if row is None:
        return None
    hand = row.get("hand") or {}
    if "keypoints_2d" not in hand:
        return None
    keypoints = np.asarray(hand["keypoints_2d"], dtype=np.float64)
    if keypoints.shape[0] < 21:
        return None
    return keypoints[:21]


def _draw_skeleton(
    canvas: np.ndarray,
    keypoints: np.ndarray,
    *,
    color: tuple[int, int, int],
    radius: int,
    thickness: int,
) -> None:
    h, w = canvas.shape[:2]
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = keypoints[start_idx]
        end = keypoints[end_idx]
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            continue
        if not (0 <= start[0] < w and 0 <= start[1] < h and 0 <= end[0] < w and 0 <= end[1] < h):
            continue
        cv2.line(
            canvas,
            tuple(np.round(start).astype(int)),
            tuple(np.round(end).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
    for idx, point in enumerate(keypoints):
        if not np.all(np.isfinite(point)):
            continue
        if not (0 <= point[0] < w and 0 <= point[1] < h):
            continue
        point_color = (255, 0, 255) if idx in PALM_INDICES else color
        cv2.circle(canvas, tuple(np.round(point).astype(int)), radius, point_color, -1, cv2.LINE_AA)


def _draw_bbox(canvas: np.ndarray, row: dict[str, Any] | None) -> None:
    if row is None:
        return
    bbox = (row.get("hand") or {}).get("bbox_xyxy")
    if bbox is None or len(bbox) != 4:
        return
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float64)
    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return
    cv2.rectangle(
        canvas,
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_motion_arrows(canvas: np.ndarray, prev_kp: np.ndarray | None, curr_kp: np.ndarray | None) -> None:
    if prev_kp is None or curr_kp is None:
        return
    h, w = canvas.shape[:2]
    for idx in PALM_INDICES:
        start = prev_kp[idx]
        end = curr_kp[idx]
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            continue
        if not (0 <= start[0] < w and 0 <= start[1] < h and 0 <= end[0] < w and 0 <= end[1] < h):
            continue
        cv2.arrowedLine(
            canvas,
            tuple(np.round(start).astype(int)),
            tuple(np.round(end).astype(int)),
            (0, 0, 255),
            3,
            cv2.LINE_AA,
            tipLength=0.25,
        )


def _palm_center(row: dict[str, Any] | None) -> np.ndarray | None:
    keypoints = _safe_keypoints2d(row)
    if keypoints is None:
        return None
    center = np.mean(keypoints[PALM_INDICES], axis=0)
    return center if np.all(np.isfinite(center)) else None


def _draw_trail(canvas: np.ndarray, rows_by_local: dict[int, dict[str, Any]], local_idx: int, trail: int) -> None:
    points = []
    for idx in range(max(0, local_idx - trail + 1), local_idx + 1):
        center = _palm_center(rows_by_local.get(idx))
        if center is not None:
            points.append(center)
    if len(points) < 2:
        return
    pts = np.round(np.asarray(points)).astype(np.int32)
    cv2.polylines(canvas, [pts], False, (0, 255, 0), 3, cv2.LINE_AA)
    for i, point in enumerate(pts):
        alpha = (i + 1) / len(pts)
        color = (0, int(120 + 135 * alpha), int(255 * (1.0 - alpha)))
        cv2.circle(canvas, tuple(point), 5, color, -1, cv2.LINE_AA)


def _put_text_block(canvas: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thickness = 2
    line_h = 29
    width = min(canvas.shape[1], 900)
    height = 16 + line_h * len(lines)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)
    for idx, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (14, 28 + idx * line_h),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def _metric_lines(
    *,
    episode_name: str,
    local_idx: int,
    raw_idx: int,
    selected_pair_local: int,
    pair: dict[str, Any] | None,
) -> list[str]:
    header = f"{episode_name[:70]} | local {local_idx} raw {raw_idx}"
    if pair is None:
        return [header, "No previous-step metric for this frame"]
    marker = "SELECTED SPIKE" if int(pair["local_index"]) == selected_pair_local else "step"
    return [
        header,
        (
            f"{marker} {int(pair['local_index'])}->{int(pair['local_index']) + 1}: "
            f"EE {pair['old_ee_step_deg']:.2f} deg | palm Kabsch {pair['kabsch_palm5_rot_deg']:.2f} deg | "
            f"all21 Kabsch {pair['kabsch_all21_rot_deg']:.2f} deg"
        ),
        (
            f"all21 residual {pair['kabsch_all21_residual_norm']:.3f} | "
            f"bone rel {pair['bone_rms_delta_rel']:.3f} | "
            f"2D centered {pair['keypoints2d_centered_mean_px']:.1f}px | "
            f"det {pair['det_score_min']:.2f}"
        ),
    ]


def _build_side_panel(pair: dict[str, Any] | None, height: int, width: int = 520) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    lines = [
        "Overlay legend",
        "cyan: current 2D skeleton",
        "orange: previous frame",
        "red arrows: palm keypoint motion",
        "green trail: palm center",
        "",
    ]
    if pair is None:
        lines.append("No metric for this frame")
    else:
        metrics = [
            ("EE step", pair["old_ee_step_deg"], "deg"),
            ("palm Kabsch", pair["kabsch_palm5_rot_deg"], "deg"),
            ("all21 Kabsch", pair["kabsch_all21_rot_deg"], "deg"),
            ("all21 residual", pair["kabsch_all21_residual_norm"], ""),
            ("bone rel", pair["bone_rms_delta_rel"], ""),
            ("2D jitter", pair["keypoints2d_centered_mean_px"], "px"),
        ]
        lines.append("Current step metrics")
        for name, value, suffix in metrics:
            lines.append(f"{name}: {float(value):.3f} {suffix}".rstrip())

    y = 34
    for line in lines:
        color = (220, 220, 220)
        if line.endswith("SPIKE"):
            color = (0, 0, 255)
        cv2.putText(panel, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        y += 34
    return panel


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read video frame {frame_idx}")
    return frame


def _video_rotation_degrees(video_path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return 0
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return 0
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return 0
    for stream in payload.get("streams", []):
        tags = stream.get("tags") or {}
        if "rotate" in tags:
            try:
                return int(round(float(tags["rotate"]))) % 360
            except (TypeError, ValueError):
                pass
        for side_data in stream.get("side_data_list") or []:
            if "rotation" in side_data:
                try:
                    return int(round(float(side_data["rotation"]))) % 360
                except (TypeError, ValueError):
                    pass
    return 0


def _apply_video_rotation(frame: np.ndarray, rotation_deg: int) -> np.ndarray:
    rotation_deg = int(rotation_deg) % 360
    if rotation_deg == 0:
        return frame
    if rotation_deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation_deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation_deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == height:
        return image
    scale = height / float(h)
    return cv2.resize(image, (int(round(w * scale)), height), interpolation=cv2.INTER_AREA)


def _write_video(
    *,
    video_path: Path,
    output_path: Path,
    episode_name: str,
    rows_by_local: dict[int, dict[str, Any]],
    pairs_by_local: dict[int, dict[str, Any]],
    raw_frame_start: int,
    window_start: int,
    window_end: int,
    selected_pair_local: int,
    output_height: int,
    fps: float,
    repeat_frames: int,
    trail: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    write_path = (
        output_path.with_name(f"{output_path.stem}.opencv_tmp{output_path.suffix}")
        if ffmpeg is not None
        else output_path
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    source_rotation_deg = _video_rotation_degrees(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for local_idx in range(window_start, window_end + 1):
            raw_idx = raw_frame_start + local_idx
            frame = _apply_video_rotation(_read_frame(cap, raw_idx), source_rotation_deg)
            curr = rows_by_local.get(local_idx)
            prev = rows_by_local.get(local_idx - 1)
            curr_kp = _safe_keypoints2d(curr)
            prev_kp = _safe_keypoints2d(prev)
            if prev_kp is not None:
                ghost = frame.copy()
                _draw_skeleton(ghost, prev_kp, color=(0, 140, 255), radius=5, thickness=2)
                cv2.addWeighted(ghost, 0.45, frame, 0.55, 0, frame)
            if curr_kp is not None:
                _draw_skeleton(frame, curr_kp, color=(255, 255, 0), radius=6, thickness=3)
            _draw_motion_arrows(frame, prev_kp, curr_kp)
            _draw_trail(frame, rows_by_local, local_idx, trail)
            _draw_bbox(frame, curr)
            pair = pairs_by_local.get(local_idx - 1)
            _put_text_block(
                frame,
                _metric_lines(
                    episode_name=episode_name,
                    local_idx=local_idx,
                    raw_idx=raw_idx,
                    selected_pair_local=selected_pair_local,
                    pair=pair,
                ),
            )
            frame = _resize_to_height(frame, output_height)
            side = _build_side_panel(pair, frame.shape[0])
            composed = np.concatenate([frame, side], axis=1)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(write_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (composed.shape[1], composed.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {write_path}")
            for _ in range(max(1, repeat_frames)):
                writer.write(composed)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if ffmpeg is not None:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(write_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-crf",
            "20",
            str(output_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg H.264 transcode failed:\n{proc.stderr}")
        write_path.unlink(missing_ok=True)


def _plot_timeline(
    *,
    output_path: Path,
    pairs: list[dict[str, Any]],
    window_start: int,
    window_end: int,
    selected_pair_local: int,
) -> None:
    selected = [
        row
        for row in pairs
        if window_start <= int(row["local_index"]) <= window_end - 1
    ]
    if not selected:
        raise RuntimeError("No pair metrics found inside selected window")
    xs = np.asarray([row["local_index"] for row in selected], dtype=np.int64)

    fig, axes = plt.subplots(4, 1, figsize=(12.5, 10), sharex=True, constrained_layout=True)
    axes[0].plot(xs, [row["old_ee_step_deg"] for row in selected], marker="o", label="EE step")
    axes[0].plot(xs, [row["kabsch_palm5_rot_deg"] for row in selected], marker="o", label="palm5 Kabsch")
    axes[0].plot(xs, [row["kabsch_all21_rot_deg"] for row in selected], marker="o", label="all21 Kabsch")
    axes[0].set_ylabel("rotation deg")
    axes[0].legend(loc="upper right")

    axes[1].plot(xs, [row["kabsch_all21_residual_norm"] for row in selected], marker="o", color="#d62728")
    axes[1].set_ylabel("all21 residual norm")

    axes[2].plot(xs, [row["bone_rms_delta_rel"] for row in selected], marker="o", color="#9467bd", label="bone rel")
    axes[2].plot(xs, [row["kabsch_palm5_residual_norm"] for row in selected], marker="o", color="#2ca02c", label="palm5 residual")
    axes[2].set_ylabel("relative residual")
    axes[2].legend(loc="upper right")

    axes[3].plot(xs, [row["keypoints2d_centered_mean_px"] for row in selected], marker="o", color="#ff7f0e")
    axes[3].set_ylabel("2D centered px")
    axes[3].set_xlabel("local pair index")

    for ax in axes:
        ax.axvline(selected_pair_local, color="black", linestyle="--", linewidth=1.2)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Human keypoint jitter window around pair {selected_pair_local}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_html(
    *,
    output_path: Path,
    episode_name: str,
    selected_pair_local: int,
    video_path: Path,
    timeline_path: Path,
    source_video_path: Path,
    diagnostic_json: Path,
) -> None:
    rel_video = video_path.name
    rel_timeline = timeline_path.name
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Human Keypoint Jitter - {episode_name}</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #111; color: #eee; }}
    video, img {{ max-width: 100%; border: 1px solid #444; }}
    code {{ color: #d7eaff; }}
  </style>
</head>
<body>
  <h1>Human Keypoint Jitter</h1>
  <p><code>{episode_name}</code></p>
  <p>Selected pair: <code>{selected_pair_local}->{selected_pair_local + 1}</code></p>
  <video src="{rel_video}" controls loop muted></video>
  <h2>Metrics Timeline</h2>
  <img src="{rel_timeline}" alt="metrics timeline">
  <p>Source video: <code>{source_video_path}</code></p>
  <p>Diagnostic JSON: <code>{diagnostic_json}</code></p>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize human hand keypoint jitter around a Kabsch/EE rotation spike."
    )
    parser.add_argument(
        "--diagnostic-json",
        default="logs/so100_hpt/rotation_noise_probe/keypoint_cloud_instability_rotation_top20_zarr_capped_2026-05-30.json",
    )
    parser.add_argument("--episode-name-substr", default=None)
    parser.add_argument("--pair-rank", type=int, default=0)
    parser.add_argument("--local-index", type=int, default=None)
    parser.add_argument("--window-radius", type=int, default=8)
    parser.add_argument("--side", default="right", choices=("left", "right"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--repeat-frames", type=int, default=1)
    parser.add_argument("--trail", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostic_json = Path(args.diagnostic_json)
    payload = _load_payload(diagnostic_json)
    report = _select_report(payload, args.episode_name_substr)
    top_pairs = report.get("top_old_ee_step_pairs") or []
    if args.local_index is None:
        if args.pair_rank < 0 or args.pair_rank >= len(top_pairs):
            raise ValueError(f"--pair-rank out of range: {args.pair_rank}; have {len(top_pairs)}")
        selected_pair_local = int(top_pairs[args.pair_rank]["local_index"])
    else:
        selected_pair_local = int(args.local_index)

    raw_path = Path(report["raw"])
    raw_frame_start = int(report["raw_frame_start"])
    window_start = max(0, selected_pair_local - int(args.window_radius))
    window_end = selected_pair_local + 1 + int(args.window_radius)
    frames = _collect_frame_rows(
        raw_path,
        side=args.side,
        raw_frame_start=raw_frame_start,
        max_frames=window_end + 1,
    )
    pairs = _pair_rows(frames)
    rows_by_local = {int(row["local_index"]): row for row in frames}
    pairs_by_local = {int(row["local_index"]): row for row in pairs}
    if selected_pair_local not in pairs_by_local:
        raise RuntimeError(f"Selected local pair {selected_pair_local} not found in raw records")

    video_path = _find_video(raw_path)
    tag = f"{report['episode_name']}_pair{selected_pair_local:06d}"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("logs/so100_hpt/rotation_noise_probe/jitter_viz")
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / f"{tag}.mp4"
    output_timeline = output_dir / f"{tag}_timeline.png"
    output_html = output_dir / f"{tag}.html"

    _write_video(
        video_path=video_path,
        output_path=output_video,
        episode_name=report["episode_name"],
        rows_by_local=rows_by_local,
        pairs_by_local=pairs_by_local,
        raw_frame_start=raw_frame_start,
        window_start=window_start,
        window_end=window_end,
        selected_pair_local=selected_pair_local,
        output_height=args.output_height,
        fps=args.fps,
        repeat_frames=args.repeat_frames,
        trail=args.trail,
    )
    _plot_timeline(
        output_path=output_timeline,
        pairs=pairs,
        window_start=window_start,
        window_end=window_end,
        selected_pair_local=selected_pair_local,
    )
    _write_html(
        output_path=output_html,
        episode_name=report["episode_name"],
        selected_pair_local=selected_pair_local,
        video_path=output_video,
        timeline_path=output_timeline,
        source_video_path=video_path,
        diagnostic_json=diagnostic_json,
    )

    print(
        json.dumps(
            {
                "episode_name": report["episode_name"],
                "selected_pair_local": selected_pair_local,
                "window": [window_start, window_end],
                "source_video": str(video_path),
                "output_video": str(output_video),
                "output_timeline": str(output_timeline),
                "output_html": str(output_html),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
