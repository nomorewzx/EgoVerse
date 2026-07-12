from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import zarr
except ImportError:  # pragma: no cover
    zarr = None

from egomimic.robot.pingti_hpt_joint_receding_rollout import (
    DEFAULT_PORT,
    DEFAULT_ROBOT_ID,
    DEFAULT_TOP_CAMERA,
    DEFAULT_TOP_CAMERA_KEY,
    DEFAULT_WRIST_CAMERA,
    DEFAULT_WRIST_CAMERA_KEY,
    AsyncDualVideoRecorder,
    JointSafety,
    LivePingtiSource,
)
from egomimic.robot.robot_utils import RateLoop
from egomimic.robot.so100_hpt_receding_rollout import (
    JsonlLogger,
    decode_jpeg_value,
    step_record_base,
    timed,
)


DEFAULT_ZARR_ROOT = "/home/zxwang/egoverse-zarr/bimanual-pick-apricot-0828-right-arm-joint"
DEFAULT_FREQUENCY = 30.0


class ZarrEpisode:
    def __init__(self, zarr_root: str | Path, episode: int, zarr_episode: str | Path | None = None):
        if zarr is None:
            raise ImportError("zarr is required. Run this with the lerobot_py312 conda environment.")
        if zarr_episode is None:
            self.path = Path(zarr_root).expanduser() / f"episode_{episode:06d}.zarr"
        else:
            self.path = Path(zarr_episode).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.group = zarr.open_group(str(self.path), mode="r")
        for key in ("obs_joint_pos", "cmd_joint_pos"):
            if key not in self.group:
                raise KeyError(f"{self.path} is missing {key}")
        self.state = np.asarray(self.group["obs_joint_pos"], dtype=np.float64)
        self.action = np.asarray(self.group["cmd_joint_pos"], dtype=np.float64)
        if self.state.shape != self.action.shape or self.state.ndim != 2 or self.state.shape[1] != 6:
            raise ValueError(
                f"Expected state/action shape (T, 6), got {self.state.shape} and {self.action.shape}"
            )
        self.num_frames = int(self.action.shape[0])
        self.attrs = dict(self.group.attrs)
        self.fps = float(self.attrs.get("fps", DEFAULT_FREQUENCY))

    def target_array(self, source: str) -> np.ndarray:
        if source == "action":
            return self.action
        if source == "state":
            return self.state
        raise ValueError(f"Unsupported source {source!r}")

    def image(self, key: str, frame_index: int) -> np.ndarray | None:
        if key not in self.group:
            return None
        return decode_jpeg_value(self.group[key][frame_index])


def build_paths(args: argparse.Namespace, episode_path: Path) -> tuple[Path | None, dict[str, Path | None]]:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = episode_path.stem.replace(".zarr", "")
    log_path = None
    if not args.no_log:
        log_path = (
            Path(args.log_jsonl).expanduser()
            if args.log_jsonl
            else Path("logs/pingti_hpt/replay_logs") / f"pingti_replay_{tag}_{stamp}.jsonl"
        )

    paths: dict[str, Path | None] = {
        "live_top": None,
        "live_wrist": None,
        "dataset_top": None,
        "dataset_wrist": None,
    }
    if args.record_video:
        video_dir = Path(args.video_dir).expanduser()
        paths["live_top"] = video_dir / f"pingti_replay_{tag}_{stamp}_live_top.mp4"
        paths["live_wrist"] = video_dir / f"pingti_replay_{tag}_{stamp}_live_wrist.mp4"
        if not args.no_dataset_video:
            paths["dataset_top"] = video_dir / f"pingti_replay_{tag}_{stamp}_dataset_top.mp4"
            paths["dataset_wrist"] = video_dir / f"pingti_replay_{tag}_{stamp}_dataset_wrist.mp4"
    return log_path, paths


def current_joint_pos(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray([*obs["q"], obs["gripper"]], dtype=np.float64)


def make_safety(args: argparse.Namespace) -> JointSafety:
    joint_min = None if args.joint_min is None else np.asarray(args.joint_min, dtype=np.float64)
    joint_max = None if args.joint_max is None else np.asarray(args.joint_max, dtype=np.float64)
    return JointSafety(
        max_joint_delta=float(args.max_joint_delta),
        max_gripper_delta=float(args.max_gripper_delta),
        gripper_min=float(args.gripper_min),
        gripper_max=float(args.gripper_max),
        joint_min=joint_min,
        joint_max=joint_max,
        joint_limit_margin=float(args.joint_limit_margin),
    )


def maybe_move_to_start(
    *,
    args: argparse.Namespace,
    source: LivePingtiSource,
    safety: JointSafety,
    first_target: np.ndarray,
    dry_run: bool,
    logger: JsonlLogger,
) -> None:
    if args.move_to_start_s <= 0:
        return
    steps = max(1, int(round(float(args.move_to_start_s) * float(args.frequency))))
    obs = source.observe()
    start = current_joint_pos(obs)
    print(
        f"[pingti-replay] moving to first target over {args.move_to_start_s:.2f}s "
        f"({steps} steps), dry_run={dry_run}"
    )
    with RateLoop(frequency=args.frequency, max_iterations=steps, verbose=args.verbose_rate) as loop:
        for step in loop:
            alpha = float(step + 1) / float(steps)
            desired = start + alpha * (first_target - start)
            obs_t = timed(source.observe)
            current = current_joint_pos(obs_t.value)
            safe, safety_debug = safety.clip(current, desired)
            sent = {"dry_run": True} if dry_run else source.send(safe)
            logger.write(
                {
                    **step_record_base(-steps + step, dry_run),
                    "event": "move_to_start",
                    "obs_ms": obs_t.ms,
                    "joint_pos": current.tolist(),
                    "desired_joint_pos": desired.tolist(),
                    "safe_joint_pos": safe.tolist(),
                    "safety": safety_debug,
                    "sent_action": sent,
                }
            )


def run(args: argparse.Namespace) -> None:
    if args.camera is not None:
        args.top_camera = args.camera
    if args.camera_key is not None:
        args.top_camera_key = args.camera_key
    if args.max_joint_delta_deg is not None:
        args.max_joint_delta = args.max_joint_delta_deg
    if args.joint_limit_margin_deg is not None:
        args.joint_limit_margin = args.joint_limit_margin_deg
    if args.video_fps is None:
        args.video_fps = args.frequency
    if args.dry_run and args.enable_motors:
        raise ValueError("Use only one of --dry-run or --enable-motors.")
    dry_run = not args.enable_motors

    episode = ZarrEpisode(args.zarr_root, args.episode, args.zarr_episode)
    targets = episode.target_array(args.source)
    start = max(0, int(args.start_frame))
    stop = episode.num_frames if args.end_frame is None else min(episode.num_frames, int(args.end_frame))
    if args.max_steps is not None:
        stop = min(stop, start + int(args.max_steps))
    if not start < stop:
        raise ValueError(f"Invalid frame range: start={start}, stop={stop}, num_frames={episode.num_frames}")
    frame_indices = np.arange(start, stop, dtype=int)

    safety = make_safety(args)
    source = LivePingtiSource(args)
    log_path, video_paths = build_paths(args, episode.path)

    print(f"[pingti-replay] episode: {episode.path}")
    print(f"[pingti-replay] frames: {start}:{stop} / {episode.num_frames}, source={args.source}")
    print(f"[pingti-replay] frequency={args.frequency} dataset_fps={episode.fps}")
    print(f"[pingti-replay] dry_run={dry_run} enable_motors={args.enable_motors}")
    print(f"[pingti-replay] keep_torque_on_disconnect={args.keep_torque_on_disconnect}")
    print(f"[pingti-replay] top_camera={args.top_camera} rotate180={args.top_rotate_180} wrist_camera={args.wrist_camera}")
    print(f"[pingti-replay] log_jsonl: {log_path}")
    print(f"[pingti-replay] videos: {video_paths}")
    if dry_run:
        print("[pingti-replay] motors are disabled; add --enable-motors to send replay commands")

    source.connect(calibrate=args.calibrate)
    try:
        with JsonlLogger(log_path) as logger, AsyncDualVideoRecorder(
            video_paths["live_top"],
            video_paths["live_wrist"],
            fps=args.video_fps,
            codec=args.video_codec,
            queue_size=args.video_queue_size,
            every_n_steps=args.video_every_n_steps,
        ) as live_recorder, AsyncDualVideoRecorder(
            video_paths["dataset_top"],
            video_paths["dataset_wrist"],
            fps=args.video_fps,
            codec=args.video_codec,
            queue_size=args.video_queue_size,
            every_n_steps=args.video_every_n_steps,
        ) as dataset_recorder:
            logger.write(
                {
                    **step_record_base(-1, dry_run),
                    "event": "replay_start",
                    "argv": list(sys.argv),
                    "episode_path": str(episode.path),
                    "episode_attrs": episode.attrs,
                    "source": args.source,
                    "start_frame": int(start),
                    "end_frame": int(stop),
                    "frequency": float(args.frequency),
                    "robot_id": args.robot_id,
                    "port": args.port,
                    "use_degrees": bool(args.use_degrees),
                    "keep_torque_on_disconnect": bool(args.keep_torque_on_disconnect),
                    "top_camera": args.top_camera,
                    "wrist_camera": args.wrist_camera,
                    "top_rotate_180": bool(args.top_rotate_180),
                    "wrist_rotate_180": bool(args.wrist_rotate_180),
                }
            )
            maybe_move_to_start(
                args=args,
                source=source,
                safety=safety,
                first_target=targets[start],
                dry_run=dry_run,
                logger=logger,
            )

            with RateLoop(
                frequency=args.frequency,
                max_iterations=len(frame_indices),
                verbose=args.verbose_rate,
            ) as loop:
                for step in loop:
                    loop_start = time.perf_counter()
                    frame_index = int(frame_indices[step])
                    obs_t = timed(source.observe)
                    obs = obs_t.value
                    current = current_joint_pos(obs)
                    target = np.asarray(targets[frame_index], dtype=np.float64)
                    dataset_state = np.asarray(episode.state[frame_index], dtype=np.float64)
                    dataset_action = np.asarray(episode.action[frame_index], dtype=np.float64)
                    safe, safety_debug = safety.clip(current, target)

                    live_enqueued = live_recorder.enqueue(step, obs["image"], obs.get("wrist_image"))
                    dataset_top = episode.image("images.front_1", frame_index)
                    dataset_wrist = episode.image("images.front_2", frame_index)
                    dataset_enqueued = dataset_recorder.enqueue(step, dataset_top, dataset_wrist)

                    sent = {"dry_run": True} if dry_run else source.send(safe)
                    record = {
                        **step_record_base(step, dry_run),
                        "frame_index": frame_index,
                        "obs_ms": obs_t.ms,
                        "loop_ms": (time.perf_counter() - loop_start) * 1000.0,
                        "joint_pos": current.tolist(),
                        "target_joint_pos": target.tolist(),
                        "safe_joint_pos": safe.tolist(),
                        "dataset_state": dataset_state.tolist(),
                        "dataset_action": dataset_action.tolist(),
                        "dataset_action_minus_state": (dataset_action - dataset_state).tolist(),
                        "target_minus_current": (target - current).tolist(),
                        "dataset_video_enqueued": dataset_enqueued,
                        "live_video_enqueued": live_enqueued,
                        "safety": safety_debug,
                        "sent_action": sent,
                    }
                    logger.write(record)
                    if step % args.print_every == 0:
                        gap = np.abs(dataset_action - dataset_state)
                        print(
                            f"[pingti-replay] step={step:05d} frame={frame_index:06d} "
                            f"q={np.round(current, 2).tolist()} target={np.round(target, 2).tolist()} "
                            f"safe={np.round(safe, 2).tolist()} "
                            f"|action-state|mean={float(gap.mean()):.2f} max={float(gap.max()):.2f}"
                        )
            logger.write(
                {
                    **step_record_base(len(frame_indices), dry_run),
                    "event": "replay_end",
                    "live_video": live_recorder.summary(),
                    "dataset_video": dataset_recorder.summary(),
                }
            )
    finally:
        source.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one exported EgoVerse PingTi training episode on the PingTi follower arm."
    )
    parser.add_argument("--zarr-root", default=DEFAULT_ZARR_ROOT)
    parser.add_argument("--zarr-episode", default=None, help="Direct path to an episode_XXXXXX.zarr.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--source", choices=("action", "state"), default="action")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)

    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--use-degrees", action="store_true")
    parser.add_argument(
        "--keep-torque-on-disconnect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep motors powered when the script disconnects. This is the safe default for PingTi.",
    )
    parser.add_argument(
        "--disable-torque-on-disconnect",
        action="store_false",
        dest="keep_torque_on_disconnect",
        help="Explicitly power off motors on disconnect. Use only when the arm is physically supported.",
    )
    parser.add_argument("--lerobot-max-relative-target", type=float, default=None)
    parser.add_argument("--action-filter-type", default="none", choices=("none", "lowpass", "moving_average", "adaptive"))
    parser.add_argument("--action-filter-alpha", type=float, default=0.3)
    parser.add_argument("--action-filter-window-size", type=int, default=3)
    parser.add_argument("--action-filter-adaptation-threshold", type=float, default=0.1)

    parser.add_argument("--top-camera", default=DEFAULT_TOP_CAMERA)
    parser.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA)
    parser.add_argument("--top-camera-key", default=DEFAULT_TOP_CAMERA_KEY)
    parser.add_argument("--wrist-camera-key", default=DEFAULT_WRIST_CAMERA_KEY)
    parser.add_argument("--camera-type", default="opencv", choices=("opencv",))
    parser.add_argument("--camera", default=None, help="Alias for --top-camera.")
    parser.add_argument("--camera-key", default=None, help="Alias for --top-camera-key.")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-fourcc", default=None)
    parser.add_argument("--top-rotate-180", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wrist-rotate-180", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--frequency", type=float, default=DEFAULT_FREQUENCY)
    parser.add_argument("--move-to-start-s", type=float, default=0.0)
    parser.add_argument("--max-joint-delta", type=float, default=10.0)
    parser.add_argument("--max-joint-delta-deg", type=float, default=None)
    parser.add_argument("--max-gripper-delta", type=float, default=30.0)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=100.0)
    parser.add_argument("--joint-min", nargs=5, type=float, default=None)
    parser.add_argument("--joint-max", nargs=5, type=float, default=None)
    parser.add_argument("--joint-limit-margin", type=float, default=0.0)
    parser.add_argument("--joint-limit-margin-deg", type=float, default=None)

    parser.add_argument("--enable-motors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--log-jsonl", default=None)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--no-dataset-video", action="store_true")
    parser.add_argument("--video-dir", default="logs/pingti_hpt/replay_videos")
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--video-queue-size", type=int, default=128)
    parser.add_argument("--video-every-n-steps", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--verbose-rate", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
