from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from egomimic.robot.robot_utils import RateLoop
from egomimic.robot.so100_hpt_receding_rollout import (
    ACTION_REPRESENTATION_JOINT,
    AsyncVideoRecorder,
    JsonlLogger,
    SO100HPTPolicy,
    blend_replanned_chunk,
    current_joint_pos_from_observation,
    numeric_sequence_stats,
    resample_action_chunk,
    step_record_base,
    timed,
)

PINGTI_ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
GRIPPER_NAME = "gripper"
DEFAULT_ROBOT_TYPE = "pingti_lite_follower"
DEFAULT_ROBOT_ID = "bimanual_pingti_lite_follower_right"
DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_TOP_CAMERA = "4"
DEFAULT_WRIST_CAMERA = "6"
DEFAULT_TOP_CAMERA_KEY = "observation.images.top"
DEFAULT_WRIST_CAMERA_KEY = "observation.images.right"
DEFAULT_QUERY_FREQUENCY = 30
DEFAULT_RESAMPLED_ACTION_LEN = 45
JOINT_LOAD_RAW_KEY = "observations.state.joint_load"


@dataclass
class JointSafety:
    max_joint_delta: float
    max_gripper_delta: float
    gripper_min: float
    gripper_max: float
    joint_min: np.ndarray | None
    joint_max: np.ndarray | None
    joint_limit_margin: float

    def clip(self, current: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        current = np.asarray(current, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if current.shape != (6,) or target.shape != (6,):
            raise ValueError(
                f"Expected current/target joint shape (6,), got {current.shape} and {target.shape}"
            )

        raw_delta = target - current
        delta = raw_delta.copy()
        delta[:5] = np.clip(delta[:5], -self.max_joint_delta, self.max_joint_delta)
        delta[5] = np.clip(delta[5], -self.max_gripper_delta, self.max_gripper_delta)
        safe = current + delta
        before_gripper_range = float(safe[5])
        safe[5] = np.clip(safe[5], self.gripper_min, self.gripper_max)

        joint_range_clipped = False
        if self.joint_min is not None and self.joint_max is not None:
            lo = self.joint_min + self.joint_limit_margin
            hi = self.joint_max - self.joint_limit_margin
            before = safe[:5].copy()
            safe[:5] = np.clip(safe[:5], lo, hi)
            joint_range_clipped = bool(np.any(np.abs(safe[:5] - before) > 1e-6))

        debug = {
            "raw_target": target.tolist(),
            "safe_target": safe.tolist(),
            "raw_delta": raw_delta.tolist(),
            "safe_delta": (safe - current).tolist(),
            "max_joint_delta": float(self.max_joint_delta),
            "max_gripper_delta": float(self.max_gripper_delta),
            "joint_delta_limited": bool(np.any(np.abs(delta[:5] - raw_delta[:5]) > 1e-6)),
            "gripper_delta_limited": bool(abs(delta[5] - raw_delta[5]) > 1e-6),
            "gripper_range_clipped": bool(abs(safe[5] - before_gripper_range) > 1e-6),
            "joint_range_clipped": joint_range_clipped,
        }
        return safe.astype(np.float64), debug


class AsyncDualVideoRecorder:
    def __init__(
        self,
        top_path: str | Path | None,
        wrist_path: str | Path | None,
        *,
        fps: float,
        codec: str,
        queue_size: int,
        every_n_steps: int,
    ):
        self.top = AsyncVideoRecorder(
            top_path,
            fps=fps,
            codec=codec,
            queue_size=queue_size,
            every_n_steps=every_n_steps,
            input_color="rgb",
        )
        self.wrist = AsyncVideoRecorder(
            wrist_path,
            fps=fps,
            codec=codec,
            queue_size=queue_size,
            every_n_steps=every_n_steps,
            input_color="rgb",
        )

    def __enter__(self):
        self.top.__enter__()
        self.wrist.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.top.__exit__(exc_type, exc, tb)
        self.wrist.__exit__(exc_type, exc, tb)

    def enqueue(self, step: int, top_image: np.ndarray, wrist_image: np.ndarray | None) -> dict[str, bool]:
        return {
            "top": self.top.enqueue(step, top_image),
            "wrist": False if wrist_image is None else self.wrist.enqueue(step, wrist_image),
        }

    def summary(self) -> dict[str, Any]:
        return {"top": self.top.summary(), "wrist": self.wrist.summary()}


class LivePingtiSource:
    def __init__(self, args: argparse.Namespace):
        from lerobot.cameras import ColorMode, Cv2Rotation
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from pingti.robots.pingti_follower.config_pingti_follower import (
            PingtiFollowerConfig,
            PingtiLiteFollowerConfig,
        )
        from pingti.robots.pingti_follower.pingti_follower import (
            PingtiFollower,
            PingtiLiteFollower,
        )

        top_camera = OpenCVCameraConfig(
            index_or_path=self._camera_index_or_path(args.top_camera),
            fps=args.camera_fps,
            width=args.camera_width,
            height=args.camera_height,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.ROTATE_180 if args.top_rotate_180 else Cv2Rotation.NO_ROTATION,
            fourcc=args.camera_fourcc,
        )
        wrist_camera = OpenCVCameraConfig(
            index_or_path=self._camera_index_or_path(args.wrist_camera),
            fps=args.camera_fps,
            width=args.camera_width,
            height=args.camera_height,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.ROTATE_180 if args.wrist_rotate_180 else Cv2Rotation.NO_ROTATION,
            fourcc=args.camera_fourcc,
        )
        cameras = {
            args.top_camera_key: top_camera,
            args.wrist_camera_key: wrist_camera,
        }
        if args.robot_type == "pingti_lite_follower":
            config_cls = PingtiLiteFollowerConfig
            robot_cls = PingtiLiteFollower
        elif args.robot_type == "pingti_follower":
            config_cls = PingtiFollowerConfig
            robot_cls = PingtiFollower
        else:
            raise ValueError(f"Unsupported --robot-type {args.robot_type!r}")

        cfg = config_cls(
            port=args.port,
            id=args.robot_id,
            cameras=cameras,
            use_degrees=bool(args.use_degrees),
            disable_torque_on_disconnect=not args.keep_torque_on_disconnect,
            max_relative_target=args.lerobot_max_relative_target,
            calibration_dir=Path(args.calibration_dir).expanduser() if args.calibration_dir else None,
            use_force_proxy=bool(args.use_force_proxy),
            action_filter_type=args.action_filter_type,
            action_filter_alpha=args.action_filter_alpha,
            action_filter_window_size=args.action_filter_window_size,
            action_filter_adaptation_threshold=args.action_filter_adaptation_threshold,
        )
        self.robot = robot_cls(cfg)
        self.top_camera_key = args.top_camera_key
        self.wrist_camera_key = args.wrist_camera_key

    @staticmethod
    def _camera_index_or_path(value: str) -> int | Path:
        return int(value) if str(value).isdigit() else Path(value).expanduser()

    def connect(self, calibrate: bool) -> None:
        self.robot.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        self.robot.disconnect()

    def observe(self) -> dict[str, Any]:
        obs = self.robot.get_observation()
        missing = [key for key in [self.top_camera_key, self.wrist_camera_key] if key not in obs]
        if missing:
            raise KeyError(f"Missing camera keys {missing}; observation keys={list(obs)}")
        q = np.asarray([float(obs[f"{name}.pos"]) for name in PINGTI_ARM_JOINT_NAMES], dtype=np.float64)
        gripper = float(obs[f"{GRIPPER_NAME}.pos"])
        load = None
        if self.robot.config.use_force_proxy:
            load_names = [*PINGTI_ARM_JOINT_NAMES, GRIPPER_NAME]
            load_keys = [f"{name}.present_load" for name in load_names]
            missing_load = [key for key in load_keys if key not in obs]
            if missing_load:
                raise KeyError(
                    f"Missing present_load keys {missing_load}; observation keys={list(obs)}"
                )
            load = np.asarray([float(obs[key]) for key in load_keys], dtype=np.float64)
        return {
            "image": np.asarray(obs[self.top_camera_key]),
            "wrist_image": np.asarray(obs[self.wrist_camera_key]),
            "q": q,
            "gripper": gripper,
            "load": load,
            "raw": obs,
        }

    def send(self, joint_pos: np.ndarray) -> dict[str, Any]:
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        if joint_pos.shape != (6,):
            raise ValueError(f"Expected joint_pos shape (6,), got {joint_pos.shape}")
        action = {f"{name}.pos": float(joint_pos[i]) for i, name in enumerate(PINGTI_ARM_JOINT_NAMES)}
        action[f"{GRIPPER_NAME}.pos"] = float(joint_pos[5])
        return self.robot.send_action(action)


def build_log_path(args: argparse.Namespace) -> Path | None:
    if args.log_jsonl is not None:
        return Path(args.log_jsonl).expanduser()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("logs/pingti_hpt/rollout_logs") / f"pingti_hpt_joint_rollout_{stamp}.jsonl"


def build_video_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    if not args.record_video:
        return None, None
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.video_path is not None:
        top_path = Path(args.video_path).expanduser()
    else:
        top_path = Path("logs/pingti_hpt/rollout_videos") / f"pingti_hpt_joint_top_{stamp}.mp4"
    if args.wrist_video_path is not None:
        wrist_path = Path(args.wrist_video_path).expanduser()
    else:
        wrist_path = top_path.with_name(top_path.stem + "_wrist" + top_path.suffix)
    return top_path, wrist_path


def rollout_start_record(
    args: argparse.Namespace,
    *,
    dry_run: bool,
    log_path: Path | None,
    top_video_path: Path | None,
    wrist_video_path: Path | None,
    policy: SO100HPTPolicy,
) -> dict[str, Any]:
    record = step_record_base(-1, dry_run)
    record.update(
        {
            "event": "rollout_start",
            "argv": list(sys.argv),
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "robot": args.robot_type,
            "robot_id": args.robot_id,
            "port": args.port,
            "use_degrees": bool(args.use_degrees),
            "keep_torque_on_disconnect": bool(args.keep_torque_on_disconnect),
            "use_force_proxy": bool(args.use_force_proxy),
            "top_camera": args.top_camera,
            "wrist_camera": args.wrist_camera,
            "top_camera_key": args.top_camera_key,
            "wrist_camera_key": args.wrist_camera_key,
            "top_rotate_180": bool(args.top_rotate_180),
            "wrist_rotate_180": bool(args.wrist_rotate_180),
            "frequency": float(args.frequency),
            "query_frequency": int(args.query_frequency),
            "resampled_action_len": int(args.resampled_action_len),
            "action_start_index": int(args.action_start_index),
            "action_stride": int(args.action_stride),
            "rollout_blend_steps": int(args.rollout_blend_steps),
            "max_steps": int(args.max_steps),
            "max_joint_delta": float(args.max_joint_delta),
            "max_gripper_delta": float(args.max_gripper_delta),
            "gripper_min": float(args.gripper_min),
            "gripper_max": float(args.gripper_max),
            "lerobot_max_relative_target": args.lerobot_max_relative_target,
            "log_jsonl": None if log_path is None else str(log_path),
            "top_video_path": None if top_video_path is None else str(top_video_path),
            "wrist_video_path": None if wrist_video_path is None else str(wrist_video_path),
            "action_representation": ACTION_REPRESENTATION_JOINT,
            "action_key": policy.prediction_key,
            "raw_state_key": policy.raw_state_key,
            "raw_action_key": policy.raw_action_key,
            "policy_head_class": policy.head.__class__.__name__,
            "policy_head_action_horizon": (
                None if policy.head_action_horizon is None else int(policy.head_action_horizon)
            ),
            "policy_rollout_action_horizon": int(policy.action_horizon),
            "policy_num_inference_steps": (
                None if policy.num_inference_steps is None else int(policy.num_inference_steps)
            ),
            "policy_expected_camera_keys": list(policy.expected_camera_keys),
            "policy_expected_raw_camera_keys": dict(policy.expected_raw_camera_keys),
            "policy_requires_wrist_image": bool(policy.requires_wrist_image),
            "policy_expected_proprio_keys": list(policy.expected_proprio_keys),
            "policy_extra_raw_state_keys": list(policy.extra_raw_state_keys),
        }
    )
    return record


def joint_load_from_observation(obs: dict[str, Any]) -> np.ndarray | None:
    load = obs.get("load")
    if load is None:
        return None
    load = np.asarray(load, dtype=np.float64)
    if load.shape != (6,):
        raise ValueError(f"Expected load shape (6,), got {load.shape}")
    return load


def policy_extra_raw_states_from_observation(
    policy: SO100HPTPolicy,
    obs: dict[str, Any],
) -> dict[str, np.ndarray]:
    if not policy.expects_raw_state_key(JOINT_LOAD_RAW_KEY):
        return {}
    load = joint_load_from_observation(obs)
    if load is None:
        raise KeyError(
            f"Checkpoint expects {JOINT_LOAD_RAW_KEY}, but live observation did not include load. "
            "Enable PingTi force proxy / Present_Load collection."
        )
    return {JOINT_LOAD_RAW_KEY: load}


def run(args: argparse.Namespace) -> None:
    if args.camera is not None:
        args.top_camera = args.camera
    if args.camera_key is not None:
        args.top_camera_key = args.camera_key
    if args.max_joint_delta_deg is not None:
        args.max_joint_delta = args.max_joint_delta_deg
    if args.joint_limit_margin_deg is not None:
        args.joint_limit_margin = args.joint_limit_margin_deg
    if args.action_representation != ACTION_REPRESENTATION_JOINT:
        print(
            "[pingti] WARNING: this checkpoint/script only supports joint action. "
            f"Ignoring --action-representation {args.action_representation!r} and using joint."
        )

    if args.dry_run and args.enable_motors:
        raise ValueError("Use only one of --dry-run or --enable-motors.")
    dry_run = not args.enable_motors
    if args.query_frequency <= 0:
        raise ValueError("--query-frequency must be positive")
    if args.resampled_action_len < 0:
        raise ValueError("--resampled-action-len must be non-negative")
    if args.resampled_action_len > 0 and args.action_start_index + args.query_frequency > args.resampled_action_len:
        raise ValueError("--action-start-index + --query-frequency must fit in --resampled-action-len")
    if args.video_fps is None:
        args.video_fps = args.frequency

    joint_min = None if args.joint_min is None else np.asarray(args.joint_min, dtype=np.float64)
    joint_max = None if args.joint_max is None else np.asarray(args.joint_max, dtype=np.float64)
    safety = JointSafety(
        max_joint_delta=args.max_joint_delta,
        max_gripper_delta=args.max_gripper_delta,
        gripper_min=args.gripper_min,
        gripper_max=args.gripper_max,
        joint_min=joint_min,
        joint_max=joint_max,
        joint_limit_margin=args.joint_limit_margin,
    )
    policy = SO100HPTPolicy(
        checkpoint=args.checkpoint,
        device=args.device,
        precision=args.precision,
        action_horizon=args.action_horizon,
        bgr_to_rgb=False,
        action_representation=ACTION_REPRESENTATION_JOINT,
    )
    if args.use_force_proxy is None:
        args.use_force_proxy = policy.expects_raw_state_key(JOINT_LOAD_RAW_KEY)
    source = LivePingtiSource(args)
    action_queue: deque[np.ndarray] = deque()
    last_action_chunk: np.ndarray | None = None
    log_path = None if args.no_log else build_log_path(args)
    top_video_path, wrist_video_path = build_video_paths(args)

    print(f"[pingti] checkpoint: {args.checkpoint}")
    print(f"[pingti] dry_run: {dry_run}  enable_motors: {args.enable_motors}")
    print(
        "[pingti] policy cameras: "
        f"{policy.expected_raw_camera_keys} "
        f"requires_wrist_image={policy.requires_wrist_image}"
    )
    print(
        f"[pingti] robot_type={args.robot_type} port={args.port} id={args.robot_id} "
        f"use_degrees={args.use_degrees} "
        f"keep_torque_on_disconnect={args.keep_torque_on_disconnect} "
        f"use_force_proxy={args.use_force_proxy}"
    )
    print(
        f"[pingti] top_camera={args.top_camera} rotate180={args.top_rotate_180} "
        f"wrist_camera={args.wrist_camera}"
    )
    print(
        f"[pingti] frequency={args.frequency} query_frequency={args.query_frequency} "
        f"resampled_action_len={args.resampled_action_len} blend_steps={args.rollout_blend_steps}"
    )
    if args.max_ee_delta_m is not None or args.max_ik_pos_error_m is not None:
        print("[pingti] NOTE: EE/IK safety args are accepted for SO100 command compatibility but ignored for joint rollout.")
    print(f"[pingti] log_jsonl: {log_path}")
    print(f"[pingti] top_video: {top_video_path} wrist_video: {wrist_video_path}")
    if dry_run:
        print("[pingti] motors are disabled; add --enable-motors to send commands")

    source.connect(calibrate=args.calibrate)
    last_safe_target: np.ndarray | None = None
    try:
        if args.warmup_iters > 0:
            warm_obs = source.observe()
            warm_state = current_joint_pos_from_observation(warm_obs)
            warm_extra_raw_states = policy_extra_raw_states_from_observation(policy, warm_obs)
            start = time.perf_counter()
            for _ in range(args.warmup_iters):
                _ = policy.predict(
                    warm_obs["image"],
                    warm_state,
                    warm_extra_raw_states,
                    wrist_image=warm_obs.get("wrist_image"),
                )
            print(
                f"[pingti] warmed policy with {args.warmup_iters} iterations "
                f"in {(time.perf_counter() - start) * 1000.0:.1f} ms"
            )

        with JsonlLogger(log_path) as logger, AsyncDualVideoRecorder(
            top_video_path,
            wrist_video_path,
            fps=args.video_fps,
            codec=args.video_codec,
            queue_size=args.video_queue_size,
            every_n_steps=args.video_every_n_steps,
        ) as video_recorder, RateLoop(
            frequency=args.frequency,
            max_iterations=args.max_steps,
            verbose=args.verbose_rate,
        ) as loop:
            logger.write(
                rollout_start_record(
                    args,
                    dry_run=dry_run,
                    log_path=log_path,
                    top_video_path=top_video_path,
                    wrist_video_path=wrist_video_path,
                    policy=policy,
                )
            )
            for step in loop:
                loop_start = time.perf_counter()
                record = step_record_base(step, dry_run)
                obs_t = timed(source.observe)
                obs = obs_t.value
                current_joint_pos = current_joint_pos_from_observation(obs).astype(np.float64)
                current_joint_load = joint_load_from_observation(obs)
                extra_raw_states = policy_extra_raw_states_from_observation(policy, obs)
                video_enqueued = video_recorder.enqueue(step, obs["image"], obs.get("wrist_image"))
                record.update(
                    {
                        "obs_ms": obs_t.ms,
                        "video_enqueued": video_enqueued,
                        "joint_pos": current_joint_pos.tolist(),
                        "joint_load": None
                        if current_joint_load is None
                        else current_joint_load.tolist(),
                        "policy_state": current_joint_pos.tolist(),
                        "policy_extra_raw_states": {
                            key: np.asarray(value, dtype=np.float64).tolist()
                            for key, value in extra_raw_states.items()
                        },
                    }
                )

                should_query = step % args.query_frequency == 0 or not action_queue
                if should_query:
                    infer_t = timed(
                        lambda: policy.predict(
                            obs["image"],
                            current_joint_pos,
                            extra_raw_states,
                            wrist_image=obs.get("wrist_image"),
                        )
                    )
                    prediction = infer_t.value
                    if args.resampled_action_len > 0:
                        action_chunk = resample_action_chunk(
                            prediction,
                            args.resampled_action_len,
                            ACTION_REPRESENTATION_JOINT,
                        )
                        pre_blend = action_chunk.copy()
                        action_chunk, blend_info = blend_replanned_chunk(
                            action_chunk,
                            last_action_chunk,
                            tail_start=args.action_start_index + args.query_frequency,
                            blend_steps=args.rollout_blend_steps,
                            dims=args.rollout_blend_dims,
                            action_representation=ACTION_REPRESENTATION_JOINT,
                        )
                        start = args.action_start_index
                        stop = start + args.query_frequency
                        selected = action_chunk[start:stop]
                        action_indices = np.arange(start, stop, dtype=int)
                        selection_mode = "resample"
                    else:
                        action_chunk = prediction
                        pre_blend = action_chunk
                        blend_info = {"applied": False, "steps_applied": 0}
                        action_indices = (
                            args.action_start_index
                            + np.arange(args.query_frequency) * args.action_stride
                        )
                        action_indices = action_indices[action_indices < prediction.shape[0]]
                        if len(action_indices) == 0:
                            raise ValueError("No action index left in predicted chunk")
                        selected = prediction[action_indices]
                        selection_mode = "stride"

                    action_queue.clear()
                    action_queue.extend(selected)
                    last_action_chunk = action_chunk.copy()
                    gripper_index = prediction.shape[1] - 1
                    record.update(
                        {
                            "query": True,
                            "inference_ms": infer_t.ms,
                            "pred_shape": list(prediction.shape),
                            "action_selection_mode": selection_mode,
                            "action_indices": action_indices.astype(int).tolist(),
                            "action_chunk_shape": list(action_chunk.shape),
                            "queued_actions": len(action_queue),
                            "rollout_blend_info": blend_info,
                            "pred_gripper_chunk_stats": numeric_sequence_stats(prediction[:, gripper_index]),
                            "action_chunk_gripper_stats": numeric_sequence_stats(
                                action_chunk[:, gripper_index]
                            ),
                            "pre_blend_action_chunk_first": pre_blend[0].tolist(),
                        }
                    )
                else:
                    record["query"] = False

                target = np.asarray(action_queue.popleft(), dtype=np.float64)
                safe_target, safety_debug = safety.clip(current_joint_pos, target)
                last_safe_target = safe_target.copy()
                record.update(
                    {
                        "target_joint_pos": target.tolist(),
                        "safe_joint_pos": safe_target.tolist(),
                        "target_delta": (target - current_joint_pos).tolist(),
                        "safe_delta": (safe_target - current_joint_pos).tolist(),
                        "safety": safety_debug,
                    }
                )
                if dry_run:
                    sent = {"dry_run": True}
                else:
                    sent = source.send(safe_target)
                record["sent_action"] = sent
                record["loop_ms"] = (time.perf_counter() - loop_start) * 1000.0
                logger.write(record)

                if step % args.print_every == 0:
                    print(
                        f"[pingti] step={step:05d} q={np.round(current_joint_pos, 2).tolist()} "
                        f"target={np.round(target, 2).tolist()} safe={np.round(safe_target, 2).tolist()} "
                        f"query={record['query']} loop_ms={record['loop_ms']:.1f}"
                    )
            logger.write(
                {
                    **step_record_base(args.max_steps, dry_run),
                    "event": "rollout_end",
                    "video": video_recorder.summary(),
                }
            )
        if args.post_rollout_hold_s > 0 and not dry_run and last_safe_target is not None:
            hold_until = time.perf_counter() + float(args.post_rollout_hold_s)
            period = 1.0 / max(float(args.frequency), 1e-6)
            print(f"[pingti] holding final target for {args.post_rollout_hold_s:.2f}s")
            while time.perf_counter() < hold_until:
                source.send(last_safe_target)
                time.sleep(period)
    finally:
        source.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out an EgoVerse HPT joint policy on a PingTi follower arm."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the trained Lightning checkpoint.")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument(
        "--robot-type",
        default=DEFAULT_ROBOT_TYPE,
        choices=("pingti_lite_follower", "pingti_follower"),
    )
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Allow interactive calibration if needed. Defaults to using existing calibration only.",
    )
    parser.add_argument(
        "--use-degrees",
        action="store_true",
        help="Use degree-normalized PingTi joints. Leave unset for datasets collected with the shown teleop command.",
    )
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
    parser.add_argument(
        "--use-force-proxy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Read Present_Load from PingTi motors. Defaults to auto: enabled when the "
            "checkpoint expects observations.state.joint_load."
        ),
    )
    parser.add_argument(
        "--action-filter-type",
        default="none",
        choices=("none", "lowpass", "moving_average", "adaptive"),
    )
    parser.add_argument("--action-filter-alpha", type=float, default=0.3)
    parser.add_argument("--action-filter-window-size", type=int, default=3)
    parser.add_argument("--action-filter-adaptation-threshold", type=float, default=0.1)

    parser.add_argument("--top-camera", default=DEFAULT_TOP_CAMERA)
    parser.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA)
    parser.add_argument("--top-camera-key", default=DEFAULT_TOP_CAMERA_KEY)
    parser.add_argument("--wrist-camera-key", default=DEFAULT_WRIST_CAMERA_KEY)
    parser.add_argument("--camera-type", default="opencv", choices=("opencv",), help="SO100 command compatibility.")
    parser.add_argument("--camera", default=None, help="Alias for --top-camera.")
    parser.add_argument("--camera-key", default=None, help="Alias for --top-camera-key.")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-fourcc", default=None)
    parser.add_argument("--top-rotate-180", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wrist-rotate-180", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--action-representation",
        default=ACTION_REPRESENTATION_JOINT,
        help="Accepted for SO100 command compatibility. PingTi rollout always uses joint.",
    )
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--query-frequency", type=int, default=DEFAULT_QUERY_FREQUENCY)
    parser.add_argument("--resampled-action-len", type=int, default=DEFAULT_RESAMPLED_ACTION_LEN)
    parser.add_argument("--action-start-index", type=int, default=0)
    parser.add_argument("--action-stride", type=int, default=1)
    parser.add_argument("--rollout-blend-steps", type=int, default=5)
    parser.add_argument("--rollout-blend-dims", choices=("pose", "all"), default="all")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--warmup-iters", type=int, default=1)

    parser.add_argument("--max-joint-delta", type=float, default=8.0)
    parser.add_argument("--max-joint-delta-deg", type=float, default=None, help="Alias for --max-joint-delta.")
    parser.add_argument("--max-gripper-delta", type=float, default=8.0)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=100.0)
    parser.add_argument("--joint-min", nargs=5, type=float, default=None)
    parser.add_argument("--joint-max", nargs=5, type=float, default=None)
    parser.add_argument("--joint-limit-margin", type=float, default=0.0)
    parser.add_argument("--joint-limit-margin-deg", type=float, default=None, help="Alias for --joint-limit-margin.")
    parser.add_argument("--max-ee-delta-m", type=float, default=None, help="Ignored; SO100 command compatibility.")
    parser.add_argument("--max-ik-pos-error-m", type=float, default=None, help="Ignored; SO100 command compatibility.")
    parser.add_argument("--stuck-action", default="warn", choices=("warn", "stop", "ignore"), help="Accepted but not currently enforced.")
    parser.add_argument("--stuck-window", type=int, default=30, help="Accepted but not currently enforced.")
    parser.add_argument("--post-rollout-hold-s", type=float, default=0.0)

    parser.add_argument(
        "--enable-motors",
        action="store_true",
        help="Actually send actions to the PingTi arm. Default is dry-run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run.")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--log-jsonl", default=None)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-path", default=None, help="Top-camera rollout video path.")
    parser.add_argument("--wrist-video-path", default=None)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--video-queue-size", type=int, default=128)
    parser.add_argument("--video-every-n-steps", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--verbose-rate", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
