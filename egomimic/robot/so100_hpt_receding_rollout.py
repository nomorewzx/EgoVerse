from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.robot.robot_utils import RateLoop
from egomimic.utils.egomimicUtils import interpolate_arr_euler

try:
    import zarr
except ImportError:  # pragma: no cover - offline dry-run helper only
    zarr = None

try:
    import simplejpeg
except ImportError:  # pragma: no cover - optional fast JPEG decoder
    simplejpeg = None

try:
    import cv2
except ImportError:  # pragma: no cover - fallback JPEG decoder only
    cv2 = None


DEFAULT_CHECKPOINT = (
    "logs/so100_hpt/hpt_base_steps100k_2026-04-30_19-42-21/"
    "checkpoints/step_80000.ckpt"
)
DEFAULT_CONVERSION_METADATA = "/home/zxwang/so100_calib/ee_pose_conversion.json"
DEFAULT_URDF = "/home/zxwang/repos/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
DEFAULT_ROBOT_ID = "so100_hpt_follower"

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
GRIPPER_NAME = "gripper"
ACTION_KEY = "so100_singlearm_actions_cartesian"
DEFAULT_QUERY_FREQUENCY = 30
DEFAULT_RESAMPLED_ACTION_LEN = 45


def load_lerobot_arm_joint_limits_deg(calibration_fpath: str | Path) -> np.ndarray:
    calibration_fpath = Path(calibration_fpath).expanduser()
    data = json.loads(calibration_fpath.read_text(encoding="utf-8"))
    limits = []
    max_res = 4095.0
    for joint_name in ARM_JOINT_NAMES:
        item = data[joint_name]
        mid = (float(item["range_min"]) + float(item["range_max"])) / 2.0
        min_degrees = (float(item["range_min"]) - mid) * 360.0 / max_res
        max_degrees = (float(item["range_max"]) - mid) * 360.0 / max_res
        limits.append([min_degrees, max_degrees])
    return np.asarray(limits, dtype=np.float64)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {transform.shape}")
    out = np.eye(4, dtype=np.float64)
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out


def pose7_ypr_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Expected pose shape (7,), got {pose.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_euler("ZYX", pose[3:6], degrees=False).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def matrix_to_pose7_ypr(transform: np.ndarray, gripper: float) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {transform.shape}")
    ypr = R.from_matrix(transform[:3, :3]).as_euler("ZYX", degrees=False)
    return np.concatenate([transform[:3, 3], ypr, [float(gripper)]]).astype(np.float32)


def rotvec_pose7_to_ypr_pose7(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (7,):
        raise ValueError(f"Expected pose shape (7,), got {pose.shape}")
    ypr = R.from_rotvec(pose[3:6]).as_euler("ZYX", degrees=False).astype(np.float32)
    return np.concatenate([pose[:3], ypr, pose[6:7]], axis=0).astype(np.float32)


def decode_jpeg_value(value: object) -> np.ndarray:
    while isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, np.ndarray) and value.dtype == np.uint8:
        jpeg_bytes = value.tobytes()
    elif isinstance(value, (bytes, bytearray, memoryview)):
        jpeg_bytes = bytes(value)
    else:
        arr = np.asarray(value)
        if arr.ndim == 3:
            return arr
        raise ValueError(
            f"Cannot decode image value of type {type(value)} with shape {getattr(arr, 'shape', None)}"
        )

    if simplejpeg is not None:
        return simplejpeg.decode_jpeg(jpeg_bytes, colorspace="RGB")
    if cv2 is None:
        raise ImportError("Need simplejpeg or cv2 to decode JPEG zarr images")
    decoded_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise ValueError("cv2.imdecode returned None")
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def as_float_array(values: list[float] | None, expected_len: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (expected_len,):
        raise ValueError(f"{name} expects {expected_len} values, got {arr.shape}")
    return arr


def finite_or_none(arr: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def clip_rotation_towards(
    current_rot: R,
    target_rot: R,
    max_delta_rad: float | None,
) -> R:
    if max_delta_rad is None or max_delta_rad <= 0:
        return target_rot
    delta = target_rot * current_rot.inv()
    rotvec = delta.as_rotvec()
    norm = float(np.linalg.norm(rotvec))
    if norm <= max_delta_rad or norm == 0.0:
        return target_rot
    clipped_delta = R.from_rotvec(rotvec / norm * max_delta_rad)
    return clipped_delta * current_rot


@dataclass
class TimedResult:
    value: Any
    ms: float


def timed(fn) -> TimedResult:
    start = time.perf_counter()
    value = fn()
    return TimedResult(value=value, ms=(time.perf_counter() - start) * 1000.0)


class SO100FrameBridge:
    def __init__(self, conversion_metadata: str | Path):
        payload = json.loads(Path(conversion_metadata).expanduser().read_text(encoding="utf-8"))
        calibration = payload.get("calibration", {})
        if "T_cam_base" not in calibration:
            raise KeyError(f"{conversion_metadata} does not contain calibration.T_cam_base")
        self.cam_T_base = np.asarray(calibration["T_cam_base"], dtype=np.float64)
        if self.cam_T_base.shape != (4, 4):
            raise ValueError(f"T_cam_base must be 4x4, got {self.cam_T_base.shape}")
        self.base_T_cam = invert_transform(self.cam_T_base)

    def base_T_ee_to_camera_ypr(self, base_T_ee: np.ndarray, gripper: float) -> np.ndarray:
        cam_T_ee = self.cam_T_base @ np.asarray(base_T_ee, dtype=np.float64)
        return matrix_to_pose7_ypr(cam_T_ee, gripper)

    def camera_ypr_to_base_T_ee(self, camera_pose_ypr: np.ndarray) -> np.ndarray:
        cam_T_ee = pose7_ypr_to_matrix(camera_pose_ypr)
        return self.base_T_cam @ cam_T_ee


class SO100Kinematics:
    def __init__(
        self,
        urdf_path: str | Path,
        target_frame_name: str,
        joint_names: list[str],
        position_weight: float,
        orientation_weight: float,
    ):
        from lerobot.model.kinematics import RobotKinematics

        self.joint_names = list(joint_names)
        self.position_weight = float(position_weight)
        self.orientation_weight = float(orientation_weight)
        self.kinematics = RobotKinematics(
            urdf_path=str(Path(urdf_path).expanduser()),
            target_frame_name=target_frame_name,
            joint_names=self.joint_names,
        )

    def fk(self, q_deg: np.ndarray) -> np.ndarray:
        q_deg = np.asarray(q_deg, dtype=np.float64)
        return np.asarray(self.kinematics.forward_kinematics(q_deg), dtype=np.float64)

    def ik(self, q_current_deg: np.ndarray, target_base_T_ee: np.ndarray) -> np.ndarray | None:
        q_current_deg = np.asarray(q_current_deg, dtype=np.float64)
        q = self.kinematics.inverse_kinematics(
            current_joint_pos=q_current_deg,
            desired_ee_pose=np.asarray(target_base_T_ee, dtype=np.float64),
            position_weight=self.position_weight,
            orientation_weight=self.orientation_weight,
        )
        return finite_or_none(q[: len(self.joint_names)])


class SO100HPTPolicy:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str,
        precision: str,
        action_horizon: int | None,
        bgr_to_rgb: bool,
    ):
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self.device = requested_device
        self.precision = precision
        self.bgr_to_rgb = bool(bgr_to_rgb)

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

        self.wrapper = ModelWrapper.load_from_checkpoint(
            str(Path(checkpoint).expanduser()),
            weights_only=False,
            map_location="cpu",
        )
        self.wrapper = self.wrapper.to(self.device)
        self.wrapper.eval()
        self.wrapper.model.device = self.device
        self.head = self.wrapper.model.nets["policy"].heads["so100_singlearm"]
        self.diffusion = bool(getattr(self.wrapper.model, "diffusion", False))
        self.head_action_horizon = getattr(self.head, "action_horizon", None)
        self.num_inference_steps = getattr(self.head, "num_inference_steps", None)

        if action_horizon is None:
            if self.head_action_horizon is None:
                raise ValueError(
                    "Checkpoint head does not expose action_horizon; pass --action-horizon explicitly."
                )
            self.action_horizon = int(self.head_action_horizon)
        else:
            self.action_horizon = int(action_horizon)
            if (
                self.head_action_horizon is not None
                and self.action_horizon != int(self.head_action_horizon)
            ):
                print(
                    "[so100] WARNING: --action-horizon "
                    f"{self.action_horizon} differs from checkpoint head horizon "
                    f"{self.head_action_horizon}; forward_eval will crop/pad by the dummy action length."
                )

        print(
            "[so100] policy head: "
            f"{self.head.__class__.__name__} diffusion={self.diffusion} "
            f"head_action_horizon={self.head_action_horizon} "
            f"rollout_action_horizon={self.action_horizon} "
            f"num_inference_steps={self.num_inference_steps}"
        )

    def _autocast(self):
        if self.device.type != "cuda" or self.precision == "fp32":
            return nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _image_to_chw_float(self, image: np.ndarray) -> torch.Tensor:
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Expected image shape (H,W,C) or (C,H,W), got {image.shape}")
        if image.shape[-1] == 3:
            if self.bgr_to_rgb:
                image = image[..., [2, 1, 0]]
            front = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        elif image.shape[0] == 3:
            front = torch.from_numpy(np.ascontiguousarray(image))
        else:
            raise ValueError(f"Expected 3-channel image, got {image.shape}")
        front = front.to(dtype=torch.float32)
        if float(front.max()) > 1.5:
            front = front / 255.0
        return front

    def _build_raw_batch(self, image: np.ndarray, ee_camera_ypr: np.ndarray) -> dict:
        front = self._image_to_chw_float(image)
        ee = torch.as_tensor(np.asarray(ee_camera_ypr, dtype=np.float32))
        dummy_actions = ee.view(1, 7).repeat(self.action_horizon, 1)
        return {
            "so100_singlearm": {
                "observations.images.front_img_1": front.unsqueeze(0),
                "observations.state.ee_pose": ee.unsqueeze(0),
                "actions_cartesian": dummy_actions.unsqueeze(0),
            }
        }

    def predict(self, image: np.ndarray, ee_camera_ypr: np.ndarray) -> np.ndarray:
        raw_batch = self._build_raw_batch(image, ee_camera_ypr)
        with torch.inference_mode(), self._autocast():
            processed = self.wrapper.model.process_batch_for_training(raw_batch)
            preds = self.wrapper.model.forward_eval(processed)
        if ACTION_KEY not in preds:
            raise KeyError(f"Expected prediction key {ACTION_KEY}, got {list(preds.keys())}")
        chunk = preds[ACTION_KEY].detach().float().cpu().numpy().squeeze(0)
        if chunk.ndim != 2 or chunk.shape[1] != 7:
            raise ValueError(f"Expected action chunk shape (T,7), got {chunk.shape}")
        return chunk.astype(np.float32, copy=False)


class SO100Safety:
    def __init__(
        self,
        *,
        workspace_min: np.ndarray | None,
        workspace_max: np.ndarray | None,
        max_ee_delta_m: float,
        max_rot_delta_deg: float,
        max_joint_delta_deg: float,
        max_gripper_delta: float,
        gripper_min: float,
        gripper_max: float,
        max_ik_pos_error_m: float,
    ):
        self.workspace_min = workspace_min
        self.workspace_max = workspace_max
        self.max_ee_delta_m = float(max_ee_delta_m)
        self.max_rot_delta_rad = math.radians(float(max_rot_delta_deg))
        self.max_joint_delta_deg = float(max_joint_delta_deg)
        self.max_gripper_delta = float(max_gripper_delta)
        self.gripper_min = float(gripper_min)
        self.gripper_max = float(gripper_max)
        self.max_ik_pos_error_m = float(max_ik_pos_error_m)
        self.joint_min_deg: np.ndarray | None = None
        self.joint_max_deg: np.ndarray | None = None
        self.joint_limit_margin_deg = 0.0

    def set_joint_limits(self, limits_deg: np.ndarray, margin_deg: float) -> None:
        limits = np.asarray(limits_deg, dtype=np.float64)
        if limits.shape != (len(ARM_JOINT_NAMES), 2):
            raise ValueError(f"Expected joint limits shape {(len(ARM_JOINT_NAMES), 2)}, got {limits.shape}")
        self.joint_min_deg = limits[:, 0]
        self.joint_max_deg = limits[:, 1]
        self.joint_limit_margin_deg = max(0.0, float(margin_deg))

    def joint_limit_margins(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        q = np.asarray(q, dtype=np.float64)
        if self.joint_min_deg is None or self.joint_max_deg is None:
            nan = np.full_like(q, np.nan, dtype=np.float64)
            return nan, nan, float("nan")
        low_margin = q - self.joint_min_deg
        high_margin = self.joint_max_deg - q
        min_margin = float(np.min(np.minimum(low_margin, high_margin)))
        return low_margin, high_margin, min_margin

    def clip_target_pose(self, current_base_T_ee: np.ndarray, target_base_T_ee: np.ndarray) -> np.ndarray:
        current = np.asarray(current_base_T_ee, dtype=np.float64)
        target = np.asarray(target_base_T_ee, dtype=np.float64).copy()

        if self.workspace_min is not None:
            target[:3, 3] = np.maximum(target[:3, 3], self.workspace_min)
        if self.workspace_max is not None:
            target[:3, 3] = np.minimum(target[:3, 3], self.workspace_max)

        delta = target[:3, 3] - current[:3, 3]
        delta_norm = float(np.linalg.norm(delta))
        if self.max_ee_delta_m > 0 and delta_norm > self.max_ee_delta_m:
            target[:3, 3] = current[:3, 3] + delta / delta_norm * self.max_ee_delta_m

        current_rot = R.from_matrix(current[:3, :3])
        target_rot = R.from_matrix(target[:3, :3])
        clipped_rot = clip_rotation_towards(current_rot, target_rot, self.max_rot_delta_rad)
        target[:3, :3] = clipped_rot.as_matrix()
        return target

    def clip_joint_target(
        self,
        q_current: np.ndarray,
        q_target: np.ndarray,
        current_gripper: float,
        target_gripper: float,
    ) -> tuple[np.ndarray, float]:
        q_current = np.asarray(q_current, dtype=np.float64)
        q_target = np.asarray(q_target, dtype=np.float64)
        q_delta = np.clip(
            q_target - q_current,
            -self.max_joint_delta_deg,
            self.max_joint_delta_deg,
        )
        q_safe = q_current + q_delta
        if self.joint_min_deg is not None and self.joint_max_deg is not None:
            lower = self.joint_min_deg + self.joint_limit_margin_deg
            upper = self.joint_max_deg - self.joint_limit_margin_deg
            q_safe = np.clip(q_safe, lower, upper)
            q_safe = np.clip(
                q_safe,
                q_current - self.max_joint_delta_deg,
                q_current + self.max_joint_delta_deg,
            )

        gripper, _ = self.clip_gripper_target(current_gripper, target_gripper)
        return q_safe, gripper

    def clip_gripper_target(
        self,
        current_gripper: float,
        target_gripper: float,
    ) -> tuple[float, dict[str, Any]]:
        current = float(current_gripper)
        raw_target = float(target_gripper)
        range_clipped_target = float(np.clip(raw_target, self.gripper_min, self.gripper_max))
        raw_delta = raw_target - current
        range_clipped_delta = range_clipped_target - current
        delta_limited_delta = float(
            np.clip(
                range_clipped_delta,
                -self.max_gripper_delta,
                self.max_gripper_delta,
            )
        )
        safe_target = current + delta_limited_delta
        debug = {
            "current": current,
            "raw_target": raw_target,
            "range_clipped_target": range_clipped_target,
            "safe_target": safe_target,
            "gripper_min": float(self.gripper_min),
            "gripper_max": float(self.gripper_max),
            "max_gripper_delta": float(self.max_gripper_delta),
            "raw_delta_from_current": raw_delta,
            "range_clipped_delta_from_current": range_clipped_delta,
            "delta_limited_delta_from_current": delta_limited_delta,
            "range_clip_delta": range_clipped_target - raw_target,
            "delta_clip_delta": delta_limited_delta - range_clipped_delta,
            "total_clip_delta_from_raw": safe_target - raw_target,
            "range_clipped": bool(not np.isclose(range_clipped_target, raw_target)),
            "delta_limited": bool(not np.isclose(delta_limited_delta, range_clipped_delta)),
        }
        return safe_target, debug

    def ik_position_error(self, solved_base_T_ee: np.ndarray, target_base_T_ee: np.ndarray) -> float:
        return float(
            np.linalg.norm(
                np.asarray(solved_base_T_ee, dtype=np.float64)[:3, 3]
                - np.asarray(target_base_T_ee, dtype=np.float64)[:3, 3]
            )
        )


class JsonlLogger:
    def __init__(self, path: str | Path | None):
        self.path = Path(path).expanduser() if path else None
        self.fp = None

    def __enter__(self):
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fp = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fp is not None:
            self.fp.close()

    def write(self, record: dict[str, Any]) -> None:
        if self.fp is None:
            return
        self.fp.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.fp.flush()


class AsyncVideoRecorder:
    def __init__(
        self,
        path: str | Path | None,
        *,
        fps: float,
        codec: str,
        queue_size: int,
        every_n_steps: int,
        input_color: str,
    ):
        self.path = Path(path).expanduser() if path else None
        self.fps = float(fps)
        self.codec = str(codec)
        self.every_n_steps = max(1, int(every_n_steps))
        self.input_color = input_color
        self.queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max(1, int(queue_size)))
        self.thread: threading.Thread | None = None
        self.writer = None
        self.enabled = self.path is not None
        self.frames_enqueued = 0
        self.frames_written = 0
        self.frames_dropped = 0
        self.error: str | None = None

    def __enter__(self):
        if not self.enabled:
            return self
        if cv2 is None:
            raise ImportError("cv2 is required for --record-video")
        if self.fps <= 0:
            raise ValueError("--video-fps must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._worker, name="async-video-recorder", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def enqueue(self, step: int, image: np.ndarray) -> bool:
        if not self.enabled or step % self.every_n_steps != 0:
            return False
        try:
            # Copy before returning to the control loop so the writer thread never
            # observes a camera buffer that is reused by the capture backend.
            frame = np.ascontiguousarray(image).copy()
            self.queue.put_nowait(frame)
            self.frames_enqueued += 1
            return True
        except queue.Full:
            self.frames_dropped += 1
            return False

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self.queue.put(None, timeout=1.0)
        except queue.Full:
            self.frames_dropped += 1
            try:
                _ = self.queue.get_nowait()
                self.queue.put_nowait(None)
            except queue.Empty:
                pass
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def summary(self) -> dict[str, Any]:
        return {
            "video_path": str(self.path) if self.path is not None else None,
            "video_frames_enqueued": self.frames_enqueued,
            "video_frames_written": self.frames_written,
            "video_frames_dropped": self.frames_dropped,
            "video_error": self.error,
        }

    def _worker(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            try:
                frame_bgr = self._prepare_frame_bgr(item)
                if self.writer is None:
                    height, width = frame_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*self.codec)
                    self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
                    if not self.writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer: {self.path}")
                self.writer.write(frame_bgr)
                self.frames_written += 1
            except Exception as exc:
                self.error = str(exc)
                self.frames_dropped += 1
            finally:
                self.queue.task_done()

    def _prepare_frame_bgr(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.ndim != 3:
            raise ValueError(f"Expected video frame shape (H,W,C) or (C,H,W), got {frame.shape}")
        if frame.shape[0] == 3 and frame.shape[-1] != 3:
            frame = np.transpose(frame, (1, 2, 0))
        if frame.shape[-1] != 3:
            raise ValueError(f"Expected 3-channel video frame, got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = frame.astype(np.float32, copy=False)
            if float(np.nanmax(frame)) <= 1.5:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if self.input_color == "rgb":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self.input_color == "bgr":
            return np.ascontiguousarray(frame)
        raise ValueError(f"Unsupported input_color: {self.input_color}")


def append_jsonl(path: str | Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, separators=(",", ":")) + "\n")


class LiveSO100Source:
    def __init__(self, args: argparse.Namespace):
        from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

        cameras = self._build_cameras(args)
        robot_id = args.robot_id
        calibration_dir = Path(args.calibration_dir).expanduser() if args.calibration_dir else None
        if args.calibration_file is not None:
            calibration_file = Path(args.calibration_file).expanduser()
            if not calibration_file.is_file():
                raise FileNotFoundError(calibration_file)
            calibration_dir = calibration_file.parent
            if robot_id == DEFAULT_ROBOT_ID:
                robot_id = calibration_file.stem
            elif robot_id != calibration_file.stem:
                raise ValueError(
                    f"--robot-id {robot_id!r} does not match --calibration-file stem "
                    f"{calibration_file.stem!r}. Use --robot-id {calibration_file.stem} "
                    "or omit --robot-id."
                )

        config_kwargs: dict[str, Any] = {
            "port": args.port,
            "id": robot_id,
            "cameras": cameras,
            "use_degrees": True,
            "disable_torque_on_disconnect": not args.keep_torque_on_disconnect,
            "max_relative_target": args.lerobot_max_relative_target,
        }
        if calibration_dir is not None:
            config_kwargs["calibration_dir"] = calibration_dir
        self.robot = SO100Follower(SO100FollowerConfig(**config_kwargs))
        self.camera_key = args.camera_key
        self.calibration_fpath = self.robot.calibration_fpath
        if not self.calibration_fpath.is_file():
            raise FileNotFoundError(
                f"LeRobot calibration file not found for robot id {self.robot.id!r}: "
                f"{self.calibration_fpath}. Pass --robot-id matching your calibration "
                "file, or pass --calibration-file /path/to/<id>.json."
            )
        self.arm_joint_limits_deg = load_lerobot_arm_joint_limits_deg(self.calibration_fpath)

    @staticmethod
    def _build_cameras(args: argparse.Namespace) -> dict[str, Any]:
        if args.camera_type == "opencv":
            from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

            index_or_path: int | Path
            if str(args.camera).isdigit():
                index_or_path = int(args.camera)
            else:
                index_or_path = Path(args.camera).expanduser()
            camera_cfg = OpenCVCameraConfig(
                index_or_path=index_or_path,
                fps=args.camera_fps,
                width=args.camera_width,
                height=args.camera_height,
            )
        elif args.camera_type == "realsense":
            from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

            camera_cfg = RealSenseCameraConfig(
                serial_number_or_name=str(args.camera),
                fps=args.camera_fps,
                width=args.camera_width,
                height=args.camera_height,
            )
        else:
            raise ValueError(f"Unsupported camera type: {args.camera_type}")
        return {args.camera_key: camera_cfg}

    def connect(self, calibrate: bool) -> None:
        self.robot.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        self.robot.disconnect()

    def read_motor_registers(self) -> dict[str, dict[str, Any]]:
        registers = [
            "P_Coefficient",
            "I_Coefficient",
            "D_Coefficient",
            "Max_Torque_Limit",
            "Torque_Limit",
            "Overload_Torque",
            "Torque_Enable",
        ]
        out: dict[str, dict[str, Any]] = {}
        for motor in ARM_JOINT_NAMES + [GRIPPER_NAME]:
            values: dict[str, Any] = {}
            for register in registers:
                try:
                    values[register] = self.robot.bus.read(register, motor, normalize=False)
                except Exception as exc:
                    values[register] = f"unavailable: {exc}"
            out[motor] = values
        return out

    def observe(self) -> dict[str, Any]:
        obs = self.robot.get_observation()
        if self.camera_key not in obs:
            raise KeyError(f"Camera key {self.camera_key!r} not found in observation keys {list(obs)}")
        q = np.asarray([float(obs[f"{name}.pos"]) for name in ARM_JOINT_NAMES], dtype=np.float64)
        gripper = float(obs[f"{GRIPPER_NAME}.pos"])
        return {
            "image": np.asarray(obs[self.camera_key]),
            "q": q,
            "gripper": gripper,
            "raw": obs,
        }

    def send(self, q: np.ndarray, gripper: float) -> dict[str, Any]:
        action = {f"{name}.pos": float(q[i]) for i, name in enumerate(ARM_JOINT_NAMES)}
        action[f"{GRIPPER_NAME}.pos"] = float(gripper)
        return self.robot.send_action(action)


class OfflineZarrSource:
    def __init__(self, args: argparse.Namespace):
        if zarr is None:
            raise ImportError("zarr is required for --offline-zarr-episode")
        self.path = Path(args.offline_zarr_episode).expanduser()
        self.group = zarr.open(str(self.path), mode="r")
        self.frame = int(args.offline_frame_index)
        self.q = as_float_array(args.offline_current_joints, 5, "--offline-current-joints")
        if self.q is None:
            self.q = np.zeros(5, dtype=np.float64)
        self.gripper = float(args.offline_gripper)

    def connect(self, calibrate: bool) -> None:
        del calibrate

    def disconnect(self) -> None:
        pass

    def observe(self) -> dict[str, Any]:
        idx = self.frame % int(self.group["obs_ee_pose_cam_rotvec"].shape[0])
        self.frame += 1
        image = decode_jpeg_value(self.group["images.front_1"][idx])
        ee_ypr = rotvec_pose7_to_ypr_pose7(np.asarray(self.group["obs_ee_pose_cam_rotvec"][idx]))
        self.gripper = float(ee_ypr[6])
        return {
            "image": image,
            "q": self.q.copy(),
            "gripper": self.gripper,
            "ee_camera_ypr_override": ee_ypr,
            "raw": {"offline_frame_index": idx},
        }

    def send(self, q: np.ndarray, gripper: float) -> dict[str, Any]:
        self.q = np.asarray(q, dtype=np.float64).copy()
        self.gripper = float(gripper)
        return {f"{name}.pos": float(self.q[i]) for i, name in enumerate(ARM_JOINT_NAMES)} | {
            f"{GRIPPER_NAME}.pos": self.gripper
        }


def build_log_path(args: argparse.Namespace) -> Path | None:
    if args.log_jsonl is not None:
        return Path(args.log_jsonl).expanduser()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("logs/so100_hpt/rollout_logs") / f"so100_hpt_rollout_{stamp}.jsonl"


def build_video_path(args: argparse.Namespace) -> Path | None:
    if not args.record_video:
        return None
    if args.video_path is not None:
        return Path(args.video_path).expanduser()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("logs/so100_hpt/rollout_videos") / f"so100_hpt_rollout_{stamp}.mp4"


def make_source(args: argparse.Namespace):
    if args.offline_zarr_episode is not None:
        return OfflineZarrSource(args)
    if not args.port:
        raise ValueError("--port is required unless --offline-zarr-episode is set")
    return LiveSO100Source(args)


def step_record_base(step: int, dry_run: bool) -> dict[str, Any]:
    return {
        "step": int(step),
        "time": time.time(),
        "dry_run": bool(dry_run),
    }


def numeric_sequence_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    stats: dict[str, Any] = {
        "count": int(arr.size),
        "finite_count": int(finite.size),
    }
    if arr.size == 0 or finite.size == 0:
        stats.update(
            {
                "first": None,
                "last": None,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "range": None,
            }
        )
        return stats
    stats.update(
        {
            "first": float(arr[0]),
            "last": float(arr[-1]),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "range": float(np.max(finite) - np.min(finite)),
        }
    )
    return stats


def resample_camera_ypr_chunk(chunk: np.ndarray, target_len: int) -> np.ndarray:
    actions = np.asarray(chunk, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected action chunk shape (T, 7), got {actions.shape}")
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if actions.shape[0] == target_len:
        return actions.astype(np.float32, copy=False)
    return interpolate_arr_euler(actions[None, ...], target_len)[0].astype(np.float32)


def angle_delta_norm(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.linalg.norm(delta))


def blend_replanned_chunk(
    new_chunk: np.ndarray,
    previous_chunk: np.ndarray | None,
    *,
    tail_start: int,
    blend_steps: int,
    dims: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if blend_steps <= 0 or previous_chunk is None:
        return new_chunk, {
            "applied": False,
            "steps_applied": 0,
            "tail_start_index": int(tail_start),
        }

    if dims not in {"pose", "all"}:
        raise ValueError(f"Unsupported blend dims: {dims}")

    old_tail = previous_chunk[tail_start:]
    n = min(int(blend_steps), len(old_tail), len(new_chunk))
    info: dict[str, Any] = {
        "applied": n > 0,
        "steps_applied": int(n),
        "tail_start_index": int(tail_start),
        "reference_chunk_shape": list(previous_chunk.shape),
        "requested_steps": int(blend_steps),
        "dims": dims,
    }
    if n <= 0:
        return new_chunk, info

    out = new_chunk.copy()
    dim_slice = slice(0, 6) if dims == "pose" else slice(None)
    pre = new_chunk[:n].copy()
    ref = old_tail[:n].copy()
    for i in range(n):
        alpha = float(i + 1) / float(n)
        out[i, dim_slice] = alpha * new_chunk[i, dim_slice] + (1.0 - alpha) * ref[i, dim_slice]

    info["alphas"] = [float(i + 1) / float(n) for i in range(n)]
    info["pre_first_pos_gap_m"] = float(np.linalg.norm(pre[0, :3] - ref[0, :3]))
    info["post_first_pos_gap_m"] = float(np.linalg.norm(out[0, :3] - ref[0, :3]))
    info["pre_first_rot_gap_rad"] = angle_delta_norm(pre[0, 3:6], ref[0, 3:6])
    info["post_first_rot_gap_rad"] = angle_delta_norm(out[0, 3:6], ref[0, 3:6])
    info["max_pose_delta_m"] = float(np.max(np.linalg.norm(out[:n, :3] - pre[:, :3], axis=1)))
    info["max_rot_delta_rad"] = float(
        np.max([angle_delta_norm(out[i, 3:6], pre[i, 3:6]) for i in range(n)])
    )
    info["max_gripper_delta"] = float(np.max(np.abs(out[:n, 6] - pre[:, 6])))
    return out, info


def rollout_start_record(
    args: argparse.Namespace,
    *,
    dry_run: bool,
    log_path: Path | None,
    video_path: Path | None,
    policy: SO100HPTPolicy,
    arm_joint_limits_deg: np.ndarray | None,
) -> dict[str, Any]:
    record = step_record_base(-1, dry_run)
    record["event"] = "rollout_start"
    record["argv"] = list(sys.argv)
    record["checkpoint"] = str(Path(args.checkpoint).expanduser())
    record["conversion_metadata"] = str(Path(args.conversion_metadata).expanduser())
    record["urdf"] = str(Path(args.urdf).expanduser())
    record["target_frame_name"] = args.target_frame_name
    record["robot_id"] = args.robot_id
    record["port"] = args.port
    record["camera_type"] = args.camera_type
    record["camera"] = args.camera
    record["camera_key"] = args.camera_key
    record["frequency"] = float(args.frequency)
    record["query_frequency"] = int(args.query_frequency)
    record["action_start_index"] = int(args.action_start_index)
    record["action_stride"] = int(args.action_stride)
    record["resampled_action_len"] = int(args.resampled_action_len)
    record["action_selection_mode"] = (
        "resample" if args.resampled_action_len > 0 else "stride"
    )
    record["rollout_blend_steps"] = int(args.rollout_blend_steps)
    record["rollout_blend_dims"] = args.rollout_blend_dims
    record["max_steps"] = int(args.max_steps)
    record["max_ee_delta_m"] = float(args.max_ee_delta_m)
    record["max_rot_delta_deg"] = float(args.max_rot_delta_deg)
    record["max_joint_delta_deg"] = float(args.max_joint_delta_deg)
    record["joint_limit_margin_deg"] = float(args.joint_limit_margin_deg)
    record["max_gripper_delta"] = float(args.max_gripper_delta)
    record["gripper_min"] = float(args.gripper_min)
    record["gripper_max"] = float(args.gripper_max)
    record["lerobot_max_relative_target"] = (
        None
        if args.lerobot_max_relative_target is None
        else float(args.lerobot_max_relative_target)
    )
    record["max_ik_pos_error_m"] = float(args.max_ik_pos_error_m)
    record["stuck_action"] = args.stuck_action
    record["stuck_window"] = int(args.stuck_window)
    record["log_jsonl"] = None if log_path is None else str(log_path)
    record["record_video"] = bool(args.record_video)
    record["video_path"] = None if video_path is None else str(video_path)
    record["action_key"] = ACTION_KEY
    record["policy_head_class"] = policy.head.__class__.__name__
    record["policy_diffusion"] = bool(policy.diffusion)
    record["policy_head_action_horizon"] = (
        None if policy.head_action_horizon is None else int(policy.head_action_horizon)
    )
    record["policy_rollout_action_horizon"] = int(policy.action_horizon)
    record["policy_num_inference_steps"] = (
        None if policy.num_inference_steps is None else int(policy.num_inference_steps)
    )
    if arm_joint_limits_deg is not None:
        record["arm_joint_limits_deg"] = np.asarray(arm_joint_limits_deg, dtype=np.float64).tolist()
    return record


def current_camera_pose_from_observation(
    obs: dict[str, Any],
    kinematics: SO100Kinematics,
    bridge: SO100FrameBridge,
) -> tuple[np.ndarray, np.ndarray]:
    q_current = np.asarray(obs["q"], dtype=np.float64)
    current_base_T_ee = kinematics.fk(q_current)
    if "ee_camera_ypr_override" in obs:
        current_ee_camera_ypr = np.asarray(obs["ee_camera_ypr_override"], dtype=np.float32)
        current_base_T_ee = bridge.camera_ypr_to_base_T_ee(current_ee_camera_ypr)
    else:
        current_ee_camera_ypr = bridge.base_T_ee_to_camera_ypr(
            current_base_T_ee,
            float(obs["gripper"]),
        )
    return current_base_T_ee, current_ee_camera_ypr


def run(args: argparse.Namespace) -> None:
    if args.dry_run and args.enable_motors:
        raise ValueError("Use only one of --dry-run or --enable-motors.")
    dry_run = not args.enable_motors
    if args.enable_motors and args.offline_zarr_episode is not None:
        raise ValueError("--enable-motors is incompatible with --offline-zarr-episode")
    if args.query_frequency <= 0:
        raise ValueError("--query-frequency must be positive")
    if args.resampled_action_len < 0:
        raise ValueError("--resampled-action-len must be non-negative")
    if args.action_stride <= 0:
        raise ValueError("--action-stride must be positive")
    if args.action_start_index < 0:
        raise ValueError("--action-start-index must be non-negative")
    if (
        args.resampled_action_len > 0
        and args.action_start_index + args.query_frequency > args.resampled_action_len
    ):
        raise ValueError(
            "--action-start-index + --query-frequency must fit within "
            "--resampled-action-len when using official resample mode. "
            f"Got {args.action_start_index} + {args.query_frequency} > "
            f"{args.resampled_action_len}. Use the official default "
            f"--query-frequency {DEFAULT_QUERY_FREQUENCY} --resampled-action-len "
            f"{DEFAULT_RESAMPLED_ACTION_LEN}, or pass --resampled-action-len 0 "
            "to use legacy action-stride mode."
        )
    if args.rollout_blend_steps < 0:
        raise ValueError("--rollout-blend-steps must be non-negative")
    if args.rollout_blend_steps > 0 and args.resampled_action_len <= 0:
        raise ValueError("--rollout-blend-steps requires resample mode")
    if (
        args.rollout_blend_steps > 0
        and args.action_start_index + args.query_frequency >= args.resampled_action_len
    ):
        raise ValueError(
            "--rollout-blend-steps needs unexecuted tail actions from the previous "
            "resampled chunk. Use a resample length larger than "
            "--action-start-index + --query-frequency, e.g. "
            "--query-frequency 30 --resampled-action-len 45 "
            "--rollout-blend-steps 5."
        )
    if args.video_fps is None:
        args.video_fps = args.frequency
    if args.video_every_n_steps <= 0:
        raise ValueError("--video-every-n-steps must be positive")
    if args.video_queue_size <= 0:
        raise ValueError("--video-queue-size must be positive")

    workspace_min = as_float_array(args.workspace_min, 3, "--workspace-min")
    workspace_max = as_float_array(args.workspace_max, 3, "--workspace-max")
    bridge = SO100FrameBridge(args.conversion_metadata)
    kinematics = SO100Kinematics(
        urdf_path=args.urdf,
        target_frame_name=args.target_frame_name,
        joint_names=ARM_JOINT_NAMES,
        position_weight=args.ik_position_weight,
        orientation_weight=args.ik_orientation_weight,
    )
    policy = SO100HPTPolicy(
        checkpoint=args.checkpoint,
        device=args.device,
        precision=args.precision,
        action_horizon=args.action_horizon,
        bgr_to_rgb=args.bgr_to_rgb,
    )
    safety = SO100Safety(
        workspace_min=workspace_min,
        workspace_max=workspace_max,
        max_ee_delta_m=args.max_ee_delta_m,
        max_rot_delta_deg=args.max_rot_delta_deg,
        max_joint_delta_deg=args.max_joint_delta_deg,
        max_gripper_delta=args.max_gripper_delta,
        gripper_min=args.gripper_min,
        gripper_max=args.gripper_max,
        max_ik_pos_error_m=args.max_ik_pos_error_m,
    )
    source = make_source(args)
    action_queue: deque[np.ndarray] = deque()
    last_prediction: np.ndarray | None = None
    last_action_chunk: np.ndarray | None = None
    initial_observed_q: np.ndarray | None = None
    initial_observed_gripper: float | None = None
    last_observed_q: np.ndarray | None = None
    last_observed_gripper: float | None = None
    last_sent_q: np.ndarray | None = None
    last_sent_gripper: float | None = None
    last_target_camera_ypr: np.ndarray | None = None
    last_step: int | None = None
    stuck_counter = 0
    log_path = None if args.no_log else build_log_path(args)
    video_path = build_video_path(args)

    print(f"[so100] checkpoint: {args.checkpoint}")
    print(f"[so100] dry_run: {dry_run}  enable_motors: {args.enable_motors}")
    print(
        f"[so100] query_frequency: {args.query_frequency} at {args.frequency} Hz "
        f"resampled_action_len={args.resampled_action_len} "
        f"action_stride={args.action_stride} "
        f"mode={'resample' if args.resampled_action_len > 0 else 'stride'} "
        f"blend_steps={args.rollout_blend_steps} "
        f"blend_dims={args.rollout_blend_dims}"
    )
    print(f"[so100] log_jsonl: {log_path}")
    print(f"[so100] video_path: {video_path}")
    if dry_run:
        print("[so100] motors are disabled; add --enable-motors to send commands")

    source.connect(calibrate=args.calibrate)
    try:
        if args.print_motor_registers and hasattr(source, "read_motor_registers"):
            print("[so100] motor registers:")
            for motor, values in source.read_motor_registers().items():
                print(f"[so100]   {motor}: {values}")

        arm_joint_limits_deg = getattr(source, "arm_joint_limits_deg", None)
        if arm_joint_limits_deg is not None:
            safety.set_joint_limits(
                np.asarray(arm_joint_limits_deg, dtype=np.float64),
                args.joint_limit_margin_deg,
            )
            print(
                "[so100] joint limits enabled "
                f"margin={args.joint_limit_margin_deg:.2f}deg "
                f"limits={np.round(np.asarray(arm_joint_limits_deg), 2).tolist()}"
            )

        if args.warmup_iters > 0:
            warm_obs = source.observe()
            _, warm_ee_camera_ypr = current_camera_pose_from_observation(
                warm_obs,
                kinematics,
                bridge,
            )
            start = time.perf_counter()
            for _ in range(args.warmup_iters):
                _ = policy.predict(warm_obs["image"], warm_ee_camera_ypr)
            print(
                f"[so100] warmed policy with {args.warmup_iters} iterations "
                f"in {(time.perf_counter() - start) * 1000.0:.1f} ms"
            )

        with JsonlLogger(log_path) as logger, AsyncVideoRecorder(
            video_path,
            fps=args.video_fps,
            codec=args.video_codec,
            queue_size=args.video_queue_size,
            every_n_steps=args.video_every_n_steps,
            input_color=args.video_input_color,
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
                    video_path=video_path,
                    policy=policy,
                    arm_joint_limits_deg=arm_joint_limits_deg,
                )
            )
            for step in loop:
                loop_start = time.perf_counter()
                record = step_record_base(step, dry_run)

                obs_t = timed(source.observe)
                obs = obs_t.value
                video_enqueued = video_recorder.enqueue(step, obs["image"])
                q_current = np.asarray(obs["q"], dtype=np.float64)
                current_gripper = float(obs["gripper"])
                previous_observed_q = last_observed_q.copy() if last_observed_q is not None else None
                previous_observed_gripper = last_observed_gripper
                previous_sent_gripper = last_sent_gripper
                current_low_margin, current_high_margin, current_min_margin = safety.joint_limit_margins(
                    q_current
                )
                if initial_observed_q is None:
                    initial_observed_q = q_current.copy()
                    initial_observed_gripper = current_gripper
                    if np.isfinite(current_min_margin):
                        print(
                            "[so100] initial joint limit margins "
                            f"low={np.round(current_low_margin, 2).tolist()} "
                            f"high={np.round(current_high_margin, 2).tolist()} "
                            f"min={current_min_margin:.2f}deg"
                        )
                last_observed_q = q_current.copy()
                last_observed_gripper = current_gripper
                record["obs_ms"] = obs_t.ms
                record["video_enqueued"] = video_enqueued
                record["video_frames_dropped"] = video_recorder.frames_dropped
                record["q_current"] = q_current.tolist()
                record["gripper_current"] = current_gripper
                record["q_current_limit_margin_low_deg"] = current_low_margin.tolist()
                record["q_current_limit_margin_high_deg"] = current_high_margin.tolist()
                record["q_current_min_limit_margin_deg"] = current_min_margin
                if isinstance(obs.get("raw"), dict):
                    record["obs_raw_info"] = {
                        k: v for k, v in obs["raw"].items() if isinstance(v, (int, float, str, bool))
                    }

                current_base_T_ee, current_ee_camera_ypr = current_camera_pose_from_observation(
                    obs,
                    kinematics,
                    bridge,
                )
                record["ee_camera_ypr"] = current_ee_camera_ypr.tolist()

                should_query = step % args.query_frequency == 0 or not action_queue
                if should_query:
                    infer_t = timed(lambda: policy.predict(obs["image"], current_ee_camera_ypr))
                    last_prediction = infer_t.value
                    pred_gripper_chunk = last_prediction[:, 6].astype(np.float64, copy=False)
                    if args.resampled_action_len > 0:
                        action_chunk = resample_camera_ypr_chunk(
                            last_prediction,
                            args.resampled_action_len,
                        )
                        pre_blend_action_chunk = action_chunk.copy()
                        blend_tail_start = args.action_start_index + args.query_frequency
                        action_chunk, blend_info = blend_replanned_chunk(
                            action_chunk,
                            last_action_chunk,
                            tail_start=blend_tail_start,
                            blend_steps=args.rollout_blend_steps,
                            dims=args.rollout_blend_dims,
                        )
                        start = args.action_start_index
                        stop = start + args.query_frequency
                        selected_prediction = action_chunk[start:stop]
                        action_indices = np.arange(start, stop, dtype=int)
                        action_selection_mode = "resample"
                    else:
                        action_chunk = last_prediction
                        pre_blend_action_chunk = action_chunk
                        blend_info = {
                            "applied": False,
                            "steps_applied": 0,
                            "tail_start_index": None,
                        }
                        start = args.action_start_index
                        action_indices = (
                            start + np.arange(args.query_frequency) * args.action_stride
                        )
                        action_indices = action_indices[action_indices < last_prediction.shape[0]]
                        if len(action_indices) == 0:
                            raise ValueError(
                                f"action_start_index {start} with action_stride {args.action_stride} "
                                f"leaves no actions in chunk shape {last_prediction.shape}"
                            )
                        selected_prediction = last_prediction[action_indices]
                        action_selection_mode = "stride"

                    action_chunk_gripper = action_chunk[:, 6].astype(np.float64, copy=False)
                    pre_blend_action_chunk_gripper = pre_blend_action_chunk[:, 6].astype(
                        np.float64,
                        copy=False,
                    )
                    executed_gripper_chunk = selected_prediction[:, 6].astype(np.float64, copy=False)
                    action_queue.clear()
                    action_queue.extend(selected_prediction)
                    last_action_chunk = action_chunk.copy()
                    record["query"] = True
                    record["inference_ms"] = infer_t.ms
                    record["pred_shape"] = list(last_prediction.shape)
                    record["action_selection_mode"] = action_selection_mode
                    record["resampled_action_len"] = int(args.resampled_action_len)
                    record["rollout_blend_steps"] = int(args.rollout_blend_steps)
                    record["rollout_blend_dims"] = args.rollout_blend_dims
                    record["rollout_blend_info"] = blend_info
                    record["rollout_blend_applied"] = bool(blend_info.get("applied"))
                    record["rollout_blend_steps_applied"] = int(
                        blend_info.get("steps_applied", 0)
                    )
                    record["action_chunk_shape"] = list(action_chunk.shape)
                    record["action_indices"] = action_indices.astype(int).tolist()
                    record["action_indices_source"] = (
                        "resampled_chunk" if args.resampled_action_len > 0 else "raw_chunk"
                    )
                    record["action_stride"] = args.action_stride
                    record["queued_actions"] = len(action_queue)
                    record["policy_input_gripper"] = float(current_ee_camera_ypr[6])
                    record["pred_gripper_chunk"] = pred_gripper_chunk.tolist()
                    record["pred_gripper_chunk_stats"] = numeric_sequence_stats(pred_gripper_chunk)
                    record["pre_blend_action_chunk_gripper"] = (
                        pre_blend_action_chunk_gripper.tolist()
                    )
                    record["pre_blend_action_chunk_gripper_stats"] = numeric_sequence_stats(
                        pre_blend_action_chunk_gripper
                    )
                    record["action_chunk_gripper"] = action_chunk_gripper.tolist()
                    record["action_chunk_gripper_stats"] = numeric_sequence_stats(
                        action_chunk_gripper
                    )
                    record["executed_gripper_chunk"] = executed_gripper_chunk.tolist()
                    record["executed_gripper_chunk_stats"] = numeric_sequence_stats(
                        executed_gripper_chunk
                    )
                    record["executed_gripper_delta_from_current_chunk"] = (
                        executed_gripper_chunk - current_gripper
                    ).tolist()
                    record["executed_gripper_delta_from_policy_input_chunk"] = (
                        executed_gripper_chunk - float(current_ee_camera_ypr[6])
                    ).tolist()
                else:
                    record["query"] = False

                if not action_queue:
                    raise RuntimeError("action_queue is empty after query handling")
                target_camera_ypr = np.asarray(action_queue.popleft(), dtype=np.float64)
                last_target_camera_ypr = target_camera_ypr.copy()
                target_base_T_ee = bridge.camera_ypr_to_base_T_ee(target_camera_ypr)
                raw_target_base_xyz = target_base_T_ee[:3, 3].copy()
                current_base_xyz = current_base_T_ee[:3, 3].copy()
                clipped_base_T_ee = safety.clip_target_pose(current_base_T_ee, target_base_T_ee)
                target_gripper = float(target_camera_ypr[6])

                ik_t = timed(lambda: kinematics.ik(q_current, clipped_base_T_ee))
                q_ik = ik_t.value
                record["ik_ms"] = ik_t.ms
                record["target_camera_ypr"] = target_camera_ypr.tolist()
                record["target_base_xyz"] = clipped_base_T_ee[:3, 3].tolist()
                record["target_gripper"] = target_gripper
                record["target_gripper_delta_from_current"] = target_gripper - current_gripper
                record["target_gripper_delta_from_policy_input"] = (
                    target_gripper - float(current_ee_camera_ypr[6])
                )

                if q_ik is None:
                    record["skip_reason"] = "ik_nonfinite"
                    logger.write(record)
                    print(f"[so100] step {step}: IK returned non-finite values; skipping")
                    continue

                solved_base_T_ee = kinematics.fk(q_ik)
                ik_pos_error = safety.ik_position_error(solved_base_T_ee, clipped_base_T_ee)
                record["ik_pos_error_m"] = ik_pos_error
                record["q_ik"] = q_ik.tolist()
                if ik_pos_error > safety.max_ik_pos_error_m:
                    record["ik_warning"] = "ik_pos_error"
                    if not dry_run:
                        record["skip_reason"] = "ik_pos_error"
                        logger.write(record)
                        print(
                            f"[so100] step {step}: IK pos error {ik_pos_error:.4f} m "
                            f"> {safety.max_ik_pos_error_m:.4f} m; skipping"
                        )
                        continue

                q_safe, gripper_safe = safety.clip_joint_target(
                    q_current,
                    q_ik,
                    current_gripper,
                    target_gripper,
                )
                _, gripper_clip_debug = safety.clip_gripper_target(
                    current_gripper,
                    target_gripper,
                )
                target_delta_xyz_m = float(
                    np.linalg.norm(target_camera_ypr[:3] - current_ee_camera_ypr[:3])
                )
                raw_target_base_delta_xyz_m = float(
                    np.linalg.norm(raw_target_base_xyz - current_base_xyz)
                )
                clipped_target_delta_xyz_m = float(
                    np.linalg.norm(clipped_base_T_ee[:3, 3] - current_base_xyz)
                )
                target_pose_clip_delta_xyz_m = float(
                    np.linalg.norm(clipped_base_T_ee[:3, 3] - raw_target_base_xyz)
                )
                max_joint_delta_deg = float(np.max(np.abs(q_safe - q_current)))
                safe_low_margin, safe_high_margin, safe_min_margin = safety.joint_limit_margins(q_safe)
                observed_motion_since_last = (
                    float(np.max(np.abs(q_current - previous_observed_q)))
                    if previous_observed_q is not None
                    else float("nan")
                )
                observed_gripper_delta_since_last = (
                    current_gripper - previous_observed_gripper
                    if previous_observed_gripper is not None
                    else float("nan")
                )
                if (
                    args.stuck_action != "none"
                    and np.isfinite(observed_motion_since_last)
                    and max_joint_delta_deg >= args.stuck_command_threshold_deg
                    and observed_motion_since_last <= args.stuck_motion_threshold_deg
                ):
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                record["q_safe"] = q_safe.tolist()
                record["gripper_safe"] = gripper_safe
                record["gripper_raw_target"] = target_gripper
                record["gripper_range_clipped_target"] = gripper_clip_debug[
                    "range_clipped_target"
                ]
                record["gripper_delta_limited_target"] = gripper_clip_debug["safe_target"]
                record["gripper_clip_debug"] = gripper_clip_debug
                record["gripper_safe_delta_from_current"] = gripper_safe - current_gripper
                record["gripper_safe_delta_from_target"] = gripper_safe - target_gripper
                record["gripper_range_clip_delta"] = gripper_clip_debug["range_clip_delta"]
                record["gripper_delta_clip_delta"] = gripper_clip_debug["delta_clip_delta"]
                record["gripper_total_clip_delta_from_raw"] = gripper_clip_debug[
                    "total_clip_delta_from_raw"
                ]
                record["gripper_range_clipped"] = gripper_clip_debug["range_clipped"]
                record["gripper_delta_limited"] = gripper_clip_debug["delta_limited"]
                record["gripper_command_delta_from_previous_sent"] = (
                    gripper_safe - previous_sent_gripper
                    if previous_sent_gripper is not None
                    else float("nan")
                )
                record["observed_gripper_delta_since_last"] = observed_gripper_delta_since_last
                record["observed_gripper_abs_delta_since_last"] = abs(
                    observed_gripper_delta_since_last
                )
                record["target_delta_xyz_m"] = target_delta_xyz_m
                record["target_camera_delta_xyz_m"] = target_delta_xyz_m
                record["raw_target_base_xyz"] = raw_target_base_xyz.tolist()
                record["raw_target_base_delta_xyz_m"] = raw_target_base_delta_xyz_m
                record["clipped_target_delta_xyz_m"] = clipped_target_delta_xyz_m
                record["clipped_target_base_xyz"] = clipped_base_T_ee[:3, 3].tolist()
                record["clipped_target_base_delta_xyz_m"] = clipped_target_delta_xyz_m
                record["target_pose_clip_delta_xyz_m"] = target_pose_clip_delta_xyz_m
                record["target_pose_position_clipped"] = target_pose_clip_delta_xyz_m > 1e-9
                record["max_joint_delta_deg"] = max_joint_delta_deg
                record["observed_motion_since_last_deg"] = observed_motion_since_last
                record["stuck_counter"] = stuck_counter
                record["q_safe_limit_margin_low_deg"] = safe_low_margin.tolist()
                record["q_safe_limit_margin_high_deg"] = safe_high_margin.tolist()
                record["q_safe_min_limit_margin_deg"] = safe_min_margin

                if dry_run:
                    sent = {f"{name}.pos": float(q_safe[i]) for i, name in enumerate(ARM_JOINT_NAMES)}
                    sent[f"{GRIPPER_NAME}.pos"] = float(gripper_safe)
                    send_ms = 0.0
                else:
                    send_t = timed(lambda: source.send(q_safe, gripper_safe))
                    sent = send_t.value
                    send_ms = send_t.ms
                sent_gripper = sent.get(f"{GRIPPER_NAME}.pos") if isinstance(sent, dict) else None
                if sent_gripper is None:
                    sent_gripper = gripper_safe
                sent_gripper = float(sent_gripper)
                record["send_ms"] = send_ms
                record["sent"] = sent
                record["sent_gripper"] = sent_gripper
                record["sent_gripper_delta_from_current"] = sent_gripper - current_gripper
                record["sent_gripper_delta_from_target"] = sent_gripper - target_gripper
                record["sent_gripper_delta_from_safe"] = sent_gripper - gripper_safe
                record["sent_gripper_delta_from_previous_sent"] = (
                    sent_gripper - previous_sent_gripper
                    if previous_sent_gripper is not None
                    else float("nan")
                )
                record["loop_ms"] = (time.perf_counter() - loop_start) * 1000.0
                last_sent_q = q_safe.copy()
                last_sent_gripper = sent_gripper
                last_step = step
                logger.write(record)

                if step % args.print_every == 0:
                    query_txt = "query" if record["query"] else "reuse"
                    inference_ms = float(record.get("inference_ms", 0.0))
                    print(
                        f"[so100] step={step:05d} {query_txt} "
                        f"loop={record['loop_ms']:.1f}ms obs={record['obs_ms']:.1f}ms "
                        f"inf={inference_ms:.1f}ms ik={record['ik_ms']:.1f}ms "
                        f"err={ik_pos_error:.4f}m "
                        f"dxyz={target_delta_xyz_m * 100:.1f}/{clipped_target_delta_xyz_m * 100:.1f}cm "
                        f"maxdq={max_joint_delta_deg:.1f}deg "
                        f"minlim={safe_min_margin:.1f}deg "
                        f"stuck={stuck_counter}/{args.stuck_window} "
                        f"q_safe={np.round(q_safe, 2).tolist()} g={gripper_safe:.1f}"
                    )
                if stuck_counter >= args.stuck_window:
                    msg = (
                        f"[so100] stuck detected for {stuck_counter} steps: "
                        f"cmd={max_joint_delta_deg:.2f}deg "
                        f"observed_motion={observed_motion_since_last:.2f}deg"
                    )
                    if args.stuck_action == "stop":
                        record["stop_reason"] = "stuck"
                        logger.write(record)
                        print(msg + "; stopping rollout")
                        break
                    if args.stuck_action == "warn":
                        print(msg)

        if last_prediction is not None:
            print(f"[so100] last prediction shape: {last_prediction.shape}")
        if video_path is not None:
            video_summary = video_recorder.summary()
            print(f"[so100] video summary: {video_summary}")
        if args.post_rollout_hold_s > 0:
            print(
                f"[so100] holding final target for {args.post_rollout_hold_s:.2f}s "
                "before disconnect"
            )
            time.sleep(args.post_rollout_hold_s)
            if not dry_run and last_sent_q is not None:
                try:
                    readback = source.observe()
                    q_after = np.asarray(readback["q"], dtype=np.float64)
                    gripper_after = float(readback["gripper"])
                    _, ee_after_camera_ypr = current_camera_pose_from_observation(
                        readback,
                        kinematics,
                        bridge,
                    )
                    hold_moved = (
                        float(np.max(np.abs(q_after - last_observed_q)))
                        if last_observed_q is not None
                        else float("nan")
                    )
                    total_moved = (
                        float(np.max(np.abs(q_after - initial_observed_q)))
                        if initial_observed_q is not None
                        else float("nan")
                    )
                    remaining = float(np.max(np.abs(q_after - last_sent_q)))
                    hold_gripper_moved = (
                        abs(gripper_after - last_observed_gripper)
                        if last_observed_gripper is not None
                        else float("nan")
                    )
                    total_gripper_moved = (
                        abs(gripper_after - initial_observed_gripper)
                        if initial_observed_gripper is not None
                        else float("nan")
                    )
                    gripper_remaining = (
                        abs(gripper_after - last_sent_gripper)
                        if last_sent_gripper is not None
                        else float("nan")
                    )
                    print(
                        "[so100] final readback "
                        f"hold_moved={hold_moved:.2f}deg total_moved={total_moved:.2f}deg "
                        f"max_remaining={remaining:.2f}deg "
                        f"hold_gripper_moved={hold_gripper_moved:.2f} "
                        f"total_gripper_moved={total_gripper_moved:.2f} "
                        f"gripper_remaining={gripper_remaining:.2f} "
                        f"q_after={np.round(q_after, 2).tolist()} g={gripper_after:.1f}"
                    )
                    final_record = step_record_base(
                        (last_step + 1) if last_step is not None else 0,
                        dry_run,
                    )
                    final_record["event"] = "final_readback"
                    final_record["readback_only"] = True
                    final_record["q_current"] = q_after.tolist()
                    final_record["gripper_current"] = gripper_after
                    final_record["ee_camera_ypr"] = ee_after_camera_ypr.tolist()
                    final_record["q_safe"] = last_sent_q.tolist()
                    final_record["gripper_safe"] = last_sent_gripper
                    final_record["sent_gripper"] = last_sent_gripper
                    if last_target_camera_ypr is not None:
                        final_record["target_camera_ypr"] = last_target_camera_ypr.tolist()
                        final_record["target_gripper"] = float(last_target_camera_ypr[6])
                    final_record["final_readback_hold_moved_deg"] = hold_moved
                    final_record["final_readback_total_moved_deg"] = total_moved
                    final_record["final_readback_max_remaining_deg"] = remaining
                    final_record["final_readback_hold_gripper_moved"] = hold_gripper_moved
                    final_record["final_readback_total_gripper_moved"] = total_gripper_moved
                    final_record["final_readback_gripper_remaining"] = gripper_remaining
                    append_jsonl(log_path, final_record)
                except Exception as exc:
                    print(f"[so100] final readback failed: {exc}")
    finally:
        source.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SO100 HPT online receding-horizon rollout runner."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--conversion-metadata", default=DEFAULT_CONVERSION_METADATA)
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--target-frame-name", default="gripper_frame_link")

    parser.add_argument("--port", default=None, help="SO100 follower serial port.")
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument(
        "--calibration-file",
        default=None,
        help=(
            "Exact LeRobot calibration JSON path. If --robot-id is omitted, "
            "the file stem is used as robot id."
        ),
    )
    parser.add_argument("--calibrate", action="store_true", help="Allow LeRobot calibration prompt.")
    parser.add_argument("--camera-type", default="opencv", choices=["opencv", "realsense"])
    parser.add_argument("--camera", default="0", help="OpenCV index/path or RealSense serial/name.")
    parser.add_argument("--camera-key", default="front")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--bgr-to-rgb", action="store_true")
    parser.add_argument(
        "--print-motor-registers",
        action="store_true",
        help="Print Feetech PID/torque registers after LeRobot connect/configure.",
    )

    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--query-frequency", type=int, default=DEFAULT_QUERY_FREQUENCY)
    parser.add_argument(
        "--resampled-action-len",
        type=int,
        default=DEFAULT_RESAMPLED_ACTION_LEN,
        help=(
            "Official EgoVerse-style action chunk resample length. A predicted chunk "
            "is linearly resampled to this length, then the first --query-frequency "
            "steps are executed before replanning. Pass 0 to disable and use legacy "
            "--action-stride indexing."
        ),
    )
    parser.add_argument(
        "--action-start-index",
        type=int,
        default=0,
        help=(
            "First action index to execute after resampling, or in the raw chunk when "
            "--resampled-action-len 0."
        ),
    )
    parser.add_argument(
        "--action-stride",
        type=int,
        default=1,
        help=(
            "Legacy stride through predicted chunk actions, used only when "
            "--resampled-action-len 0. For example, "
            "--query-frequency 50 --action-stride 2 executes indices 0,2,...,98 "
            "at the control frequency, compressing a 100-frame chunk to 50 steps."
        ),
    )
    parser.add_argument(
        "--rollout-blend-steps",
        type=int,
        default=0,
        help=(
            "Blend the first N actions of a newly replanned resampled chunk with "
            "the unexecuted tail of the previous chunk. This requires "
            "--resampled-action-len > --action-start-index + --query-frequency."
        ),
    )
    parser.add_argument(
        "--rollout-blend-dims",
        choices=("pose", "all"),
        default="pose",
        help=(
            "Dimensions to blend when --rollout-blend-steps is active. The default "
            "'pose' blends xyz+ypr only and leaves gripper commands unchanged."
        ),
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help=(
            "Length of the dummy action sequence used for normalization/eval cropping. "
            "Defaults to the checkpoint head action_horizon, e.g. 100 for FM/diffusion "
            "heads and 64 for older MLP heads."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument(
        "--post-rollout-hold-s",
        type=float,
        default=0.0,
        help=(
            "Keep the final commanded target active for this many seconds before "
            "disconnecting. Useful for single-step hardware tests."
        ),
    )

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])

    parser.add_argument("--ik-position-weight", type=float, default=1.0)
    parser.add_argument("--ik-orientation-weight", type=float, default=0.01)
    parser.add_argument("--max-ik-pos-error-m", type=float, default=0.03)
    parser.add_argument("--max-ee-delta-m", type=float, default=0.01)
    parser.add_argument("--max-rot-delta-deg", type=float, default=5.0)
    parser.add_argument("--max-joint-delta-deg", type=float, default=2.0)
    parser.add_argument(
        "--joint-limit-margin-deg",
        type=float,
        default=0.25,
        help=(
            "Soft margin inside LeRobot calibration joint limits. Commands are "
            "clipped before send_action to avoid repeated calibration-limit clamps."
        ),
    )
    parser.add_argument("--max-gripper-delta", type=float, default=3.0)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=100.0)
    parser.add_argument("--workspace-min", nargs=3, type=float, default=None)
    parser.add_argument("--workspace-max", nargs=3, type=float, default=None)
    parser.add_argument("--lerobot-max-relative-target", type=float, default=None)
    parser.add_argument(
        "--stuck-action",
        choices=["none", "warn", "stop"],
        default="warn",
        help="Warn or stop when commands remain non-trivial but observed joints stop moving.",
    )
    parser.add_argument("--stuck-window", type=int, default=10)
    parser.add_argument("--stuck-command-threshold-deg", type=float, default=1.0)
    parser.add_argument("--stuck-motion-threshold-deg", type=float, default=0.05)

    parser.add_argument(
        "--enable-motors",
        action="store_true",
        help="Actually send q_safe to SO100. Without this flag the runner is dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send motor commands. This is the default unless --enable-motors is set.",
    )
    parser.add_argument(
        "--keep-torque-on-disconnect",
        action="store_true",
        help=(
            "Pass disable_torque_on_disconnect=False to LeRobot. Leave unset for "
            "normal safety behavior."
        ),
    )
    parser.add_argument("--verbose-rate", action="store_true")
    parser.add_argument("--log-jsonl", default=None)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Asynchronously write observed camera frames to an mp4 during rollout.",
    )
    parser.add_argument(
        "--video-path",
        default=None,
        help="Output video path. Defaults to logs/so100_hpt/rollout_videos/<timestamp>.mp4.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Output video FPS. Defaults to --frequency when omitted.",
    )
    parser.add_argument(
        "--video-codec",
        default="mp4v",
        help="OpenCV fourcc codec for video writing, e.g. mp4v or avc1.",
    )
    parser.add_argument(
        "--video-queue-size",
        type=int,
        default=180,
        help="Max frames buffered for async video writing before new frames are dropped.",
    )
    parser.add_argument(
        "--video-every-n-steps",
        type=int,
        default=1,
        help="Record one frame every N control steps to reduce enqueue/copy overhead.",
    )
    parser.add_argument(
        "--video-input-color",
        choices=["rgb", "bgr"],
        default="rgb",
        help="Color order of obs image before writing. OpenCV output is converted to BGR.",
    )

    parser.add_argument(
        "--offline-zarr-episode",
        default=None,
        help="Use a converted SO100 .zarr episode as image/EE input and never connect to hardware.",
    )
    parser.add_argument("--offline-frame-index", type=int, default=0)
    parser.add_argument("--offline-current-joints", nargs=5, type=float, default=None)
    parser.add_argument("--offline-gripper", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
