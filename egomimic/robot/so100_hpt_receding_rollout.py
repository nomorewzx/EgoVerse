from __future__ import annotations

import argparse
import json
import math
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
        action_horizon: int,
        bgr_to_rgb: bool,
    ):
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self.device = requested_device
        self.precision = precision
        self.action_horizon = int(action_horizon)
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

        gripper = float(np.clip(target_gripper, self.gripper_min, self.gripper_max))
        g_delta = float(
            np.clip(
                gripper - float(current_gripper),
                -self.max_gripper_delta,
                self.max_gripper_delta,
            )
        )
        return q_safe, float(current_gripper) + g_delta

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
    dry_run = not args.enable_motors
    if args.enable_motors and args.offline_zarr_episode is not None:
        raise ValueError("--enable-motors is incompatible with --offline-zarr-episode")
    if args.query_frequency <= 0:
        raise ValueError("--query-frequency must be positive")
    if args.action_stride <= 0:
        raise ValueError("--action-stride must be positive")
    if args.action_start_index < 0:
        raise ValueError("--action-start-index must be non-negative")

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

    print(f"[so100] checkpoint: {args.checkpoint}")
    print(f"[so100] dry_run: {dry_run}  enable_motors: {args.enable_motors}")
    print(
        f"[so100] query_frequency: {args.query_frequency} at {args.frequency} Hz "
        f"action_stride={args.action_stride}"
    )
    print(f"[so100] log_jsonl: {log_path}")
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

        with JsonlLogger(log_path) as logger, RateLoop(
            frequency=args.frequency,
            max_iterations=args.max_steps,
            verbose=args.verbose_rate,
        ) as loop:
            for step in loop:
                loop_start = time.perf_counter()
                record = step_record_base(step, dry_run)

                obs_t = timed(source.observe)
                obs = obs_t.value
                q_current = np.asarray(obs["q"], dtype=np.float64)
                current_gripper = float(obs["gripper"])
                previous_observed_q = last_observed_q.copy() if last_observed_q is not None else None
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
                    start = args.action_start_index
                    action_indices = start + np.arange(args.query_frequency) * args.action_stride
                    action_indices = action_indices[action_indices < last_prediction.shape[0]]
                    if len(action_indices) == 0:
                        raise ValueError(
                            f"action_start_index {start} with action_stride {args.action_stride} "
                            f"leaves no actions in chunk shape {last_prediction.shape}"
                        )
                    action_queue.clear()
                    action_queue.extend(last_prediction[action_indices])
                    record["query"] = True
                    record["inference_ms"] = infer_t.ms
                    record["pred_shape"] = list(last_prediction.shape)
                    record["action_indices"] = action_indices.astype(int).tolist()
                    record["action_stride"] = args.action_stride
                    record["queued_actions"] = len(action_queue)
                else:
                    record["query"] = False

                if not action_queue:
                    raise RuntimeError("action_queue is empty after query handling")
                target_camera_ypr = np.asarray(action_queue.popleft(), dtype=np.float64)
                last_target_camera_ypr = target_camera_ypr.copy()
                target_base_T_ee = bridge.camera_ypr_to_base_T_ee(target_camera_ypr)
                clipped_base_T_ee = safety.clip_target_pose(current_base_T_ee, target_base_T_ee)
                target_gripper = float(target_camera_ypr[6])

                ik_t = timed(lambda: kinematics.ik(q_current, clipped_base_T_ee))
                q_ik = ik_t.value
                record["ik_ms"] = ik_t.ms
                record["target_camera_ypr"] = target_camera_ypr.tolist()
                record["target_base_xyz"] = clipped_base_T_ee[:3, 3].tolist()
                record["target_gripper"] = target_gripper

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
                target_delta_xyz_m = float(
                    np.linalg.norm(target_camera_ypr[:3] - current_ee_camera_ypr[:3])
                )
                clipped_target_delta_xyz_m = float(
                    np.linalg.norm(clipped_base_T_ee[:3, 3] - current_base_T_ee[:3, 3])
                )
                max_joint_delta_deg = float(np.max(np.abs(q_safe - q_current)))
                safe_low_margin, safe_high_margin, safe_min_margin = safety.joint_limit_margins(q_safe)
                observed_motion_since_last = (
                    float(np.max(np.abs(q_current - previous_observed_q)))
                    if previous_observed_q is not None
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
                record["target_delta_xyz_m"] = target_delta_xyz_m
                record["clipped_target_delta_xyz_m"] = clipped_target_delta_xyz_m
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
                record["send_ms"] = send_ms
                record["sent"] = sent
                record["loop_ms"] = (time.perf_counter() - loop_start) * 1000.0
                last_sent_q = q_safe.copy()
                last_sent_gripper = gripper_safe
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
                    if last_target_camera_ypr is not None:
                        final_record["target_camera_ypr"] = last_target_camera_ypr.tolist()
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
    parser.add_argument("--query-frequency", type=int, default=5)
    parser.add_argument("--action-start-index", type=int, default=0)
    parser.add_argument(
        "--action-stride",
        type=int,
        default=1,
        help=(
            "Stride through predicted chunk actions. For example, "
            "--query-frequency 32 --action-stride 2 executes indices 0,2,...,62 "
            "at the control frequency, compressing a 64-frame chunk to 32 steps."
        ),
    )
    parser.add_argument("--action-horizon", type=int, default=64)
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
