"""
Embodiment-dependent action chunk transforms for ZarrDataset.

Replicates the prestacking transformations from aria_to_lerobot.py / eva_to_lerobot.py,
applied at load time instead of at data creation time. Raw action frames are loaded
as (action_horizon, action_dim) and interpolated to (chunk_length, action_dim).

Translation (xyz) and gripper dimensions use linear interpolation.
Rotation (euler ypr) dimensions use np.unwrap before interpolation and rewrap after,
matching the behaviour of egomimicUtils.interpolate_arr_euler.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

try:
    from projectaria_tools.core.sophus import SE3
except ModuleNotFoundError:  # pragma: no cover - exercised only without Aria deps
    class SE3:
        """Small SE3 fallback that covers the matrix API used by these transforms."""

        def __init__(self, matrix):
            self._matrix = np.asarray(matrix, dtype=np.float64)

        @classmethod
        def from_matrix(cls, matrix):
            return cls(matrix)

        def inverse(self):
            mats = np.asarray(self._matrix, dtype=np.float64)
            single = mats.ndim == 2
            if single:
                mats = mats[None, ...]
            inv = np.empty_like(mats)
            rot = mats[:, :3, :3]
            trans = mats[:, :3, 3]
            rot_inv = np.swapaxes(rot, -1, -2)
            inv[:, :3, :3] = rot_inv
            inv[:, :3, 3] = -np.einsum("bij,bj->bi", rot_inv, trans)
            inv[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])
            return SE3(inv[0] if single else inv)

        def __matmul__(self, other):
            return SE3(np.matmul(self._matrix, other._matrix))

        def to_matrix(self):
            return np.asarray(self._matrix)

from egomimic.utils.pose_utils import (
    _interpolate_euler,
    _interpolate_linear,
    _interpolate_quat_wxyz,
    _interpolate_xyz,
    _matrix_to_xyz,
    _matrix_to_xyzwxyz,
    _matrix_to_xyzypr,
    _xyz_to_matrix,
    _xyzwxyz_to_matrix,
    _xyzypr_to_matrix,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

# ---------------------------------------------------------------------------
# Base Transform
# ---------------------------------------------------------------------------


class Transform:
    """Base Class for all transforms."""

    @abstractmethod
    def transform(self, batch: dict) -> dict:
        """Transform the data."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Interpolation Transforms
# ---------------------------------------------------------------------------


class InterpolatePose(Transform):
    """Interpolate a pose chunk of shape (T, 6) or (T, 7)."""

    def __init__(
        self,
        new_chunk_length: int,
        action_key: str,
        output_action_key: str,
        stride: int = 1,
        mode: Literal["xyzwxyz", "xyzypr"] = "xyzwxyz",
        is_quat: bool | None = None,
    ):
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if is_quat is not None:
            mode = "xyzwxyz" if is_quat else "xyzypr"
        self.new_chunk_length = new_chunk_length
        self.action_key = action_key
        self.output_action_key = output_action_key
        self.stride = int(stride)
        self.mode = mode

    def transform(self, batch: dict) -> dict:
        actions = np.asarray(batch[self.action_key])
        actions = actions[:: self.stride]
        if self.mode == "xyzwxyz":
            if actions.ndim != 2 or actions.shape[-1] != 7:
                raise ValueError(
                    f"InterpolatePose expects (T, 7) when is_quat=True, got "
                    f"{actions.shape} for key '{self.action_key}'"
                )
            batch[self.output_action_key] = _interpolate_quat_wxyz(
                actions, self.new_chunk_length
            )
        elif self.mode == "xyzypr":
            if actions.ndim != 2 or actions.shape[-1] != 6:
                raise ValueError(
                    f"InterpolatePose expects (T, 6), got {actions.shape} for key "
                    f"'{self.action_key}'"
                )
            batch[self.output_action_key] = _interpolate_euler(
                actions, self.new_chunk_length
            )
        else:
            if actions.shape[-1] != 3:
                raise ValueError(
                    f"InterpolatePose expects (T, 3) or (T, K, 3), got {actions.shape} for key "
                    f"'{self.action_key}'"
                )
            batch[self.output_action_key] = _interpolate_xyz(
                actions, self.new_chunk_length
            )
        return batch


class InterpolateLinear(Transform):
    """Interpolate any chunk of shape (T, D) with linear interpolation."""

    def __init__(
        self,
        new_chunk_length: int,
        action_key: str,
        output_action_key: str,
        stride: int = 1,
    ):
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self.new_chunk_length = new_chunk_length
        self.action_key = action_key
        self.output_action_key = output_action_key
        self.stride = int(stride)

    def transform(self, batch: dict) -> dict:
        actions = np.asarray(batch[self.action_key])
        if actions.ndim != 2:
            raise ValueError(
                f"InterpolateLinear expects (T, D), got {actions.shape} for key "
                f"'{self.action_key}'"
            )
        actions = actions[:: self.stride]
        batch[self.output_action_key] = _interpolate_linear(
            actions, self.new_chunk_length
        )
        return batch


class SmoothQuaternionPoseRotations(Transform):
    """Smooth only the rotation part of a pose chunk stored as xyz + quat(wxyz)."""

    def __init__(
        self,
        pose_key: str,
        output_key: str | None = None,
        window: int = 9,
        sigma: float | None = 2.0,
    ):
        if window < 1 or window % 2 != 1:
            raise ValueError(f"window must be a positive odd integer, got {window}")
        self.pose_key = pose_key
        self.output_key = output_key or pose_key
        self.window = int(window)
        self.sigma = sigma

    def transform(self, batch: dict) -> dict:
        poses = np.asarray(batch[self.pose_key])
        if poses.ndim != 2 or poses.shape[-1] != 7:
            raise ValueError(
                f"SmoothQuaternionPoseRotations expects (T, 7), got {poses.shape} "
                f"for key '{self.pose_key}'"
            )
        if len(poses) <= 1 or self.window == 1:
            batch[self.output_key] = poses.copy()
            return batch

        out = poses.astype(np.float64, copy=True)
        quat_wxyz = out[:, 3:7]
        norms = np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
        if np.any(norms <= 0.0):
            raise ValueError(f"Encountered zero-norm quaternion in '{self.pose_key}'")

        rotations = R.from_quat((quat_wxyz / norms)[:, [1, 2, 3, 0]])
        half = self.window // 2
        base_weights = self._weights()
        smoothed = []
        for idx in range(len(rotations)):
            lo = max(0, idx - half)
            hi = min(len(rotations), idx + half + 1)
            weight_lo = half - (idx - lo)
            weight_hi = weight_lo + (hi - lo)
            smoothed.append(
                rotations[lo:hi].mean(weights=base_weights[weight_lo:weight_hi])
            )

        quat_xyzw = R.concatenate(smoothed).as_quat()
        out[:, 3:7] = quat_xyzw[:, [3, 0, 1, 2]]
        batch[self.output_key] = out.astype(poses.dtype, copy=False)
        return batch

    def _weights(self) -> np.ndarray:
        if self.sigma is None:
            return np.ones(self.window, dtype=np.float64) / float(self.window)
        half = self.window // 2
        offsets = np.arange(-half, half + 1, dtype=np.float64)
        weights = np.exp(-0.5 * (offsets / max(float(self.sigma), 1e-12)) ** 2)
        return weights / np.sum(weights)


class SmoothPoseXYZInternalRatioAdaptive(Transform):
    """Adaptively smooth XYZ only on internal high-frequency pose-chunk frames.

    This transform is intended for human action chunks after they have been
    expressed in the current head frame and before YPR conversion/interpolation.
    It uses an HF-ratio gate plus an absolute residual gate so nearly-stationary
    tiny denominators do not smooth the whole chunk.
    """

    _AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}

    def __init__(
        self,
        pose_key: str,
        output_key: str | None = None,
        axes: str = "xyz",
        window: int = 9,
        sigma: float | None = 2.0,
        ratio_window: int = 17,
        ratio_smooth_window: int = 5,
        ratio_smooth_sigma: float | None = 2.0,
        ratio_threshold: float = 0.5,
        residual_threshold_m: float = 0.003,
        max_alpha: float = 1.0,
        internal_margin: int | None = None,
        exclude_tail_padding: bool = True,
        tail_pad_atol_m: float = 1e-6,
        oscillation_window: int = 9,
        min_sign_flips: int = 2,
        oscillation_deadband_m: float = 5e-4,
    ):
        for name, value in (
            ("window", window),
            ("ratio_window", ratio_window),
            ("ratio_smooth_window", ratio_smooth_window),
            ("oscillation_window", oscillation_window),
        ):
            if value < 1 or value % 2 != 1:
                raise ValueError(f"{name} must be a positive odd integer, got {value}")
        if ratio_threshold < 0.0:
            raise ValueError(
                f"ratio_threshold must be non-negative, got {ratio_threshold}"
            )
        if residual_threshold_m < 0.0:
            raise ValueError(
                f"residual_threshold_m must be non-negative, got {residual_threshold_m}"
            )
        if not (0.0 <= max_alpha <= 1.0):
            raise ValueError(f"max_alpha must be in [0, 1], got {max_alpha}")

        axis_indices = []
        for axis in axes.lower():
            if axis not in self._AXIS_TO_INDEX:
                raise ValueError(
                    f"Unsupported XYZ smoothing axis '{axis}' in axes={axes!r}"
                )
            axis_indices.append(self._AXIS_TO_INDEX[axis])
        if not axis_indices:
            raise ValueError("axes must contain at least one of x, y, z")

        self.pose_key = pose_key
        self.output_key = output_key or pose_key
        self.axis_indices = tuple(dict.fromkeys(axis_indices))
        self.window = int(window)
        self.sigma = sigma
        self.ratio_window = int(ratio_window)
        self.ratio_smooth_window = int(ratio_smooth_window)
        self.ratio_smooth_sigma = ratio_smooth_sigma
        self.ratio_threshold = float(ratio_threshold)
        self.residual_threshold_m = float(residual_threshold_m)
        self.max_alpha = float(max_alpha)
        self.internal_margin = (
            int(internal_margin)
            if internal_margin is not None
            else max(self.window // 2, self.ratio_window // 2)
        )
        self.exclude_tail_padding = bool(exclude_tail_padding)
        self.tail_pad_atol_m = float(tail_pad_atol_m)
        self.oscillation_window = int(oscillation_window)
        self.min_sign_flips = int(min_sign_flips)
        self.oscillation_deadband_m = float(oscillation_deadband_m)

    def transform(self, batch: dict) -> dict:
        poses = np.asarray(batch[self.pose_key])
        if poses.ndim != 2 or poses.shape[-1] < 3:
            raise ValueError(
                f"SmoothPoseXYZInternalRatioAdaptive expects (T, D>=3), got "
                f"{poses.shape} for key '{self.pose_key}'"
            )
        if len(poses) <= 2 or self.window == 1 or self.max_alpha <= 0.0:
            batch[self.output_key] = poses.copy()
            return batch

        out = poses.astype(np.float64, copy=True)
        xyz = out[:, :3]
        smooth_xyz = self._smooth_array(xyz, self.window, self.sigma)
        residual = xyz - smooth_xyz
        residual_norm = np.linalg.norm(residual[:, self.axis_indices], axis=1)
        ratio = self._local_hf_ratio(
            xyz[:, self.axis_indices],
            smooth_xyz[:, self.axis_indices],
            self.ratio_window,
        )
        ratio_smooth = self._smooth_ratio(ratio)

        valid = self._internal_valid_mask(xyz)
        mask = (
            valid
            & (ratio_smooth >= self.ratio_threshold)
            & (residual_norm >= self.residual_threshold_m)
        )
        if self.min_sign_flips > 0:
            flips = self._local_sign_flips(
                residual[:, self.axis_indices],
                self.oscillation_window,
                self.oscillation_deadband_m,
            )
            mask &= flips >= self.min_sign_flips

        if np.any(mask):
            ratio_score = (ratio_smooth - self.ratio_threshold) / max(
                self.ratio_threshold, 1e-12
            )
            residual_score = (residual_norm - self.residual_threshold_m) / max(
                self.residual_threshold_m, 1e-12
            )
            alpha = np.clip(np.minimum(ratio_score, residual_score), 0.0, 1.0)
            alpha = (alpha * self.max_alpha)[:, None]
            blended = (1.0 - alpha) * xyz + alpha * smooth_xyz
            for axis_idx in self.axis_indices:
                out[mask, axis_idx] = blended[mask, axis_idx]

        batch[self.output_key] = out.astype(poses.dtype, copy=False)
        return batch

    @staticmethod
    def _weights(window: int, sigma: float | None) -> np.ndarray:
        if sigma is None:
            return np.ones(window, dtype=np.float64) / float(window)
        half = window // 2
        offsets = np.arange(-half, half + 1, dtype=np.float64)
        weights = np.exp(-0.5 * (offsets / max(float(sigma), 1e-12)) ** 2)
        return weights / np.sum(weights)

    @classmethod
    def _smooth_array(
        cls,
        values: np.ndarray,
        window: int,
        sigma: float | None,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if len(values) <= 1 or window <= 1:
            return values.copy()
        half = window // 2
        base_weights = cls._weights(window, sigma)
        out = np.empty_like(values, dtype=np.float64)
        for idx in range(len(values)):
            lo = max(0, idx - half)
            hi = min(len(values), idx + half + 1)
            weight_lo = half - (idx - lo)
            weight_hi = weight_lo + (hi - lo)
            weights = base_weights[weight_lo:weight_hi]
            weights = weights / np.sum(weights)
            out[idx] = np.sum(values[lo:hi] * weights[:, None], axis=0)
        return out

    @classmethod
    def _smooth_scalar(
        cls,
        values: np.ndarray,
        window: int,
        sigma: float | None,
    ) -> np.ndarray:
        return cls._smooth_array(values[:, None], window, sigma)[:, 0]

    @staticmethod
    def _median_filter_1d(values: np.ndarray, window: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if len(values) <= 1 or window <= 1:
            return values.copy()
        half = window // 2
        out = np.empty_like(values, dtype=np.float64)
        for idx in range(len(values)):
            lo = max(0, idx - half)
            hi = min(len(values), idx + half + 1)
            out[idx] = np.median(values[lo:hi])
        return out

    def _smooth_ratio(self, ratio: np.ndarray) -> np.ndarray:
        smoothed = np.asarray(ratio, dtype=np.float64)
        if self.ratio_smooth_window > 1:
            smoothed = self._median_filter_1d(smoothed, self.ratio_smooth_window)
            smoothed = self._smooth_scalar(
                smoothed,
                self.ratio_smooth_window,
                self.ratio_smooth_sigma,
            )
        return smoothed

    @staticmethod
    def _local_hf_ratio(
        xyz: np.ndarray,
        smooth_xyz: np.ndarray,
        window: int,
    ) -> np.ndarray:
        residual = xyz - smooth_xyz
        half = window // 2
        ratio = np.zeros(len(xyz), dtype=np.float64)
        for idx in range(len(xyz)):
            lo = max(0, idx - half)
            hi = min(len(xyz), idx + half + 1)
            if hi - lo < 3:
                ratio[idx] = 0.0
                continue
            residual_energy = np.mean(np.sum(residual[lo:hi] ** 2, axis=1))
            centered = xyz[lo:hi] - np.mean(xyz[lo:hi], axis=0, keepdims=True)
            centered_energy = np.mean(np.sum(centered**2, axis=1))
            ratio[idx] = residual_energy / max(centered_energy, 1e-12)
        return ratio

    def _internal_valid_mask(self, xyz: np.ndarray) -> np.ndarray:
        count = len(xyz)
        valid_hi = count
        if self.exclude_tail_padding and count >= 2:
            step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
            idx = len(step) - 1
            while idx >= 0 and step[idx] <= self.tail_pad_atol_m:
                idx -= 1
            if idx < len(step) - 1:
                valid_hi = max(1, idx + 2)

        lo = self.internal_margin
        hi = max(lo, valid_hi - self.internal_margin)
        frame_ids = np.arange(count)
        return (frame_ids >= lo) & (frame_ids < hi)

    @staticmethod
    def _local_sign_flips(
        values: np.ndarray,
        window: int,
        deadband: float,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        half = window // 2
        flips = np.zeros(len(values), dtype=np.int64)
        for idx in range(len(values)):
            lo = max(0, idx - half)
            hi = min(len(values), idx + half + 1)
            if hi - lo < 4:
                continue
            local_max = 0
            diffs = np.diff(values[lo:hi], axis=0)
            for axis_idx in range(diffs.shape[1]):
                axis_diff = diffs[:, axis_idx]
                signs = np.sign(axis_diff)
                signs[np.abs(axis_diff) <= deadband] = 0.0
                signs = signs[signs != 0.0]
                if len(signs) >= 2:
                    local_max = max(local_max, int(np.sum(signs[1:] != signs[:-1])))
            flips[idx] = local_max
        return flips


# ---------------------------------------------------------------------------
# Coordinate Transforms
# ---------------------------------------------------------------------------


class ActionChunkCoordinateFrameTransform(Transform):
    def __init__(
        self,
        target_world: str,
        chunk_world: str,
        transformed_key_name: str,
        extra_batch_key: dict = None,
        mode: Literal["xyz", "xyzwxyz", "xyzypr"] = "xyzwxyz",
        inverse: bool = True,
        is_quat: bool | None = None,
    ):
        """
        args:
            target_world:
            chunk_world:
            transformed_key_name:
            is_quat: if True, inputs are xyz + quat(wxyz); otherwise xyz + ypr.
        """
        self.target_world = target_world
        self.chunk_world = chunk_world
        self.transformed_key_name = transformed_key_name
        self.extra_batch_key = extra_batch_key
        if is_quat is not None:
            mode = "xyzwxyz" if is_quat else "xyzypr"
        self.mode = mode
        self.inverse = inverse

    def transform(self, batch):
        """
        args:
            batch:
                if is_quat=False, inputs are xyz + ypr.
                if is_quat=True, inputs are xyz + quat(wxyz).
                Input shape validation is delegated to the selected to-matrix helper.
                transformed_key_name: str, name of the new key to store the transformed chunk world in

        returns
            batch with new key containing transformed chunk world in target frame:
                if is_quat=False: (T, 6) xyz + ypr
                if is_quat=True: (T, 7) xyz + quat(wxyz)
        """
        # flatten to (T, D)
        # target world is head pose, chunk world is keypoints
        batch.update(self.extra_batch_key or {})
        target_world = np.asarray(batch[self.target_world])
        chunk_world = np.asarray(batch[self.chunk_world])
        chunk_world_shape = None

        if chunk_world.ndim > 2:
            chunk_world_shape = chunk_world.shape
            chunk_world = chunk_world.reshape(-1, chunk_world_shape[-1])

        to_matrix_fn = None
        if self.mode == "xyzwxyz":
            to_matrix_fn = _xyzwxyz_to_matrix
        elif self.mode == "xyzypr":
            to_matrix_fn = _xyzypr_to_matrix
        elif self.mode == "xyz":
            to_matrix_fn = _xyz_to_matrix
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        target_world_to_matrix_fn = (
            _xyzwxyz_to_matrix if target_world.shape[-1] == 7 else _xyzypr_to_matrix
        )
        # Convert to SE3 for transformation
        target_se3 = SE3.from_matrix(
            target_world_to_matrix_fn(target_world[None, :])[0]
        )  # (4, 4)
        chunk_se3 = SE3.from_matrix(to_matrix_fn(chunk_world))  # (T, 4, 4)

        # Compute relative transform and apply to chunk
        if self.inverse:
            chunk_in_target_frame = target_se3.inverse() @ chunk_se3
        else:
            chunk_in_target_frame = target_se3 @ chunk_se3
        chunk_mats = chunk_in_target_frame.to_matrix()
        if chunk_mats.ndim == 2:
            chunk_mats = chunk_mats[None, ...]

        if self.mode == "xyzwxyz":
            chunk_in_target_frame = _matrix_to_xyzwxyz(chunk_mats)
        elif self.mode == "xyzypr":
            chunk_in_target_frame = _matrix_to_xyzypr(chunk_mats)
        elif self.mode == "xyz":
            chunk_in_target_frame = _matrix_to_xyz(chunk_mats)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if chunk_world_shape is not None:
            chunk_in_target_frame = chunk_in_target_frame.reshape(*chunk_world_shape)

        # Store transformed chunk back in batch
        batch[self.transformed_key_name] = chunk_in_target_frame

        return batch


class QuaternionPoseToYPR(Transform):
    """Convert a single pose from xyz + quat(x,y,z,w) to xyz + ypr."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.shape != (7,):
            raise ValueError(
                f"QuaternionPoseToYPR expects shape (7,), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:3]
        xyzw = wxyz_to_xyzw(pose[3:7])
        ypr = R.from_quat(xyzw).as_euler("ZYX", degrees=False)
        batch[self.output_key] = np.concatenate([xyz, ypr], axis=0)
        return batch


class YPRToQuaternionPose(Transform):
    """Convert a single pose from xyz + ypr to xyz + quat(x,y,z,w)."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.shape != (6,):
            raise ValueError(
                f"YPRToQuaternionPose expects shape (6,), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:3]
        quat = R.from_euler("ZYX", pose[3:6], degrees=False).as_quat()  # (x,y,z,w)
        quat = xyzw_to_wxyz(quat)
        batch[self.output_key] = np.concatenate([xyz, quat], axis=0)
        return batch


class BatchQuaternionPoseToYPR(Transform):
    """Convert a batch of poses from xyz + quat(x,y,z,w) to xyz + ypr."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.ndim != 2 or pose.shape[-1] != 7:
            raise ValueError(
                f"BatchQuaternionPoseToYPR expects shape (N, 7), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:, :3]
        xyzw = wxyz_to_xyzw(pose[:, 3:7])
        ypr = R.from_quat(xyzw).as_euler("ZYX", degrees=False)  # (N, 3)
        batch[self.output_key] = np.concatenate([xyz, ypr], axis=1)
        return batch


class RotVecPoseToYPR(Transform):
    """Convert a single pose from xyz + rotvec to xyz + yaw/pitch/roll."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.shape != (6,):
            raise ValueError(
                f"RotVecPoseToYPR expects shape (6,), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:3]
        ypr = R.from_rotvec(pose[3:6]).as_euler("ZYX", degrees=False)
        batch[self.output_key] = np.concatenate([xyz, ypr], axis=0)
        return batch


class BatchRotVecPoseToYPR(Transform):
    """Convert a chunk of poses from xyz + rotvec to xyz + yaw/pitch/roll."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.ndim != 2 or pose.shape[-1] != 6:
            raise ValueError(
                f"BatchRotVecPoseToYPR expects shape (N, 6), got {pose.shape} "
                f"for key '{self.pose_key}'"
            )
        xyz = pose[:, :3]
        ypr = R.from_rotvec(pose[:, 3:6]).as_euler("ZYX", degrees=False)
        batch[self.output_key] = np.concatenate([xyz, ypr], axis=1)
        return batch


class BatchYPRToQuaternionPose(Transform):
    """Convert a batch of poses from xyz + ypr to xyz + quat(x,y,z,w)."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.ndim != 2 or pose.shape[-1] != 6:
            raise ValueError(
                f"BatchYPRToQuaternionPose expects shape (N, 6), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:, :3]
        quat = R.from_euler("ZYX", pose[:, 3:6], degrees=False).as_quat()  # (N, 4)
        quat = xyzw_to_wxyz(quat)
        batch[self.output_key] = np.concatenate([xyz, quat], axis=1)
        return batch


class PoseCoordinateFrameTransform(Transform):
    """Transform a single pose into a target frame pose."""

    def __init__(
        self,
        target_world: str,
        pose_world: str,
        transformed_key_name: str,
        mode: Literal["xyzwxyz", "xyzypr", "xyz"] = "xyzwxyz",
    ):
        self.target_world = target_world
        self.pose_world = pose_world
        self.transformed_key_name = transformed_key_name
        self.mode = mode
        self._chunk_transform = ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=pose_world,
            transformed_key_name=transformed_key_name,
            mode=mode,
        )

    def transform(self, batch: dict) -> dict:
        pose_world = np.asarray(batch[self.pose_world])
        transformed = self._chunk_transform.transform(
            {
                self.target_world: batch[self.target_world],
                self.pose_world: pose_world[None, :],
            }
        )
        batch[self.transformed_key_name] = np.asarray(
            transformed[self.transformed_key_name]
        )[0]
        return batch


class DeleteKeys(Transform):
    def __init__(self, keys_to_delete):
        self.keys_to_delete = keys_to_delete

    def transform(self, batch):
        for key in self.keys_to_delete:
            batch.pop(key, None)
        return batch


class XYZWXYZ_to_XYZYPR(Transform):
    """Convert listed keys from xyz+quat(wxyz) to xyz+ypr in-place."""

    def __init__(self, keys: list[str]):
        self.keys = list(keys)

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            value = np.asarray(batch[key])
            if value.ndim == 1 and value.shape[0] == 7:
                batch[key] = _matrix_to_xyzypr(_xyzwxyz_to_matrix(value[None, :]))[0]
            elif value.ndim == 2 and value.shape[1] == 7:
                batch[key] = _matrix_to_xyzypr(_xyzwxyz_to_matrix(value))
            else:
                raise ValueError(
                    f"XYZWXYZ_to_XYZYPR expects key '{key}' to have shape (7,) "
                    f"or (T, 7), got {value.shape}"
                )
        return batch


class CartesianWithGripperCoordinateTransform(Transform):
    def __init__(
        self,
        left_target_world: str,
        right_target_world: str,
        chunk_world: str,
        transformed_key_name: str,
        extra_batch_key: dict = None,
    ):
        """
        args:
            left_target_world: string key for left target world pose in batch (6D: xyz + ypr)
            right_target_world: string key for right target world pose in batch (6D: xyz + ypr)
            chunk_world: string key for chunk world pose in batch (14D: xyz + ypr + gripper * 2 arms)
            transformed_key_name: string key to store transformed chunk world in batch (14D)
        """
        self.left_target_world = left_target_world
        self.right_target_world = right_target_world
        self.chunk_world = chunk_world
        self.transformed_key_name = transformed_key_name
        self.extra_batch_key = extra_batch_key

    def transform(self, batch):
        """
        args:
            batch:
                left_target_world: numpy(6): xyz + ypr
                right_target_world: numpy(6): xyz + ypr
                chunk_world: numpy(T, 14): [left xyz+ypr+gripper, right xyz+ypr+gripper]
                transformed_key_name: str, name of the new key to store the transformed chunk world in

        returns
            batch with new key containing transformed chunk world in target frame: (T, 14)
        """
        batch.update(self.extra_batch_key or {})
        left_target_world = batch[self.left_target_world]
        right_target_world = batch[self.right_target_world]
        chunk_world = batch[self.chunk_world]

        if left_target_world.shape != (6,):
            raise ValueError(
                f"Expected left_target_world shape (6,), got {left_target_world.shape}"
            )
        if right_target_world.shape != (6,):
            raise ValueError(
                f"Expected right_target_world shape (6,), got {right_target_world.shape}"
            )
        if chunk_world.ndim != 2 or chunk_world.shape[1] != 14:
            raise ValueError(
                f"Expected chunk_world shape (T, 14), got {chunk_world.shape}"
            )

        # Chunk layout: [left xyz+ypr+gripper, right xyz+ypr+gripper]
        left_pose_world = chunk_world[:, :6]
        right_pose_world = chunk_world[:, 7:13]

        left_target_se3 = SE3.from_matrix(
            _xyzypr_to_matrix(left_target_world[None, :])[0]
        )
        right_target_se3 = SE3.from_matrix(
            _xyzypr_to_matrix(right_target_world[None, :])[0]
        )
        left_target_inv = left_target_se3.inverse()
        right_target_inv = right_target_se3.inverse()

        left_pose_in_target = _matrix_to_xyzypr(
            (
                left_target_inv @ SE3.from_matrix(_xyzypr_to_matrix(left_pose_world))
            ).to_matrix()
        )
        right_pose_in_target = _matrix_to_xyzypr(
            (
                right_target_inv @ SE3.from_matrix(_xyzypr_to_matrix(right_pose_world))
            ).to_matrix()
        )

        chunk_in_target_frame = np.empty_like(chunk_world)
        chunk_in_target_frame[:, :6] = left_pose_in_target
        chunk_in_target_frame[:, 6] = chunk_world[:, 6]  # left gripper unchanged
        chunk_in_target_frame[:, 7:13] = right_pose_in_target
        chunk_in_target_frame[:, 13] = chunk_world[:, 13]  # right gripper unchanged

        batch[self.transformed_key_name] = chunk_in_target_frame
        return batch


# ---------------------------------------------------------------------------
# Shape Transforms
# ---------------------------------------------------------------------------
class SplitKeys(Transform):
    def __init__(self, input_key: str, output_key_list: list[(str, int)]):
        self.input_key = input_key
        self.output_key_list = list(output_key_list)

    def transform(self, batch: dict) -> dict:
        prev_end = 0
        for key, size in self.output_key_list:
            batch[key] = batch[self.input_key][..., prev_end : prev_end + size]
            prev_end += size
        return batch


class ConcatKeys(Transform):
    def __init__(self, key_list, new_key_name, delete_old_keys=False):
        self.key_list = list(key_list)
        self.new_key_name = new_key_name
        self.delete_old_keys = delete_old_keys

    def transform(self, batch):
        arrays = [np.asarray(batch[k]) for k in self.key_list]
        try:
            batch[self.new_key_name] = np.concatenate(arrays, axis=-1)
        except ValueError as e:
            shapes = {k: np.asarray(batch[k]).shape for k in self.key_list}
            raise ValueError(
                f"ConcatKeys failed for keys {self.key_list} with shapes {shapes}"
            ) from e

        if self.delete_old_keys:
            for k in self.key_list:
                batch.pop(k, None)

        return batch


class Reshape(Transform):
    def __init__(self, input_key: str, output_key: str, shape: tuple):
        self.input_key = input_key
        self.output_key = output_key
        self.shape = shape

    def transform(self, batch: dict) -> dict:
        batch[self.output_key] = batch[self.input_key].reshape(*self.shape)
        return batch


# ---------------------------------------------------------------------------
# Type Transforms
# ---------------------------------------------------------------------------


class NumpyToTensor(Transform):
    def __init__(self, keys: list[str]):
        self.keys = keys

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            if isinstance(batch[key], np.ndarray):
                batch[key] = torch.from_numpy(batch[key])
            elif isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].clone()
            else:
                raise ValueError(
                    f"NumpyToTensor expects key '{key}' to be a numpy array or torch tensor, got {type(batch[key])}"
                )
        return batch


def build_so100_singlearm_transform_list(
    *,
    obs_raw_key: str = "obs_ee_pose_cam_rotvec",
    action_raw_key: str = "cmd_ee_pose_cam_rotvec",
    force_proxy_key: str | None = None,
    obs_pose_key: str = "so100.obs_pose_rotvec",
    obs_gripper_key: str = "so100.obs_gripper",
    action_pose_key: str = "so100.action_pose_rotvec",
    action_gripper_key: str = "so100.action_gripper",
    obs_pose_ypr_key: str = "so100.obs_pose_ypr",
    action_pose_ypr_key: str = "so100.action_pose_ypr",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 64,
    stride: int = 1,
) -> list[Transform]:
    """Build the SO100 fixed-camera single-arm transform.

    Raw arrays are already expressed in the fixed camera frame as
    ``[x, y, z, wx, wy, wz, gripper]``. The transform only converts rotvec to
    yaw/pitch/roll and turns the command stream into a future action chunk.
    """

    tensor_keys = [actions_key, obs_key]
    if force_proxy_key is not None:
        tensor_keys.append(force_proxy_key)

    return [
        SplitKeys(
            input_key=obs_raw_key,
            output_key_list=[(obs_pose_key, 6), (obs_gripper_key, 1)],
        ),
        SplitKeys(
            input_key=action_raw_key,
            output_key_list=[(action_pose_key, 6), (action_gripper_key, 1)],
        ),
        RotVecPoseToYPR(pose_key=obs_pose_key, output_key=obs_pose_ypr_key),
        BatchRotVecPoseToYPR(pose_key=action_pose_key, output_key=action_pose_ypr_key),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=action_pose_ypr_key,
            output_action_key=action_pose_ypr_key,
            stride=stride,
            mode="xyzypr",
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=action_gripper_key,
            output_action_key=action_gripper_key,
            stride=stride,
        ),
        ConcatKeys(
            key_list=[action_pose_ypr_key, action_gripper_key],
            new_key_name=actions_key,
            delete_old_keys=True,
        ),
        ConcatKeys(
            key_list=[obs_pose_ypr_key, obs_gripper_key],
            new_key_name=obs_key,
            delete_old_keys=True,
        ),
        DeleteKeys(
            keys_to_delete=[
                obs_raw_key,
                action_raw_key,
                obs_pose_key,
                action_pose_key,
            ]
        ),
        NumpyToTensor(keys=tensor_keys),
    ]


def build_so100_singlearm_joint_transform_list(
    *,
    obs_raw_key: str = "obs_joint_pos",
    action_raw_key: str = "cmd_joint_pos",
    actions_key: str = "actions_joint",
    obs_key: str = "observations.state.joint_pos",
    chunk_length: int = 64,
    stride: int = 1,
) -> list[Transform]:
    """Build the SO100 joint-space single-arm transform.

    Raw arrays are ``[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
    wrist_roll, gripper]``. Joint-space chunks use linear interpolation for all
    dimensions because there is no pose rotation representation to unwrap.
    """

    return [
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=action_raw_key,
            output_action_key=actions_key,
            stride=stride,
        ),
        ConcatKeys(
            key_list=[obs_raw_key],
            new_key_name=obs_key,
            delete_old_keys=True,
        ),
        DeleteKeys(keys_to_delete=[action_raw_key]),
        NumpyToTensor(keys=[actions_key, obs_key]),
    ]
