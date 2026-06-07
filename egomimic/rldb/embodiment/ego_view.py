from __future__ import annotations

from typing import Literal

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    DeleteKeys,
    InterpolateLinear,
    InterpolatePose,
    NumpyToTensor,
    PoseCoordinateFrameTransform,
    SmoothPoseXYZInternalRatioAdaptive,
    SmoothQuaternionPoseRotations,
    Transform,
    XYZWXYZ_to_XYZYPR,
)


class EgoViewRightArm(Embodiment):
    """Right-arm human ego-view dataset transformed into SO100-style HPT keys."""

    VIZ_INTRINSICS_KEY = "base"
    ACTION_HORIZON_REAL = 30
    ACTION_CHUNK_LENGTH = 100
    ACTION_STRIDE = 1

    @classmethod
    def get_transform_list(
        cls,
        mode: Literal["cartesian", "cartesian_no_gripper"] = "cartesian",
        chunk_length: int | None = None,
        stride: int | None = None,
        smooth_action_rot: bool = False,
        smooth_action_rot_window: int = 9,
        smooth_action_rot_sigma: float | None = 2.0,
        smooth_action_xyz_internal_ratio: bool = False,
        smooth_action_xyz_window: int = 9,
        smooth_action_xyz_sigma: float | None = 2.0,
        smooth_action_xyz_ratio_window: int = 17,
        smooth_action_xyz_ratio_smooth_window: int = 5,
        smooth_action_xyz_ratio_smooth_sigma: float | None = 2.0,
        smooth_action_xyz_ratio_threshold: float = 0.5,
        smooth_action_xyz_residual_threshold_m: float = 0.003,
        smooth_action_xyz_max_alpha: float = 1.0,
        smooth_action_xyz_internal_margin: int | None = None,
        smooth_action_xyz_min_sign_flips: int = 2,
    ) -> list[Transform]:
        if mode not in {"cartesian", "cartesian_no_gripper"}:
            raise ValueError(
                "Unsupported ego-view right-arm transform mode "
                f"'{mode}'. Expected one of ('cartesian', 'cartesian_no_gripper')."
            )
        return _build_ego_view_right_arm_cartesian_transform_list(
            chunk_length=chunk_length or cls.ACTION_CHUNK_LENGTH,
            stride=stride or cls.ACTION_STRIDE,
            include_gripper=(mode == "cartesian"),
            smooth_action_rot=smooth_action_rot,
            smooth_action_rot_window=smooth_action_rot_window,
            smooth_action_rot_sigma=smooth_action_rot_sigma,
            smooth_action_xyz_internal_ratio=smooth_action_xyz_internal_ratio,
            smooth_action_xyz_window=smooth_action_xyz_window,
            smooth_action_xyz_sigma=smooth_action_xyz_sigma,
            smooth_action_xyz_ratio_window=smooth_action_xyz_ratio_window,
            smooth_action_xyz_ratio_smooth_window=smooth_action_xyz_ratio_smooth_window,
            smooth_action_xyz_ratio_smooth_sigma=smooth_action_xyz_ratio_smooth_sigma,
            smooth_action_xyz_ratio_threshold=smooth_action_xyz_ratio_threshold,
            smooth_action_xyz_residual_threshold_m=smooth_action_xyz_residual_threshold_m,
            smooth_action_xyz_max_alpha=smooth_action_xyz_max_alpha,
            smooth_action_xyz_internal_margin=smooth_action_xyz_internal_margin,
            smooth_action_xyz_min_sign_flips=smooth_action_xyz_min_sign_flips,
        )

    @classmethod
    def _get_keymap(
        cls,
        keymap_mode: Literal["cartesian", "cartesian_no_gripper"] = "cartesian",
    ):
        if keymap_mode not in {"cartesian", "cartesian_no_gripper"}:
            raise ValueError(
                "Unsupported ego-view right-arm keymap mode "
                f"'{keymap_mode}'. Expected one of ('cartesian', 'cartesian_no_gripper')."
            )
        key_map = {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
            "right.action_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "right.obs_ee_pose",
                "horizon": cls.ACTION_HORIZON_REAL,
            },
            "right.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_ee_pose",
            },
            "obs_head_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "obs_head_pose",
            },
        }
        if keymap_mode == "cartesian":
            key_map["right.action_gripper"] = {
                "key_type": "action_keys",
                "zarr_key": "right.gripper",
                "horizon": cls.ACTION_HORIZON_REAL,
            }
            key_map["right.obs_gripper"] = {
                "key_type": "proprio_keys",
                "zarr_key": "right.gripper",
            }
        return key_map


def _build_ego_view_right_arm_cartesian_transform_list(
    *,
    target_world: str = "obs_head_pose",
    action_world: str = "right.action_ee_pose",
    action_gripper: str = "right.action_gripper",
    obs_pose: str = "right.obs_ee_pose",
    obs_gripper: str = "right.obs_gripper",
    action_headframe: str = "right.action_ee_pose_headframe",
    obs_headframe: str = "right.obs_ee_pose_headframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 64,
    stride: int = 1,
    include_gripper: bool = True,
    smooth_action_rot: bool = False,
    smooth_action_rot_window: int = 9,
    smooth_action_rot_sigma: float | None = 2.0,
    smooth_action_xyz_internal_ratio: bool = False,
    smooth_action_xyz_window: int = 9,
    smooth_action_xyz_sigma: float | None = 2.0,
    smooth_action_xyz_ratio_window: int = 17,
    smooth_action_xyz_ratio_smooth_window: int = 5,
    smooth_action_xyz_ratio_smooth_sigma: float | None = 2.0,
    smooth_action_xyz_ratio_threshold: float = 0.5,
    smooth_action_xyz_residual_threshold_m: float = 0.003,
    smooth_action_xyz_max_alpha: float = 1.0,
    smooth_action_xyz_internal_margin: int | None = None,
    smooth_action_xyz_min_sign_flips: int = 2,
) -> list[Transform]:
    """Build future observed right-hand pose chunks in the current head frame."""
    transforms: list[Transform] = [
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=action_world,
            transformed_key_name=action_headframe,
            mode="xyzwxyz",
        ),
    ]
    if smooth_action_rot:
        transforms.append(
            SmoothQuaternionPoseRotations(
                pose_key=action_headframe,
                window=smooth_action_rot_window,
                sigma=smooth_action_rot_sigma,
            )
        )
    if smooth_action_xyz_internal_ratio:
        transforms.append(
            SmoothPoseXYZInternalRatioAdaptive(
                pose_key=action_headframe,
                window=smooth_action_xyz_window,
                sigma=smooth_action_xyz_sigma,
                ratio_window=smooth_action_xyz_ratio_window,
                ratio_smooth_window=smooth_action_xyz_ratio_smooth_window,
                ratio_smooth_sigma=smooth_action_xyz_ratio_smooth_sigma,
                ratio_threshold=smooth_action_xyz_ratio_threshold,
                residual_threshold_m=smooth_action_xyz_residual_threshold_m,
                max_alpha=smooth_action_xyz_max_alpha,
                internal_margin=smooth_action_xyz_internal_margin,
                min_sign_flips=smooth_action_xyz_min_sign_flips,
            )
        )
    transforms.extend(
        [
            PoseCoordinateFrameTransform(
                target_world=target_world,
                pose_world=obs_pose,
                transformed_key_name=obs_headframe,
                mode="xyzwxyz",
            ),
            XYZWXYZ_to_XYZYPR(keys=[action_headframe, obs_headframe]),
            InterpolatePose(
                new_chunk_length=chunk_length,
                action_key=action_headframe,
                output_action_key=action_headframe,
                stride=stride,
                mode="xyzypr",
            ),
        ]
    )
    if include_gripper:
        transforms.extend(
            [
                InterpolateLinear(
                    new_chunk_length=chunk_length,
                    action_key=action_gripper,
                    output_action_key=action_gripper,
                    stride=stride,
                ),
                ConcatKeys(
                    key_list=[action_headframe, action_gripper],
                    new_key_name=actions_key,
                    delete_old_keys=True,
                ),
                ConcatKeys(
                    key_list=[obs_headframe, obs_gripper],
                    new_key_name=obs_key,
                    delete_old_keys=True,
                ),
            ]
        )
    else:
        transforms.extend(
            [
                ConcatKeys(
                    key_list=[action_headframe],
                    new_key_name=actions_key,
                    delete_old_keys=True,
                ),
                ConcatKeys(
                    key_list=[obs_headframe],
                    new_key_name=obs_key,
                    delete_old_keys=True,
                ),
            ]
        )
    transforms.extend(
        [
            DeleteKeys(
                keys_to_delete=[
                    target_world,
                    action_world,
                    obs_pose,
                    action_gripper,
                    obs_gripper,
                ]
            ),
            NumpyToTensor(keys=[actions_key, obs_key]),
        ]
    )
    return transforms
