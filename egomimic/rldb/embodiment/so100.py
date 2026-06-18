from __future__ import annotations

from typing import Literal

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    Transform,
    build_so100_singlearm_joint_transform_list,
    build_so100_singlearm_transform_list,
)


class So100SingleArm(Embodiment):
    """SO100 right-arm dataset with fixed-camera EE state and command actions."""

    VIZ_INTRINSICS_KEY = "base"
    ACTION_HORIZON_REAL = 30
    ACTION_CHUNK_LENGTH = 100
    ACTION_STRIDE = 1

    @classmethod
    def get_transform_list(
        cls,
        mode: Literal[
            "camera_frame_ypr",
            "camera_frame_ypr_force",
            "base_frame_ypr",
            "joint",
        ] = "camera_frame_ypr",
        chunk_length: int | None = None,
        stride: int | None = None,
    ) -> list[Transform]:
        if mode not in {
            "camera_frame_ypr",
            "camera_frame_ypr_force",
            "base_frame_ypr",
            "joint",
        }:
            raise ValueError(
                f"Unsupported SO100 transform mode '{mode}'. "
                "Expected 'camera_frame_ypr', 'camera_frame_ypr_force', "
                "'base_frame_ypr', or 'joint'."
            )
        if mode == "joint":
            return build_so100_singlearm_joint_transform_list(
                chunk_length=chunk_length or cls.ACTION_CHUNK_LENGTH,
                stride=stride or cls.ACTION_STRIDE,
            )

        obs_raw_key = (
            "obs_ee_pose_base_rotvec"
            if mode == "base_frame_ypr"
            else "obs_ee_pose_cam_rotvec"
        )
        action_raw_key = (
            "cmd_ee_pose_base_rotvec"
            if mode == "base_frame_ypr"
            else "cmd_ee_pose_cam_rotvec"
        )
        return build_so100_singlearm_transform_list(
            obs_raw_key=obs_raw_key,
            action_raw_key=action_raw_key,
            force_proxy_key="observations.state.force_proxy"
            if mode == "camera_frame_ypr_force"
            else None,
            chunk_length=chunk_length or cls.ACTION_CHUNK_LENGTH,
            stride=stride or cls.ACTION_STRIDE,
        )

    @classmethod
    def _get_keymap(
        cls,
        keymap_mode: Literal[
            "camera_frame_ypr",
            "camera_frame_ypr_force",
            "base_frame_ypr",
            "joint",
        ] = "camera_frame_ypr",
    ):
        if keymap_mode not in {
            "camera_frame_ypr",
            "camera_frame_ypr_force",
            "base_frame_ypr",
            "joint",
        }:
            raise ValueError(
                f"Unsupported SO100 keymap mode '{keymap_mode}'. "
                "Expected 'camera_frame_ypr', 'camera_frame_ypr_force', "
                "'base_frame_ypr', or 'joint'."
            )
        keymap = {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
        }
        if keymap_mode == "joint":
            keymap.update(
                {
                    "obs_joint_pos": {
                        "key_type": "proprio_keys",
                        "zarr_key": "obs_joint_pos",
                    },
                    "cmd_joint_pos": {
                        "key_type": "action_keys",
                        "zarr_key": "cmd_joint_pos",
                        "horizon": cls.ACTION_HORIZON_REAL,
                    },
                }
            )
            return keymap

        obs_zarr_key = (
            "obs_ee_pose_base_rotvec"
            if keymap_mode == "base_frame_ypr"
            else "obs_ee_pose_cam_rotvec"
        )
        action_zarr_key = (
            "cmd_ee_pose_base_rotvec"
            if keymap_mode == "base_frame_ypr"
            else "cmd_ee_pose_cam_rotvec"
        )
        keymap.update(
            {
                obs_zarr_key: {
                    "key_type": "proprio_keys",
                    "zarr_key": obs_zarr_key,
                },
                action_zarr_key: {
                    "key_type": "action_keys",
                    "zarr_key": action_zarr_key,
                    "horizon": cls.ACTION_HORIZON_REAL,
                },
            }
        )
        if keymap_mode == "camera_frame_ypr_force":
            keymap["observations.state.force_proxy"] = {
                "key_type": "proprio_keys",
                "zarr_key": "observations.state.force_proxy",
            }
        return keymap
