from pathlib import Path

import cv2
import numpy as np
import torch
import zarr

from egomimic.rldb.embodiment.ego_view import EgoViewRightArm
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset


def _encode_jpeg(frame_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    assert ok
    return encoded.tobytes()


def _write_ego_view_right_arm_zarr(root: Path) -> Path:
    episode_path = root / "ego_view_right_arm_episode_000000.zarr"
    store = zarr.open_group(str(episode_path), mode="w", zarr_format=2)

    right_pose = np.array(
        [
            [0.00, 0.00, 0.30, 1.0, 0.0, 0.0, 0.0],
            [0.01, 0.02, 0.31, 1.0, 0.0, 0.0, 0.0],
            [0.02, 0.04, 0.32, 1.0, 0.0, 0.0, 0.0],
            [0.03, 0.06, 0.33, 1.0, 0.0, 0.0, 0.0],
            [0.04, 0.08, 0.34, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    head_pose = np.tile(
        np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        (5, 1),
    )
    gripper = np.linspace(0.2, 1.0, 5, dtype=np.float32).reshape(-1, 1)
    store.create_dataset(
        "right.obs_ee_pose",
        shape=right_pose.shape,
        chunks=(5, 7),
        dtype=right_pose.dtype,
    )
    store["right.obs_ee_pose"][:] = right_pose
    store.create_dataset(
        "obs_head_pose",
        shape=head_pose.shape,
        chunks=(5, 7),
        dtype=head_pose.dtype,
    )
    store["obs_head_pose"][:] = head_pose
    store.create_dataset(
        "right.gripper",
        shape=gripper.shape,
        chunks=(5, 1),
        dtype=gripper.dtype,
    )
    store["right.gripper"][:] = gripper

    encoded_values = []
    for idx in range(5):
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame[..., 0] = idx * 20
        frame[..., 1] = 40
        encoded_values.append(_encode_jpeg(frame))
    encoded = np.asarray(encoded_values, dtype=f"S{max(len(item) for item in encoded_values)}")
    store.create_dataset(
        "images.front_1",
        shape=(5,),
        chunks=(1,),
        dtype=encoded.dtype,
    )
    store["images.front_1"][:] = encoded

    store.attrs.update(
        {
            "embodiment": "ego_view_right_arm",
            "robot_name": "ego_view_right_arm",
            "source_episode_mode": "single_arm_right",
            "task_name": "apricot_human_in_domain",
            "total_frames": 5,
            "fps": 30,
            "features": {
                "right.obs_ee_pose": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
                },
                "obs_head_pose": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
                },
                "right.gripper": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": ["gripper"],
                },
                "images.front_1": {
                    "dtype": "jpeg",
                    "shape": [12, 16, 3],
                    "names": ["height", "width", "channel"],
                },
            },
        }
    )
    return episode_path


def test_ego_view_right_arm_zarr_loads_and_emits_canonical_chunk(tmp_path: Path) -> None:
    _write_ego_view_right_arm_zarr(tmp_path)
    resolver = LocalEpisodeResolver(
        folder_path=tmp_path,
        key_map=EgoViewRightArm.get_keymap(mode="cartesian"),
        transform_list=EgoViewRightArm.get_transform_list(
            mode="cartesian",
            chunk_length=64,
        ),
    )
    dataset = MultiDataset._from_resolver(
        resolver=resolver,
        filters=DatasetFilter(
            filter_lambdas=[
                "lambda row: row.get('embodiment') == 'ego_view_right_arm'",
                "lambda row: row.get('source_episode_mode') == 'single_arm_right'",
            ]
        ),
        mode="total",
        valid_ratio=0.0,
    )

    sample = dataset[0]

    assert tuple(sample["observations.images.front_img_1"].shape) == (3, 12, 16)
    assert tuple(sample["observations.state.ee_pose"].shape) == (7,)
    assert tuple(sample["actions_cartesian"].shape) == (64, 7)
    assert int(sample["embodiment"]) == get_embodiment_id("ego_view_right_arm")
    assert torch.isfinite(sample["observations.state.ee_pose"]).all()
    assert torch.isfinite(sample["actions_cartesian"]).all()

    np.testing.assert_allclose(
        sample["observations.state.ee_pose"].numpy(),
        np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 0.2], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        sample["actions_cartesian"][0].numpy(),
        np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 0.2], dtype=np.float32),
        atol=1e-6,
    )
    assert sample["actions_cartesian"][-1, 1] > sample["actions_cartesian"][0, 1]


def test_ego_view_right_arm_zarr_no_gripper_mode_emits_6d_pose(
    tmp_path: Path,
) -> None:
    _write_ego_view_right_arm_zarr(tmp_path)
    resolver = LocalEpisodeResolver(
        folder_path=tmp_path,
        key_map=EgoViewRightArm.get_keymap(mode="cartesian_no_gripper"),
        transform_list=EgoViewRightArm.get_transform_list(
            mode="cartesian_no_gripper",
            chunk_length=64,
        ),
    )
    dataset = MultiDataset._from_resolver(
        resolver=resolver,
        filters=DatasetFilter(
            filter_lambdas=[
                "lambda row: row.get('embodiment') == 'ego_view_right_arm'",
                "lambda row: row.get('source_episode_mode') == 'single_arm_right'",
            ]
        ),
        mode="total",
        valid_ratio=0.0,
    )

    sample = dataset[0]

    assert "right.action_gripper" not in EgoViewRightArm.get_keymap(
        mode="cartesian_no_gripper"
    )
    assert "right.obs_gripper" not in EgoViewRightArm.get_keymap(
        mode="cartesian_no_gripper"
    )
    assert tuple(sample["observations.state.ee_pose"].shape) == (6,)
    assert tuple(sample["actions_cartesian"].shape) == (64, 6)
    assert torch.isfinite(sample["observations.state.ee_pose"]).all()
    assert torch.isfinite(sample["actions_cartesian"]).all()

    np.testing.assert_allclose(
        sample["observations.state.ee_pose"].numpy(),
        np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        sample["actions_cartesian"][0].numpy(),
        np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
