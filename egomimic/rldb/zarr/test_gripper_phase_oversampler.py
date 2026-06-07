from types import SimpleNamespace

import numpy as np

from egomimic.rldb.zarr.zarr_dataset_multi import GripperPhaseOversampler


class _DummyDataset:
    def __init__(self, gripper):
        actions = np.zeros((len(gripper), 7), dtype=np.float32)
        actions[:, 6] = np.asarray(gripper, dtype=np.float32)
        self.episode_path = "dummy.zarr"
        self.episode_reader = SimpleNamespace(_store={"cmd_ee_pose_cam_rotvec": actions})

    def __len__(self):
        return len(self.episode_reader._store["cmd_ee_pose_cam_rotvec"])


def test_gripper_phase_oversampler_builds_50_25_25_epoch():
    dataset = _DummyDataset([10, 12, 20, 34, 39, 40, 41, 41, 20, 10])
    base_index_map = [("ep0", i) for i in range(len(dataset))]
    sampler = GripperPhaseOversampler(
        phases=[
            {"name": "full", "fraction": 0.5},
            {"name": "closing", "fraction": 0.25},
            {"name": "closed_hold", "fraction": 0.25},
        ],
        horizon=4,
        closed_threshold=38.0,
        open_threshold=35.0,
        min_close_delta=5.0,
        hold_min_fraction=0.5,
        seed=0,
    )

    sampled = sampler.build_index_map({"ep0": dataset}, base_index_map)

    assert len(sampled) == 20
    assert sampler.last_summary["phases"]["full"]["sampled"] == 10
    assert sampler.last_summary["phases"]["closing"]["sampled"] == 5
    assert sampler.last_summary["phases"]["closed_hold"]["sampled"] == 5
    assert sampler.last_summary["phases"]["closing"]["available"] > 0
    assert sampler.last_summary["phases"]["closed_hold"]["available"] > 0
