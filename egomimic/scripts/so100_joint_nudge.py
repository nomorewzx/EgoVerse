#!/usr/bin/env python3
"""Small SO100 joint command diagnostic using the LeRobot follower API."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROBOT_ID = "so100_hpt_follower"
ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
GRIPPER_NAME = "gripper"
ALL_JOINTS = ARM_JOINT_NAMES + [GRIPPER_NAME]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nudge one SO100 joint by a fixed LeRobot command and report readback. "
            "This is for hardware command-path diagnostics, not policy rollout."
        )
    )
    parser.add_argument("--port", required=True)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument(
        "--calibration-file",
        default=None,
        help="Exact LeRobot calibration JSON path. The file stem is used as robot id if needed.",
    )
    parser.add_argument("--joint", choices=ALL_JOINTS + ["all"], default="all")
    parser.add_argument(
        "--delta",
        type=float,
        default=1.0,
        help="Command delta in degrees for arm joints, or LeRobot gripper units for gripper.",
    )
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument(
        "--restore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send the pre-nudge command back after each test.",
    )
    parser.add_argument("--restore-hold-s", type=float, default=1.0)
    parser.add_argument(
        "--keep-torque-on-disconnect",
        action="store_true",
        help="Pass disable_torque_on_disconnect=False to LeRobot.",
    )
    return parser.parse_args()


def resolve_robot_id_and_calibration(args: argparse.Namespace) -> tuple[str, Path | None]:
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
                f"{calibration_file.stem!r}"
            )
    return robot_id, calibration_dir


def read_state(robot: Any) -> tuple[np.ndarray, float, dict[str, Any]]:
    obs = robot.get_observation()
    q = np.asarray([float(obs[f"{name}.pos"]) for name in ARM_JOINT_NAMES], dtype=np.float64)
    gripper = float(obs[f"{GRIPPER_NAME}.pos"])
    return q, gripper, obs


def action_from_state(q: np.ndarray, gripper: float) -> dict[str, float]:
    action = {f"{name}.pos": float(q[i]) for i, name in enumerate(ARM_JOINT_NAMES)}
    action[f"{GRIPPER_NAME}.pos"] = float(gripper)
    return action


def nudge_joint(robot: Any, joint: str, delta: float, hold_s: float, restore: bool, restore_hold_s: float) -> None:
    q_before, g_before, _ = read_state(robot)
    q_target = q_before.copy()
    g_target = g_before
    if joint == GRIPPER_NAME:
        g_target = g_before + delta
    else:
        q_target[ARM_JOINT_NAMES.index(joint)] += delta

    sent = robot.send_action(action_from_state(q_target, g_target))
    time.sleep(hold_s)
    q_after, g_after, _ = read_state(robot)

    q_move = q_after - q_before
    q_remaining = q_target - q_after
    g_move = g_after - g_before
    g_remaining = g_target - g_after
    if joint == GRIPPER_NAME:
        moved = g_move
        remaining = g_remaining
    else:
        idx = ARM_JOINT_NAMES.index(joint)
        moved = q_move[idx]
        remaining = q_remaining[idx]

    print(
        f"[nudge] joint={joint} target_delta={delta:.3f} "
        f"moved={moved:.3f} remaining={remaining:.3f} "
        f"q_before={np.round(q_before, 3).tolist()} g_before={g_before:.3f} "
        f"q_after={np.round(q_after, 3).tolist()} g_after={g_after:.3f}"
    )
    print(f"[nudge] sent={sent}")

    if restore:
        robot.send_action(action_from_state(q_before, g_before))
        time.sleep(restore_hold_s)
        q_restored, g_restored, _ = read_state(robot)
        q_restore_error = q_restored - q_before
        g_restore_error = g_restored - g_before
        print(
            f"[nudge] restored joint={joint} "
            f"max_q_error={float(np.max(np.abs(q_restore_error))):.3f} "
            f"g_error={g_restore_error:.3f} "
            f"q_restored={np.round(q_restored, 3).tolist()} g_restored={g_restored:.3f}"
        )


def main() -> None:
    args = parse_args()
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    robot_id, calibration_dir = resolve_robot_id_and_calibration(args)
    config_kwargs: dict[str, Any] = {
        "port": args.port,
        "id": robot_id,
        "cameras": {},
        "use_degrees": True,
        "disable_torque_on_disconnect": not args.keep_torque_on_disconnect,
    }
    if calibration_dir is not None:
        config_kwargs["calibration_dir"] = calibration_dir

    robot = SO100Follower(SO100FollowerConfig(**config_kwargs))
    robot.connect(calibrate=False)
    try:
        joints = ALL_JOINTS if args.joint == "all" else [args.joint]
        print(f"[nudge] robot_id={robot.id} calibration={robot.calibration_fpath}")
        for joint in joints:
            nudge_joint(
                robot,
                joint=joint,
                delta=args.delta,
                hold_s=args.hold_s,
                restore=args.restore,
                restore_hold_s=args.restore_hold_s,
            )
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
