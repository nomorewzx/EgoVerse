#!/usr/bin/env python3
"""Analyze SO100 HPT rollout JSONL logs and export trend charts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a SO100 HPT rollout JSONL log and export trend charts."
    )
    parser.add_argument("jsonl", type=Path, help="Path to rollout .jsonl")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <jsonl parent>/<jsonl stem>_analysis.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if "step" in record:
                records.append(record)
    if not records:
        raise ValueError(f"No step records found in {path}")
    records.sort(key=lambda item: int(item["step"]))
    return records


def array_field(record: dict[str, Any], key: str, length: int) -> np.ndarray:
    value = record.get(key)
    if value is None:
        return np.full(length, np.nan, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (length,):
        return np.full(length, np.nan, dtype=np.float64)
    return arr


def norm_or_nan(values: np.ndarray) -> float:
    if not np.isfinite(values).all():
        return math.nan
    return float(np.linalg.norm(values))


def cosine_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return math.nan
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0.0 or b_norm == 0.0:
        return math.nan
    return float(np.dot(a, b) / (a_norm * b_norm))


def build_step_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        q_current = array_field(record, "q_current", 5)
        q_safe = array_field(record, "q_safe", 5)
        ee_current = array_field(record, "ee_camera_ypr", 7)
        ee_target = array_field(record, "target_camera_ypr", 7)

        q_delta = q_safe - q_current
        ee_target_delta = ee_target[:3] - ee_current[:3]
        next_q = array_field(records[idx + 1], "q_current", 5) if idx + 1 < len(records) else None
        next_ee = array_field(records[idx + 1], "ee_camera_ypr", 7) if idx + 1 < len(records) else None

        row: dict[str, Any] = {
            "step": int(record["step"]),
            "dry_run": bool(record.get("dry_run", False)),
            "query": bool(record.get("query", False)),
            "loop_ms": record.get("loop_ms", math.nan),
            "obs_ms": record.get("obs_ms", math.nan),
            "inference_ms": record.get("inference_ms", math.nan),
            "ik_ms": record.get("ik_ms", math.nan),
            "send_ms": record.get("send_ms", math.nan),
            "ik_pos_error_cm": float(record.get("ik_pos_error_m", math.nan)) * 100.0,
            "target_delta_xyz_cm": norm_or_nan(ee_target_delta) * 100.0,
            "clipped_target_delta_xyz_cm": float(
                record.get("clipped_target_delta_xyz_m", math.nan)
            )
            * 100.0,
            "command_max_abs_joint_delta_deg": float(np.nanmax(np.abs(q_delta))),
            "gripper_current": record.get("gripper_current", math.nan),
            "gripper_safe": record.get("gripper_safe", math.nan),
        }

        for joint_idx, name in enumerate(ARM_JOINT_NAMES):
            row[f"q_current/{name}"] = q_current[joint_idx]
            row[f"q_safe/{name}"] = q_safe[joint_idx]
            row[f"q_delta/{name}"] = q_delta[joint_idx]
            if next_q is not None:
                row[f"next_tracking_error/{name}"] = next_q[joint_idx] - q_safe[joint_idx]

        for axis_idx, axis in enumerate(["x", "y", "z"]):
            row[f"ee_current_{axis}_m"] = ee_current[axis_idx]
            row[f"ee_target_{axis}_m"] = ee_target[axis_idx]
            row[f"ee_target_delta_{axis}_cm"] = ee_target_delta[axis_idx] * 100.0

        if next_q is not None:
            next_q_delta = next_q - q_current
            row["next_actual_max_joint_move_deg"] = float(np.nanmax(np.abs(next_q_delta)))
            row["next_joint_tracking_error_max_deg"] = float(np.nanmax(np.abs(next_q - q_safe)))
            row["next_joint_direction_agreement"] = cosine_or_nan(q_delta, next_q_delta)
        if next_ee is not None:
            next_ee_delta = next_ee[:3] - ee_current[:3]
            row["next_ee_move_cm"] = norm_or_nan(next_ee_delta) * 100.0
            row["next_ee_progress_cosine"] = cosine_or_nan(ee_target_delta, next_ee_delta)
            row["next_ee_target_distance_change_cm"] = (
                norm_or_nan(next_ee[:3] - ee_target[:3]) - norm_or_nan(ee_current[:3] - ee_target[:3])
            ) * 100.0

        rows.append(row)
    return pd.DataFrame(rows)


def finite_summary(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(values.count()),
        "min": float(values.min()),
        "p50": float(values.quantile(0.50)),
        "p90": float(values.quantile(0.90)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def write_report(df: pd.DataFrame, out_path: Path, source: Path) -> None:
    metrics = [
        "loop_ms",
        "inference_ms",
        "ik_pos_error_cm",
        "target_delta_xyz_cm",
        "clipped_target_delta_xyz_cm",
        "command_max_abs_joint_delta_deg",
        "next_actual_max_joint_move_deg",
        "next_joint_tracking_error_max_deg",
        "next_joint_direction_agreement",
        "next_ee_move_cm",
        "next_ee_progress_cosine",
        "next_ee_target_distance_change_cm",
    ]
    lines = [
        "# SO100 Rollout Analysis",
        "",
        f"- source: `{source}`",
        f"- steps: {len(df)}",
        f"- dry_run: {bool(df['dry_run'].iloc[0]) if 'dry_run' in df else 'unknown'}",
        "",
        "## Summary",
        "",
        "| metric | count | min | p50 | p90 | max | mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        if metric not in df:
            continue
        summary = finite_summary(df[metric])
        if summary["count"] == 0:
            continue
        lines.append(
            "| "
            + metric
            + " | "
            + " | ".join(
                [
                    str(summary["count"]),
                    f"{summary['min']:.4g}",
                    f"{summary['p50']:.4g}",
                    f"{summary['p90']:.4g}",
                    f"{summary['max']:.4g}",
                    f"{summary['mean']:.4g}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Reading The Trend",
            "",
            "- `next_joint_direction_agreement > 0` means the next observed joint motion generally follows the command direction.",
            "- `next_ee_progress_cosine > 0` means the next observed camera-frame EE motion points toward the previous target.",
            "- `next_ee_target_distance_change_cm < 0` means the EE got closer to the previous target.",
            "- A one-step rollout cannot estimate these next-step metrics; run at least 10 steps.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_time_series(
    df: pd.DataFrame,
    cols: list[str],
    title: str,
    ylabel: str,
    out_path: Path,
    dpi: int,
) -> None:
    present = [col for col in cols if col in df and pd.to_numeric(df[col], errors="coerce").notna().any()]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for col in present:
        ax.plot(df["step"], df[col], marker="o", markersize=2.5, linewidth=1.4, label=col)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def export_plots(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    plot_time_series(
        df,
        ["loop_ms", "inference_ms", "obs_ms", "ik_ms", "send_ms"],
        "Timing",
        "ms",
        out_dir / "timing.png",
        dpi,
    )
    plot_time_series(
        df,
        ["target_delta_xyz_cm", "ik_pos_error_cm", "next_ee_move_cm"],
        "EE and IK Magnitudes",
        "cm",
        out_dir / "ee_ik_magnitudes.png",
        dpi,
    )
    plot_time_series(
        df,
        ["next_joint_direction_agreement", "next_ee_progress_cosine"],
        "Next-Step Direction Agreement",
        "cosine",
        out_dir / "direction_agreement.png",
        dpi,
    )
    plot_time_series(
        df,
        ["next_joint_tracking_error_max_deg", "next_actual_max_joint_move_deg"],
        "Joint Tracking",
        "deg",
        out_dir / "joint_tracking.png",
        dpi,
    )
    for axis in ["x", "y", "z"]:
        plot_time_series(
            df,
            [f"ee_current_{axis}_m", f"ee_target_{axis}_m"],
            f"Camera EE {axis.upper()} Current vs Target",
            "m",
            out_dir / f"ee_{axis}_current_vs_target.png",
            dpi,
        )
    for name in ARM_JOINT_NAMES:
        plot_time_series(
            df,
            [f"q_current/{name}", f"q_safe/{name}"],
            f"{name} Current vs Command",
            "deg",
            out_dir / f"joint_{name}.png",
            dpi,
        )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.jsonl.expanduser().with_suffix("").parent / (
        args.jsonl.expanduser().stem + "_analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.jsonl)
    df = build_step_dataframe(records)
    df.to_csv(out_dir / "rollout_step_metrics.csv", index=False)

    summary_rows = []
    for col in df.columns:
        if col in {"dry_run", "query"}:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            row = {"metric": col}
            row.update(finite_summary(values))
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "rollout_summary.csv", index=False)

    export_plots(df, out_dir, args.dpi)
    write_report(df, out_dir / "report.md", args.jsonl)

    print(f"[rollout-analysis] records: {len(df)}")
    print(f"[rollout-analysis] out_dir: {out_dir}")
    for metric in [
        "target_delta_xyz_cm",
        "clipped_target_delta_xyz_cm",
        "ik_pos_error_cm",
        "next_joint_direction_agreement",
        "next_ee_progress_cosine",
        "next_ee_target_distance_change_cm",
    ]:
        if metric in df:
            summary = finite_summary(df[metric])
            if summary["count"]:
                print(
                    f"[rollout-analysis] {metric}: "
                    f"p50={summary['p50']:.4g} p90={summary['p90']:.4g} "
                    f"min={summary['min']:.4g} max={summary['max']:.4g}"
                )


if __name__ == "__main__":
    main()
