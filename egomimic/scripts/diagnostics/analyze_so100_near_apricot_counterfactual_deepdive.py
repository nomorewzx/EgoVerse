from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.scripts.diagnostics.probe_so100_rollout_counterfactual import (
    expected_action_len,
    image_to_tensor,
    load_calibration,
    load_query_records,
    project_points,
    read_video_frame,
)
from egomimic.scripts.diagnostics.probe_so100_val_action_quality import (
    ACTION_KEY,
    EMBODIMENT,
    resample_chunk,
)


DEFAULT_SUMMARY_JSON = (
    "logs/so100_hpt/rollout_counterfactual_probe/"
    "so100_two_rollouts_near_apricot_exhaustive_counterfactual_threshold70_2026-05-25_summary.json"
)
DEFAULT_TAG = "so100_cotrain_vs_so100_near_apricot_deepdive_2026-05-25"
DEFAULT_MODELS = (
    "fm_cotrain_human_so100_80k",
    "fm_so100_60k",
    "fm_so100_80k",
)
PHASES = ("close_0_30", "near_30_50", "approach_50_70")
METRICS = (
    "first10_cos_to_target_mean",
    "exec_cos_to_target_mean",
    "exec_best_progress_px",
    "exec_final_progress_px",
    "exec_min_distance_px",
)
HIGHER_IS_BETTER = {
    "first10_cos_to_target_mean": True,
    "exec_cos_to_target_mean": True,
    "exec_best_progress_px": True,
    "exec_final_progress_px": True,
    "exec_min_distance_px": False,
}
PRETTY = {
    "fm_cotrain_human_so100_80k": "FM Cotrain 80k",
    "fm_so100_60k": "FM SO100 60k",
    "fm_so100_80k": "FM SO100 80k",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deep-dive counterfactual analysis for near-apricot SO100 rollouts: "
            "paired cotrain-vs-SO100 deltas, frame mining, and optional per-horizon "
            "trajectory decomposition."
        )
    )
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", action="append", default=list(DEFAULT_MODELS))
    parser.add_argument("--skip-horizon-inference", action="store_true")
    parser.add_argument("--output-dir", default="logs/so100_hpt/rollout_counterfactual_probe")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def paired_scalar_analysis(
    frame_model_means: pd.DataFrame,
    *,
    output_dir: Path,
    tag: str,
) -> dict[str, Path]:
    cotrain = "fm_cotrain_human_so100_80k"
    baselines = ("fm_so100_60k", "fm_so100_80k")
    rows: list[dict] = []
    mining_rows: list[dict] = []

    wide = frame_model_means.set_index(["rollout", "step", "model"])
    for (rollout, step), frame in frame_model_means.groupby(["rollout", "step"]):
        if cotrain not in set(frame["model"]):
            continue
        c = frame[frame["model"] == cotrain].iloc[0]
        for baseline in baselines:
            if baseline not in set(frame["model"]):
                continue
            b = frame[frame["model"] == baseline].iloc[0]
            base_info = {
                "rollout": rollout,
                "step": int(step),
                "time_s": float(c["time_s"]),
                "phase": c["phase"],
                "comparison": f"{cotrain}_minus_{baseline}",
                "baseline": baseline,
                "target_distance_px": float(c["target_distance_px"]),
                "state_gripper": float(c["state_gripper"]),
                "n_apricot_detections": int(c["n_apricot_detections"]),
            }
            for metric in METRICS:
                delta = float(c[metric]) - float(b[metric])
                better = delta > 0 if HIGHER_IS_BETTER[metric] else delta < 0
                rows.append(
                    {
                        **base_info,
                        "metric": metric,
                        "cotrain_value": float(c[metric]),
                        "baseline_value": float(b[metric]),
                        "delta": delta,
                        "cotrain_better": bool(better),
                    }
                )
            mining_rows.append(
                {
                    **base_info,
                    "cotrain_best_progress_px": float(c["exec_best_progress_px"]),
                    "baseline_best_progress_px": float(b["exec_best_progress_px"]),
                    "delta_best_progress_px": float(c["exec_best_progress_px"])
                    - float(b["exec_best_progress_px"]),
                    "cotrain_final_progress_px": float(c["exec_final_progress_px"]),
                    "baseline_final_progress_px": float(b["exec_final_progress_px"]),
                    "delta_final_progress_px": float(c["exec_final_progress_px"])
                    - float(b["exec_final_progress_px"]),
                    "cotrain_first10_cos": float(c["first10_cos_to_target_mean"]),
                    "baseline_first10_cos": float(b["first10_cos_to_target_mean"]),
                    "delta_first10_cos": float(c["first10_cos_to_target_mean"])
                    - float(b["first10_cos_to_target_mean"]),
                    "cotrain_exec_cos": float(c["exec_cos_to_target_mean"]),
                    "baseline_exec_cos": float(b["exec_cos_to_target_mean"]),
                    "delta_exec_cos": float(c["exec_cos_to_target_mean"])
                    - float(b["exec_cos_to_target_mean"]),
                    "cotrain_pred_close_rate": float(c["pred_close_rate"]),
                    "baseline_pred_close_rate": float(b["pred_close_rate"]),
                    "delta_pred_close_rate": float(c["pred_close_rate"])
                    - float(b["pred_close_rate"]),
                    "cotrain_min_distance_idx": float(c["min_distance_idx"]),
                    "baseline_min_distance_idx": float(b["min_distance_idx"]),
                }
            )

    delta_df = pd.DataFrame(rows)
    mining_df = pd.DataFrame(mining_rows)
    delta_path = output_dir / f"{tag}_paired_scalar_deltas.csv"
    mining_path = output_dir / f"{tag}_frame_mining_all.csv"
    delta_df.to_csv(delta_path, index=False)
    mining_df.to_csv(mining_path, index=False)

    summary_rows: list[dict] = []
    for keys, group in delta_df.groupby(["comparison", "metric"]):
        comparison, metric = keys
        for phase_name, phase_group in [("all", group)] + [
            (phase, group[group["phase"] == phase]) for phase in PHASES
        ]:
            if phase_group.empty:
                continue
            summary_rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "phase": phase_name,
                    "n_frames": int(len(phase_group)),
                    "delta_mean": float(phase_group["delta"].mean()),
                    "delta_median": float(phase_group["delta"].median()),
                    "delta_p25": float(phase_group["delta"].quantile(0.25)),
                    "delta_p75": float(phase_group["delta"].quantile(0.75)),
                    "cotrain_win_rate": float(phase_group["cotrain_better"].mean()),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"{tag}_paired_scalar_delta_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    top_rows: list[dict] = []
    for comparison, group in mining_df.groupby("comparison"):
        for metric, column in (
            ("best_progress", "delta_best_progress_px"),
            ("final_progress", "delta_final_progress_px"),
            ("first10_cos", "delta_first10_cos"),
            ("exec_cos", "delta_exec_cos"),
        ):
            top_win = group.sort_values(column, ascending=False).head(12)
            top_loss = group.sort_values(column, ascending=True).head(12)
            for label, selected in (("cotrain_win", top_win), ("cotrain_loss", top_loss)):
                for rank, (_, row) in enumerate(selected.iterrows(), start=1):
                    out = row.to_dict()
                    out["mined_metric"] = metric
                    out["bucket"] = label
                    out["rank"] = rank
                    top_rows.append(out)
    top_df = pd.DataFrame(top_rows)
    top_path = output_dir / f"{tag}_frame_mining_top_wins_losses.csv"
    top_df.to_csv(top_path, index=False)

    plot_path = output_dir / f"{tag}_paired_scalar_delta_boxplots.png"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    plot_metrics = (
        ("exec_best_progress_px", "Best progress delta (px)"),
        ("exec_final_progress_px", "Final progress delta (px)"),
        ("exec_cos_to_target_mean", "30-step direction cosine delta"),
        ("first10_cos_to_target_mean", "First10 direction cosine delta"),
    )
    phase_order = list(PHASES)
    for ax, (metric, title) in zip(axes.flat, plot_metrics):
        metric_df = delta_df[delta_df["metric"] == metric]
        positions = []
        data = []
        labels = []
        pos = 1
        for comparison in sorted(metric_df["comparison"].unique()):
            for phase in phase_order:
                vals = metric_df[
                    (metric_df["comparison"] == comparison) & (metric_df["phase"] == phase)
                ]["delta"].to_numpy()
                data.append(vals)
                positions.append(pos)
                labels.append(
                    ("60k" if comparison.endswith("fm_so100_60k") else "80k")
                    + "\n"
                    + phase.replace("_", "\n")
                )
                pos += 1
            pos += 0.8
        ax.boxplot(data, positions=positions, showfliers=False)
        ax.axhline(0.0, color="#222", linewidth=0.8)
        ax.set_xticks(positions, labels, fontsize=7)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Cotrain minus SO100-only paired deltas by phase", fontsize=13)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    return {
        "paired_scalar_deltas": delta_path,
        "paired_scalar_delta_summary": summary_path,
        "frame_mining_all": mining_path,
        "frame_mining_top_wins_losses": top_path,
        "paired_scalar_delta_boxplots": plot_path,
    }


def predict_chunk(
    wrapper: ModelWrapper,
    frame_bgr: np.ndarray,
    ee_camera_ypr: np.ndarray,
    expected_t: int,
    *,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    front = image_to_tensor(frame_bgr).to(device)
    ee = torch.as_tensor(np.asarray(ee_camera_ypr, dtype=np.float32), device=device)
    dummy_actions = ee.view(1, 7).repeat(expected_t, 1)
    raw_batch = {
        EMBODIMENT: {
            "observations.images.front_img_1": front.unsqueeze(0),
            "observations.state.ee_pose": ee.unsqueeze(0),
            "actions_cartesian": dummy_actions.unsqueeze(0),
        }
    }
    with torch.inference_mode():
        processed = wrapper.model.process_batch_for_training(raw_batch)
        pred = wrapper.model.forward_eval(processed)[ACTION_KEY]
    return pred.detach().float().cpu().numpy().squeeze(0)


def horizon_rows_for_traj(
    *,
    rollout: str,
    step: int,
    time_s: float,
    phase: str,
    model: str,
    seed: int,
    traj_px: np.ndarray,
    gripper: np.ndarray,
    ee_px: np.ndarray,
    target_px: np.ndarray,
    state_gripper: float,
    n_apricot_detections: int,
) -> list[dict]:
    target_vec = target_px - ee_px
    target_distance = float(np.linalg.norm(target_vec))
    if target_distance < 1e-6:
        target_unit = np.zeros(2, dtype=np.float64)
    else:
        target_unit = target_vec / target_distance
    lateral_unit = np.asarray([-target_unit[1], target_unit[0]], dtype=np.float64)
    rows: list[dict] = []
    for idx, point in enumerate(traj_px):
        disp = point - ee_px
        disp_norm = float(np.linalg.norm(disp))
        target_axis = float(np.dot(disp, target_unit))
        lateral_signed = float(np.dot(disp, lateral_unit))
        distance = float(np.linalg.norm(point - target_px))
        cos = "" if disp_norm < 1e-6 else float(target_axis / disp_norm)
        rows.append(
            {
                "rollout": rollout,
                "step": int(step),
                "time_s": float(time_s),
                "phase": phase,
                "model": model,
                "seed": int(seed),
                "horizon_step": idx + 1,
                "state_gripper": float(state_gripper),
                "target_distance_px": target_distance,
                "n_apricot_detections": int(n_apricot_detections),
                "distance_px": distance,
                "progress_px": target_distance - distance,
                "target_axis_px": target_axis,
                "lateral_signed_px": lateral_signed,
                "lateral_abs_px": abs(lateral_signed),
                "displacement_px": disp_norm,
                "cos_to_target": cos,
                "gripper": float(gripper[idx]),
            }
        )
    return rows


def run_horizon_inference(
    *,
    summary: dict,
    model_names: list[str],
    output_dir: Path,
    tag: str,
    device: torch.device,
) -> dict[str, Path]:
    candidate_csv = Path(summary["candidate_csv"])
    candidates = pd.read_csv(candidate_csv)
    model_meta = {item["model"]: item for item in summary["models"]}
    missing = [name for name in model_names if name not in model_meta]
    if missing:
        raise KeyError(f"Missing models in summary metadata: {missing}")

    camera_matrix, _ = load_calibration(
        Path("/home/zxwang/so100_calib/selected_calibration_drop0007_5x7_29_21.json")
    )
    query_records = {
        rollout: load_query_records(Path(meta["log_jsonl"]))
        for rollout, meta in summary["rollouts"].items()
    }
    videos = {rollout: Path(meta["video"]) for rollout, meta in summary["rollouts"].items()}

    wrappers = {}
    for model_name in model_names:
        checkpoint = Path(model_meta[model_name]["checkpoint"])
        print(f"Loading {model_name}: {checkpoint}", flush=True)
        wrapper = ModelWrapper.load_from_checkpoint(
            str(checkpoint), weights_only=False, map_location="cpu"
        )
        wrapper = wrapper.to(device)
        wrapper.eval()
        wrapper.model.device = device
        wrappers[model_name] = (wrapper, expected_action_len(wrapper))

    seeds = [int(seed) for seed in summary.get("seeds", [0, 1, 2])]
    resampled_action_len = int(summary.get("resampled_action_len", 45))
    execute_steps = int(summary.get("execute_steps", 30))
    seed_rows: list[dict] = []
    total = len(candidates)
    for idx, candidate in candidates.iterrows():
        if idx == 0 or (idx + 1) % 10 == 0 or idx + 1 == total:
            print(
                f"Processing candidate {idx + 1}/{total}: "
                f"{candidate['rollout']} step={int(candidate['step'])}",
                flush=True,
            )
        rollout = str(candidate["rollout"])
        step = int(candidate["step"])
        record = query_records[rollout][step]
        state = np.asarray(record["ee_camera_ypr"], dtype=np.float64)
        ee_px = project_points(state[:3][None, :], camera_matrix)[0]
        target_px = np.asarray([candidate["target_x"], candidate["target_y"]], dtype=np.float64)
        frame_bgr = read_video_frame(videos[rollout], step)
        for model_name, (wrapper, expected_t) in wrappers.items():
            for seed in seeds:
                chunk = predict_chunk(
                    wrapper,
                    frame_bgr,
                    state,
                    expected_t,
                    seed=seed,
                    device=device,
                )
                exec_chunk = resample_chunk(chunk, resampled_action_len)[:execute_steps]
                traj_px = project_points(exec_chunk[:, :3], camera_matrix)
                seed_rows.extend(
                    horizon_rows_for_traj(
                        rollout=rollout,
                        step=step,
                        time_s=float(candidate["time_s"]),
                        phase=str(candidate["phase"]),
                        model=model_name,
                        seed=seed,
                        traj_px=traj_px,
                        gripper=exec_chunk[:, 6],
                        ee_px=ee_px,
                        target_px=target_px,
                        state_gripper=float(candidate["state_gripper"]),
                        n_apricot_detections=int(candidate["n_apricot_detections"]),
                    )
                )

    seed_path = output_dir / f"{tag}_horizon_seed_rows.csv"
    write_csv(seed_path, seed_rows)
    seed_df = pd.DataFrame(seed_rows)
    mean_cols = [
        "distance_px",
        "progress_px",
        "target_axis_px",
        "lateral_signed_px",
        "lateral_abs_px",
        "displacement_px",
        "cos_to_target",
        "gripper",
    ]
    meta_cols = [
        "rollout",
        "step",
        "time_s",
        "phase",
        "model",
        "horizon_step",
        "state_gripper",
        "target_distance_px",
        "n_apricot_detections",
    ]
    mean_df = (
        seed_df.groupby(meta_cols, dropna=False)[mean_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    mean_df.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in mean_df.columns
    ]
    mean_path = output_dir / f"{tag}_horizon_frame_model_means.csv"
    mean_df.to_csv(mean_path, index=False)

    pair_rows: list[dict] = []
    cotrain = "fm_cotrain_human_so100_80k"
    baselines = [name for name in model_names if name != cotrain]
    for keys, group in mean_df.groupby(["rollout", "step", "horizon_step"]):
        rollout, step, horizon_step = keys
        models_here = set(group["model"])
        if cotrain not in models_here:
            continue
        c = group[group["model"] == cotrain].iloc[0]
        for baseline in baselines:
            if baseline not in models_here:
                continue
            b = group[group["model"] == baseline].iloc[0]
            row = {
                "rollout": rollout,
                "step": int(step),
                "time_s": float(c["time_s"]),
                "phase": c["phase"],
                "horizon_step": int(horizon_step),
                "comparison": f"{cotrain}_minus_{baseline}",
                "baseline": baseline,
                "target_distance_px": float(c["target_distance_px"]),
                "state_gripper": float(c["state_gripper"]),
            }
            for metric in mean_cols:
                row[f"cotrain_{metric}"] = float(c[f"{metric}_mean"])
                row[f"baseline_{metric}"] = float(b[f"{metric}_mean"])
                row[f"delta_{metric}"] = float(c[f"{metric}_mean"]) - float(b[f"{metric}_mean"])
            pair_rows.append(row)
    pair_df = pd.DataFrame(pair_rows)
    pair_path = output_dir / f"{tag}_horizon_pairwise_deltas.csv"
    pair_df.to_csv(pair_path, index=False)

    summary_rows: list[dict] = []
    for keys, group in pair_df.groupby(["comparison", "phase", "horizon_step"]):
        comparison, phase, horizon_step = keys
        summary_rows.append(
            {
                "comparison": comparison,
                "phase": phase,
                "horizon_step": int(horizon_step),
                "n_frames": int(len(group)),
                "delta_progress_px_mean": float(group["delta_progress_px"].mean()),
                "delta_progress_px_median": float(group["delta_progress_px"].median()),
                "cotrain_progress_win_rate": float((group["delta_progress_px"] > 0).mean()),
                "delta_target_axis_px_mean": float(group["delta_target_axis_px"].mean()),
                "delta_lateral_abs_px_mean": float(group["delta_lateral_abs_px"].mean()),
                "delta_cos_to_target_mean": float(group["delta_cos_to_target"].mean()),
                "delta_displacement_px_mean": float(group["delta_displacement_px"].mean()),
                "delta_gripper_mean": float(group["delta_gripper"].mean()),
            }
        )
    horizon_summary = pd.DataFrame(summary_rows)
    horizon_summary_path = output_dir / f"{tag}_horizon_delta_summary.csv"
    horizon_summary.to_csv(horizon_summary_path, index=False)
    curve_stats_path, compact_steps_path = write_horizon_compact_tables(
        mean_df, horizon_summary, output_dir=output_dir, tag=tag
    )

    model_curve_path = output_dir / f"{tag}_horizon_model_progress_curves.png"
    delta_curve_path = output_dir / f"{tag}_horizon_delta_decomposition_curves.png"
    plot_model_progress_curves(mean_df, model_curve_path)
    plot_delta_decomposition_curves(horizon_summary, delta_curve_path)

    return {
        "horizon_seed_rows": seed_path,
        "horizon_frame_model_means": mean_path,
        "horizon_pairwise_deltas": pair_path,
        "horizon_delta_summary": horizon_summary_path,
        "horizon_curve_stats": curve_stats_path,
        "horizon_delta_steps_1_5_10_30": compact_steps_path,
        "horizon_model_progress_curves": model_curve_path,
        "horizon_delta_decomposition_curves": delta_curve_path,
    }


def write_horizon_compact_tables(
    mean_df: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    *,
    output_dir: Path,
    tag: str,
) -> tuple[Path, Path]:
    pretty = {
        "fm_cotrain_human_so100_80k": "FM Cotrain 80k",
        "fm_so100_60k": "FM SO100 60k",
        "fm_so100_80k": "FM SO100 80k",
    }
    rows: list[dict] = []
    for (phase, model), group in mean_df.groupby(["phase", "model"]):
        curve = group.groupby("horizon_step").agg(
            {
                "progress_px_mean": "mean",
                "target_axis_px_mean": "mean",
                "lateral_abs_px_mean": "mean",
                "displacement_px_mean": "mean",
                "cos_to_target_mean": "mean",
                "gripper_mean": "mean",
            }
        )
        peak_step = int(curve["progress_px_mean"].idxmax())
        final = curve.loc[30]
        rows.append(
            {
                "phase": phase,
                "model": pretty.get(model, model),
                "mean_curve_peak_progress_px": float(curve.loc[peak_step, "progress_px_mean"]),
                "mean_curve_peak_step": peak_step,
                "final_progress_px": float(final["progress_px_mean"]),
                "final_target_axis_px": float(final["target_axis_px_mean"]),
                "final_lateral_abs_px": float(final["lateral_abs_px_mean"]),
                "final_displacement_px": float(final["displacement_px_mean"]),
                "final_cos_to_target": float(final["cos_to_target_mean"]),
                "final_gripper": float(final["gripper_mean"]),
            }
        )
    curve_stats = pd.DataFrame(rows).sort_values(["phase", "model"])
    curve_stats_path = output_dir / f"{tag}_horizon_curve_stats.csv"
    curve_stats.to_csv(curve_stats_path, index=False)

    compact = horizon_summary[horizon_summary["horizon_step"].isin([1, 5, 10, 30])].copy()
    compact["baseline"] = compact["comparison"].str.split("_minus_").str[-1].map(pretty)
    compact = compact[
        [
            "baseline",
            "phase",
            "horizon_step",
            "n_frames",
            "delta_progress_px_mean",
            "cotrain_progress_win_rate",
            "delta_target_axis_px_mean",
            "delta_lateral_abs_px_mean",
            "delta_cos_to_target_mean",
            "delta_displacement_px_mean",
            "delta_gripper_mean",
        ]
    ]
    compact_path = output_dir / f"{tag}_horizon_delta_steps_1_5_10_30.csv"
    compact.to_csv(compact_path, index=False)
    return curve_stats_path, compact_path


def plot_model_progress_curves(mean_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, constrained_layout=True)
    model_order = ["fm_cotrain_human_so100_80k", "fm_so100_60k", "fm_so100_80k"]
    colors = {
        "fm_cotrain_human_so100_80k": "#8c4b7e",
        "fm_so100_60k": "#2f6f73",
        "fm_so100_80k": "#2458a6",
    }
    for ax, phase in zip(axes, PHASES):
        phase_df = mean_df[mean_df["phase"] == phase]
        for model in model_order:
            group = phase_df[phase_df["model"] == model]
            curve = group.groupby("horizon_step")["progress_px_mean"].mean()
            ax.plot(
                curve.index,
                curve.values,
                label=PRETTY[model],
                linewidth=2.0,
                color=colors[model],
            )
        ax.axhline(0.0, color="#222", linewidth=0.8)
        ax.set_title(phase.replace("_", " "))
        ax.set_xlabel("horizon step")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean progress toward apricot (px)")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Per-horizon progress curves by phase", fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_delta_decomposition_curves(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, constrained_layout=True)
    comparisons = sorted(summary_df["comparison"].unique())
    colors = {"fm_so100_60k": "#2f6f73", "fm_so100_80k": "#2458a6"}
    row_metrics = (
        ("delta_progress_px_mean", "progress delta px"),
        ("delta_target_axis_px_mean", "target-axis delta px"),
        ("delta_lateral_abs_px_mean", "lateral abs delta px"),
    )
    for col, comparison in enumerate(comparisons):
        baseline = comparison.split("_minus_")[-1]
        comp_df = summary_df[summary_df["comparison"] == comparison]
        for row, (metric, title) in enumerate(row_metrics):
            ax = axes[row, col]
            for phase in PHASES:
                phase_df = comp_df[comp_df["phase"] == phase].sort_values("horizon_step")
                ax.plot(phase_df["horizon_step"], phase_df[metric], label=phase, linewidth=1.8)
            ax.axhline(0.0, color="#222", linewidth=0.8)
            ax.set_title(f"Cotrain - {PRETTY.get(baseline, baseline)}: {title}")
            ax.grid(alpha=0.25)
            if row == 2:
                ax.set_xlabel("horizon step")
            if col == 0:
                ax.set_ylabel(title)
    axes[0, -1].legend(loc="best", fontsize=8)
    fig.suptitle("Cotrain-vs-SO100 per-horizon decomposition", fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_report(
    *,
    output_path: Path,
    artifacts: dict[str, Path],
    scalar_summary_path: Path,
    horizon_summary_path: Path | None,
) -> None:
    scalar = pd.read_csv(scalar_summary_path)
    primary = scalar[
        scalar["metric"].isin(
            [
                "exec_best_progress_px",
                "exec_final_progress_px",
                "first10_cos_to_target_mean",
                "exec_cos_to_target_mean",
            ]
        )
    ].copy()
    primary["baseline"] = primary["comparison"].str.split("_minus_").str[-1].map(PRETTY)
    primary["metric"] = primary["metric"].replace(
        {
            "exec_best_progress_px": "best progress",
            "exec_final_progress_px": "final progress",
            "first10_cos_to_target_mean": "first10 direction cosine",
            "exec_cos_to_target_mean": "30-step direction cosine",
        }
    )
    primary = primary[
        [
            "baseline",
            "phase",
            "metric",
            "n_frames",
            "delta_mean",
            "delta_median",
            "cotrain_win_rate",
        ]
    ].sort_values(["baseline", "metric", "phase"])

    def table(df: pd.DataFrame) -> str:
        d = df.copy()
        for col in d.columns:
            if pd.api.types.is_float_dtype(d[col]):
                d[col] = d[col].map(lambda x: f"{x:.3f}")
        return d.to_html(index=False, classes="data", border=0, escape=False)

    horizon_takeaway = ""
    if horizon_summary_path is not None and horizon_summary_path.exists():
        horizon = pd.read_csv(horizon_summary_path)
        final = horizon[horizon["horizon_step"] == horizon["horizon_step"].max()]
        lines = []
        for _, row in final.iterrows():
            baseline = PRETTY.get(row["comparison"].split("_minus_")[-1], row["comparison"])
            lines.append(
                f"{row['phase']} vs {baseline}: final progress delta "
                f"{row['delta_progress_px_mean']:.2f}px, target-axis delta "
                f"{row['delta_target_axis_px_mean']:.2f}px, lateral-abs delta "
                f"{row['delta_lateral_abs_px_mean']:.2f}px."
            )
        horizon_takeaway = "<ul>" + "".join(f"<li>{line}</li>" for line in lines) + "</ul>"
    curve_stats_html = ""
    if "horizon_curve_stats" in artifacts and artifacts["horizon_curve_stats"].exists():
        curve_stats_html = "<h2>Curve Stats</h2>" + table(pd.read_csv(artifacts["horizon_curve_stats"]))
    compact_steps_html = ""
    if (
        "horizon_delta_steps_1_5_10_30" in artifacts
        and artifacts["horizon_delta_steps_1_5_10_30"].exists()
    ):
        compact_steps_html = (
            "<h2>Step 1/5/10/30 Delta Decomposition</h2>"
            "<p class=\"note\">Positive target-axis delta means Cotrain moves farther along "
            "the gripper-to-apricot direction. Positive lateral-abs delta means more sideways "
            "deviation.</p>"
            + table(pd.read_csv(artifacts["horizon_delta_steps_1_5_10_30"]))
        )

    css = """
body { margin:0; background:#f7f8fa; color:#17202a; font:14px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1180px; margin:0 auto; padding:28px 24px 56px; }
h1 { margin:0 0 8px; font-size:28px; line-height:1.2; letter-spacing:0; }
h2 { margin:28px 0 10px; font-size:20px; letter-spacing:0; }
p { margin:8px 0; }
.note { color:#596575; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; margin:14px 0 20px; }
.callout { background:#fff; border:1px solid #d8dee8; border-left:4px solid #2f6f73; padding:12px 14px; }
.callout.warn { border-left-color:#9a5b00; }
img { max-width:100%; height:auto; border:1px solid #d8dee8; background:#fff; margin:8px 0 18px; }
table.data { border-collapse:collapse; width:100%; background:#fff; margin:8px 0 18px; font-size:13px; }
table.data th, table.data td { border:1px solid #d8dee8; padding:6px 8px; text-align:right; vertical-align:top; }
table.data th:first-child, table.data td:first-child, table.data th:nth-child(2), table.data td:nth-child(2), table.data th:nth-child(3), table.data td:nth-child(3) { text-align:left; }
table.data th { background:#eef2f5; font-weight:650; }
code { background:#eef2f5; padding:1px 4px; border-radius:4px; }
"""
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SO100 Cotrain vs SO100-only Counterfactual Deep Dive</title>
  <style>{css}</style>
</head>
<body>
<main>
  <h1>SO100 Cotrain vs SO100-only Counterfactual Deep Dive</h1>
  <p class="note">Same 165 near-apricot query frames. FM seeds 0/1/2 are averaged per frame. Deltas are always Cotrain minus SO100-only.</p>

  <h2>Key Findings</h2>
  <div class="grid">
    <div class="callout"><b>Not uniformly worse at every horizon.</b><br/>In approach 50-70px, Cotrain can have positive early progress/direction deltas.</div>
    <div class="callout warn"><b>Final approach is worse.</b><br/>At horizon step 30, Cotrain final-progress deltas are negative against SO100-only baselines across the distance phases.</div>
    <div class="callout"><b>Likely difference.</b><br/>Cotrain tends to close more strongly, but its target-axis progress is less sustained and close-range lateral offset is larger.</div>
  </div>

  <h2>What Changed</h2>
  <div class="grid">
    <div class="callout"><b>Paired deltas.</b><br/>Every frame is compared against the same image/state/target under Cotrain and SO100-only policies.</div>
    <div class="callout"><b>Frame mining.</b><br/>Top Cotrain wins/losses are exported with distance, phase, direction, progress, and close-rate fields.</div>
    <div class="callout warn"><b>Per-horizon decomposition.</b><br/>Progress is split into target-axis motion, lateral drift, displacement magnitude, direction cosine, and gripper value.</div>
  </div>

  <h2>Scalar Paired Delta Summary</h2>
  <img src="{artifacts['paired_scalar_delta_boxplots']}" alt="paired scalar delta boxplots" />
  {table(primary)}

  <h2>Per-Horizon Curves</h2>
  {('<img src="' + str(artifacts['horizon_model_progress_curves']) + '" alt="horizon model progress curves" />') if 'horizon_model_progress_curves' in artifacts else '<p class="note">Horizon inference was skipped.</p>'}
  {('<img src="' + str(artifacts['horizon_delta_decomposition_curves']) + '" alt="horizon delta decomposition curves" />') if 'horizon_delta_decomposition_curves' in artifacts else ''}
  {horizon_takeaway}
  {curve_stats_html}
  {compact_steps_html}

  <h2>Artifacts</h2>
  <p><code>{artifacts['paired_scalar_deltas']}</code></p>
  <p><code>{artifacts['paired_scalar_delta_summary']}</code></p>
  <p><code>{artifacts['frame_mining_top_wins_losses']}</code></p>
  {''.join(f'<p><code>{path}</code></p>' for key, path in artifacts.items() if key.startswith('horizon_'))}
</main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json)
    summary = json.loads(summary_path.read_text())
    frame_model_means = pd.read_csv(summary["outputs"]["frame_model_means"])

    artifacts = paired_scalar_analysis(frame_model_means, output_dir=output_dir, tag=args.tag)
    horizon_summary_path: Path | None = None
    if not args.skip_horizon_inference:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        artifacts.update(
            run_horizon_inference(
                summary=summary,
                model_names=list(dict.fromkeys(args.model)),
                output_dir=output_dir,
                tag=args.tag,
                device=device,
            )
        )
        horizon_summary_path = artifacts["horizon_delta_summary"]

    report_path = Path(f"{args.tag}.html")
    render_report(
        output_path=report_path,
        artifacts=artifacts,
        scalar_summary_path=artifacts["paired_scalar_delta_summary"],
        horizon_summary_path=horizon_summary_path,
    )
    out_summary = {
        "summary_json": str(summary_path),
        "tag": args.tag,
        "models": list(dict.fromkeys(args.model)),
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "report": str(report_path),
    }
    out_path = output_dir / f"{args.tag}_summary.json"
    out_path.write_text(json.dumps(out_summary, indent=2))
    print(json.dumps(out_summary, indent=2))


if __name__ == "__main__":
    main()
