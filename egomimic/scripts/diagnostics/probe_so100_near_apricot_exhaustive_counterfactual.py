from __future__ import annotations

import argparse
import csv
import json
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
    DEFAULT_CALIBRATION,
    expected_action_len,
    gripper_metrics,
    load_calibration,
    load_query_records,
    predict_chunk,
    project_points,
    read_video_frame,
    trajectory_metrics,
    write_csv,
)
from egomimic.scripts.diagnostics.probe_so100_val_action_quality import (
    parse_model_spec,
    resample_chunk,
)


DEFAULT_BASE_SUMMARY = (
    "logs/so100_hpt/rollout_counterfactual_probe/"
    "so100_two_rollouts_near_apricot_exhaustive_counterfactual_threshold70_2026-05-25_summary.json"
)

METRICS = [
    "first10_cos_to_target_mean",
    "exec_cos_to_target_mean",
    "first10_progress_px",
    "exec_best_progress_px",
    "exec_final_progress_px",
    "exec_min_distance_px",
    "pred_gripper_max",
    "pred_close_rate",
    "gripper_at_min_distance",
    "distance_at_first_close_px",
]

HIGHER_IS_BETTER = {
    "first10_cos_to_target_mean": True,
    "exec_cos_to_target_mean": True,
    "first10_progress_px": True,
    "exec_best_progress_px": True,
    "exec_final_progress_px": True,
    "exec_min_distance_px": False,
    "pred_gripper_max": True,
    "pred_close_rate": True,
    "gripper_at_min_distance": True,
    "distance_at_first_close_px": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run arbitrary SO100 checkpoints on the fixed near-apricot candidate frames "
            "from rollout logs, then summarize paired counterfactual metrics."
        )
    )
    parser.add_argument("--base-summary-json", default=DEFAULT_BASE_SUMMARY)
    parser.add_argument("--candidate-csv", default=None)
    parser.add_argument("--model", action="append", required=True, help="NAME=CHECKPOINT")
    parser.add_argument(
        "--pairwise-a",
        action="append",
        default=[],
        help="Model name to compare as model_a. Defaults to all non-baseline models.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="Baseline model name for pairwise comparisons.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--resampled-action-len", type=int, default=None)
    parser.add_argument("--execute-steps", type=int, default=None)
    parser.add_argument("--calibration-json", default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", default="logs/so100_hpt/rollout_counterfactual_probe")
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def extra_metrics(traj_px: np.ndarray, target_px: np.ndarray, gripper: np.ndarray) -> dict:
    distances = np.linalg.norm(traj_px - target_px[None, :], axis=1)
    min_idx = int(np.argmin(distances))
    close_hits = np.flatnonzero(gripper >= 30.0)
    if close_hits.size:
        first_close = int(close_hits[0])
        distance_at_first_close: float | str = float(distances[first_close])
    else:
        distance_at_first_close = ""
    return {
        "min_distance_idx": min_idx,
        "gripper_at_min_distance": float(gripper[min_idx]),
        "distance_at_first_close_px": distance_at_first_close,
        "final_gripper": float(gripper[-1]),
    }


def summarize_frame_means(seed_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        "rollout",
        "step",
        "time_s",
        "phase",
        "model",
        "state_gripper",
        "target_distance_px",
        "ee_px_x",
        "ee_px_y",
        "target_x",
        "target_y",
        "n_apricot_detections",
    ]
    numeric_cols = [
        col
        for col in seed_df.columns
        if col not in set(meta_cols + ["seed", "checkpoint", "diffusion", "expected_action_len"])
        and pd.api.types.is_numeric_dtype(seed_df[col])
    ]
    grouped = seed_df.groupby(meta_cols, dropna=False)
    mean_df = grouped[numeric_cols].agg(["mean", "std"]).reset_index()
    mean_df.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in mean_df.columns
    ]
    mean_renames = {}
    for col in numeric_cols:
        if f"{col}_mean" in mean_df.columns:
            mean_renames[f"{col}_mean"] = col
    mean_df = mean_df.rename(columns=mean_renames)
    n_seeds = grouped["seed"].nunique().reset_index(name="n_seeds")
    mean_df = mean_df.merge(n_seeds, on=meta_cols, how="left")

    rename = {
        "pred_has_close": "pred_close_rate",
    }
    mean_df = mean_df.rename(columns=rename)
    if "distance_at_first_close_px_std" not in mean_df.columns:
        mean_df["distance_at_first_close_px_std"] = np.nan
    if "pred_close_rate_std" in mean_df.columns:
        mean_df = mean_df.drop(columns=["pred_close_rate_std"])
    front = meta_cols[:5] + ["n_seeds"] + meta_cols[5:]
    rest = [col for col in mean_df.columns if col not in front]
    return mean_df[front + rest].sort_values(["rollout", "step", "model"])


def aggregate_summary(frame_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    rollout_groups = [("both", frame_df)]
    rollout_groups.extend((rollout, group) for rollout, group in frame_df.groupby("rollout"))
    for rollout_name, rollout_df in rollout_groups:
        phase_groups = [("all", rollout_df)]
        phase_groups.extend((phase, group) for phase, group in rollout_df.groupby("phase"))
        for phase_name, phase_df in phase_groups:
            for model, group in phase_df.groupby("model"):
                row = {
                    "rollout": rollout_name,
                    "phase": phase_name,
                    "model": model,
                    "n_frames": int(len(group)),
                }
                for metric in METRICS:
                    if metric not in group.columns:
                        continue
                    vals = pd.to_numeric(group[metric], errors="coerce").dropna()
                    if vals.empty:
                        row[f"{metric}_mean"] = np.nan
                        row[f"{metric}_median"] = np.nan
                    else:
                        row[f"{metric}_mean"] = float(vals.mean())
                        row[f"{metric}_median"] = float(vals.median())
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["rollout", "phase", "model"])


def pairwise_summary(
    frame_df: pd.DataFrame,
    *,
    models_a: list[str],
    baselines: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    rollout_groups = [("both", frame_df)]
    rollout_groups.extend((rollout, group) for rollout, group in frame_df.groupby("rollout"))
    for rollout_name, rollout_df in rollout_groups:
        phase_groups = [("all", rollout_df)]
        phase_groups.extend((phase, group) for phase, group in rollout_df.groupby("phase"))
        for phase_name, phase_df in phase_groups:
            wide = phase_df.set_index(["rollout", "step", "model"])
            frame_keys = list(phase_df[["rollout", "step"]].drop_duplicates().itertuples(index=False))
            for metric in METRICS:
                if metric not in phase_df.columns:
                    continue
                higher = HIGHER_IS_BETTER[metric]
                for model_a in models_a:
                    for baseline in baselines:
                        if model_a == baseline:
                            continue
                        deltas = []
                        a_wins = b_wins = ties = 0
                        for frame_key in frame_keys:
                            key_a = (frame_key.rollout, frame_key.step, model_a)
                            key_b = (frame_key.rollout, frame_key.step, baseline)
                            if key_a not in wide.index or key_b not in wide.index:
                                continue
                            a = pd.to_numeric(wide.loc[key_a, metric], errors="coerce")
                            b = pd.to_numeric(wide.loc[key_b, metric], errors="coerce")
                            if pd.isna(a) or pd.isna(b):
                                continue
                            delta = float(a) - float(b)
                            advantage = delta if higher else -delta
                            deltas.append(advantage)
                            if abs(advantage) < 1e-9:
                                ties += 1
                            elif advantage > 0:
                                a_wins += 1
                            else:
                                b_wins += 1
                        if not deltas:
                            continue
                        arr = np.asarray(deltas, dtype=np.float64)
                        rows.append(
                            {
                                "rollout": rollout_name,
                                "phase": phase_name,
                                "metric": metric,
                                "model_a": model_a,
                                "model_b": baseline,
                                "a_better_frames": a_wins,
                                "b_better_frames": b_wins,
                                "ties": ties,
                                "n_frames": int(len(arr)),
                                "a_win_rate": float(a_wins / len(arr)),
                                "mean_a_advantage": float(arr.mean()),
                                "median_a_advantage": float(np.median(arr)),
                            }
                        )
    return pd.DataFrame(rows).sort_values(["rollout", "phase", "metric", "model_a", "model_b"])


def plot_overall(summary_df: pd.DataFrame, out_path: Path) -> None:
    overall = summary_df[(summary_df["rollout"] == "both") & (summary_df["phase"] == "all")]
    metrics = [
        ("exec_best_progress_px_mean", "Best progress px"),
        ("exec_final_progress_px_mean", "Final progress px"),
        ("first10_cos_to_target_mean_mean", "First10 direction cosine"),
        ("pred_close_rate_mean", "Close rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (metric, title) in zip(axes.flat, metrics):
        data = overall.sort_values(metric, ascending=metric == "exec_min_distance_px_mean")
        ax.barh(data["model"], data[metric], color="#4C78A8")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_phase_progress(summary_df: pd.DataFrame, out_path: Path) -> None:
    phase_df = summary_df[
        (summary_df["rollout"] == "both") & (summary_df["phase"] != "all")
    ].copy()
    phases = ["close_0_30", "near_30_50", "approach_50_70"]
    models = list(phase_df["model"].drop_duplicates())
    x = np.arange(len(phases))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(models):
        vals = []
        for phase in phases:
            row = phase_df[(phase_df["phase"] == phase) & (phase_df["model"] == model)]
            vals.append(float(row["exec_final_progress_px_mean"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width, label=model)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, phases)
    ax.set_ylabel("Final progress px")
    ax.set_title("Final progress by distance phase")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_report(
    *,
    out_path: Path,
    tag: str,
    summary_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    artifacts: dict[str, Path],
) -> None:
    overall = summary_df[(summary_df["rollout"] == "both") & (summary_df["phase"] == "all")]
    cols = [
        "model",
        "n_frames",
        "first10_cos_to_target_mean_mean",
        "exec_best_progress_px_mean",
        "exec_final_progress_px_mean",
        "exec_min_distance_px_mean",
        "pred_close_rate_mean",
        "pred_gripper_max_mean",
    ]
    overall_html = overall[cols].sort_values("exec_final_progress_px_mean", ascending=False).to_html(
        index=False, float_format=lambda x: f"{x:.4f}"
    )
    pw = pairwise_df[
        (pairwise_df["rollout"] == "both")
        & (pairwise_df["phase"].isin(["all", "approach_50_70"]))
        & (pairwise_df["metric"].isin(["exec_final_progress_px", "exec_best_progress_px", "first10_cos_to_target_mean"]))
    ]
    pairwise_html = pw.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{tag}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0 28px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f5f5f5; }}
    img {{ max-width: 100%; display: block; margin: 14px 0 26px; border: 1px solid #ddd; }}
    code {{ background: #f4f4f4; padding: 2px 4px; }}
    .note {{ color: #5f6368; }}
  </style>
</head>
<body>
<main>
  <h1>{tag}</h1>
  <p class="note">Same candidate frames and seeds as the referenced base summary; FM seeds are averaged per frame before model comparison.</p>
  <h2>Overall</h2>
  <img src="{artifacts['overall_plot']}" alt="overall metrics">
  {overall_html}
  <h2>Phase Final Progress</h2>
  <img src="{artifacts['phase_plot']}" alt="phase final progress">
  <h2>Selected Pairwise Comparisons</h2>
  {pairwise_html}
  <h2>Artifacts</h2>
  {''.join(f'<p><code>{path}</code></p>' for path in artifacts.values())}
</main>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_summary_path = Path(args.base_summary_json)
    base_summary = json.loads(base_summary_path.read_text())
    candidate_csv = Path(args.candidate_csv or base_summary["candidate_csv"])
    candidates = pd.read_csv(candidate_csv)
    model_specs = [parse_model_spec(spec) for spec in args.model]
    for _, checkpoint in model_specs:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_matrix, _ = load_calibration(Path(args.calibration_json).expanduser())
    query_records = {
        rollout: load_query_records(Path(meta["log_jsonl"]))
        for rollout, meta in base_summary["rollouts"].items()
    }
    videos = {rollout: Path(meta["video"]) for rollout, meta in base_summary["rollouts"].items()}
    seeds = args.seeds if args.seeds is not None else [int(s) for s in base_summary.get("seeds", [0, 1, 2])]
    resampled_action_len = int(args.resampled_action_len or base_summary.get("resampled_action_len", 45))
    execute_steps = int(args.execute_steps or base_summary.get("execute_steps", 30))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    wrappers = {}
    model_meta = []
    print(f"Loading {len(model_specs)} models...", flush=True)
    for model_name, checkpoint in model_specs:
        wrapper = ModelWrapper.load_from_checkpoint(
            str(checkpoint), weights_only=False, map_location="cpu"
        )
        wrapper = wrapper.to(device)
        wrapper.eval()
        wrapper.model.device = device
        diffusion = bool(getattr(wrapper.model, "diffusion", False))
        expected_t = expected_action_len(wrapper)
        wrappers[model_name] = {
            "wrapper": wrapper,
            "diffusion": diffusion,
            "expected_t": expected_t,
            "checkpoint": str(checkpoint),
        }
        model_meta.append(
            {
                "model": model_name,
                "checkpoint": str(checkpoint),
                "diffusion": diffusion,
                "expected_action_len": expected_t,
            }
        )
        print(f"  {model_name}: diffusion={diffusion} expected_t={expected_t}", flush=True)

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
        for model_name, meta in wrappers.items():
            run_seeds: list[int | None] = list(seeds) if meta["diffusion"] else [None]
            for seed in run_seeds:
                chunk = predict_chunk(
                    meta["wrapper"],
                    frame_bgr,
                    state,
                    meta["expected_t"],
                    seed=seed,
                    device=device,
                )
                exec_chunk = resample_chunk(chunk, resampled_action_len)[:execute_steps]
                traj_px = project_points(exec_chunk[:, :3], camera_matrix)
                row = {
                    "rollout": rollout,
                    "step": step,
                    "time_s": float(candidate["time_s"]),
                    "phase": str(candidate["phase"]),
                    "model": model_name,
                    "seed": "det" if seed is None else int(seed),
                    "checkpoint": meta["checkpoint"],
                    "diffusion": bool(meta["diffusion"]),
                    "expected_action_len": int(meta["expected_t"]),
                    "state_gripper": float(state[6]),
                    "ee_px_x": float(ee_px[0]),
                    "ee_px_y": float(ee_px[1]),
                    "target_x": float(target_px[0]),
                    "target_y": float(target_px[1]),
                    "n_apricot_detections": int(candidate["n_apricot_detections"]),
                }
                row.update(trajectory_metrics(traj_px, ee_px, target_px))
                row.update(gripper_metrics(exec_chunk[:, 6]))
                row.update(extra_metrics(traj_px, target_px, exec_chunk[:, 6]))
                seed_rows.append(row)

    seed_path = output_dir / f"{args.tag}_seed_rows.csv"
    write_csv(seed_path, seed_rows)
    seed_df = pd.DataFrame(seed_rows)
    frame_df = summarize_frame_means(seed_df)
    frame_path = output_dir / f"{args.tag}_frame_model_means.csv"
    frame_df.to_csv(frame_path, index=False)

    summary_df = aggregate_summary(frame_df)
    summary_path = output_dir / f"{args.tag}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    all_models = [name for name, _ in model_specs]
    baselines = args.baseline or [name for name in all_models if name.startswith("fm_so100")]
    models_a = args.pairwise_a or [name for name in all_models if name not in baselines]
    pairwise_df = pairwise_summary(frame_df, models_a=models_a, baselines=baselines)
    pairwise_path = output_dir / f"{args.tag}_pairwise.csv"
    pairwise_df.to_csv(pairwise_path, index=False)

    overall_plot = output_dir / f"{args.tag}_overall_metrics.png"
    phase_plot = output_dir / f"{args.tag}_phase_final_progress.png"
    plot_overall(summary_df, overall_plot)
    plot_phase_progress(summary_df, phase_plot)

    report_path = Path(f"{args.tag}.html")
    artifacts = {
        "seed_rows": seed_path,
        "frame_model_means": frame_path,
        "summary": summary_path,
        "pairwise": pairwise_path,
        "overall_plot": overall_plot,
        "phase_plot": phase_plot,
    }
    render_report(out_path=report_path, tag=args.tag, summary_df=summary_df, pairwise_df=pairwise_df, artifacts=artifacts)

    out_summary = {
        "tag": args.tag,
        "base_summary_json": str(base_summary_path),
        "candidate_csv": str(candidate_csv),
        "n_candidates": int(len(candidates)),
        "rollouts": base_summary["rollouts"],
        "models": model_meta,
        "resampled_action_len": resampled_action_len,
        "execute_steps": execute_steps,
        "seeds": seeds,
        "baselines": baselines,
        "pairwise_a": models_a,
        "outputs": {key: str(path) for key, path in artifacts.items()},
        "report": str(report_path),
    }
    out_summary_path = output_dir / f"{args.tag}_summary.json"
    out_summary_path.write_text(json.dumps(out_summary, indent=2))
    print(json.dumps(out_summary, indent=2))
    print("Success")


if __name__ == "__main__":
    main()
