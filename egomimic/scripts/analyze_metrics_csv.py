#!/usr/bin/env python3
"""Analyze Lightning CSV metrics and export charts plus a compact report."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXCLUDE_COLUMNS = {"step", "epoch"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a Lightning metrics.csv file and export charts."
    )
    parser.add_argument("metrics_csv", type=Path, help="Path to metrics.csv")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <metrics.csv parent>/analysis.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["auto", "step", "epoch", "index"],
        default="auto",
        help="X axis for plots. Auto prefers step, then epoch, then row index.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Rolling window used for smoothed curves. Use 1 to disable.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG export DPI.",
    )
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "metric"


def read_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"metrics CSV not found: {path}")
    df = pd.read_csv(path)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(axis=1, how="all")
    return df


def choose_x_axis(df: pd.DataFrame, requested: str) -> tuple[str, pd.Series]:
    if requested == "auto":
        if "step" in df.columns and df["step"].notna().any():
            requested = "step"
        elif "epoch" in df.columns and df["epoch"].notna().any():
            requested = "epoch"
        else:
            requested = "index"

    if requested == "index":
        return "index", pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    if requested not in df.columns or not df[requested].notna().any():
        raise ValueError(f"requested x-axis {requested!r} is not available in {df.columns.tolist()}")
    return requested, df[requested]


def metric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLUMNS and df[c].notna().any()]


def is_timing_metric(metric: str) -> bool:
    lower = metric.lower()
    return lower.startswith("timing/") or "sec" in lower or "time" in lower


def is_loss_metric(metric: str) -> bool:
    return "loss" in metric.lower() and not is_timing_metric(metric)


def group_metrics(metrics: list[str]) -> dict[str, list[str]]:
    groups = {
        "loss": [],
        "validation": [],
        "grad_norm": [],
        "timing": [],
        "optimizer": [],
        "other": [],
    }
    for col in metrics:
        lower = col.lower()
        if lower.startswith("val/") or lower.startswith("valid/"):
            groups["validation"].append(col)
        elif is_timing_metric(col):
            groups["timing"].append(col)
        elif is_loss_metric(col):
            groups["loss"].append(col)
        elif "grad_norm" in lower or "gradnorm" in lower:
            groups["grad_norm"].append(col)
        elif lower.startswith("optimizer/") or "lr" in lower:
            groups["optimizer"].append(col)
        else:
            groups["other"].append(col)
    return {k: v for k, v in groups.items() if v}


def summarize_metric(df: pd.DataFrame, x: pd.Series, col: str) -> dict[str, float | str | int]:
    y = df[col]
    mask = y.notna()
    yy = y[mask]
    xx = x[mask]
    if yy.empty:
        return {}

    first = float(yy.iloc[0])
    last = float(yy.iloc[-1])
    min_pos = yy.idxmin()
    max_pos = yy.idxmax()
    delta_abs = last - first
    delta_pct = math.nan if first == 0 else delta_abs / abs(first) * 100.0

    return {
        "metric": col,
        "count": int(yy.count()),
        "first_x": float(xx.iloc[0]),
        "last_x": float(xx.iloc[-1]),
        "first": first,
        "last": last,
        "min": float(yy.min()),
        "min_x": float(x.loc[min_pos]),
        "max": float(yy.max()),
        "max_x": float(x.loc[max_pos]),
        "mean": float(yy.mean()),
        "std": float(yy.std(ddof=0)) if yy.count() > 1 else 0.0,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
    }


def write_summary_csv(summary: pd.DataFrame, out_path: Path) -> None:
    rounded = summary.copy()
    for col in rounded.columns:
        if col not in {"metric", "count"}:
            rounded[col] = rounded[col].astype(float).round(8)
    rounded.to_csv(out_path, index=False)


def format_float(value: float, digits: int = 6) -> str:
    if pd.isna(value):
        return "nan"
    if value == 0:
        return "0"
    if abs(value) < 1e-4 or abs(value) >= 1e5:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def plot_metric_group(
    df: pd.DataFrame,
    x_name: str,
    x: pd.Series,
    cols: list[str],
    title: str,
    out_path: Path,
    rolling_window: int,
    dpi: int,
    log_y: bool = False,
) -> None:
    if not cols:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for col in cols:
        y = df[col]
        mask = y.notna() & x.notna()
        if mask.sum() == 0:
            continue
        xx = x[mask]
        yy = y[mask]
        if rolling_window > 1 and len(yy) >= rolling_window:
            ax.plot(xx, yy, alpha=0.22, linewidth=1.0)
            smooth = yy.rolling(rolling_window, min_periods=1).mean()
            ax.plot(xx, smooth, label=f"{col} (roll{rolling_window})", linewidth=1.8)
        else:
            ax.plot(xx, yy, label=col, linewidth=1.6)

    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.grid(True, alpha=0.25)
    if log_y:
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_all_metrics(
    df: pd.DataFrame,
    x_name: str,
    x: pd.Series,
    cols: list[str],
    out_path: Path,
    rolling_window: int,
    dpi: int,
) -> None:
    if not cols:
        return
    ncols = 2
    nrows = math.ceil(len(cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, max(3.2, 2.8 * nrows)))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        y = df[col]
        mask = y.notna() & x.notna()
        xx = x[mask]
        yy = y[mask]
        ax.plot(xx, yy, alpha=0.35, linewidth=1.0)
        if rolling_window > 1 and len(yy) >= rolling_window:
            ax.plot(xx, yy.rolling(rolling_window, min_periods=1).mean(), linewidth=1.6)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel(x_name)
        ax.grid(True, alpha=0.22)
    for ax in axes[len(cols) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def write_markdown_report(
    out_path: Path,
    metrics_csv: Path,
    df: pd.DataFrame,
    x_name: str,
    summary: pd.DataFrame,
    charts: list[Path],
) -> None:
    lines = []
    lines.append("# Metrics Analysis")
    lines.append("")
    lines.append(f"- Source: `{metrics_csv}`")
    lines.append(f"- Rows: `{len(df)}`")
    lines.append(f"- Numeric metric columns: `{len(metric_columns(df))}`")
    lines.append(f"- X axis: `{x_name}`")
    if "step" in df.columns and df["step"].notna().any():
        lines.append(
            f"- Step range: `{format_float(float(df['step'].min()), 0)}` to `{format_float(float(df['step'].max()), 0)}`"
        )
    if "epoch" in df.columns and df["epoch"].notna().any():
        lines.append(
            f"- Epoch range: `{format_float(float(df['epoch'].min()), 0)}` to `{format_float(float(df['epoch'].max()), 0)}`"
        )
    lines.append("")

    loss_rows = summary[summary["metric"].map(is_loss_metric)]
    if not loss_rows.empty:
        lines.append("## Loss Metrics")
        lines.append("")
        lines.append("| metric | first | last | min | min_x | delta_pct |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in loss_rows.itertuples(index=False):
            lines.append(
                "| {metric} | {first} | {last} | {minv} | {minx} | {delta} |".format(
                    metric=row.metric,
                    first=format_float(row.first),
                    last=format_float(row.last),
                    minv=format_float(row.min),
                    minx=format_float(row.min_x, 0),
                    delta=format_float(row.delta_pct, 3) + "%",
                )
            )
        lines.append("")

    key_patterns = ("grad_norm", "timing", "optimizer", "lr", "val/")
    key_rows = summary[
        summary["metric"]
        .str.lower()
        .map(lambda s: any(p in s for p in key_patterns) or is_loss_metric(s))
    ]
    if not key_rows.empty:
        lines.append("## Metric Summary")
        lines.append("")
        lines.append("| metric | count | first_x | last_x | first | last | min | max | mean |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in key_rows.itertuples(index=False):
            lines.append(
                "| {metric} | {count} | {first_x} | {last_x} | {first} | {last} | {minv} | {maxv} | {mean} |".format(
                    metric=row.metric,
                    count=row.count,
                    first_x=format_float(row.first_x, 0),
                    last_x=format_float(row.last_x, 0),
                    first=format_float(row.first),
                    last=format_float(row.last),
                    minv=format_float(row.min),
                    maxv=format_float(row.max),
                    mean=format_float(row.mean),
                )
            )
        lines.append("")

    lines.append("## Charts")
    lines.append("")
    for chart in charts:
        lines.append(f"- [{chart.name}]({chart.relative_to(out_path.parent)})")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics_csv = args.metrics_csv.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else metrics_csv.parent / "analysis"
    )
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    df = read_metrics(metrics_csv)
    x_name, x = choose_x_axis(df, args.x_axis)
    metrics = metric_columns(df)
    if not metrics:
        raise ValueError(f"No numeric metric columns found in {metrics_csv}")

    df.to_csv(out_dir / "metrics_clean.csv", index=False)

    summary_rows = [summarize_metric(df, x, col) for col in metrics]
    summary = pd.DataFrame([row for row in summary_rows if row])
    summary = summary.sort_values("metric")
    write_summary_csv(summary, out_dir / "metric_summary.csv")

    groups = group_metrics(metrics)
    charts: list[Path] = []
    for group_name, cols in groups.items():
        chart = charts_dir / f"{group_name}.png"
        plot_metric_group(
            df,
            x_name,
            x,
            cols,
            title=group_name.replace("_", " ").title(),
            out_path=chart,
            rolling_window=args.rolling_window,
            dpi=args.dpi,
            log_y=group_name == "loss",
        )
        charts.append(chart)

    all_chart = charts_dir / "all_metrics.png"
    plot_all_metrics(
        df,
        x_name,
        x,
        metrics,
        all_chart,
        rolling_window=args.rolling_window,
        dpi=args.dpi,
    )
    charts.append(all_chart)

    write_markdown_report(
        out_dir / "summary.md",
        metrics_csv,
        df,
        x_name,
        summary,
        charts,
    )

    print(f"Wrote analysis to: {out_dir}")
    print(f"Summary: {out_dir / 'summary.md'}")
    print(f"Charts: {charts_dir}")


if __name__ == "__main__":
    main()
