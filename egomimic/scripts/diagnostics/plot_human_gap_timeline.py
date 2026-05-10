from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _parse_point(spec: str) -> tuple[int, Path]:
    try:
        step_str, path_str = spec.split("=", 1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid point spec {spec!r}. Expected format STEP=/abs/path/to/file.json"
        ) from exc
    step = int(step_str)
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"Diagnostics file not found: {path}")
    return step, path


def _load_series(points: list[str], domain: str) -> list[dict]:
    rows = []
    for spec in points:
        step, path = _parse_point(spec)
        data = json.loads(path.read_text())
        metrics = data["results"]["train"][domain]["normalized"]["overall"]
        train_loss = float(metrics["smooth_l1_mean"])
        metrics = data["results"]["valid"][domain]["normalized"]["overall"]
        valid_loss = float(metrics["smooth_l1_mean"])
        rows.append(
            {
                "step": step,
                "path": str(path),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "gap_abs": valid_loss - train_loss,
                "gap_ratio": valid_loss / train_loss if train_loss > 0 else float("inf"),
            }
        )
    rows.sort(key=lambda row: row["step"])
    return rows


def _write_csv(path: Path, cotrain: list[dict], scratch: list[dict]) -> None:
    fieldnames = [
        "series",
        "step",
        "train_loss",
        "valid_loss",
        "gap_abs",
        "gap_ratio",
        "source_json",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for series_name, rows in (("cotrain", cotrain), ("scratch", scratch)):
            for row in rows:
                writer.writerow(
                    {
                        "series": series_name,
                        "step": row["step"],
                        "train_loss": row["train_loss"],
                        "valid_loss": row["valid_loss"],
                        "gap_abs": row["gap_abs"],
                        "gap_ratio": row["gap_ratio"],
                        "source_json": row["path"],
                    }
                )


def _plot_series(ax, xs, ys, *, color: str, label: str, linestyle: str) -> None:
    ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=6, color=color, linestyle=linestyle, label=label)


def _annotate_points(ax, xs, ys, fmt: str) -> None:
    for x, y in zip(xs, ys):
        ax.annotate(
            fmt.format(y),
            (x, y),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )


def _extract_xy(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    return [row["step"] for row in rows], [row[key] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot matched human train/valid/gap timelines from diagnostics JSON files."
    )
    parser.add_argument(
        "--cotrain-point",
        action="append",
        required=True,
        help="One cotrain point in the format STEP=/abs/path/to/json",
    )
    parser.add_argument(
        "--scratch-point",
        action="append",
        required=True,
        help="One scratch point in the format STEP=/abs/path/to/json",
    )
    parser.add_argument(
        "--domain",
        default="ego_view_right_arm",
        help="Domain key inside diagnostics JSON. Default: ego_view_right_arm",
    )
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--title",
        default="Human Train/Valid Gap Timeline",
        help="Figure title",
    )
    args = parser.parse_args()

    cotrain_rows = _load_series(args.cotrain_point, args.domain)
    scratch_rows = _load_series(args.scratch_point, args.domain)

    output_png = Path(args.output_png)
    output_csv = Path(args.output_csv)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    _write_csv(output_csv, cotrain_rows, scratch_rows)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(args.title, fontsize=15)

    cotrain_color = "#1f77b4"
    scratch_color = "#d95f02"

    x_ct, y_ct_train = _extract_xy(cotrain_rows, "train_loss")
    _, y_ct_valid = _extract_xy(cotrain_rows, "valid_loss")
    x_sc, y_sc_train = _extract_xy(scratch_rows, "train_loss")
    _, y_sc_valid = _extract_xy(scratch_rows, "valid_loss")

    _plot_series(
        axes[0], x_ct, y_ct_train, color=cotrain_color, label="Cotrain train", linestyle="-"
    )
    _plot_series(
        axes[0], x_ct, y_ct_valid, color=cotrain_color, label="Cotrain valid", linestyle="--"
    )
    _plot_series(
        axes[0], x_sc, y_sc_train, color=scratch_color, label="Scratch train", linestyle="-"
    )
    _plot_series(
        axes[0], x_sc, y_sc_valid, color=scratch_color, label="Scratch valid", linestyle="--"
    )
    axes[0].set_ylabel("Normalized Smooth L1")
    axes[0].set_title("Human train and valid loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncols=2, fontsize=9)
    _annotate_points(axes[0], x_ct, y_ct_valid, "{:.3f}")
    _annotate_points(axes[0], x_sc, y_sc_valid, "{:.3f}")

    x_ct, y_ct_gap_abs = _extract_xy(cotrain_rows, "gap_abs")
    x_sc, y_sc_gap_abs = _extract_xy(scratch_rows, "gap_abs")
    _plot_series(
        axes[1], x_ct, y_ct_gap_abs, color=cotrain_color, label="Cotrain gap abs", linestyle="-"
    )
    _plot_series(
        axes[1], x_sc, y_sc_gap_abs, color=scratch_color, label="Scratch gap abs", linestyle="-"
    )
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.4)
    axes[1].set_ylabel("Valid - Train")
    axes[1].set_title("Absolute gap")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)
    _annotate_points(axes[1], x_ct, y_ct_gap_abs, "{:.3f}")
    _annotate_points(axes[1], x_sc, y_sc_gap_abs, "{:.3f}")

    x_ct, y_ct_gap_ratio = _extract_xy(cotrain_rows, "gap_ratio")
    x_sc, y_sc_gap_ratio = _extract_xy(scratch_rows, "gap_ratio")
    _plot_series(
        axes[2], x_ct, y_ct_gap_ratio, color=cotrain_color, label="Cotrain gap ratio", linestyle="-"
    )
    _plot_series(
        axes[2], x_sc, y_sc_gap_ratio, color=scratch_color, label="Scratch gap ratio", linestyle="-"
    )
    axes[2].axhline(1.0, color="black", linewidth=1, alpha=0.4)
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Valid / Train")
    axes[2].set_title("Relative gap")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=9)
    _annotate_points(axes[2], x_ct, y_ct_gap_ratio, "{:.2f}")
    _annotate_points(axes[2], x_sc, y_sc_gap_ratio, "{:.2f}")

    plt.tight_layout()
    fig.subplots_adjust(top=0.94)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote figure to {output_png}")
    print(f"Wrote summary CSV to {output_csv}")


if __name__ == "__main__":
    main()
