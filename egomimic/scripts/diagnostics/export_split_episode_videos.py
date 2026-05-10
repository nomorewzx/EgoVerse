from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import hydra
import numpy as np
import zarr
from omegaconf import OmegaConf

from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset

try:
    import simplejpeg
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    simplejpeg = None


def _load_cfg(args: argparse.Namespace):
    if args.run_dir is not None:
        config_path = Path(args.run_dir) / ".hydra" / "config.yaml"
    elif args.config is not None:
        config_path = Path(args.config)
    else:
        raise ValueError("Provide either --run-dir or --config")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return OmegaConf.load(config_path), config_path


def _instantiate_dataset(cfg, split_name: str, domain: str) -> MultiDataset:
    split_cfg = cfg.data.train_datasets if split_name == "train" else cfg.data.valid_datasets
    return hydra.utils.instantiate(split_cfg[domain])


def _find_camera_zarr_key(dataset) -> str:
    for spec in dataset.key_map.values():
        if spec.get("key_type") == "camera_keys":
            return spec["zarr_key"]
    raise ValueError(f"No camera key found for episode {dataset.episode_path}")


def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    if simplejpeg is not None:
        return simplejpeg.decode_jpeg(jpeg_bytes, colorspace="RGB")
    decoded_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise ValueError("cv2.imdecode returned None")
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def _write_episode_video(
    *,
    dataset,
    output_path: Path,
    annotate: bool,
) -> dict:
    store = zarr.open_group(str(dataset.episode_path), mode="r")
    metadata = dict(store.attrs)
    camera_key = _find_camera_zarr_key(dataset)
    image_arr = store[camera_key]
    num_frames = int(metadata["total_frames"])
    fps = float(metadata.get("fps", 30))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_rgb = _decode_jpeg(image_arr[0:1][0])
    height, width = first_rgb.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        for frame_idx in range(num_frames):
            rgb = _decode_jpeg(image_arr[frame_idx : frame_idx + 1][0])
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if annotate:
                cv2.putText(
                    bgr,
                    f"{Path(dataset.episode_path).stem} | frame {frame_idx}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(bgr)
    finally:
        writer.release()

    return {
        "episode_name": Path(dataset.episode_path).stem,
        "episode_path": str(dataset.episode_path),
        "camera_zarr_key": camera_key,
        "num_frames": num_frames,
        "fps": fps,
        "output_video": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw episode videos for a configured train/valid split."
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--domain", default="ego_view_right_arm")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--episode-name-substr", default=None)
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, config_path = _load_cfg(args)
    dataset = _instantiate_dataset(cfg, args.split, args.domain)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else (Path(args.run_dir) / f"{args.split}_{args.domain}_videos")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    items = sorted(dataset.datasets.items(), key=lambda kv: Path(kv[1].episode_path).name)
    if args.episode_name_substr:
        items = [
            item
            for item in items
            if args.episode_name_substr in Path(item[1].episode_path).name
        ]
    if args.max_episodes is not None:
        items = items[: args.max_episodes]

    results = []
    for _, leaf_dataset in items:
        episode_name = Path(leaf_dataset.episode_path).stem
        output_path = output_dir / f"{episode_name}.mp4"
        if output_path.exists() and not args.overwrite:
            results.append(
                {
                    "episode_name": episode_name,
                    "episode_path": str(leaf_dataset.episode_path),
                    "output_video": str(output_path),
                    "skipped_existing": True,
                }
            )
            continue
        results.append(
            _write_episode_video(
                dataset=leaf_dataset,
                output_path=output_path,
                annotate=args.annotate,
            )
        )

    payload = {
        "config_path": str(config_path),
        "split": args.split,
        "domain": args.domain,
        "num_episodes": len(results),
        "output_dir": str(output_dir),
        "episodes": results,
    }

    text = json.dumps(payload, indent=2)
    if args.output_json is not None:
        Path(args.output_json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
