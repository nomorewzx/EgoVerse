from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import zarr
from scipy.spatial.transform import Rotation as R

from egomimic.pl_utils.pl_model import ModelWrapper

try:
    import simplejpeg
except ImportError:  # pragma: no cover - optional fast JPEG decoder
    simplejpeg = None

try:
    import cv2
except ImportError:  # pragma: no cover - fallback only when simplejpeg is missing
    cv2 = None


DEFAULT_CKPT = (
    "logs/so100_hpt/hpt_base_steps100k_2026-04-30_19-42-21/"
    "checkpoints/step_80000.ckpt"
)
DEFAULT_ZARR_ROOT = "/home/zxwang/so100-ee-cam-egoverse-zarr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark single-sample SO100 HPT rollout inference latency. "
            "Measures CPU batch construction, process_batch_for_training, "
            "forward_eval, and synchronous end-to-end query time."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--zarr-root", default=DEFAULT_ZARR_ROOT)
    parser.add_argument("--episode", default=None, help="Optional .zarr episode path/name.")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--precision",
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="Autocast precision used around forward_eval on CUDA.",
    )
    parser.add_argument("--action-horizon", type=int, default=64)
    parser.add_argument(
        "--bgr-to-rgb",
        action="store_true",
        help="Use this if the sampled image is BGR. Zarr images are usually RGB.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path for a JSON latency report.",
    )
    return parser.parse_args()


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(device: torch.device, fn: Callable[[], object]) -> tuple[float, object]:
    cuda_sync(device)
    t0 = time.perf_counter()
    out = fn()
    cuda_sync(device)
    return (time.perf_counter() - t0) * 1000.0, out


def stats_ms(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def resolve_episode(zarr_root: Path, episode: str | None) -> Path:
    if episode is not None:
        candidate = Path(episode).expanduser()
        if not candidate.is_absolute():
            candidate = zarr_root / candidate
        if candidate.suffix != ".zarr":
            with_suffix = candidate.with_suffix(".zarr")
            if with_suffix.exists():
                candidate = with_suffix
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    episodes = sorted(zarr_root.glob("*.zarr"))
    if not episodes:
        raise FileNotFoundError(f"No .zarr episodes found under {zarr_root}")
    return episodes[0]


def rotvec_pose7_to_ypr_pose7(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (7,):
        raise ValueError(f"Expected pose shape (7,), got {pose.shape}")
    ypr = R.from_rotvec(pose[3:6]).as_euler("ZYX", degrees=False).astype(np.float32)
    return np.concatenate([pose[:3], ypr, pose[6:7]], axis=0).astype(np.float32)


def decode_jpeg_value(value: object) -> np.ndarray:
    while isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        value = value.item()
    if isinstance(value, np.ndarray) and value.dtype == np.uint8:
        jpeg_bytes = value.tobytes()
    elif isinstance(value, (bytes, bytearray, memoryview)):
        jpeg_bytes = bytes(value)
    else:
        arr = np.asarray(value)
        if arr.ndim == 3:
            return arr
        raise ValueError(
            f"Cannot decode image value of type {type(value)} with shape {getattr(arr, 'shape', None)}"
        )

    if simplejpeg is not None:
        return simplejpeg.decode_jpeg(jpeg_bytes, colorspace="RGB")
    if cv2 is None:
        raise ImportError("Need simplejpeg or cv2 to decode JPEG zarr images")
    decoded_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise ValueError("cv2.imdecode returned None")
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def load_zarr_sample(
    zarr_root: Path,
    episode: str | None,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    episode_path = resolve_episode(zarr_root, episode)
    group = zarr.open(str(episode_path), mode="r")
    image_arr = group["images.front_1"]
    pose_arr = group["obs_ee_pose_cam_rotvec"]
    num_frames = int(pose_arr.shape[0])
    if frame_index < 0 or frame_index >= num_frames:
        raise IndexError(
            f"frame-index {frame_index} out of range for {episode_path.name}: "
            f"0..{num_frames - 1}"
        )
    image = decode_jpeg_value(image_arr[frame_index])
    ee_ypr = rotvec_pose7_to_ypr_pose7(np.asarray(pose_arr[frame_index]))
    metadata = {
        "episode": str(episode_path),
        "frame_index": int(frame_index),
        "image_shape": list(image.shape),
        "ee_ypr": ee_ypr.tolist(),
    }
    return image, ee_ypr, metadata


def build_live_like_batch(
    image: np.ndarray,
    ee_ypr: np.ndarray,
    action_horizon: int,
    *,
    bgr_to_rgb: bool,
) -> dict[str, dict[str, torch.Tensor]]:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image shape (H,W,C) or (C,H,W), got {image.shape}")
    if image.shape[-1] == 3:
        if bgr_to_rgb:
            image = image[..., [2, 1, 0]]
        front = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    elif image.shape[0] == 3:
        front = torch.from_numpy(np.ascontiguousarray(image))
    else:
        raise ValueError(f"Expected 3-channel image, got {image.shape}")

    front = front.to(dtype=torch.float32)
    if float(front.max()) > 1.5:
        front = front / 255.0

    ee = torch.as_tensor(np.asarray(ee_ypr, dtype=np.float32))
    actions = ee.view(1, 7).repeat(action_horizon, 1)

    return {
        "so100_singlearm": {
            "observations.images.front_img_1": front.unsqueeze(0),
            "observations.state.ee_pose": ee.unsqueeze(0),
            "actions_cartesian": actions.unsqueeze(0),
        }
    }


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = requested_device

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    image, ee_ypr, sample_metadata = load_zarr_sample(
        Path(args.zarr_root).expanduser(),
        args.episode,
        args.frame_index,
    )

    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    load_t0 = time.perf_counter()
    wrapper = ModelWrapper.load_from_checkpoint(
        str(ckpt_path), weights_only=False, map_location="cpu"
    )
    wrapper = wrapper.to(device)
    wrapper.eval()
    wrapper.model.device = device
    model_load_s = time.perf_counter() - load_t0

    raw = build_live_like_batch(
        image,
        ee_ypr,
        args.action_horizon,
        bgr_to_rgb=args.bgr_to_rgb,
    )

    with torch.inference_mode(), autocast_context(device, args.precision):
        processed = wrapper.model.process_batch_for_training(raw)
        _ = wrapper.model.forward_eval(processed)
        cuda_sync(device)

        for _ in range(args.warmup):
            raw = build_live_like_batch(
                image,
                ee_ypr,
                args.action_horizon,
                bgr_to_rgb=args.bgr_to_rgb,
            )
            processed = wrapper.model.process_batch_for_training(raw)
            _ = wrapper.model.forward_eval(processed)
        cuda_sync(device)

        build_times: list[float] = []
        process_times: list[float] = []
        forward_times: list[float] = []
        end_to_end_times: list[float] = []

        processed_for_forward = wrapper.model.process_batch_for_training(
            build_live_like_batch(
                image,
                ee_ypr,
                args.action_horizon,
                bgr_to_rgb=args.bgr_to_rgb,
            )
        )

        for _ in range(args.iters):
            t_ms, _ = time_call(
                torch.device("cpu"),
                lambda: build_live_like_batch(
                    image,
                    ee_ypr,
                    args.action_horizon,
                    bgr_to_rgb=args.bgr_to_rgb,
                ),
            )
            build_times.append(t_ms)

        for _ in range(args.iters):
            raw = build_live_like_batch(
                image,
                ee_ypr,
                args.action_horizon,
                bgr_to_rgb=args.bgr_to_rgb,
            )
            t_ms, _ = time_call(device, lambda: wrapper.model.process_batch_for_training(raw))
            process_times.append(t_ms)

        for _ in range(args.iters):
            t_ms, _ = time_call(
                device,
                lambda: wrapper.model.forward_eval(processed_for_forward),
            )
            forward_times.append(t_ms)

        for _ in range(args.iters):
            def end_to_end():
                raw_batch = build_live_like_batch(
                    image,
                    ee_ypr,
                    args.action_horizon,
                    bgr_to_rgb=args.bgr_to_rgb,
                )
                processed_batch = wrapper.model.process_batch_for_training(raw_batch)
                return wrapper.model.forward_eval(processed_batch)

            t_ms, _ = time_call(device, end_to_end)
            end_to_end_times.append(t_ms)

    forward_p50 = stats_ms(forward_times)["p50_ms"]
    e2e_p50 = stats_ms(end_to_end_times)["p50_ms"]
    report: dict[str, object] = {
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": args.precision,
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "model_load_s": float(model_load_s),
        "sample": sample_metadata,
        "latency": {
            "build_live_batch_cpu": stats_ms(build_times),
            "process_batch": stats_ms(process_times),
            "forward_eval": stats_ms(forward_times),
            "end_to_end_sync_query": stats_ms(end_to_end_times),
        },
        "derived": {
            "fps_from_forward_p50": float(1000.0 / forward_p50) if forward_p50 > 0 else math.inf,
            "fps_from_end_to_end_p50": float(1000.0 / e2e_p50) if e2e_p50 > 0 else math.inf,
            "frame_budget_30fps_ms": 1000.0 / 30.0,
        },
    }
    return report


def print_report(report: dict[str, object]) -> None:
    print(json.dumps(report, indent=2))
    latency = report["latency"]
    derived = report["derived"]
    print("\nSummary:")
    for key in [
        "build_live_batch_cpu",
        "process_batch",
        "forward_eval",
        "end_to_end_sync_query",
    ]:
        values = latency[key]
        print(
            f"  {key:24s} p50={values['p50_ms']:.2f} ms "
            f"p90={values['p90_ms']:.2f} ms p99={values['p99_ms']:.2f} ms"
        )
    print(
        "  30 FPS frame budget: "
        f"{derived['frame_budget_30fps_ms']:.2f} ms"
    )
    print(
        "  p50 sync query FPS: "
        f"{derived['fps_from_end_to_end_p50']:.1f}"
    )


def main() -> None:
    args = parse_args()
    report = benchmark(args)
    if args.json_out:
        out_path = Path(args.json_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report)


if __name__ == "__main__":
    main()
