from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from egomimic.pl_utils.pl_data_utils import annotation_collate
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id
from egomimic.rldb.zarr.utils import DataSchematic, set_global_seed


def _build_model_config_tree(cfg):
    model_cfg = copy.deepcopy(cfg.model)
    if (
        "robomimic_model" in model_cfg
        and OmegaConf.is_config(model_cfg.robomimic_model)
        and "data_schematic" in model_cfg.robomimic_model
    ):
        model_cfg.robomimic_model.data_schematic = None
    return OmegaConf.create({"model": model_cfg})


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


def _maybe_use_run_norm_cache(cfg, args: argparse.Namespace):
    if OmegaConf.select(cfg, "norm_stats.precomputed_norm_path", default=None) is not None:
        return
    if args.run_dir is None:
        return
    cache_path = Path(args.run_dir) / "norm_stats" / "norm_stats.json"
    if cache_path.is_file():
        cfg.norm_stats.precomputed_norm_path = str(cache_path)


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        ckpt = Path(args.checkpoint)
    elif args.run_dir is not None:
        ckpt = Path(args.run_dir) / "checkpoints" / "last.ckpt"
    else:
        raise ValueError("Provide --checkpoint when not using --run-dir")

    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def _instantiate_split_datasets(cfg, split_name: str):
    datasets = {}
    split_cfg = cfg.data.train_datasets if split_name == "train" else cfg.data.valid_datasets
    for dataset_name in split_cfg:
        datasets[dataset_name] = hydra.utils.instantiate(split_cfg[dataset_name])
    return datasets


def _infer_data_schematic(cfg, train_datasets):
    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)
    for dataset_name, dataset in train_datasets.items():
        data_schematic.infer_shapes_from_batch(dataset[0])

        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        keymap_cfg = instantiate_copy.resolver.key_map
        km = OmegaConf.to_container(keymap_cfg, resolve=False)
        km["norm_mode"] = True
        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)

        data_schematic.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
            num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
            precomputed_norm_path=OmegaConf.select(
                cfg, "norm_stats.precomputed_norm_path", default=None
            ),
        )
    return data_schematic


def _dim_names(domain: str, dim: int) -> list[str]:
    if dim == 6:
        return ["x", "y", "z", "yaw", "pitch", "roll"]
    if dim == 7:
        return ["x", "y", "z", "yaw", "pitch", "roll", "gripper"]
    return [f"dim_{i}" for i in range(dim)]


@dataclass
class MetricAccumulator:
    dim_names: list[str]
    device: torch.device

    def __post_init__(self):
        dim = len(self.dim_names)
        self.loss_sum = torch.zeros(dim, device=self.device)
        self.mae_sum = torch.zeros(dim, device=self.device)
        self.mse_sum = torch.zeros(dim, device=self.device)
        self.loss_t0_sum = torch.zeros(dim, device=self.device)
        self.mae_t0_sum = torch.zeros(dim, device=self.device)
        self.mse_t0_sum = torch.zeros(dim, device=self.device)
        self.count = 0
        self.count_t0 = 0
        self.num_batches = 0
        self.num_sequences = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, *, beta: float):
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
        self.num_batches += 1
        self.num_sequences += int(pred.shape[0])
        self.count += int(pred.shape[0] * pred.shape[1])
        self.count_t0 += int(pred.shape[0])

        diff = pred - target
        loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
        self.loss_sum += loss.sum(dim=(0, 1))
        self.mae_sum += diff.abs().sum(dim=(0, 1))
        self.mse_sum += diff.square().sum(dim=(0, 1))

        diff_t0 = diff[:, 0, :]
        loss_t0 = loss[:, 0, :]
        self.loss_t0_sum += loss_t0.sum(dim=0)
        self.mae_t0_sum += diff_t0.abs().sum(dim=0)
        self.mse_t0_sum += diff_t0.square().sum(dim=0)

    def finalize(self) -> dict:
        per_dim = {}
        for idx, name in enumerate(self.dim_names):
            per_dim[name] = {
                "smooth_l1_mean": float(self.loss_sum[idx].item() / max(self.count, 1)),
                "mae_mean": float(self.mae_sum[idx].item() / max(self.count, 1)),
                "rmse": float((self.mse_sum[idx].item() / max(self.count, 1)) ** 0.5),
                "t0_smooth_l1_mean": float(
                    self.loss_t0_sum[idx].item() / max(self.count_t0, 1)
                ),
                "t0_mae_mean": float(
                    self.mae_t0_sum[idx].item() / max(self.count_t0, 1)
                ),
                "t0_rmse": float(
                    (self.mse_t0_sum[idx].item() / max(self.count_t0, 1)) ** 0.5
                ),
            }

        return {
            "num_batches": self.num_batches,
            "num_sequences": self.num_sequences,
            "num_timestep_vectors": self.count,
            "overall": {
                "smooth_l1_mean": float(self.loss_sum.sum().item() / max(self.count * len(self.dim_names), 1)),
                "mae_mean": float(self.mae_sum.sum().item() / max(self.count * len(self.dim_names), 1)),
                "rmse": float(
                    (self.mse_sum.sum().item() / max(self.count * len(self.dim_names), 1))
                    ** 0.5
                ),
                "t0_smooth_l1_mean": float(
                    self.loss_t0_sum.sum().item()
                    / max(self.count_t0 * len(self.dim_names), 1)
                ),
                "t0_mae_mean": float(
                    self.mae_t0_sum.sum().item()
                    / max(self.count_t0 * len(self.dim_names), 1)
                ),
                "t0_rmse": float(
                    (
                        self.mse_t0_sum.sum().item()
                        / max(self.count_t0 * len(self.dim_names), 1)
                    )
                    ** 0.5
                ),
            },
            "per_dim": per_dim,
        }


def _make_hpt_batch(algo, processed_domain_batch: dict, embodiment_id: int) -> dict:
    embodiment_name = get_embodiment(embodiment_id).lower()
    cam_keys = algo.camera_keys[embodiment_id]
    proprio_keys = algo.proprio_keys[embodiment_id]
    lang_keys = algo.lang_keys[embodiment_id]
    ac_key = algo.ac_keys[embodiment_id]
    aux_ac_keys = algo.auxiliary_ac_keys.get(embodiment_name, [])
    data = algo._robomimic_to_hpt_data(
        processed_domain_batch,
        cam_keys,
        proprio_keys,
        lang_keys,
        ac_key,
        aux_ac_keys,
    )
    return {"domain": embodiment_name, "data": data}


@torch.no_grad()
def _evaluate_domain(
    *,
    algo,
    data_schematic: DataSchematic,
    domain_name: str,
    dataset,
    loader_params,
    device: torch.device,
    max_batches: int | None,
    shuffle: bool,
    beta: float,
):
    loader = DataLoader(
        dataset,
        batch_size=int(loader_params.batch_size),
        shuffle=shuffle,
        num_workers=int(loader_params.num_workers),
        collate_fn=annotation_collate,
        pin_memory=bool(getattr(loader_params, "pin_memory", False)),
        persistent_workers=bool(getattr(loader_params, "persistent_workers", False))
        if int(loader_params.num_workers) > 0
        else False,
        prefetch_factor=(
            int(loader_params.prefetch_factor)
            if int(loader_params.num_workers) > 0
            and getattr(loader_params, "prefetch_factor", None) is not None
            else None
        ),
    )

    norm_acc = None
    raw_acc = None
    progress_total = max_batches if max_batches is not None else len(loader)
    for batch_idx, raw_domain_batch in enumerate(
        tqdm(loader, desc=f"{domain_name}", total=progress_total)
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break

        processed = algo.process_batch_for_training({domain_name: raw_domain_batch})
        embodiment_id = next(iter(processed.keys()))
        processed_domain_batch = processed[embodiment_id]
        embodiment_name = get_embodiment(embodiment_id).lower()
        ac_key = algo.ac_keys[embodiment_id]
        hpt_batch = _make_hpt_batch(algo, processed_domain_batch, embodiment_id)
        actions = algo.nets["policy"].forward(hpt_batch["domain"], hpt_batch["data"])

        pred_norm = actions[embodiment_name]
        target_norm = processed_domain_batch[ac_key]
        _, horizon, dim = target_norm.shape
        pred_norm = pred_norm[:, :horizon, :dim]

        pred_raw = data_schematic.unnormalize_data({ac_key: pred_norm}, embodiment_id)[ac_key]
        target_raw = raw_domain_batch[ac_key].to(device).float()

        if norm_acc is None:
            names = _dim_names(domain_name, dim)
            norm_acc = MetricAccumulator(names, device)
            raw_acc = MetricAccumulator(names, device)

        norm_acc.update(pred_norm, target_norm, beta=beta)
        raw_acc.update(pred_raw, target_raw, beta=beta)

    if norm_acc is None or raw_acc is None:
        raise ValueError(f"No batches processed for domain {domain_name}")

    return {
        "normalized": norm_acc.finalize(),
        "raw_units": raw_acc.finalize(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect per-domain and per-dimension checkpoint errors for cotraining runs."
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        choices=["train", "valid"],
    )
    parser.add_argument("--max-train-batches", type=int, default=256)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    parser.add_argument("--train-shuffle", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    set_global_seed(args.seed)

    cfg, config_path = _load_cfg(args)
    _maybe_use_run_norm_cache(cfg, args)
    ckpt_path = _resolve_checkpoint(args)
    device = torch.device(args.device)

    train_datasets = _instantiate_split_datasets(cfg, "train")
    valid_datasets = _instantiate_split_datasets(cfg, "valid")
    data_schematic = _infer_data_schematic(cfg, train_datasets)

    wrapper = ModelWrapper(
        config_tree=_build_model_config_tree(cfg),
        data_schematic_state=data_schematic.to_state(),
        viz_func=None,
        scheduler_interval=cfg.model.get("scheduler_interval", "step"),
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = wrapper.load_state_dict(checkpoint["state_dict"], strict=False)
    wrapper = wrapper.to(device)
    wrapper.model.device = device
    wrapper.model.nets["policy"].device = device
    wrapper.eval()
    wrapper.model.nets.eval()

    beta = 0.05
    results = {}
    for split_name in args.splits:
        split_datasets = train_datasets if split_name == "train" else valid_datasets
        split_loader_params = (
            cfg.data.train_dataloader_params
            if split_name == "train"
            else cfg.data.valid_dataloader_params
        )
        max_batches = args.max_train_batches if split_name == "train" else args.max_valid_batches
        split_results = {}
        for domain_name, dataset in split_datasets.items():
            split_results[domain_name] = _evaluate_domain(
                algo=wrapper.model,
                data_schematic=data_schematic,
                domain_name=domain_name,
                dataset=dataset,
                loader_params=split_loader_params[domain_name],
                device=device,
                max_batches=max_batches,
                shuffle=bool(args.train_shuffle and split_name == "train"),
                beta=beta,
            )
        results[split_name] = split_results

    payload = {
        "config_path": str(config_path),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "seed": args.seed,
        "device": str(device),
        "results": results,
    }

    text = json.dumps(payload, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
