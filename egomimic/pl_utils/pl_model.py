import random
import time
from collections import OrderedDict, deque
from fnmatch import fnmatch
from typing import Any, Dict

import hydra
import numpy as np
import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

import egomimic.utils.tensor_utils as TensorUtils
from egomimic.rldb.zarr.utils import DataSchematic


def _barrier_if_distributed(rank: int, label: str) -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    print(
        f"Rank {rank} on {label}, waiting for all ranks to synchronize",
        flush=True,
    )
    torch.distributed.barrier()
    print(
        f"Rank {rank} on {label}, all ranks synchronized",
        flush=True,
    )


class ModelWrapper(LightningModule):
    """
    Wrapper class around robomimic models to ensure compatibility with Pytorch Lightning.
    """

    debug_loss_spike = False
    debug_loss_spike_factor = 1000.0
    debug_loss_spike_prob = 0.03
    grad_norm_mad_scale = 3.0
    grad_norm_mad_min_count = 100
    grad_norm_mad_window = 200

    def __init__(
        self,
        robomimic_model=None,
        optimizer=None,
        scheduler=None,
        config_tree=None,
        data_schematic_state=None,
        scheduler_interval="step",
        scheduler_frequency: int = 1,
        viz_func=None,
        evaluator=None,
        enable_grad_norm: bool = True,
        trainable_parameter_patterns: list[str] | None = None,
        frozen_parameter_patterns: list[str] | None = None,
    ):
        """
        Args:
            model (PolicyAlgo): robomimic model to wrap.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["robomimic_model", "viz_func"])

        if config_tree is not None:
            self.model = self._instantiate_model(
                config_tree, data_schematic_state, viz_func
            )
        elif robomimic_model is not None:  # legacy support
            self.model = robomimic_model
        else:
            raise ValueError(
                "ModelWrapper requires either an instantiated robomimic_model or "
                "a config_tree with data_schematic_state."
            )
        self.nets = (
            self.model.nets
        )  # to ensure the lightning module has access to the model's parameters
        try:
            self.params = self.model.nets["policy"].params
        except Exception:
            pass
        self._apply_parameter_trainability(
            trainable_parameter_patterns=trainable_parameter_patterns,
            frozen_parameter_patterns=frozen_parameter_patterns,
        )
        self.enable_grad_norm = enable_grad_norm
        self.grad_norm_history = deque(maxlen=self.grad_norm_mad_window)

        self.epoch_memory_stats = []  # Store memory stats per epoch
        self.evaluator = evaluator

    @staticmethod
    def _as_config(cfg):
        if cfg is None:
            return None
        if isinstance(cfg, DictConfig):
            return cfg
        return OmegaConf.create(cfg)

    def _instantiate_model(self, config_tree, data_schematic_state, viz_func=None):
        cfg = self._as_config(config_tree)
        data_schematic = DataSchematic.from_state(data_schematic_state)
        return hydra.utils.instantiate(
            cfg.model.robomimic_model,
            data_schematic=data_schematic,
            viz_func=viz_func,
        )

    @staticmethod
    def _pattern_matches(name: str, patterns: list[str]) -> bool:
        return any(fnmatch(name, pattern) or pattern in name for pattern in patterns)

    def _apply_parameter_trainability(
        self,
        *,
        trainable_parameter_patterns: list[str] | None,
        frozen_parameter_patterns: list[str] | None,
    ) -> None:
        trainable_patterns = list(trainable_parameter_patterns or [])
        frozen_patterns = list(frozen_parameter_patterns or [])
        if not trainable_patterns and not frozen_patterns:
            return

        if trainable_patterns:
            for _, param in self.named_parameters():
                param.requires_grad = False
            for name, param in self.named_parameters():
                if self._pattern_matches(name, trainable_patterns):
                    param.requires_grad = True

        if frozen_patterns:
            for name, param in self.named_parameters():
                if self._pattern_matches(name, frozen_patterns):
                    param.requires_grad = False

        trainable = [
            (name, param.numel())
            for name, param in self.named_parameters()
            if param.requires_grad
        ]
        frozen = sum(
            param.numel() for _, param in self.named_parameters() if not param.requires_grad
        )
        trainable_count = sum(numel for _, numel in trainable)
        if trainable_patterns and trainable_count == 0:
            raise ValueError(
                "trainable_parameter_patterns matched no parameters: "
                f"{trainable_patterns}"
            )

        print(
            "[ModelWrapper] trainable parameters: "
            f"{trainable_count:,}; frozen parameters: {frozen:,}; "
            f"patterns={trainable_patterns or '<unchanged>'}",
            flush=True,
        )

    def _optimizer_parameters(self):
        params = [p for p in self.trainer.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters available for optimizer")
        return params

    # batch is now a dict, handle on model side
    def training_step(self, batch, batch_idx):
        self.train()
        loss_dicts = []

        t0 = time.time()
        batch = self.model.process_batch_for_training(batch)
        t1 = time.time()
        predictions = self.model.forward_training(batch)
        t2 = time.time()
        losses = self.model.compute_losses(predictions, batch)
        t3 = time.time()
        loss_dicts.append(losses)

        self.log(
            "Timing/Process_Batch_Sec",
            t1 - t0,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Timing/Forward_Pass_Sec",
            t2 - t1,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Timing/Compute_Losses_Sec",
            t3 - t2,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Average over both the hand and robot batch if applicable
        losses = OrderedDict()
        for key in loss_dicts[0].keys():
            losses[key] = torch.mean(
                torch.stack([loss_dict[key] for loss_dict in loss_dicts])
            )

        if (
            self.debug_loss_spike
            and random.random() < self.debug_loss_spike_prob
            and self.global_step > 100
        ):
            losses["action_loss"] = losses["action_loss"] * self.debug_loss_spike_factor
            if self.trainer.is_global_zero:
                print(
                    f"[LOSS_SPIKE] step={self.global_step} factor={self.debug_loss_spike_factor}",
                    flush=True,
                )

        info = {}
        info["losses"] = TensorUtils.detach(losses)
        for k, v in self.model.log_info(info).items():
            self.log("Train/" + k, v, sync_dist=True, on_step=False, on_epoch=True)

        return losses["action_loss"]

    def on_after_backward(self):
        if not self.enable_grad_norm:
            return
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        grad_norm_val = float(grad_norm)
        info = {"policy_grad_norms_raw": grad_norm_val}
        grad_norm_flagged = False

        if len(self.grad_norm_history) >= self.grad_norm_mad_min_count:
            values = np.array(self.grad_norm_history, dtype=np.float32)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            if mad > 0.0:
                threshold = median + self.grad_norm_mad_scale * mad
                info["policy_grad_norms_mad_threshold"] = threshold
                grad_norm_flagged = grad_norm_val > threshold
                info["policy_grad_norms_mad_flag"] = float(grad_norm_flagged)
                if grad_norm_flagged:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=median)
                    if self.trainer.is_global_zero:
                        print(
                            "[GRAD_NORM_SPIKE] "
                            f"step={self.global_step} "
                            f"grad_norm={grad_norm_val:.4f} "
                            f"median={median:.4f} "
                            f"mad={mad:.4f} "
                            f"threshold={threshold:.4f}",
                            flush=True,
                        )

        if not grad_norm_flagged:
            self.grad_norm_history.append(grad_norm_val)
        for k, v in info.items():
            self.log("Train/" + k, v, on_step=False, on_epoch=True, sync_dist=True)

    def on_before_optimizer_step(self, optimizer):
        if not self.enable_grad_norm:
            return
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        self.log(
            "Train/policy_grad_norms_clipped",
            float(grad_norm),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def on_validation_start(self):
        if self.evaluator is None:
            return
        self.model.device = self.device

        self.evaluator.on_validation_start()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Run a validation step on the batch, and save that batch of images into the val_image_buffer.  Once the buffer hits 1000 images, save that as a 30fps video using torchvision.io.write_video.
        """
        batch = self.model.process_batch_for_training(batch)
        predictions = self.model.forward_training(batch)
        losses = self.model.compute_losses(predictions, batch)

        info = {"losses": TensorUtils.detach(losses)}
        for k, v in self.model.log_info(info).items():
            self.log("Val/" + k, v, sync_dist=True, on_step=False, on_epoch=True)

        if self.evaluator is not None:
            print(
                f"[VAL_STEP] rank={self.global_rank}, batch_idx={batch_idx}",
                flush=True,
            )
            self.evaluator.on_validation_step(batch, batch_idx, dataloader_idx)

        return losses["action_loss"]

    def on_validation_end(self):
        print(f"[ON_VALIDATION_END] rank={self.global_rank}", flush=True)
        if self.evaluator is not None:
            self.evaluator.on_validation_end()

        _barrier_if_distributed(self.global_rank, "validation end")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        config_tree = getattr(self.hparams, "config_tree", None)
        if config_tree is not None:
            cfg = self._as_config(config_tree)
            optimizer = hydra.utils.instantiate(
                cfg.model.optimizer,
                params=self._optimizer_parameters(),
            )
            if callable(optimizer):
                optimizer = optimizer()
            scheduler_cfg = cfg.model.get("scheduler")
            if scheduler_cfg is not None:
                scheduler = hydra.utils.instantiate(
                    scheduler_cfg,
                    optimizer=optimizer,
                )
                if callable(scheduler):
                    scheduler = scheduler()
            else:
                scheduler = None
        else:
            optimizer = self.hparams.optimizer(params=self._optimizer_parameters())
            scheduler = (
                self.hparams.scheduler(optimizer=optimizer)
                if self.hparams.scheduler is not None
                else None
            )

        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler_interval,
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}

    def on_fit_start(self):
        self.model.device = self.device
        _barrier_if_distributed(self.global_rank, "fit start")

    def on_train_epoch_start(self):
        for i, param_group in enumerate(self.optimizers().param_groups):
            self.log(
                f"Optimizer/param_group_{i}_lr",
                param_group["lr"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return super().on_train_epoch_start()
