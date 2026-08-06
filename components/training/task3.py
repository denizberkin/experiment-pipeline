from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from eval_pipeline.components.trainers.base import Trainer
from eval_pipeline.context import TrainingContext
from eval_pipeline.registry import register_component


def block_mask(images: torch.Tensor, ratio: float, patch_size: int) -> torch.Tensor:
    if not 0.0 < ratio < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")
    if patch_size < 1:
        raise ValueError("mask_patch_size must be positive")
    height, width = images.shape[-2:]
    grid = torch.rand(
        images.size(0),
        1,
        (height + patch_size - 1) // patch_size,
        (width + patch_size - 1) // patch_size,
        device=images.device,
    ) < ratio
    return grid.repeat_interleave(patch_size, 2).repeat_interleave(patch_size, 3)[..., :height, :width]


def weighted_reconstruction_loss(
    losses: dict[str, Any], prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    if not losses:
        raise ValueError("Task 3 training requires at least one configured loss")
    total = prediction.new_zeros(())
    values = {}
    for name, item in losses.items():
        loss = item["loss_fn"](prediction, target)
        total = total + float(item.get("weight", 1.0)) * loss
        values[name] = float(loss.detach())
    return total, values


@register_component("task3_unet_trainer", category="training")
class Task3UNetTrainer(Trainer[dict[str, Any], torch.nn.Module, dict[str, Any]]):
    def train(
        self,
        *,
        data: dict[str, Any],
        model: torch.nn.Module,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: TrainingContext,
    ) -> dict[str, Any]:
        del metrics
        device = self._device()
        model = model.to(device)
        result: dict[str, Any] = {"device": str(device)}

        pretrain_steps = int(self.params.get("pretrain_steps", 0))
        if pretrain_steps:
            if "pretrain" not in data:
                raise ValueError("pretrain_steps > 0 requires data.params.include_retro=true")
            result["pretrain"] = self._pretrain(
                model, data["pretrain"], device, pretrain_steps, context
            )

        result["finetune"] = self._finetune(model, data["train"], losses, device, context)
        context.state["model"] = model
        return result

    def _pretrain(
        self,
        model: torch.nn.Module,
        loader: Any,
        device: torch.device,
        steps: int,
        context: TrainingContext,
    ) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.params.get("pretrain_learning_rate", 1e-4)),
            betas=(
                float(self.params.get("pretrain_beta1", 0.0)),
                float(self.params.get("pretrain_beta2", 0.99)),
            ),
            weight_decay=float(self.params.get("pretrain_weight_decay", 1e-4)),
        )
        ratio = float(self.params.get("mask_ratio", 0.5))
        patch_size = int(self.params.get("mask_patch_size", 16))
        save_every = int(self.params.get("pretrain_save_every", 10000))
        print_every = max(1, int(self.params.get("pretrain_print_every", 1000)))
        step = 0
        last_loss = 0.0
        use_amp = bool(self.params.get("amp", True)) and device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        model.train()
        print(f"Pretraining: {steps} steps", flush=True)

        while step < steps:
            for batch in loader:
                images = batch["image"].to(device)
                domains = batch["domain"].to(device).long()
                mask = block_mask(images, ratio, patch_size)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    prediction = model(images.masked_fill(mask, 0.0), domains)
                    selected = mask.expand_as(images) & (images > -0.99)
                    if not selected.any():
                        selected = mask.expand_as(images)
                    loss = (prediction - images).abs()[selected].mean()

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1
                last_loss = float(loss.detach())

                if step % print_every == 0 or step == steps:
                    context.tracker.log_loss("masked_l1", last_loss, step=step, stage="pretraining")
                    print(f"Pretraining [{step}/{steps}] masked_l1={last_loss:.6f}", flush=True)
                if save_every > 0 and step % save_every == 0:
                    checkpoint = self._save(model, optimizer, context.paths.artifacts_dir, "pretrain", step)
                    context.tracker.log_artifact(checkpoint, artifact_path="checkpoints")
                if step >= steps:
                    break

        checkpoint = self._save(model, optimizer, context.paths.artifacts_dir, "pretrain", step)
        context.tracker.log_artifact(checkpoint, artifact_path="checkpoints")
        return {"steps": step, "loss": last_loss, "checkpoint": str(checkpoint)}

    def _finetune(
        self,
        model: torch.nn.Module,
        loader: Any,
        losses: dict[str, Any],
        device: torch.device,
        context: TrainingContext,
    ) -> dict[str, Any]:
        epochs = int(self.params.get("epochs", 50))
        if epochs < 1:
            raise ValueError("epochs must be positive")
        save_every = int(self.params.get("save_every", 10))
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.params.get("learning_rate", 5e-5)),
            betas=(float(self.params.get("beta1", 0.5)), float(self.params.get("beta2", 0.999))),
        )
        last_loss = 0.0
        checkpoint: Path | None = None
        use_amp = bool(self.params.get("amp", True)) and device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        print(f"Fine-tuning: {epochs} epochs", flush=True)
        show_progress = bool(self.params.get("progress", True))
        progress_interval = float(self.params.get("progress_interval", 5.0))

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            running_values = {name: 0.0 for name in losses}
            batches = 0
            progress = tqdm(
                loader,
                desc=f"Epoch {epoch}/{epochs}",
                unit="batch",
                mininterval=progress_interval,
                disable=not show_progress,
            )
            for batch in progress:
                source = batch["source"].to(device)
                target = batch["target"].to(device)
                source_domain = batch["source_domain"].to(device).long()
                target_domain = batch["target_domain"].to(device).long()
                with torch.amp.autocast(device.type, enabled=use_amp):
                    prediction = model(source, target_domain, source_domain)
                    loss, values = weighted_reconstruction_loss(losses, prediction, target)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.detach())
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}", refresh=False)
                for name, value in values.items():
                    running_values[name] += value
                batches += 1

            last_loss = running_loss / max(batches, 1)
            context.tracker.log_loss("total", last_loss, step=epoch, stage="finetuning")
            for name, value in running_values.items():
                context.tracker.log_loss(name, value / max(batches, 1), step=epoch, stage="finetuning")
            print(f"Fine-tuning [{epoch}/{epochs}] total={last_loss:.6f}", flush=True)
            if save_every > 0 and epoch % save_every == 0:
                checkpoint = self._save(model, optimizer, context.paths.artifacts_dir, "finetune", epoch)
                context.tracker.log_artifact(checkpoint, artifact_path="checkpoints")

        checkpoint = self._save(model, optimizer, context.paths.artifacts_dir, "finetune", epochs)
        context.tracker.log_artifact(checkpoint, artifact_path="checkpoints")
        return {"epochs": epochs, "loss": last_loss, "checkpoint": str(checkpoint)}

    def _device(self) -> torch.device:
        configured = self.params.get("device")
        if isinstance(configured, list):
            configured = configured[0]
            if isinstance(configured, int):
                configured = f"cuda:{configured}"
        device = torch.device(configured or ("cuda" if torch.cuda.is_available() else "cpu"))
        return torch.device("cpu") if device.type == "cuda" and not torch.cuda.is_available() else device

    @staticmethod
    def _save(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        directory: Path,
        phase: str,
        progress: int,
    ) -> Path:
        path = directory / f"task3_unet_{phase}_{progress}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "phase": phase,
                "progress": progress,
            },
            path,
        )
        return path
