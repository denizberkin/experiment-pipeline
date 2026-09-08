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

        result["finetune"] = self._finetune(
            model, data["train"], losses, device, context, data.get("validation")
        )
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
        log_checkpoints = bool(self.params.get("log_checkpoints", True))
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
                    if log_checkpoints:
                        context.tracker.log_artifact(checkpoint, artifact_path="checkpoints")
                if step >= steps:
                    break

        checkpoint = self._save(model, optimizer, context.paths.artifacts_dir, "pretrain", step)
        if log_checkpoints:
            context.tracker.log_artifact(checkpoint, artifact_path="checkpoints")
        return {"steps": step, "loss": last_loss, "checkpoint": str(checkpoint)}

    def _finetune(
        self,
        model: torch.nn.Module,
        loader: Any,
        losses: dict[str, Any],
        device: torch.device,
        context: TrainingContext,
        validation_loader: Any = None,
    ) -> dict[str, Any]:
        epochs = int(self.params.get("epochs", 50))
        if epochs < 1:
            raise ValueError("epochs must be positive")
        save_every = int(self.params.get("save_every", 10))
        # log_artifact copies each checkpoint into mlruns/, so a tracked run costs
        # twice its checkpoints on disk. The copy is byte-identical to the one in
        # runs/<name>/artifacts/ and only feeds the MLflow artifact browser, so it
        # is the first thing to drop when space is tight. Default stays True so
        # existing configs are unaffected.
        log_checkpoints = bool(self.params.get("log_checkpoints", True))
        learning_rate = float(self.params.get("learning_rate", 5e-5))
        encoder_scale = float(self.params.get("encoder_lr_scale", 1.0))
        # A model carrying a pretrained backbone can expose its own parameter
        # groups so the encoder steps slower than the freshly initialised
        # decoder. Anything else keeps the original single-group optimizer.
        if encoder_scale != 1.0 and hasattr(model, "optimizer_param_groups"):
            parameters = model.optimizer_param_groups(learning_rate, encoder_scale)
            print(f"Discriminative lr: decoder={learning_rate:g} encoder={learning_rate * encoder_scale:g}", flush=True)
        else:
            parameters = model.parameters()
        optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=(float(self.params.get("beta1", 0.5)), float(self.params.get("beta2", 0.999))),
            weight_decay=float(self.params.get("weight_decay", 0.0)),
        )
        last_loss = 0.0
        checkpoint: Path | None = None
        use_amp = bool(self.params.get("amp", True)) and device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        print(f"Fine-tuning: {epochs} epochs", flush=True)
        show_progress = bool(self.params.get("progress", True))
        progress_interval = float(self.params.get("progress_interval", 5.0))

        # Early stopping. patience <= 0 disables it entirely, preserving the
        # original behaviour of running every epoch.
        patience = int(self.params.get("early_stopping_patience", 0))
        min_delta = float(self.params.get("early_stopping_min_delta", 0.0))
        restore_best = bool(self.params.get("early_stopping_restore_best", True))
        best_train = float("inf")
        train_wait = 0
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stopped_early = False
        stop_reason = None
        if patience > 0:
            configured = str(self.params.get("early_stopping_monitor", "validation"))
            monitor = configured if validation_loader is not None else "train"
            note = "" if validation_loader is not None else "  (no validation loader)"
            print(
                f"Early stopping: monitor={monitor} patience={patience} "
                f"min_delta={min_delta:g} restore_best={restore_best}{note}",
                flush=True,
            )

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

            # A held-out subject is the only signal that can actually see
            # overfitting; training loss falls right through it.
            monitored = last_loss
            if validation_loader is not None:
                model.eval()
                total, seen = 0.0, 0
                with torch.no_grad():
                    for batch in validation_loader:
                        with torch.amp.autocast(device.type, enabled=use_amp):
                            prediction = model(
                                batch["source"].to(device),
                                batch["target_domain"].to(device).long(),
                                batch["source_domain"].to(device).long(),
                            )
                            loss, _ = weighted_reconstruction_loss(
                                losses, prediction, batch["target"].to(device)
                            )
                        total += float(loss.detach())
                        seen += 1
                validation_loss = total / max(seen, 1)
                context.tracker.log_loss("validation", validation_loss, step=epoch, stage="finetuning")
                print(f"Validation [{epoch}/{epochs}] total={validation_loss:.6f}", flush=True)
                if str(self.params.get("early_stopping_monitor", "validation")) == "validation":
                    monitored = validation_loss

            if patience > 0:
                # best/best_epoch/best_state must move together, so the
                # restored weights are the ones the reported loss belongs to.
                if monitored < best_train - min_delta:
                    best_train = monitored
                    best_epoch = epoch
                    if restore_best:
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }
                    train_wait = 0
                else:
                    train_wait += 1

                # With a holdout_subjects loader this watches validation loss and
                # so does stop on overfitting; without one it falls back to the
                # training loss and stops only on convergence. Name whichever it
                # actually watched -- a run that reports the wrong one cannot be
                # told apart from a run that had no held-out signal at all.
                if train_wait >= patience:
                    stop_reason = f"{monitor} loss flat for {patience} epochs"
                    stopped_early = True
                    print(
                        f"Early stopping at epoch {epoch}/{epochs}: {stop_reason}. "
                        f"Best epoch {best_epoch}.",
                        flush=True,
                    )

            if save_every > 0 and epoch % save_every == 0:
                checkpoint = self._save(model, optimizer, context.paths.artifacts_dir, "finetune", epoch)
                if log_checkpoints:
                    context.tracker.log_artifact(checkpoint, artifact_path="checkpoints")

            if stopped_early:
                completed_epochs = epoch
                break
        else:
            completed_epochs = epochs

        if restore_best and best_state is not None and best_epoch != completed_epochs:
            model.load_state_dict(best_state)
            print(f"Restored best weights from epoch {best_epoch}.", flush=True)

        checkpoint = self._save(
            model, optimizer, context.paths.artifacts_dir, "finetune", completed_epochs
        )
        if log_checkpoints:
            context.tracker.log_artifact(checkpoint, artifact_path="checkpoints")
        summary: dict[str, Any] = {
            "epochs": completed_epochs,
            "loss": last_loss,
            "checkpoint": str(checkpoint),
        }
        if patience > 0:
            summary["stopped_early"] = stopped_early
            summary["best_epoch"] = best_epoch
            if stop_reason:
                summary["stop_reason"] = stop_reason
            if best_train != float("inf"):
                summary[f"best_{monitor}_loss"] = best_train
        return summary


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
