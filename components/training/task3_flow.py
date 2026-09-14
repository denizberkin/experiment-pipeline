"""Conditional flow matching trainer for Task 3 (section 38).

Implements the cross-field finetuning stage of Imre et al. (arXiv:2609.00960): bridge the
registered source slice to the target with a linear path and regress the velocity.

    x_t = (1 - t) x0 + t x1,   t ~ U[0, 1]                                    (their eq. 1)
    L   = E || v_theta(x_t, t, s, tau) - (x1 - x0) ||^2                       (their eq. 2)

Because the path is linear, its velocity is the constant x1 - x0 and needs no numerical
differentiation. The network sees only x_t: x0 enters as the t=0 state of the integration
and never as a separate input. That is the paper's "leak-free" bridge, and it is what keeps
the objective from degenerating -- given x0 alongside x_t a network can recover x1 by
linear algebra, at which point this is regression with extra steps (their Table 1: 0.906).

Sampling at validation uses ``heun_sample`` from the model module, so the integration that
is scored is the same code the submission runs. Section 20.1 is the reason that matters:
0.006 SSIM went missing to a scoring path that had drifted from the training one.

Not implemented here, deliberately: stage 1 (degradation-bridge pretraining) and stage 3
(adversarial refinement). Stage 3 is the paper's largest single gain (+0.024 SSIM, 0.885 ->
0.909) and wants a PatchGAN, a hinge loss, feature matching and backpropagation through the
Euler rollout; it goes in as its own trainer once this one is measured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from components.models.conditional_flow_unet import heun_sample
from eval_pipeline.components.trainers.base import Trainer
from eval_pipeline.context import TrainingContext
from eval_pipeline.registry import register_component
from mrixfields.audit import get_mem_usage, get_train_logger

_LOGGER = None


def audit_logger():
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = get_train_logger()
    return _LOGGER


@register_component("task3_flow_trainer", category="training")
class Task3FlowTrainer(Trainer):
    def train(
        self,
        data: dict[str, Any],
        model: torch.nn.Module,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: TrainingContext,
    ) -> dict[str, Any]:
        del metrics
        device = self._device()
        model = model.to(device)
        loader = data["train"]

        epochs = int(self.params.get("epochs", 10))
        learning_rate = float(self.params.get("learning_rate", 3e-4))
        save_every = int(self.params.get("save_every", 1))
        steps_for_sampling = int(self.params.get("heun_steps", 5))
        use_amp = bool(self.params.get("amp", True)) and device.type == "cuda"
        show_progress = bool(self.params.get("progress", True))
        ema_decay = float(self.params.get("ema_decay", 0.999))

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate,
            betas=(float(self.params.get("beta1", 0.9)), float(self.params.get("beta2", 0.999))),
            weight_decay=float(self.params.get("weight_decay", 0.0)),
        )
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

        # EMA of the weights, as the paper does (decay 0.999). Velocity regression is noisy
        # -- t is resampled every step, so consecutive batches see different points on the
        # path -- and the averaged weights are what gets sampled and saved.
        ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()
               if v.dtype.is_floating_point} if ema_decay > 0 else None

        # The pipeline hands losses over as {name: {"loss_fn": ..., "weight": ...}}, the
        # same shape weighted_reconstruction_loss unpacks. Flow matching has exactly one
        # term, so take the first rather than summing.
        loss_name, spec = next(iter(losses.items()))
        loss_fn = spec["loss_fn"] if isinstance(spec, dict) else spec
        loss_weight = float(spec.get("weight", 1.0)) if isinstance(spec, dict) else 1.0
        print(f"Flow matching: {epochs} epochs, loss {loss_name}, "
              f"AdamW lr={learning_rate:g}, EMA {ema_decay}, sampling with "
              f"{steps_for_sampling} Heun steps", flush=True)

        iteration = 0
        last_loss = 0.0
        for epoch in range(1, epochs + 1):
            model.train()
            running, batches = 0.0, 0
            progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch",
                            mininterval=float(self.params.get("progress_interval", 5.0)),
                            disable=not show_progress)
            for batch in progress:
                x0 = batch["source"].to(device, non_blocking=True)
                x1 = batch["target"].to(device, non_blocking=True)
                source_domain = batch["source_domain"].to(device).long()
                target_domain = batch["target_domain"].to(device).long()

                # One t per sample, not one per batch: a shared t would correlate every
                # example in the batch to the same point on the path and the gradient would
                # see a single time slice per step instead of the whole interval.
                t = torch.rand(x0.shape[0], device=device)
                shaped = t.reshape(-1, 1, 1, 1)
                x_t = (1.0 - shaped) * x0 + shaped * x1
                velocity = x1 - x0

                with torch.amp.autocast(device.type, enabled=use_amp):
                    predicted = model(x_t, t, target_domain, source_domain)
                    loss = loss_fn(predicted, velocity)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if ema is not None:
                    with torch.no_grad():
                        state = model.state_dict()
                        for key, shadow in ema.items():
                            shadow.mul_(ema_decay).add_(state[key].float(), alpha=1 - ema_decay)

                value = float(loss.detach())
                running += value
                batches += 1
                iteration += 1
                progress.set_postfix(loss=f"{value:.4f}", refresh=False)
                audit_logger().info(
                    f"Epoch:{epoch}, Iteration:{iteration}, "
                    f"LR:{optimizer.param_groups[0]['lr']}, Scheduler:None, "
                    f"BatchSize:{x0.shape[0]}, Loss:{value:.6f}, "
                    f"Losses:{{'{loss_name}': {value:.6f}}}, "
                    f"LossWeight:{{'{loss_name}': {loss_weight}}}, MemUsage:{get_mem_usage()}, "
                    f"Stage:flow_matching"
                )

            last_loss = running / max(batches, 1)
            context.tracker.log_loss("total", last_loss, step=epoch, stage="flow_matching")
            context.tracker.log_loss(loss_name, last_loss, step=epoch, stage="flow_matching")
            print(f"Flow matching [{epoch}/{epochs}] velocity_mse={last_loss:.6f}", flush=True)

            # Integrate one batch and compare to the target. The training loss only says
            # the velocity is regressed well pointwise; it says nothing about whether
            # integrating it actually lands on x1, which is the thing being submitted.
            # Their own ablation is the warning: the same weights score 0.817 at one step
            # and 0.909 at five, so a good velocity loss is not a good sample.
            model.eval()
            sampled = heun_sample(model, x0, target_domain, source_domain,
                                  steps=steps_for_sampling)
            identity = (x0 - x1).abs().mean()
            error = (sampled - x1).abs().mean()
            context.tracker.log_loss("sample_l1", float(error), step=epoch, stage="flow_matching")
            print(f"  Heun x{steps_for_sampling} sample L1 {float(error):.6f} "
                  f"(identity {float(identity):.6f})", flush=True)
            audit_logger().info(
                f"[Validation] Epoch:{epoch}, ValTotalLoss:{last_loss:.6f}, "
                f"ValLosses:[{last_loss:.6f}], "
                f"ValMetrics:{{'sample_l1': {float(error):.6f}, "
                f"'identity_l1': {float(identity):.6f}}}"
            )

            if save_every > 0 and epoch % save_every == 0:
                path = self._save(model, ema, optimizer, context.paths.artifacts_dir, epoch)
                audit_logger().info(
                    f"[Checkpoint] Saved to: {path} | Epoch:{epoch}, IsBest:False, "
                    f"ValLoss:{last_loss:.6f}, "
                    f"ValMetrics:{{'sample_l1': {float(error):.6f}}}"
                )

        checkpoint = self._save(model, ema, optimizer, context.paths.artifacts_dir, epochs)
        context.state["model"] = model
        return {"epochs": epochs, "loss": last_loss, "checkpoint": str(checkpoint),
                "heun_steps": steps_for_sampling}

    def _device(self) -> torch.device:
        configured = self.params.get("device")
        if isinstance(configured, list):
            configured = configured[0]
        if isinstance(configured, int):
            configured = f"cuda:{configured}"
        device = torch.device(configured or ("cuda" if torch.cuda.is_available() else "cpu"))
        return torch.device("cpu") if device.type == "cuda" and not torch.cuda.is_available() else device

    @staticmethod
    def _save(model, ema, optimizer, directory: Path, epoch: int) -> Path:
        """Write both the raw and the EMA weights.

        ``model`` is what training resumes from; ``model_ema`` is what should be sampled.
        Saving only the EMA would make the run unresumable, and saving only the raw weights
        would throw away the copy the paper actually evaluates -- so both, named so a loader
        cannot confuse them.
        """
        path = directory / f"task3_flow_{epoch}.pt"
        payload: dict[str, Any] = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "phase": "flow_matching",
            "progress": epoch,
        }
        if ema is not None:
            merged = {k: v.clone() for k, v in model.state_dict().items()}
            merged.update({k: v.clone() for k, v in ema.items()})
            payload["model_ema"] = merged
        torch.save(payload, path)
        return path
