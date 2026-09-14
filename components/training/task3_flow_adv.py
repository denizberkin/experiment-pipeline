"""Stage 3: adversarial refinement of a trained flow-matching model (section 38.4).

After Imre et al. (arXiv:2609.00960) section 2.5. This is the stage that carries their
model from 0.885 to 0.909 -- +0.024 SSIM, the largest single gain in their pipeline, and
per section 38.6 the only part that clears our 0.909353. Without it the CFM objective lands
at 0.885-0.894 on two independent measurements of theirs.

Why a squared-error velocity regression needs this at all: it fits the conditional *mean*,
so wherever the mapping is uncertain the model averages the plausible high-frequency detail
and the result is blur. No reweighting of the regression fixes that -- the mean is what L2
is for. An adversarial term supplies the missing objective: not "be close to the target" but
"be indistinguishable from a real acquisition".

Three details that are not decoration:

*The rollout is differentiable end to end.* The generator produces its sample by Euler
integration from x0, and the gradient flows back through every step, so the critic refines
the whole transport rather than the endpoint. This is also why the refined field stops
working in one step (section 38.6: `g_1` is the worst of their four configurations) -- it is
calibrated as a trajectory.

*Euler here, Heun at inference.* Heun costs two network evaluations per step; during
training, with the graph retained across the rollout, that doubles both memory and time for
an integration accuracy the adversarial gradient does not need. The paper makes the same
split.

*The velocity-regression anchor stays on, at weight 1.* It ties the output to the correct
translation. Adversarial pressure alone will happily produce a sharp, plausible, wrong
brain; the anchor is what keeps reconstruction error improving under refinement rather than
trading it away for texture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from components.models.patchgan import (
    MultiScalePatchDiscriminator,
    feature_matching_loss,
    hinge_discriminator_loss,
    hinge_generator_loss,
)
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


def euler_rollout(model, x0: torch.Tensor, target_domain: torch.Tensor,
                  source_domain: torch.Tensor, steps: int) -> torch.Tensor:
    """Integrate t: 0 -> 1 with Euler, keeping the graph. Never conditioned on the target.

    That last point is what makes this match inference: the rollout only ever sees where it
    has got to, so the refinement is judged on the trajectory the submission will actually
    produce, not on a teacher-forced one.
    """
    x = x0
    h = 1.0 / steps
    for index in range(steps):
        t = torch.full((x.shape[0],), index * h, device=x.device, dtype=torch.float32)
        x = x + h * model(x, t, target_domain, source_domain)
    return x


@register_component("task3_flow_adv_trainer", category="training")
class Task3FlowAdversarialTrainer(Trainer):
    def train(
        self,
        data: dict[str, Any],
        model: torch.nn.Module,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: TrainingContext,
    ) -> dict[str, Any]:
        del metrics, losses
        device = self._device()
        generator = model.to(device)
        loader = data["train"]

        # Stage 3 refines a converged stage-2 flow; from random init the critic would be
        # shaping noise and the comparison against stage 2 would not be like-for-like.
        init_from = self.params.get("init_from")
        if init_from:
            state = torch.load(init_from, map_location=device, weights_only=False)
            weights = state.get("model_ema") or state.get("model") or state
            missing, unexpected = generator.load_state_dict(weights, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"init_from={init_from} does not match the generator: "
                    f"{len(missing)} missing, {len(unexpected)} unexpected"
                )
            print(f"generator warm-started from {init_from}"
                  f" [{'model_ema' if 'model_ema' in state else 'model'}]")

        channels = int(self.params.get("channels", 3))
        scales = int(self.params.get("discriminator_scales", 2))
        rollout_steps = int(self.params.get("rollout_steps", 5))
        epochs = int(self.params.get("epochs", 10))
        save_every = int(self.params.get("save_every", 1))
        warmup_steps = int(self.params.get("discriminator_warmup", 5000))
        ramp_steps = int(self.params.get("adversarial_ramp", 2000))
        adversarial_weight = float(self.params.get("adversarial_weight", 1e-4))
        feature_weight = float(self.params.get("feature_matching_weight", 10.0))
        anchor_weight = float(self.params.get("velocity_weight", 1.0))
        generator_lr = float(self.params.get("generator_learning_rate", 1e-5))
        discriminator_lr = float(self.params.get("discriminator_learning_rate", 3e-4))
        ema_decay = float(self.params.get("ema_decay", 0.999))
        use_amp = bool(self.params.get("amp", True)) and device.type == "cuda"
        show_progress = bool(self.params.get("progress", True))

        # in_channels = source stack + candidate stack: the critic is conditional on x0.
        discriminator = MultiScalePatchDiscriminator(
            in_channels=channels * 2, scales=scales,
            base_channels=int(self.params.get("discriminator_channels", 64)),
            num_domains=int(self.params.get("num_domains", 5)),
        ).to(device)

        optimiser_g = torch.optim.AdamW(generator.parameters(), lr=generator_lr, betas=(0.5, 0.9))
        optimiser_d = torch.optim.AdamW(discriminator.parameters(), lr=discriminator_lr,
                                        betas=(0.5, 0.9))
        scaler_g = torch.amp.GradScaler(device.type, enabled=use_amp)
        scaler_d = torch.amp.GradScaler(device.type, enabled=use_amp)

        ema = {k: v.detach().clone().float() for k, v in generator.state_dict().items()
               if v.dtype.is_floating_point} if ema_decay > 0 else None

        # Their 5,000-step warm-up and 2,000-step ramp are 10% and 4% of a 50,000-step run
        # at batch 8. Taken literally at a different batch size they can exceed the whole
        # run, and the generator then never updates -- silently, since the only symptom is
        # a feature-matching loss that reads 0.00 forever. Clamp to the same fractions.
        total_steps = epochs * len(loader)
        if warmup_steps >= total_steps:
            clamped = max(1, int(0.10 * total_steps))
            print(f"WARNING: discriminator_warmup={warmup_steps} >= {total_steps} total "
                  f"steps; the generator would never update. Clamping to {clamped} (10%).",
                  flush=True)
            warmup_steps = clamped
        if ramp_steps >= total_steps:
            ramp_steps = max(1, int(0.04 * total_steps))

        print(f"Adversarial refinement: {epochs} epochs, rollout {rollout_steps} Euler steps, "
              f"G lr={generator_lr:g} D lr={discriminator_lr:g}, warmup {warmup_steps} steps, "
              f"adv {adversarial_weight:g} ramped over {ramp_steps}, fm {feature_weight:g}",
              flush=True)

        iteration = 0
        generator_steps = 0
        last = {}
        for epoch in range(1, epochs + 1):
            generator.train()
            discriminator.train()
            totals = {"d": 0.0, "g_adv": 0.0, "g_fm": 0.0, "g_vel": 0.0}
            batches = 0
            progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch",
                            mininterval=float(self.params.get("progress_interval", 5.0)),
                            disable=not show_progress)
            for batch in progress:
                x0 = batch["source"].to(device, non_blocking=True)
                x1 = batch["target"].to(device, non_blocking=True)
                s = batch["source_domain"].to(device).long()
                tau = batch["target_domain"].to(device).long()
                iteration += 1

                # ---- discriminator ----
                with torch.amp.autocast(device.type, enabled=use_amp):
                    with torch.no_grad():
                        fake = euler_rollout(generator, x0, tau, s, rollout_steps)
                    real_logits, _ = discriminator(torch.cat((x0, x1), 1), s, tau)
                    fake_logits, _ = discriminator(torch.cat((x0, fake.detach()), 1), s, tau)
                    loss_d = hinge_discriminator_loss(real_logits, fake_logits)
                optimiser_d.zero_grad(set_to_none=True)
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimiser_d)
                scaler_d.update()
                totals["d"] += float(loss_d.detach())

                # ---- generator, once the critic is worth listening to ----
                loss_g_adv = loss_g_fm = 0.0
                if iteration > warmup_steps:
                    generator_steps += 1
                    # Ramp from zero: a cold critic's gradient is noise, and at weight 1e-4
                    # against an anchor at 1 there is no room to absorb a bad first push.
                    ramp = min(1.0, generator_steps / max(ramp_steps, 1))
                    with torch.amp.autocast(device.type, enabled=use_amp):
                        sampled = euler_rollout(generator, x0, tau, s, rollout_steps)
                        fake_logits, fake_features = discriminator(
                            torch.cat((x0, sampled), 1), s, tau)
                        with torch.no_grad():
                            _, real_features = discriminator(torch.cat((x0, x1), 1), s, tau)

                        # The faithfulness anchor, on a fresh random t rather than the
                        # rollout: it is the same CFM objective stage 2 used, so refinement
                        # cannot drift off the translation it was trained for.
                        t = torch.rand(x0.shape[0], device=device)
                        shaped = t.reshape(-1, 1, 1, 1)
                        x_t = (1.0 - shaped) * x0 + shaped * x1
                        velocity = torch.nn.functional.mse_loss(
                            generator(x_t, t, tau, s), x1 - x0)

                        adversarial = hinge_generator_loss(fake_logits)
                        matching = feature_matching_loss(real_features, fake_features)
                        loss_g = (anchor_weight * velocity
                                  + adversarial_weight * ramp * adversarial
                                  + feature_weight * matching)
                    optimiser_g.zero_grad(set_to_none=True)
                    scaler_g.scale(loss_g).backward()
                    scaler_g.step(optimiser_g)
                    scaler_g.update()
                    loss_g_adv, loss_g_fm = float(adversarial.detach()), float(matching.detach())
                    totals["g_adv"] += loss_g_adv
                    totals["g_fm"] += loss_g_fm
                    totals["g_vel"] += float(velocity.detach())

                    if ema is not None:
                        with torch.no_grad():
                            state = generator.state_dict()
                            for key, shadow in ema.items():
                                shadow.mul_(ema_decay).add_(state[key].float(), alpha=1 - ema_decay)

                batches += 1
                progress.set_postfix(d=f"{float(loss_d.detach()):.3f}", fm=f"{loss_g_fm:.3f}",
                                     refresh=False)
                audit_logger().info(
                    f"Epoch:{epoch}, Iteration:{iteration}, "
                    f"LR:{optimiser_g.param_groups[0]['lr']}, Scheduler:None, "
                    f"BatchSize:{x0.shape[0]}, Loss:{float(loss_d.detach()):.6f}, "
                    f"Losses:{{'d_hinge': {float(loss_d.detach()):.6f}, "
                    f"'g_adv': {loss_g_adv:.6f}, 'g_fm': {loss_g_fm:.6f}}}, "
                    f"LossWeight:{{'velocity': {anchor_weight}, "
                    f"'adversarial': {adversarial_weight}, 'feature': {feature_weight}}}, "
                    f"MemUsage:{get_mem_usage()}, Stage:adversarial_refinement"
                )

            last = {k: v / max(batches, 1) for k, v in totals.items()}
            for name, value in last.items():
                context.tracker.log_loss(name, value, step=epoch, stage="adversarial_refinement")
            print(f"Refinement [{epoch}/{epochs}] d={last['d']:.4f} adv={last['g_adv']:.4f} "
                  f"fm={last['g_fm']:.4f} velocity={last['g_vel']:.6f}", flush=True)
            audit_logger().info(
                f"[Validation] Epoch:{epoch}, ValTotalLoss:{last['g_vel']:.6f}, "
                f"ValLosses:{list(last.values())}, ValMetrics:{{}}")

            if save_every > 0 and epoch % save_every == 0:
                path = self._save(generator, discriminator, ema, context.paths.artifacts_dir, epoch)
                audit_logger().info(
                    f"[Checkpoint] Saved to: {path} | Epoch:{epoch}, IsBest:False, "
                    f"ValLoss:{last['g_vel']:.6f}, ValMetrics:{{}}")

        checkpoint = self._save(generator, discriminator, ema, context.paths.artifacts_dir, epochs)
        context.state["model"] = generator
        return {"epochs": epochs, "checkpoint": str(checkpoint), **last}

    def _device(self) -> torch.device:
        configured = self.params.get("device")
        if isinstance(configured, list):
            configured = configured[0]
        if isinstance(configured, int):
            configured = f"cuda:{configured}"
        device = torch.device(configured or ("cuda" if torch.cuda.is_available() else "cpu"))
        return torch.device("cpu") if device.type == "cuda" and not torch.cuda.is_available() else device

    @staticmethod
    def _save(generator, discriminator, ema, directory: Path, epoch: int) -> Path:
        path = directory / f"task3_flow_adv_{epoch}.pt"
        payload: dict[str, Any] = {
            "model": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "phase": "adversarial_refinement",
            "progress": epoch,
        }
        if ema is not None:
            merged = {k: v.clone() for k, v in generator.state_dict().items()}
            merged.update({k: v.clone() for k, v in ema.items()})
            payload["model_ema"] = merged
        torch.save(payload, path)
        return path
