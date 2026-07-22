from __future__ import annotations

from typing import Any

import torch

from eval_pipeline.components.models import save_torch_checkpoint
from eval_pipeline.components.trainers.base import Trainer
from eval_pipeline.context import TrainingContext
from eval_pipeline.registry import register_component

from examples.cifar10.common import (
    compute_metrics,
    get_device,
    metric_fns,
    primary_loss,
    reset_metrics,
)

type Cifar10Data = dict[str, Any]
type Cifar10Result = dict[str, Any]


@register_component("cifar10_torch_trainer", category="training")
class Cifar10TorchTrainer(Trainer[Cifar10Data, Any, Cifar10Result]):
    def train(
        self,
        *,
        data: Cifar10Data,
        model: Any,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: TrainingContext,
    ) -> Cifar10Result:
        device = get_device(self.params.get("device"))
        epochs = int(self.params.get("epochs", 5))
        learning_rate = float(self.params.get("learning_rate", 1e-3))
        weight_decay = float(self.params.get("weight_decay", 1e-4))
        save_checkpoint_path = str(self.params.get("save_checkpoint", "cifar10_model.pt"))
        checkpoint_frequency = int(self.params.get("checkpoint_frequency", 0))
        if checkpoint_frequency < 0:
            raise ValueError("checkpoint_frequency must be 0 or a positive integer.")

        model = model.to(device)
        loss_fn = primary_loss(losses)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_metrics = metric_fns(metrics, "training")
        checkpoint = None

        for epoch in range(1, epochs + 1):
            model.train()
            reset_metrics(train_metrics)
            running_loss = 0.0
            seen = 0

            for images, targets in data["train"]:
                images = images.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                batch_size = int(targets.size(0))
                running_loss += float(loss.item()) * batch_size
                seen += batch_size
                for metric in train_metrics.values():
                    metric.update(outputs.detach(), targets)

            epoch_loss = running_loss / max(seen, 1)
            context.tracker.log_loss("cross_entropy", epoch_loss, step=epoch, stage=context.stage)
            for name, value in compute_metrics(train_metrics).items():
                context.tracker.log_metric(name, value, step=epoch, stage=context.stage)

            if _should_save_checkpoint(epoch=epoch, epochs=epochs, checkpoint_frequency=checkpoint_frequency):
                checkpoint = save_torch_checkpoint(model, context.paths.artifacts_dir / save_checkpoint_path)
                context.tracker.log_artifact(checkpoint, artifact_path="checkpoints")

        context.state["model"] = model
        context.state["checkpoint"] = str(checkpoint) if checkpoint else None

        return {
            "epochs": epochs,
            "device": str(device),
            "checkpoint": str(checkpoint) if checkpoint else None,
        }


def _should_save_checkpoint(*, epoch: int, epochs: int, checkpoint_frequency: int) -> bool:
    if checkpoint_frequency == 0:
        return epoch == epochs
    return epoch % checkpoint_frequency == 0
