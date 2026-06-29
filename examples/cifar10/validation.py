from __future__ import annotations

from typing import Any

from eval_pipeline.components.validators.base import Validator
from eval_pipeline.context import ValidationContext
from eval_pipeline.registry import register_component

from components.cifar10.common import compute_metrics, get_device, metric_fns, primary_loss, reset_metrics

type Cifar10Data = dict[str, Any]
type Cifar10Result = dict[str, Any]


@register_component("cifar10_validator", category="validation")
class Cifar10Validator(Validator[Cifar10Data, Any, Cifar10Result]):
    def validate(
        self,
        *,
        data: Cifar10Data,
        model: Any,
        losses: dict[str, Any],
        metrics: dict[str, Any],
        context: ValidationContext,
    ) -> Cifar10Result:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("CIFAR-10 validator requires torch.") from exc

        device = get_device(self.params.get("device"))
        model = model.to(device)
        model.eval()

        loss_fn = primary_loss(losses)
        validation_metrics = metric_fns(metrics, context.stage)
        reset_metrics(validation_metrics)
        running_loss = 0.0
        seen = 0

        with torch.no_grad():
            for images, targets in data["validation"]:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, targets)

                batch_size = int(targets.size(0))
                running_loss += float(loss.item()) * batch_size
                seen += batch_size
                for metric in validation_metrics.values():
                    metric.update(outputs, targets)

        loss_value = running_loss / max(seen, 1)
        metric_values = compute_metrics(validation_metrics)
        context.tracker.log_loss("cross_entropy", loss_value, stage=context.stage)
        context.tracker.log_metrics(metric_values, stage=context.stage)

        return {
            "loss": loss_value,
            "metrics": metric_values,
            "samples": seen,
            "device": str(device),
        }
