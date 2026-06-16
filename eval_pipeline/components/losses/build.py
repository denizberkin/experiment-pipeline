from eval_pipeline.core.config import LossConfig
from eval_pipeline.core.registry import instantiate_component
from eval_pipeline.interfaces.loss import Loss


def build_losses(configs: list[LossConfig]) -> dict[str, dict[str, object]]:
    losses: dict[str, dict[str, object]] = {}
    for config in configs:
        losses[config.label] = {
            "loss_fn": instantiate_component(config),
            "weight": config.weight,
            "config": config,
        }
    return losses


def build_loss(config: LossConfig) -> Loss:
    return instantiate_component(config)
