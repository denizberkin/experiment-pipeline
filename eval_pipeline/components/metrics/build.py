from eval_pipeline.core.config import MetricConfig
from eval_pipeline.core.registry import instantiate_component
from eval_pipeline.interfaces.metric import Metric


def build_metrics(configs: list[MetricConfig]) -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}
    for config in configs:
        metrics[config.label] = {
            "metric_fn": instantiate_component(config),
            "stages": config.stages,
            "config": config,
        }
    return metrics


def build_metric(config: MetricConfig) -> Metric:
    return instantiate_component(config)
