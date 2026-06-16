from eval_pipeline.core.config import ComponentConfig
from eval_pipeline.core.registry import instantiate_component
from eval_pipeline.interfaces.training import Trainer


def build_trainer(config: ComponentConfig) -> Trainer:
    return instantiate_component(config)
