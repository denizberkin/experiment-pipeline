from eval_pipeline.core.config import ComponentConfig
from eval_pipeline.core.registry import instantiate_component
from eval_pipeline.interfaces.model import ModelFactory


def build_model_factory(config: ComponentConfig) -> ModelFactory:
    return instantiate_component(config)
