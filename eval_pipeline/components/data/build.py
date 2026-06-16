from eval_pipeline.core.config import ComponentConfig
from eval_pipeline.core.registry import instantiate_component
from eval_pipeline.interfaces.data import DataModule


def build_data_module(config: ComponentConfig) -> DataModule:
    return instantiate_component(config)
