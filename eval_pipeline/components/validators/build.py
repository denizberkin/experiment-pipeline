from eval_pipeline.core.config import ComponentConfig
from eval_pipeline.core.registry import instantiate_component
from eval_pipeline.interfaces.validation import Validator


def build_validator(config: ComponentConfig) -> Validator:
    return instantiate_component(config)
