from eval_pipeline.core.config import ComponentConfig
from eval_pipeline.core.registry import instantiate_component
from eval_pipeline.interfaces.prediction import Predictor


def build_predictor(config: ComponentConfig) -> Predictor:
    return instantiate_component(config)
