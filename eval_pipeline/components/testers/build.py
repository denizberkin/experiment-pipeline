from eval_pipeline.core.config import ComponentConfig
from eval_pipeline.core.registry import instantiate_component
from eval_pipeline.interfaces.testing import Tester


def build_tester(config: ComponentConfig) -> Tester:
    return instantiate_component(config)
