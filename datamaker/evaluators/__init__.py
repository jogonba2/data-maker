from ..config import Config
from .base import Evaluator
from .llm import LLMThreeScaleCategoricalEvaluator

mapping = {
    "LLMThreeScaleCategoricalEvaluator": LLMThreeScaleCategoricalEvaluator
}


def get_evaluator(config: Config) -> Evaluator:
    name = config.evaluator.evaluator_class
    if name in mapping:
        return mapping[name](config)
    else:
        raise ValueError(f"{name} is not a valid evaluator")
