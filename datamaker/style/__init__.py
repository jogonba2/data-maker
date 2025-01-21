from typing import Mapping, Type

from ..config import Config
from .base import StyleGenerator
from .dummy import DummyStyle
from .llm import LLMSupervisedStyle

mapping: Mapping[str, Type[StyleGenerator]] = {
    "LLMSupervisedStyle": LLMSupervisedStyle,
    "DummyStyle": DummyStyle,
}


def get_style_generator(config: Config) -> StyleGenerator:
    name = config.style.style_class
    if name in mapping:
        return mapping[name](config)
    else:
        raise ValueError(f"{name} is not a valid style generator")
