from ..config import Config
from .base import Synthesizer
from .llm import LLMSynthesizer

mapping = {"LLMSynthesizer": LLMSynthesizer}


def get_synthesizer(config: Config) -> Synthesizer:
    name = config.synthesizer.synthesizer_class
    if name in mapping:
        return mapping[name](config)
    else:
        raise ValueError(f"{name} is not a valid synthesizer")
