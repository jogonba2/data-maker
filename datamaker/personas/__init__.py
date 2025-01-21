from ..config import Config
from .base import PersonaGenerator
from .dummy import DummyPersonas
from .persona_hub import PersonaHub

mapping = {"PersonaHub": PersonaHub, "DummyPersonas": DummyPersonas}


def get_persona_generator(config: Config) -> PersonaGenerator:
    name = config.persona.persona_class
    if name in mapping:
        return mapping[name](config)
    else:
        raise ValueError(f"{name} is not a valid persona generator")
