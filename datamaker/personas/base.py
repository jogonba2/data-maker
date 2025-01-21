from abc import ABC, abstractmethod

from ..common import OpenAIClient
from ..config import Config


class PersonaGenerator(ABC):
    def __init__(self, config: Config):
        assert config.persona, "Persona config is needed for PersonaGenerator"
        self.config = config
        self.client = OpenAIClient(config.openai) if config.openai else None

    @abstractmethod
    def _generate(self) -> list[str]: ...

    def generate(self) -> list[str]:
        generated_personas = self._generate()
        predefined_personas = self.config.persona.personas or []
        return predefined_personas + generated_personas
