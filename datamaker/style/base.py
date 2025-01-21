from abc import ABC, abstractmethod

from ..common import OpenAIClient
from ..config import Config


class StyleGenerator(ABC):
    def __init__(self, config: Config):
        assert config.style, "Style config is needed for the StyleGenerator."
        self.config = config
        self.client = OpenAIClient(config.openai) if config.openai else None

    @abstractmethod
    def _generate(self) -> str: ...

    def generate(self) -> str:
        generated_style = self._generate()
        if not generated_style:
            generated_style = self.config.style.style_description
        return generated_style
