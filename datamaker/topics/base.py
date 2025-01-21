from abc import ABC, abstractmethod

from ..common import OpenAIClient
from ..config import Config


class TopicGenerator(ABC):
    def __init__(self, config: Config):
        assert config.topic, "Topic config is needed for the TopicGenerator."
        self.config = config
        self.client = OpenAIClient(config.openai) if config.openai else None

    @abstractmethod
    def _generate(self) -> list[str]: ...

    def generate(self) -> list[str]:
        generated_topics = self._generate()
        predefined_topics = self.config.topic.topics or []
        return predefined_topics + generated_topics
