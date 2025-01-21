from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..common import OpenAIClient, is_classification, oversample
from ..config import Config
from ..evaluators import get_evaluator
from ..personas import get_persona_generator
from ..style import get_style_generator
from ..topics import get_topic_generator
from .schemas import get_task_schema


class Synthesizer(ABC):

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAIClient(config.openai) if config.openai else None

        self.persona_generator = (
            get_persona_generator(config) if config.persona else None
        )

        self.topic_generator = (
            get_topic_generator(config) if config.topic else None
        )

        self.style_generator = (
            get_style_generator(config) if config.style else None
        )

        self.evaluator = get_evaluator(config) if config.evaluator else None

        self.task_schema = get_task_schema(self.config.task)

    @abstractmethod
    def _generate(self) -> list[BaseModel]: ...

    def generate(self) -> list[BaseModel]:
        samples = self._generate()
        samples = [sample for sample in samples if sample is not None]

        if self.evaluator:
            samples = self.evaluator.run(samples)

        # Oversample per label in classification up to `num_samples`
        # per label to ensure balancing after evaluator filtering.
        if is_classification(self.task_schema):
            samples = oversample(
                samples, self.task_schema, self.config.synthesizer.num_samples
            )
        return samples
