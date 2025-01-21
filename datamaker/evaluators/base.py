from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..common import OpenAIClient, get_logger
from ..config import Config

_logger = get_logger(__name__)


class Evaluator(ABC):
    def __init__(self, config: Config):
        assert config.evaluator, "Evaluator config is needed for Evaluator"
        self.config = config
        self.client = OpenAIClient(config.openai) if config.openai else None

    @abstractmethod
    def _evaluate(self, examples: list[BaseModel]) -> list[BaseModel]: ...

    def _filter(self, score: BaseModel) -> bool:
        return (
            True
            if score in self.config.evaluator.filtering_criterion
            else False
        )

    def run(self, examples: list[BaseModel]) -> list[BaseModel]:
        eval_outputs = self._evaluate(examples)
        filtered = [
            example
            for example, output in zip(examples, eval_outputs)
            if output is not None and self._filter(output.score)
        ]
        _logger.info(
            f"{len(eval_outputs) - len(filtered)} samples do not meet the filtering criterion."
        )
        _logger.info(
            f"Your dataset has {len(filtered)} samples after filtering."
        )
        return filtered
