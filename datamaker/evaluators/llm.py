from typing import Literal

from pydantic import BaseModel

from .base import Evaluator


class QualityScale(BaseModel):
    explanation: str
    score: Literal["low", "mid", "high"]


class LLMThreeScaleCategoricalEvaluator(Evaluator):

    def _evaluate(self, examples: list[BaseModel]) -> list[BaseModel]:
        assert (
            self.config.evaluator.instruction
        ), f"An instruction is needed for {self.__class__.__name__}"
        messages = []
        for example in examples:
            user_msg = "\n".join(
                [
                    key.capitalize() + ": " + str(value)
                    for key, value in example.dict().items()
                ]
            )
            messages.append(
                [
                    {
                        "role": "system",
                        "content": self.config.evaluator.instruction,
                    },
                    {"role": "user", "content": user_msg},
                ]
            )

        completions = self.client.generate(
            messages=messages, response_format=QualityScale
        )
        return completions
