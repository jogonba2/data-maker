from typing import Literal

from datamaker import (
    EvaluatorConfig,
    PersonaConfig,
    SynthesizerConfig,
    TopicConfig,
    register_task_schema,
)
from datasets import load_dataset
from pydantic import BaseModel
from .base import BaseConfig


class IMDBSchema(BaseModel):
    text: str
    completion: Literal["It was terrible", "It was great"]


class IMDB(BaseConfig):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        num_icl_examples=[0, 4, 8],
        num_samples=1024,
    ):
        super().__init__(model_name, num_icl_examples, num_samples)
        self.build_dataset()
        self.languages = ["English"]
        self.task_name = "imdb"
        register_task_schema(self.task_name, IMDBSchema)

    def build_dataset(self):
        self.dataset = load_dataset("stanfordnlp/imdb")
        self.dataset["train"] = self.dataset["train"].select_columns(
            ["text", "label"]
        )
        self.dataset["test"] = self.dataset["test"].select_columns(
            ["text", "label"]
        )
        self.label_verbalization = {0: "It was terrible", 1: "It was great"}
        self.dataset["train"] = self.dataset["train"].add_column(
            "label_text",
            [
                self.label_verbalization[label]
                for label in self.dataset["train"]["label"]
            ],
        )
        self.dataset["test"] = self.dataset["test"].add_column(
            "label_text",
            [
                self.label_verbalization[label]
                for label in self.dataset["test"]["label"]
            ],
        )
        self.dataset["train"] = self.dataset["train"].remove_columns(["label"])
        self.dataset["test"] = self.dataset["test"].remove_columns(["label"])
        self.dataset["train"] = self.dataset["train"].rename_columns(
            {"label_text": "completion"}
        )
        self.dataset["test"] = self.dataset["test"].rename_columns(
            {"label_text": "completion"}
        )

    def get_persona_config(self):
        return {
            "persona+yes": PersonaConfig(
                persona_class="PersonaHub", num_personas=self.num_samples
            ),
            "persona+no": PersonaConfig(
                persona_class="DummyPersonas", num_personas=0
            ),
        }

    def get_topic_config(self):
        return {
            "topic+yes": TopicConfig(
                topic_class="LLMSupervisedTopics",
                num_topics=200,
                instruction=f"Extract 200 different and diverse topics from the provided texts.",
                kwargs={
                    "texts": self.dataset["train"]["text"][:50],
                    "num_texts": 50,
                },
            ),
            "topic+no": TopicConfig(topic_class="DummyTopics", num_topics=0),
        }

    def get_evaluator_config(self):
        return {
            "evaluator+yes": EvaluatorConfig(
                evaluator_class="LLMThreeScaleCategoricalEvaluator",
                filtering_criterion=["high"],
                instruction="""Assess the suitability of this example for a sentiment analysis task. The example can be only labeled as `It was terrible` or `It was great`.

Here are the criteria for evaluating the example:
- If the text is clearly labeled correctly, assign a 'high' score.
- If there is any uncertainty about the label's accuracy, assign a 'medium' score.
- If the text is clearly labeled incorrectly, assign a 'low' score.

Briefly justify your assessment in a single sentence.""",
            ),
            "evaluator+no": None,
        }

    def get_synthesizer_config(self, num_icl_examples: int):
        return SynthesizerConfig(
            synthesizer_class="LLMSynthesizer",
            num_samples=self.num_samples,
            instruction="Write a movie review on the IMDB platform to be included in a dataset for a sentiment analysis task. The review must be similar to the provided examples (if any).",
            examples=(
                self.get_incontext_examples(num_icl_examples)
                .to_pandas()
                .to_dict(orient="records")
                if num_icl_examples > 0
                else []
            ),
            kwargs={"max_tokens": 200},
        )
