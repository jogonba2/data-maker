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


class AGNewsSchema(BaseModel):
    text: str
    completion: Literal[
        "It is business news",
        "It is science news",
        "It is sports news",
        "It is world news",
    ]


class AGNews(BaseConfig):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        num_icl_examples=[0, 4, 8],
        num_samples=1024,
    ):
        super().__init__(model_name, num_icl_examples, num_samples)
        self.build_dataset()
        self.languages = ["English"]
        self.task_name = "ag_news"
        register_task_schema(self.task_name, AGNewsSchema)

    def build_dataset(self):
        self.dataset = load_dataset("fancyzhx/ag_news")
        self.dataset["train"] = self.dataset["train"].select_columns(
            ["text", "label"]
        )
        self.dataset["test"] = self.dataset["test"].select_columns(
            ["text", "label"]
        )
        self.label_verbalization = {
            0: "It is world news",
            1: "It is sports news",
            2: "It is business news",
            3: "It is science news",
        }

        self.dataset["train"] = self.dataset["train"].add_column(
            "completion",
            [
                self.label_verbalization[label]
                for label in self.dataset["train"]["label"]
            ],
        )
        self.dataset["test"] = self.dataset["test"].add_column(
            "completion",
            [
                self.label_verbalization[label]
                for label in self.dataset["test"]["label"]
            ],
        )

        self.dataset = self.dataset.remove_columns(["label"])

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
                instruction="""Assess the suitability of this example for a topic classification task. The example can be only labeled as `It is business news`, `It is science news`, `It is sports news`, or `It is world news`.

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
            instruction="Write a brief, 1-3 sentence news article to be included in a dataset for topic classification. The sentences must be similar to the provided examples (if any). Do not stick to common sentence structures; vary the sentence construction, employ different tones, and incorporate multiple perspectives.",
            examples=(
                self.get_incontext_examples(num_icl_examples)
                .to_pandas()
                .to_dict(orient="records")
                if num_icl_examples > 0
                else []
            ),
            kwargs={"max_tokens": 140},
        )
