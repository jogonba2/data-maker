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


class MRPCSchema(BaseModel):
    text: str
    completion: Literal[
        "The sentences are equivalent",
        "The sentences are not equivalent",
    ]


class MRPC(BaseConfig):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        num_icl_examples=[0, 4, 8],
        num_samples=1024,
    ):
        super().__init__(model_name, num_icl_examples, num_samples)
        self.build_dataset()
        self.languages = ["English"]
        self.task_name = "mrpc"
        register_task_schema(self.task_name, MRPCSchema)

    def _prepare_text(self, dataset):
        dataset["train"] = dataset["train"].map(
            lambda q1, q2: {"text": f"S1: {q1}\nS2: {q2}"},
            input_columns=["sentence1", "sentence2"],
            remove_columns=["sentence1", "sentence2"],
        )
        dataset["test"] = dataset["test"].map(
            lambda q1, q2: {"text": f"S1: {q1}\nS2: {q2}"},
            input_columns=["sentence1", "sentence2"],
            remove_columns=["sentence1", "sentence2"],
        )
        return dataset

    def build_dataset(self):
        self.dataset = load_dataset("nyu-mll/glue", "mrpc")
        self.dataset["train"] = self.dataset["train"].select_columns(
            ["sentence1", "sentence2", "label"]
        )

        self.dataset["test"] = self.dataset["test"].select_columns(
            ["sentence1", "sentence2", "label"]
        )
        # Remove test set since it is unlabeled
        self.dataset["test"] = self.dataset["validation"]
        del self.dataset["validation"]

        # Join the questions
        self.dataset = self._prepare_text(self.dataset)

        self.label_verbalization = {
            0: "The sentences are not equivalent",
            1: "The sentences are equivalent",
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

        self.dataset["train"] = self.dataset["train"].remove_columns(["label"])
        self.dataset["test"] = self.dataset["test"].remove_columns(["label"])

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
                instruction="""Assess the suitability of this example for a classification task that aims to determine whether two sentences are equivalent or not. The example can only be labeled as either `The sentences are equivalent` or `The sentences are not equivalent`.

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
            instruction="Write two sentences for a news article, to be included in a dataset for a classification task aimed at determining whether two sentences are equivalent or not. The sentences must be similar, both semantically and lexically, to the provided examples (if any). Do not stick to common sentence structures; vary the sentence construction, employ different tones, and incorporate multiple perspectives. Ensure that punctuation marks are separated by spaces and that your response follows this format: `S1: ...\nS2: ...`.",
            examples=(
                self.get_incontext_examples(num_icl_examples)
                .to_pandas()
                .to_dict(orient="records")
                if num_icl_examples > 0
                else []
            ),
            kwargs={"max_tokens": 50},
        )
