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


class ToxicConversationsSchema(BaseModel):
    text: str
    completion: Literal[
        "It is a toxic comment",
        "It is not a toxic comment",
    ]


class ToxicConversations(BaseConfig):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        num_icl_examples=[0, 4, 8],
        num_samples=1024,
    ):
        super().__init__(model_name, num_icl_examples, num_samples)
        self.build_dataset()
        self.languages = ["English"]
        self.task_name = "toxic_conversations"
        register_task_schema(self.task_name, ToxicConversationsSchema)

    def build_dataset(self):
        self.dataset = load_dataset("SetFit/toxic_conversations")
        self.dataset["train"] = self.dataset["train"].select_columns(
            ["text", "label_text"]
        )
        self.dataset["test"] = self.dataset["test"].select_columns(
            ["text", "label_text"]
        )
        self.label_verbalization = {
            "toxic": "It is a toxic comment",
            "not toxic": "It is not a toxic comment",
        }
        self.dataset["train"] = self.dataset["train"].map(
            lambda label: {"completion": self.label_verbalization[label]},
            input_columns=["label_text"],
        )
        self.dataset["test"] = self.dataset["test"].map(
            lambda label: {"completion": self.label_verbalization[label]},
            input_columns=["label_text"],
        )
        self.dataset = self.dataset.remove_columns(["label_text"])

        # Subsample to reduce times (original is about 50k)
        self.dataset["test"] = (
            self.dataset["test"].shuffle(42).select(range(5000))
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
                instruction="""Assess the suitability of this example for a binary classification task that aims to determine whether a text is toxic or not. The example can only be labeled as either "It is a toxic comment" or "It is not a toxic comment".

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
            instruction="""Write a comment in the Civil Comments platform to be included in a dataset for a binary classification task aimed at determining whether a text is toxic or not. The comment must be similar, both semantically and lexically, to the provided examples (if any). Do not stick to common sentence structures; vary the sentence construction, employ different tones, and incorporate multiple perspectives.""",
            examples=(
                self.get_incontext_examples(num_icl_examples)
                .to_pandas()
                .to_dict(orient="records")
                if num_icl_examples > 0
                else []
            ),
            kwargs={"max_tokens": 200},
        )
