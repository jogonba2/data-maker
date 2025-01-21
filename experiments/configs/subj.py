from typing import Literal
import random
from datamaker import (
    EvaluatorConfig,
    PersonaConfig,
    SynthesizerConfig,
    TopicConfig,
    StyleConfig,
    register_task_schema,
)
from datasets import load_dataset
from pydantic import BaseModel
from .base import BaseConfig


class SUBJSchema(BaseModel):
    text: str
    completion: Literal["It is objective", "It is subjective"]


class SUBJ(BaseConfig):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        num_icl_examples=[0, 4, 8],
        num_samples=16, #1024,
    ):
        super().__init__(model_name, num_icl_examples, num_samples)
        self.build_dataset()
        self.languages = ["English"]
        self.task_name = "subj"
        register_task_schema(self.task_name, SUBJSchema)

    def build_dataset(self):
        self.dataset = load_dataset("SetFit/subj")
        self.dataset["train"] = self.dataset["train"].select_columns(
            ["text", "label"]
        )
        self.dataset["test"] = self.dataset["test"].select_columns(
            ["text", "label"]
        )
        self.label_verbalization = {0: "It is objective", 1: "It is subjective"}
        self.dataset = self.dataset.map(
            lambda label: {"label": self.label_verbalization[label]},
            input_columns=["label"],
        )
        self.dataset = self.dataset.rename_columns({"label": "completion"})

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
                instruction=f"Extract 500 different and diverse topics from the provided texts.",
                kwargs={
                    "texts": random.sample(self.dataset["train"]["text"][:100], 100) if random.seed(42) is None else None,
                    "num_texts": 100,
                },
            ),
            "topic+no": TopicConfig(topic_class="DummyTopics", num_topics=0),
        }

    def get_style_config(self):
        return {
            "style+yes": StyleConfig(
                style_class="LLMSupervisedStyle",
                instruction=f"""Describe the writing style of the following texts. Do not focus on the topics or specific terms, just describe the style features.
                You must describe the following style features using one or two sentences:

                - Writing style: a summary of the writing style used in the texts
                - Word length: the average number of words per text (delimited by whitespaces)
                - Sentence length: the average number of sentences per text (delimited by ".")
                - Vocabulary: the choice of words, including their complexity, specificity, and uniqueness
                """,
                kwargs={
                    "texts": random.sample(self.dataset["train"]["text"][:100], 100) if random.seed(42) is None else None,
                    "num_texts": 100,
                },
            ),
            "style+no": StyleConfig(style_class="DummyStyle"),
        }
    
    def get_evaluator_config(self):
        return {
            "evaluator+yes": EvaluatorConfig(
                evaluator_class="LLMThreeScaleCategoricalEvaluator",
                filtering_criterion=["high"],
                instruction="""Assess the suitability of this example for a binary classification task that aims to determine whether a text is subjective or objective. The example can only be labeled as either "It is objective" or "It is subjective."

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
            instruction="Create a labeled text instance to be included in a dataset for a binary classification task aimed at determining whether a text is subjective or objective. Base your sentence in the provided examples (if any).",
            examples=(
                self.get_incontext_examples(num_icl_examples)
                .to_pandas()
                .to_dict(orient="records")
                if num_icl_examples > 0
                else []
            ),
            kwargs={"max_tokens": 200},
        )
