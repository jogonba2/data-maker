from random import choices
from typing import Optional

from pydantic import BaseModel

from ..common import (
    get_classification_labels,
    get_logger,
    is_classification,
    prompts_to_messages,
    truncate,
)
from .base import Synthesizer

_logger = get_logger(__name__)


class LLMSynthesizer(Synthesizer):

    def _generate(self) -> list[BaseModel]:
        assert self.client, "LLM synthesizer needs an OpenAI client"
        assert (
            self.persona_generator
        ), "LLM Synthesizer needs a persona generator"
        assert self.topic_generator, "LLM Synthesizer needs a topics generator"
        assert self.style_generator, "LLM Synthesizer needs an style generator"
        assert (
            self.config.synthesizer.instruction
        ), "LLM Synthesizer needs a prompt instruction in the config"
        personas = self.persona_generator.generate()
        topics = self.topic_generator.generate()
        style = self.style_generator.generate()
        prompts = self._prepare_prompts(personas, topics, style)
        _logger.info(f"This is how one prompt looks like:\n{prompts[0]}")
        messages = prompts_to_messages(prompts)
        return self.client.generate(messages, response_format=self.task_schema)

    def _prepare_prompt(
        self,
        language: Optional[str] = None,
        persona: Optional[str] = None,
        topic: Optional[str] = None,
        style: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:

        prompt_parts = []

        prompt_parts.append(
            f"{self.config.synthesizer.instruction.strip()}\n\n"
        )

        prompt_parts.append(
            "It is essential that you follow these instructions precisely when composing your text:\n"
        )

        if language:
            prompt_parts.append(f"- The language must be {language}.\n")

        if topic:
            prompt_parts.append(
                f"- Focus on any subtopic related to {topic}.\n"
            )

        if persona:
            prompt_parts.append(f"- Write the text as if you are {persona}.\n")

        if label:
            prompt_parts.append(
                f"- The label of your text must be `{label}`.\n"
            )

        if style:
            prompt_parts.append(
                "\nIt is also crucial that you closely follow the stylistic guidelines outlined below when writing your text:\n"
            )
            prompt_parts.append(style)

        if self.config.synthesizer.examples:
            formatted_examples = "\n".join(
                map(str, self.config.synthesizer.examples)
            )
            prompt_parts.append("\n\nHere are some examples:\n")
            prompt_parts.append(
                f"<examples>\n{formatted_examples}\n</examples>"
            )

        return "".join(prompt_parts)

    def _prepare_prompts(
        self, personas: list[str], topics: list[str], style: str
    ):
        # In classification tasks, `num_samples` will be generated per label
        num_samples = self.config.synthesizer.num_samples
        if is_classification(self.task_schema):
            labels = get_classification_labels(self.task_schema)
            allowed_labels: list[Optional[str]] = [
                label
                for label in labels
                for _ in range(self.config.synthesizer.num_samples)
            ]
            num_samples = len(labels) * self.config.synthesizer.num_samples
        else:
            allowed_labels = [None] * num_samples
        # Sample personas, topics, and languages
        sampled_personas = (
            choices(personas, k=num_samples) if personas else [""] * num_samples
        )
        sampled_topics = (
            choices(topics, k=num_samples) if topics else [""] * num_samples
        )
        sampled_languages = (
            choices(self.config.languages, k=num_samples)
            if self.config.languages
            else [""] * num_samples
        )
        # Truncate IC examples if specified
        if (
            self.config.synthesizer.examples
            and "max_tokens" in self.config.synthesizer.kwargs
        ):
            self.config.synthesizer.examples = [
                {
                    key: truncate(
                        example[key],
                        self.config.synthesizer.kwargs["max_tokens"],
                        self.client.model_name,
                    )
                    for key in example
                }
                for example in self.config.synthesizer.examples
            ]

        # Prepare the prompts
        prompts = [
            self._prepare_prompt(language, persona, topic, style, label)
            for persona, topic, language, label in zip(
                sampled_personas,
                sampled_topics,
                sampled_languages,
                allowed_labels,
            )
        ]
        return prompts
