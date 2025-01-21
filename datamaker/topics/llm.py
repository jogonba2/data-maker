from pydantic import BaseModel

from ..common import truncate
from .base import TopicGenerator


class TopicList(BaseModel):
    topics: list[str]


class LLMUnsupervisedTopics(TopicGenerator):
    def _generate(self) -> list[str]:
        assert (
            self.config.topic
        ), "Topic config is needed for {self.__class__.__name__}"
        assert (
            self.config.topic.instruction
        ), "A prompt instruction is needed for {self.__class__.__name__}"
        assert (
            self.client
        ), "OpenAI client config is needed for {self.__class__.__name__}"
        topics = self.client.generate_single(
            [{"role": "user", "content": self.config.topic.instruction}],
            TopicList,
        )
        return topics.topics


class LLMSupervisedTopics(TopicGenerator):
    def _generate(self) -> list[str]:
        assert (
            self.config.topic
        ), "Topic config is needed for {self.__class__.__name__}"
        assert (
            self.config.topic.instruction
        ), "A prompt instruction is needed for {self.__class__.__name__}"
        assert (
            self.client
        ), "OpenAI client config is needed for {self.__class__.__name__}"
        assert (
            self.config.topic.kwargs
        ), f"`kwargs` are required for {self.__class__.__name__}"
        assert (
            "texts" in self.config.topic.kwargs
        ), f"A list of texts is needed for {self.__class__.__name__}"
        assert (
            "num_texts" in self.config.topic.kwargs
        ), f"The number of texts to put in the prompt is needed for {self.__class__.__name__}"

        all_texts = self.config.topic.kwargs["texts"][
            : self.config.topic.kwargs["num_texts"]
        ]
        all_texts = [
            truncate(
                text, max_length=self.config.topic.kwargs.get("max_tokens", 200)
            )
            for text in all_texts
        ]
        texts = "\n".join(all_texts)
        prompt_texts = "# Texts\n\n" + texts + "\n\n# Topics"
        topics = self.client.generate_single(
            [
                {"role": "system", "content": self.config.topic.instruction},
                {"role": "user", "content": prompt_texts},
            ],
            TopicList,
        )
        return topics.topics
