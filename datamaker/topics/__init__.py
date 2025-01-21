from typing import Mapping, Type

from ..config import Config
from .base import TopicGenerator
from .dummy import DummyTopics
from .llm import LLMSupervisedTopics, LLMUnsupervisedTopics

mapping: Mapping[str, Type[TopicGenerator]] = {
    "LLMUnsupervisedTopics": LLMUnsupervisedTopics,
    "LLMSupervisedTopics": LLMSupervisedTopics,
    "DummyTopics": DummyTopics,
}


def get_topic_generator(config: Config) -> TopicGenerator:
    name = config.topic.topic_class
    if name in mapping:
        return mapping[name](config)
    else:
        raise ValueError(f"{name} is not a valid topic generator")
