from .base import TopicGenerator


class DummyTopics(TopicGenerator):
    def _generate(self) -> list[str]:
        return []
