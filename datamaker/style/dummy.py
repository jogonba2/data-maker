from .base import StyleGenerator


class DummyStyle(StyleGenerator):
    def _generate(self) -> str:
        return ""
