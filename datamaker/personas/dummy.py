from .base import PersonaGenerator


class DummyPersonas(PersonaGenerator):
    def _generate(self) -> list[str]:
        return []
