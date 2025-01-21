from datasets import load_dataset

from .base import PersonaGenerator


class PersonaHub(PersonaGenerator):
    def _generate(self) -> list[str]:
        ds = load_dataset("proj-persona/PersonaHub", "persona")["train"]
        ds = ds.shuffle(seed=self.config.random_seed).select(
            range(self.config.persona.num_personas)
        )
        return ds["persona"]
