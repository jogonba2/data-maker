from pydantic import BaseModel

from ..common import truncate
from .base import StyleGenerator


class StyleDescription(BaseModel):
    writing_style: str
    word_count_range: str
    sentence_count_range: str


class LLMSupervisedStyle(StyleGenerator):
    def _generate(self) -> str:
        assert (
            self.config.style
        ), "Style config is needed for {self.__class__.__name__}"
        assert (
            self.config.style.instruction
        ), "A prompt instruction is needed for {self.__class__.__name__}"
        assert (
            self.client
        ), "OpenAI client config is needed for {self.__class__.__name__}"
        assert (
            self.config.style.kwargs
        ), f"`kwargs` are required for {self.__class__.__name__}"
        assert (
            "texts" in self.config.style.kwargs
        ), f"A list of texts is needed for {self.__class__.__name__}"
        assert (
            "num_texts" in self.config.style.kwargs
        ), f"The number of texts to put in the prompt is needed for {self.__class__.__name__}"

        all_texts = self.config.style.kwargs["texts"][
            : self.config.style.kwargs["num_texts"]
        ]
        all_texts = [
            truncate(
                text, max_length=self.config.style.kwargs.get("max_tokens", 200)
            )
            for text in all_texts
        ]
        texts = "\n".join(
            [
                " ".join(
                    f"- Text {i}: {text}".replace("\n", " ").strip().split()
                )
                for i, text in enumerate(all_texts)
            ]
        )
        prompt_texts = "# Texts\n" + texts + "\n\n# Style description\n"
        style = self.client.generate_single(
            [
                {"role": "system", "content": self.config.style.instruction},
                {"role": "user", "content": prompt_texts},
            ],
            StyleDescription,
        )
        verbalized_style = "- " + "\n- ".join(style.dict().values())
        return verbalized_style
