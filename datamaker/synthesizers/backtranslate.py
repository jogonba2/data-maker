from pydantic import BaseModel

from ..common import get_logger
from .base import Synthesizer

_logger = get_logger(__name__)


class BackTranslate(Synthesizer):

    def _generate(self) -> list[BaseModel]:
        assert (
            "translate_chain" in self.config.kwargs
        ), "A translate chain is needed for BackTranslate"
        assert (
            "dataset" in self.config.kwargs
        ), "A dataset is needed for BackTranslate"
        return []
