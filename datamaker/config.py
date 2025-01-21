from typing import Any, Optional

from pydantic import BaseModel


class SynthesizerConfig(BaseModel):
    synthesizer_class: str
    num_samples: int
    instruction: Optional[str] = None
    examples: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {}


class PersonaConfig(BaseModel):
    persona_class: str
    num_personas: int = 0
    instruction: Optional[str] = None
    personas: list[str] = []
    kwargs: dict[str, Any] = {}


class TopicConfig(BaseModel):
    topic_class: str
    num_topics: int = 0
    instruction: Optional[str] = None
    topics: list[str] = []
    kwargs: dict[str, Any] = {}


class StyleConfig(BaseModel):
    style_class: str
    instruction: Optional[str] = None
    style_description: str = ""
    kwargs: dict[str, Any] = {}


class EvaluatorConfig(BaseModel):
    evaluator_class: str
    filtering_criterion: list[str]
    instruction: Optional[str] = None
    kwargs: dict[str, Any] = {}


class OpenAIConfig(BaseModel):
    model_name: str = "gpt-4o-mini"
    max_retries: int = 10
    timeout: int = 10


class Config(BaseModel):
    task: str
    languages: Optional[list[str]] = None
    synthesizer: Optional[SynthesizerConfig] = None
    random_seed: int = 42
    openai: Optional[OpenAIConfig] = None
    persona: Optional[PersonaConfig] = None
    topic: Optional[TopicConfig] = None
    style: Optional[StyleConfig] = None
    evaluator: Optional[EvaluatorConfig] = None
