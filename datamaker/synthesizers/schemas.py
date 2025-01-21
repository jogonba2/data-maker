from typing import Literal, Type

from pydantic import BaseModel


class SentimentAnalysis(BaseModel):
    text: str
    completion: Literal["positive", "negative", "neutral"]


class Translation(BaseModel):
    source_text: str
    completion: str


task_mapping: dict[str, Type[BaseModel]] = {
    "sentiment_analysis": SentimentAnalysis,
    "translation": Translation,
}


def register_task_schema(name: str, cls: BaseModel) -> None:
    task_mapping[name] = cls


def get_task_schema(name: str) -> BaseModel:
    if name in task_mapping:
        return task_mapping[name]
    else:
        raise ValueError(f"Task {name} is not supported.")
