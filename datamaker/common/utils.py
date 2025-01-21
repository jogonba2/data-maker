from typing import _LiteralGenericAlias  # type: ignore

import pandas as pd
from pydantic import BaseModel


def prompts_to_messages(prompts: list[str]) -> list[list[dict[str, str]]]:
    return [[{"role": "user", "content": prompt}] for prompt in prompts]


def is_classification(task_schema: BaseModel) -> bool:
    return "completion" in task_schema.__annotations__ and isinstance(
        task_schema.__annotations__["completion"], _LiteralGenericAlias
    )


def get_classification_labels(task_schema: BaseModel) -> list[str]:
    assert is_classification(
        task_schema
    ), "The task schema is not aimed for classification"
    return list(task_schema.__annotations__["completion"].__args__)


def oversample(
    samples: list[BaseModel], task_schema: BaseModel, n_samples: int
) -> list[BaseModel]:
    dict_samples = [sample.dict() for sample in samples]
    df = pd.DataFrame(dict_samples)

    oversampled_examples = []
    for _, group in df.groupby("completion"):
        # If the group has fewer rows than n_samples, repeat rows to reach n_samples
        if len(group) < n_samples:
            group = pd.concat(
                [
                    group,
                    group.sample(
                        n_samples - len(group), replace=True, random_state=42
                    ),
                ]
            )

        # Append the resampled group to the list
        oversampled_examples.append(group)

    # Combine oversampled examples and convert back to list of dicts
    examples = pd.concat(oversampled_examples).to_dict("records")

    # Convert back to list of task_schema instances
    return [task_schema(**example) for example in examples]
