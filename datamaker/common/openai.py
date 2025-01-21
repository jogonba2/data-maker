from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from ..config import OpenAIConfig


class OpenAIClient:
    # Client to be shared across all the pipeline
    def __init__(self, config: OpenAIConfig):
        self.model_name = config.model_name
        self.client = OpenAI(
            max_retries=config.max_retries,
            timeout=config.timeout,
        )

    def generate_single(
        self,
        messages: list[dict[str, str]],
        response_format: Optional[str | BaseModel] = None,
    ) -> str | BaseModel:
        if response_format:
            try:
                response = (
                    self.client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=messages,
                        response_format=response_format,
                    )
                    .choices[0]
                    .message
                )
                # If the model refuses to respond, returns None
                completion = None if response.refusal else response.parsed
            # In case of OpenAI errors -> None
            except Exception:
                completion = None
        else:
            try:
                completion = (
                    self.client.chat.completions.create(
                        messages=messages, model=self.model_name
                    )
                    .choices[0]
                    .message.content
                )
            # In case of OpenAI errors -> None
            except Exception:
                completion = None
        return completion

    def generate(
        self,
        messages: list[list[dict[str, str]]],
        response_format: Optional[str | BaseModel] = None,
        threads: int = 8,
    ) -> list[str] | list[BaseModel]:
        responses = []
        with ThreadPoolExecutor(
            max_workers=min(threads, len(messages))
        ) as thread_pool:
            for msg in messages:
                responses.append(
                    thread_pool.submit(
                        self.generate_single, msg, response_format
                    )
                )
            # Wait completions
            completions = [response.result() for response in tqdm(responses)]

        return completions
