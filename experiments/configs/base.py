from abc import ABC, abstractmethod
from datamaker import Config, OpenAIConfig, get_synthesizer
import pandas as pd
from datasets import Dataset


class BaseConfig(ABC):
    def __init__(
        self,
        model_name: str = "gpt-4o",
        num_icl_examples=[0, 4, 8],
        num_samples=1024,
    ):
        self.build_dataset()
        self.num_icl_examples = num_icl_examples
        self.num_samples = num_samples
        self.model_name = model_name

    @abstractmethod
    def build_dataset(self): ...

    @abstractmethod
    def get_persona_config(self): ...

    @abstractmethod
    def get_topic_config(self): ...

    @abstractmethod
    def get_style_config(self): ...

    @abstractmethod
    def get_evaluator_config(self): ...

    @abstractmethod
    def get_synthesizer_config(self, num_icl_examples: int): ...

    def get_incontext_examples(self, icl_num) -> Dataset:
        return Dataset.from_pandas(
            self.dataset["train"]
            .to_pandas()
            .groupby("completion")
            .sample(icl_num, random_state=42, replace=True),
            preserve_index=False,
        ).shuffle(seed=42)

    def get_openai_config(self):
        return OpenAIConfig(
            model_name=self.model_name, timeout=120, max_retries=20
        )

    def get_all_configs(self):
        configs = {}
        openai_config = self.get_openai_config()
        for persona_key, persona_config in self.get_persona_config().items():
            for topic_key, topic_config in self.get_topic_config().items():
                for style_key, style_config in self.get_style_config().items():
                    for (
                        evaluator_key,
                        evaluator_config,
                    ) in self.get_evaluator_config().items():
                        for num_icl in self.num_icl_examples:
                            synthesizer_config = self.get_synthesizer_config(
                                num_icl
                            )
                            configs[
                                "_".join(
                                    [
                                        persona_key,
                                        topic_key,
                                        style_key,
                                        evaluator_key,
                                        f"icexamples+{num_icl}",
                                        f"LLM+{self.model_name}",
                                        f"numsamples+{self.num_samples}",
                                    ]
                                )
                            ] = Config(
                                task=self.task_name,
                                languages=self.languages,
                                synthesizer=synthesizer_config,
                                topic=topic_config,
                                style=style_config,
                                persona=persona_config,
                                evaluator=evaluator_config,
                                openai=openai_config,
                            )
        return configs

    def synthesize(self, config):
        synthesizer = get_synthesizer(config)
        samples = synthesizer.generate()
        dict_samples = [sample.dict() for sample in samples]
        df = pd.DataFrame(dict_samples)
        df = df.sample(frac=1)
        X, y = df["text"].to_list(), df["completion"].to_list()
        return X, y

    def subsample_synthetic(self, X, y, n_subsamples):
        dataset = Dataset.from_dict({"text": X, "completion": y}).shuffle(
            seed=42
        )
        dataset = Dataset.from_pandas(
            dataset.to_pandas().groupby("completion").head(n_subsamples),
            preserve_index=False,
        ).shuffle(seed=42)
        X, y = dataset["text"], dataset["completion"]
        return X, y

    def subsample_real(self, n_subsamples):
        ds = self.get_incontext_examples(n_subsamples)
        ds = ds.shuffle(seed=42)
        X, y = ds["text"], ds["completion"]
        return X, y

    def mix_synthetic_with_real(self, X_synth, y_synth, n):
        ic_dataset = self.get_incontext_examples(n).to_pandas()
        synth_dataset = pd.DataFrame({"text": X_synth, "completion": y_synth})

        # Remove `n` samples per label
        df = (
            synth_dataset.groupby("completion")
            .apply(lambda x: x.iloc[n:])
            .reset_index(drop=True)
        )
        # Add the in context examples to the synthetic data with `n` removed
        df = pd.concat([df, ic_dataset], axis=0)
        # And shuffle
        df = df.sample(frac=1)

        return df["text"], df["completion"]

    def subsample_synthetic_and_mix_with_real(
        self, X_synth, y_synth, subsamples, n
    ):
        # Subsample synthetic
        X, y = self.subsample_synthetic(X_synth, y_synth, subsamples)
        # Mix with real data
        X, y = self.mix_synthetic_with_real(X, y, n)
        return X, y
