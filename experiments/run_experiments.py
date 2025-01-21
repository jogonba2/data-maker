# FIRST SET OPENAI API KEY
import os
import openai

openai.api_key = os.environ["SYNTHETIC_DATA_OPENAI_API_KEY"]

from datamaker import *
from configs import mapping
from models import fs_dec_tuning, fs_lr_tuning
from datasets import Dataset, load_from_disk
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import numpy as np
import os
import argparse

# REDO JUST IN CASE
openai.api_key = os.environ["SYNTHETIC_DATA_OPENAI_API_KEY"]

"""
# Hypothesis
Most of the work on synthetic data explores the generation of large amounts of data to finetune models on large data regimes of several thousands samples per label.
However, there are specific scenarios when it is not possible:

1) Models as a service which limits the amounts of data for training, e.g. 1024 samples in Symanto Brain
2) Limited budget for training: models as a service who charge for the number of tokens in the training data, e.g. OpenAI
3) Scarcity of task-specific examples, e.g., classifying family dynamics needs private conversations of parent-child or detecting coercive language in private negotiations

The behavior of few-shot models for text classification when trained on small datasets comprised of synthetic data have not been explored.

# Experiments:
models -> TF-IDF+LR and DEC (DONE)
personas -> w/ and w/o personas (DONE)
topics -> w/ and w/o topics (DONE)
icl examples -> 0, 4, 8 (DONE)
evaluator -> w and w/o filtering (DONE)
style -> w/ and w/o style (DONE)
number of samples -> create 256 samples and eval on reducing sizes one the dataset is created (128 / 64 / 32 / 16 / 8 / 4) + zero shot (DONE)
data -> full real, full synthetic, mixed (synthetic + as many real data per label as in-context examples. This one helps to fully answer question (1))
verbalizations -> same than in the LT paper for the shared datasets

# Questions:
1) If I have n human annotated examples, it is better to train a few-shot model with them, or use them as ICE to generate synthetic data?
2) Related to 1) how many samples must I generate to outperform the few-shot model trained with human-annotated examples?
3) The more synthetic data the better?
4) Do personas help to improve performance?
5) Do ICE help to improve performance?
6) Do topics help to improve performance?
7) Do the evaluator helps to improve performance?
8) the better the synthetic data, the better the mixed data under the same config?
"""


def build_row(config_name, others, replace_with_NA=False):
    values = [key.split("+") for key in config_name.split("_")]
    row = {val[0]: val[1] if not replace_with_NA else "N/A" for val in values}
    row = {**row, **others}
    return row


def cache_results(dataset_name, results, confs_cache):
    df = pd.DataFrame(results)
    results_path = Path("results")
    cache_path = Path("cache")
    results_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path / f"{dataset_name}.tsv", index=False, sep="\t")
    confs_cache = pd.DataFrame({"done_confs": confs_cache})
    confs_cache.to_csv(
        cache_path / f"done_confs_{dataset_name}.tsv", index=False, sep="\t"
    )


def get_cached_results(dataset_name):
    results, dones = [], []
    try:
        df = pd.read_csv(f"results/{dataset_name}.tsv", delimiter="\t")
        df = df.replace(np.nan, "N/A")
        results = df.to_dict("records")
    except:
        pass
    try:
        confs_cache = pd.read_csv(
            f"cache/done_confs_{dataset_name}.tsv", delimiter="\t"
        )
        dones = confs_cache["done_confs"].to_list()
    except:
        pass
    return results, dones


def cache_dataset(dataset_name, conf, X, y):
    """Caches a synthetic dataset"""
    cache_path = Path(f"cache/{dataset_name}")
    cache_path.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_dict({"text": X, "completion": y})
    ds.save_to_disk(f"cache/{dataset_name}/{conf}")


def get_cached_dataset(dataset_name, conf):
    try:
        ds = load_from_disk(f"cache/{dataset_name}/{conf}")
        return ds["text"], ds["completion"]
    except:
        return None, None


def run_experiments(dataset_name):
    cls_ = mapping[dataset_name]
    confs = cls_.get_all_configs()
    results, done_confs = get_cached_results(dataset_name)
    print(list(confs.keys()))
    exit()
    ### BEGIN TEST ###
    test_confs = ["persona+yes_topic+yes_style+yes_evaluator+yes_icexamples+0_LLM+gpt-4o-mini_numsamples+16", "persona+yes_topic+yes_style+no_evaluator+yes_icexamples+0_LLM+gpt-4o-mini_numsamples+16"]
    confs = good_confs
    ### END TEST ###
    if "dec_zs" not in done_confs:
        results += run_zero_shot_experiment(cls_)
        done_confs.append("dec_zs")
        cache_results(dataset_name, results, done_confs)
    else:
        print("**Skipping dec_zs since it is already in cache**")

    if "fs" not in done_confs:
        results += run_few_shot_experiments(cls_)
        done_confs.append("fs")
        cache_results(dataset_name, results, done_confs)
    else:
        print("**Skipping fs since it is already in cache**")

    for conf in tqdm(confs):
        if conf not in done_confs:
            print(f"** Doing {conf} since it is not in cache")
            results += run_synthetic_experiments(conf, confs[conf], cls_)
            results += run_mixed_experiments(conf, confs[conf], cls_)
            done_confs.append(conf)
            cache_results(dataset_name, results, done_confs)
        else:
            print(f"**Skipping {conf} since it is already in cache**")


def run_zero_shot_experiment(cls_):
    """
    Run experiment with zero shot model
    """
    X_te, y_te = (
        cls_.dataset["test"]["text"],
        cls_.dataset["test"]["completion"],
    )
    y_tr = cls_.dataset["train"]["completion"]
    _, zs_result = fs_dec_tuning(None, y_tr, X_te, y_te)
    return [
        {
            "persona": "N/A",
            "topic": "N/A",
            "evaluator": "N/A",
            "style": "N/A",
            "icexamples": "N/A",
            "LLM": "N/A",
            "numsamples": 0,
            "synthetic": "no",
            "dataset": cls_.task_name,
            "dec-f1": zs_result,
            "lr-f1": "N/A",
        }
    ]


def run_few_shot_experiments(cls_):
    """
    Run experiment with few-shot models using real data
    """
    results = []
    X_te, y_te = (
        cls_.dataset["test"]["text"],
        cls_.dataset["test"]["completion"],
    )
    # Run with the complete number of samples
    X, y = cls_.subsample_real(cls_.num_samples)
    print("NS:", cls_.num_samples, "X:", len(X), "Y:", Counter(y))
    _, lr_result = fs_lr_tuning(X, y, X_te, y_te)
    _, dec_result = fs_dec_tuning(X, y, X_te, y_te)
    row = {
        "persona": "N/A",
        "topic": "N/A",
        "evaluator": "N/A",
        "style": "N/A",
        "icexamples": "N/A",
        "LLM": "N/A",
        "numsamples": cls_.num_samples,
        "synthetic": "no",
        "dataset": cls_.task_name,
        "dec-f1": dec_result,
        "lr-f1": lr_result,
    }
    results.append(row)
    num_samples = [8, 4] #[512, 256, 128, 64, 32, 16, 8, 4]
    for ns in num_samples:
        X, y = cls_.subsample_real(ns)
        print("NS:", ns, "X:", len(X), "Y:", Counter(y))
        _, lr_result = fs_lr_tuning(X, y, X_te, y_te)
        _, dec_result = fs_dec_tuning(X, y, X_te, y_te)
        row = {
            "persona": "N/A",
            "topic": "N/A",
            "evaluator": "N/A",
            "style": "N/A",
            "icexamples": "N/A",
            "LLM": "N/A",
            "numsamples": ns,
            "synthetic": "no",
            "dataset": cls_.task_name,
            "dec-f1": dec_result,
            "lr-f1": lr_result,
        }
        results.append(row)
    return results


def run_synthetic_experiments(config_name, config, cls_):
    """
    Run experiments with few-shot models using fully-synthetic data
    """
    results = []
    X_te, y_te = (
        cls_.dataset["test"]["text"],
        cls_.dataset["test"]["completion"],
    )

    # Run with the complete number of samples
    ## Synthesize
    X_synth, y_synth = get_cached_dataset(cls_.task_name, config_name)
    if not X_synth:
        print("** Dataset not in cache, generating it... **")
        X_synth, y_synth = cls_.synthesize(config)
        cache_dataset(cls_.task_name, config_name, X_synth, y_synth)
    else:
        print("** Dataset loaded from cache **")
    print("NS:", cls_.num_samples, "X:", len(X_synth), "Y:", Counter(y_synth))

    ## Train model with synthetic data
    _, lr_result = fs_lr_tuning(X_synth, y_synth, X_te, y_te)
    _, dec_result = fs_dec_tuning(X_synth, y_synth, X_te, y_te)
    row = build_row(
        config_name,
        {
            "synthetic": "yes",
            "lr-f1": lr_result,
            "dec-f1": dec_result,
            "dataset": cls_.task_name,
        },
    )
    results.append(row)

    # Evaluate subsamples
    num_samples = [8, 4] #[512, 256, 128, 64, 32, 16, 8, 4]
    for ns in num_samples:
        # Train model with subsampled synthetic data
        X, y = cls_.subsample_synthetic(X_synth, y_synth, ns)
        print("NS:", ns, "X:", len(X), "Y:", Counter(y))
        _, lr_result = fs_lr_tuning(X, y, X_te, y_te)
        _, dec_result = fs_dec_tuning(X, y, X_te, y_te)
        row = build_row(
            config_name,
            {
                "synthetic": "yes",
                "dec-f1": dec_result,
                "lr-f1": lr_result,
                "dataset": cls_.task_name,
            },
        )
        row["numsamples"] = ns
        results.append(row)

    return results


def run_mixed_experiments(config_name, config, cls_):
    """
    Run experiments with mixed synthetic data and real data.
    The real data is only considered when
    """
    X_te, y_te = (
        cls_.dataset["test"]["text"],
        cls_.dataset["test"]["completion"],
    )
    results = []

    # Since we are using IC examples as real data, if the config do not use them
    # then no real data is used.
    if not config.synthesizer.examples:
        print(
            f"Skipping mixed experiments with {config_name}, since it does not use ICL"
        )
        return results

    # Load from the disk as it is already cached from `run_synthetic_experiments`
    X_synth, y_synth = get_cached_dataset(cls_.task_name, config_name)

    # Mix the synthetic data with real in context examples
    num_icl_examples = len(config.synthesizer.examples) // len(
        set(cls_.dataset["train"]["completion"])
    )
    X_mix, y_mix = cls_.mix_synthetic_with_real(
        X_synth, y_synth, num_icl_examples
    )
    print("NS:", cls_.num_samples, "X:", len(X_synth), "Y:", Counter(y_synth))

    ## Train model with mixed data
    _, lr_result = fs_lr_tuning(X_mix, y_mix, X_te, y_te)
    _, dec_result = fs_dec_tuning(X_mix, y_mix, X_te, y_te)
    row = build_row(
        config_name,
        {
            "synthetic": "mixed",
            "lr-f1": lr_result,
            "dec-f1": dec_result,
            "dataset": cls_.task_name,
        },
    )
    results.append(row)

    num_samples = [8, 4] #[512, 256, 128, 64, 32, 16, 8, 4]
    for ns in num_samples:
        # Train model with subsampled synthetic data
        # NOTE: mixing can only be applied when `num_samples` > |IC examples|
        # otherwise it will be completely real data, e.g. with 4, 8 ICE examples:
        # NUM_SAMPLES Allowed_|ICE|
        # 16          4, 8
        # 8           4
        # 4           -
        if ns <= num_icl_examples:
            print(
                f"**Skipping mixed subsampling since {ns}(num samples)<={num_icl_examples}(ICE)"
            )
            continue
        X, y = cls_.subsample_synthetic_and_mix_with_real(
            X_synth, y_synth, ns, num_icl_examples
        )
        print("NS:", ns, "X:", len(X), "Y:", Counter(y))
        _, lr_result = fs_lr_tuning(X, y, X_te, y_te)
        _, dec_result = fs_dec_tuning(X, y, X_te, y_te)
        row = build_row(
            config_name,
            {
                "synthetic": "mixed",
                "lr-f1": lr_result,
                "dec-f1": dec_result,
                "dataset": cls_.task_name,
            },
        )
        row["numsamples"] = ns
        results.append(row)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run classification experiments"
    )

    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(args.dataset)
