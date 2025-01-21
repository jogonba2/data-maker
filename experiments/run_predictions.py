from models import fs_dec_tuning
import pandas as pd
from configs import mapping
from datasets import load_from_disk
import argparse
from pathlib import Path
from tqdm import tqdm
from utils import bm25


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run predictions on synthetic datasets"
    )

    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset"
    )

    return parser.parse_args()


def main(dataset):
    # Load original dataset
    cls_ = mapping[dataset]

    X_test, y_test = (
        cls_.dataset["test"]["text"],
        cls_.dataset["test"]["completion"],
    )

    # Evaluate on the synthetic dataset
    for path in tqdm(Path(f"cache/{dataset}").iterdir()):
        config_name = path.as_posix().split("/")[-1]
        dataset_name = path.as_posix().split("/")[1]
        # Load the synthetic dataset
        synthetic_dataset = load_from_disk(path)
        X_train, y_train = (
            synthetic_dataset["text"],
            synthetic_dataset["completion"],
        )
        # Train & predict with DEC
        preds, dec_f1 = fs_dec_tuning(X_train, y_train, X_test, y_test)
        print(f"F1 at {config_name}: {dec_f1}")

        # Get the errors
        df = pd.DataFrame(
            {"text": X_test, "reference": y_test, "prediction": preds}
        )

        # Get the most similar sentence using BM25
        retrieved_texts, retrieved_labels, scores = (
            bm25(df["text"].tolist(), X_train, y_train)
        )
        df["most_similar_synthetic_text"] = retrieved_texts
        df["most_similar_synthetic_label"] = retrieved_labels
        df["most_similar_synthetic_score"] = scores

        # Store in disk
        predictions_path = Path(f"predictions/{dataset_name}")
        predictions_path.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            predictions_path / f"{config_name}.tsv", index=False, sep="\t"
        )


if __name__ == "__main__":
    args = parse_args()
    main(args.dataset)
