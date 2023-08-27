"""Evaluate contrastive learning embeddings on benchmark classification task."""
from pathlib import Path

import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from ..utils import ModelConfig


def get_eval_targets(conf: ModelConfig) -> dict[str, NDArray]:
    """Get the target labels for the evaluation training and test sets."""
    eval_train_file = next(Path(f"{conf.dirs.metadata}").glob("*eval_train*.csv"))
    eval_test_file = next(Path(f"{conf.dirs.metadata}").glob("*eval_test*.csv"))
    return {
        "train": pd.read_csv(eval_train_file)["label"].to_numpy(),
        "test": pd.read_csv(eval_test_file)["label"].to_numpy(),
    }


def evaluate_embeddings(
    embeddings: dict[str, NDArray],
    targets: dict[str, NDArray],
    random_state: int | None = None,
) -> float:
    """Evaluate the quality of the contrastive learning embeddings using XGBoost classifier.

    Args:
        embeddings (dict[str, NDArray]): Dictionary of embeddings with keys "train" and "test".
            Each value should be a 2D array-like object (samples, embeddings).
        targets (dict[str, str]): Dictionary of targets with keys "train" and "test".
            Each value should be a 1D array-like object of categorical string labels.
        random_state (int, optional): Determines random number generation for XGBoost.
            Use an int for reproducible results. If None, the randomness will depend on np.random.

    Returns:
        float: Mean accuracy of the XGBoost classifier on the test set.
    """
    train_embeddings, test_embeddings = embeddings["train"], embeddings["test"]
    train_targets, test_targets = targets["train"], targets["test"]

    label_encoder = LabelEncoder().fit(train_targets)
    train_labels = label_encoder.transform(train_targets)
    test_labels = label_encoder.transform(test_targets)

    clf_xgb = XGBClassifier(random_state=random_state)
    clf_xgb.fit(train_embeddings, train_labels)

    return clf_xgb.score(test_embeddings, test_labels)
