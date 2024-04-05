"""Evaluate contrastive learning embeddings on benchmark classification task."""

from pathlib import Path
from typing import Any

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn

from deep_neuronmorpho.data import NeuronGraphDataset
from deep_neuronmorpho.models import MACGNN
from deep_neuronmorpho.utils.model_config import Config

from .trainer_utils import Checkpoint


def create_embedding_df(dataset: NeuronGraphDataset, model: nn.Module) -> pd.DataFrame:
    """Create a DataFrame of model embeddings from a NeuronGraphDataset.

    Useful for visualizing embeddings using methods like UMAP or t-SNE.

    Args:
        dataset (NeuronGraphDataset): NeuronGraphDataset of graphs to get embeddings for.
        Dataset must contain graphs and labels.
        model (nn.Module): Model to get embeddings from.

    Returns:
        DataFrame: DataFrame of embeddings with columns "neuron_name", "target", "labels",
        and embedding dimensions.
    """
    model.eval()
    if dataset.num_classes is None:
        graphs = dataset[:]
        labels = np.zeros(len(graphs), dtype=int)
    else:
        graphs, labels = dataset[:]
    batch_graphs = dgl.batch(graphs)
    with torch.inference_mode():
        embeds = model(batch_graphs, batch_graphs.ndata["nattrs"])
    df_embed = pd.DataFrame(
        embeds.detach().numpy(),
        columns=[f"dim_{i}" for i in range(embeds.shape[1])],
    )

    neuron_names = [graph.id if graph.id is not None else "N/A" for graph in dataset.graphs]

    df_embed.insert(0, "neuron_name", neuron_names)
    if dataset.num_classes is not None and dataset.glabel_dict is not None:
        df_embed.insert(1, "target", labels)
        label_dict = dict(
            zip(dataset.glabel_dict.values(), dataset.glabel_dict.keys(), strict=True)
        )
        df_embed.insert(2, "label", [label_dict.get(i.item(), None) for i in labels])

    return df_embed


def repeated_kfold_eval(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    model: ClassifierMixin,
    n_splits: int = 5,
    n_repeats: int = 10,
    standardize: bool = True,
    random_state: int | None = None,
) -> tuple[float, float, float, float]:
    """Perform repeated k-fold cross-validation and testing.

    For small datasets, it is important to perform repeated k-fold cross-validation to get a more reliable estimate of the model's performance.
    Otherwise, the performance estimate on the test set can be highly variable depending on the random seed.
    This function performs repeated k-fold cross-validation and testing on the test set.

    Args:
        X (NDArray, pd.DataFrame): Feature matrix. If a DataFrame is provided, it will be converted to a numpy array.
        y (np.array): Target vector
        model (nn.Module): Model to evaluate
        n_splits (int, optional): Number of folds to perform CV on. Defaults to 5.
        n_repeats (int, optional): Number of times to repeat k-fold CV. Defaults to 10.
        random_state (int | None, optional): Set random seed. Defaults to None.

    Returns:
        tuple[float, float, float, float]: cv_mean, cv_std, test_mean, test_std
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    test_scores = []
    cv_scores = []
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    print(rskf)
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        cv_score = cross_val_score(model, X_train, y_train, cv=n_splits, scoring="f1_micro").mean()
        cv_scores.append(cv_score)

        model.fit(X_train, y_train)
        test_score = accuracy_score(y_test, model.predict(X_test))
        test_scores.append(test_score)

    return np.mean(cv_scores), np.std(cv_scores), np.mean(test_scores), np.std(test_scores)


def get_model_embeddings(
    model_dir: str | Path,
    epoch: int,
    dataset: DGLDataset,
) -> pd.DataFrame:
    config_file = next(iter(Path(model_dir).glob(("*.yml"))))
    ckpt_dir = Path(model_dir) / "ckpts"
    last_epoch = max(int(ckpt.stem.split("epoch_")[-1]) for ckpt in ckpt_dir.glob("*.pt"))
    if epoch > last_epoch:
        print(f"Epoch {epoch} does not exist. Using last epoch: {last_epoch} instead.")
        epoch = last_epoch
    ckpt_file = next(ckpt_dir.glob(f"*{epoch:04d}*.pt"))
    conf = Config.from_yaml(config_file)
    base_model = MACGNN(conf.model)
    model = Checkpoint.load_model(ckpt_file=ckpt_file, model=base_model)
    df_embeds = create_embedding_df(dataset, model)

    return df_embeds


def evaluate_embeddings(
    df_embeds: pd.DataFrame,
    df_label: pd.DataFrame,
    clf: ClassifierMixin,
    **kwargs: Any,
) -> tuple[float, float, float, float]:
    """Evaluate model embeddings on classification task.

    Args:
        df_embeds (pd.DataFrame): DataFrame of model embeddings.
        df_label (pd.DataFrame): DataFrame of neuron names and labels.
        clf (Classifier): Scikit-learn classifier to use for evaluation.

    Returns:
        tuple[float, float, float, float]: cv_mean, cv_std, test_mean, test_std
    """
    df_embeds = df_embeds.copy()
    label_dict = dict(zip(df_label["neuron_name"], df_label["label"], strict=True))
    df_embeds["label"] = df_embeds["neuron_name"].map(label_dict)
    df_embeds.drop(columns=["neuron_name"], inplace=True)
    labels = df_embeds.pop("label")

    return repeated_kfold_eval(df_embeds, labels, clf, **kwargs)
