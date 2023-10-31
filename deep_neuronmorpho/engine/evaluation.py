"""Evaluate contrastive learning embeddings on benchmark classification task."""

import re

import dgl
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from torch import nn

from deep_neuronmorpho.data import NeuronGraphDataset


def evaluate_embeddings(
    embeddings: dict[str, NDArray],
    targets: dict[str, NDArray],
    search: bool = True,
    random_state: int | None = None,
) -> float:
    """Evaluate the quality of the contrastive learning embeddings on benchmark classification task.

    Args:
        embeddings (dict[str, NDArray]): Dictionary of embeddings with keys "train" and "test".
            Each value should be a 2D array-like object (samples, embeddings).
        targets (dict[str, str]): Dictionary of targets with keys "train" and "test".
            Each value should be a 1D array-like object of categorical string labels.
        search (bool, optional): Whether to perform a grid search for hyperparameter tuning.
        Defaults to True.
        random_state (int, optional): Controls the pseudo random number generation for shuffling the
        data for probability estimates.
        Pass an int for reproducible output across multiple function calls. Defaults to None.

    Returns:
        float: Mean accuracy of the classifier on the test set.
    """
    train_embeddings, test_embeddings = embeddings["train"], embeddings["test"]
    train_targets, test_targets = targets["train"], targets["test"]

    label_encoder = LabelEncoder().fit(train_targets)
    train_labels = label_encoder.transform(train_targets)
    test_labels = label_encoder.transform(test_targets)

    if search:
        params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        clf = GridSearchCV(SVC(gamma="auto"), params, cv=5, scoring="accuracy", verbose=0)

    else:
        clf = SVC(gamma="auto")

    clf.fit(train_embeddings, train_labels)

    return clf.score(test_embeddings, test_labels)


def create_embedding_df(dataset: NeuronGraphDataset, model: nn.Module) -> pd.DataFrame:
    """Create a DataFrame of model embeddings from a NeuronGraphDataset.

    Useful for visualizing embeddings using methods like UMAP or t-SNE.

    Args:
        dataset (NeuronGraphDataset): NeuronGraphDataset of graphs to get embeddings for.
        Dataset must contain graphs and labels.
        model (nn.Module): Model to get embeddings from.

    Returns:
        pd.DataFrame: DataFrame of embeddings with columns "neuron", "target", "labels",
        and embedding dimensions.
    """
    graphs, labels = dataset[:]
    batch_graphs = dgl.batch(graphs)
    embeds = model(batch_graphs)
    df_embed = pd.DataFrame(
        embeds.detach().numpy(),
        columns=[f"dim_{i}" for i in range(embeds.shape[1])],
    )
    pattern = r"[^-]+-(.*?)(?:-resampled_[^\.]+)?$"
    df_embed.insert(
        0, "neuron", [re.search(pattern, graph.id).group(1) for graph in dataset.graphs]
    )
    df_embed.insert(1, "target", labels)
    df_embed.insert(2, "labels", [dataset.glabel_dict[i.item()] for i in labels])

    return df_embed
