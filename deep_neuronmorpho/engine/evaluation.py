"""Evaluate contrastive learning embeddings on benchmark classification task."""

import dgl
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    train_embeds, test_embeds = embeddings["train"], embeddings["test"]
    scaler = StandardScaler().fit(train_embeds)
    train_embeds_scaled = scaler.transform(train_embeds)
    test_embeds_scaled = scaler.transform(test_embeds)
    train_targets, test_targets = targets["train"], targets["test"]
    label_encoder = LabelEncoder().fit(train_targets)
    train_labels = label_encoder.transform(train_targets)
    test_labels = label_encoder.transform(test_targets)

    if search:
        params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        clf = GridSearchCV(SVC(gamma="auto"), params, cv=5, scoring="accuracy", verbose=0)

    else:
        clf = SVC(gamma="auto")

    clf.fit(train_embeds_scaled, train_labels)

    return clf.score(test_embeds_scaled, test_labels)


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
    if dataset.num_classes is None:
        graphs = dataset[:]
        labels = np.zeros(len(graphs), dtype=int)
    else:
        graphs, labels = dataset[:]
    batch_graphs = dgl.batch(graphs)
    embeds = model(batch_graphs, batch_graphs.ndata["nattrs"])
    df_embed = pd.DataFrame(
        embeds.detach().numpy(),
        columns=[f"dim_{i}" for i in range(embeds.shape[1])],
    )

    neuron_names = [graph.id if graph.id is not None else "N/A" for graph in dataset.graphs]

    df_embed.insert(0, "neuron_name", neuron_names)
    if dataset.num_classes is not None:
        df_embed.insert(1, "target", labels)
        df_embed.insert(
            2,
            "labels",
            [
                dataset.glabel_dict[i.item()] if dataset.glabel_dict is not None else None
                for i in labels
            ],
        )

    return df_embed
