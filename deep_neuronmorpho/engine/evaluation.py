"""Evaluate contrastive learning embeddings on benchmark classification task."""

from pathlib import Path
from typing import Any

import dgl
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance_matrix
from sklearn.base import ClassifierMixin
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from umap import UMAP

from deep_neuronmorpho.data import NeuronGraphDataset

from .contrastive_trainer import ContrastiveGraphModule


def create_embedding_df(model: nn.Module, dataset_file: str | Path) -> pd.DataFrame:
    """Create a DataFrame of model embeddings from a NeuronGraphDataset.

    Useful for visualizing embeddings using methods like UMAP or t-SNE.

    Args:
        dataset_file (NeuronGraphDataset): File to NeuronGraphDataset of graphs to get embeddings for.
        model (nn.Module): Model to get embeddings from.

    Returns:
        DataFrame: DataFrame of embeddings with columns "neuron_name", "target", "labels",
        and embedding dimensions.
    """

    dataset = NeuronGraphDataset(name=dataset_file, from_file=True)
    model.eval()
    if dataset.num_classes is None:
        graphs = dataset[:]
        labels = np.zeros(len(graphs), dtype=int)
    else:
        graphs, labels = dataset[:]
    batch_graphs = dgl.batch(graphs)
    model.eval()
    with torch.inference_mode():
        embeds = model(batch_graphs, batch_graphs.ndata["nattrs"])
    df_embed = pd.DataFrame(
        embeds.detach().numpy(),
        columns=[f"dim_{i}" for i in range(embeds.shape[1])],
    )

    neuron_names = [graph.id if graph.id is not None else "N/A" for graph in dataset.graphs]

    df_embed.insert(0, "neuron_name", neuron_names)
    if dataset.num_classes is not None and dataset.glabel_dict is not None:
        label_dict = dict(
            zip(dataset.glabel_dict.values(), dataset.glabel_dict.keys(), strict=True)
        )
        df_embed.insert(1, "label", [label_dict.get(i.item(), None) for i in labels])

    return df_embed


def repeated_kfold_eval(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    model: ClassifierMixin,
    n_splits: int = 5,
    n_repeats: int = 10,
    standardize: bool = True,
    random_state: int | None = None,
) -> tuple[float, float]:
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
        standardize (bool, optional): Whether to standardize the data. Defaults to True.
        random_state (int | None, optional): Set random seed. Defaults to None.

    Returns:
        tuple[float, float]: cv_mean, cv_std
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    scores = []
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average="micro")
        scores.append(score)

    return np.mean(scores), np.std(scores)


def get_model_embeddings(ckpt_file: str | Path, dataset_file: str | Path) -> pd.DataFrame:
    """Get model embeddings from a saved model checkpoint.

    Args:
        ckpt_file (str | Path): Path to the model checkpoint file.
        dataset_file (str | Path): Path to NeuronGraphDataset to get embeddings for.

    Returns:
        pd.DataFrame: Model embeddings as a DataFrame.
    """
    loaded_model = ContrastiveGraphModule.load_from_checkpoint(ckpt_file)
    model = loaded_model.model
    df_embeds = create_embedding_df(model, dataset_file=dataset_file)

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


def get_similar_neurons(
    df: pd.DataFrame,
    target_sample: str,
    k: int,
    closest: bool = True,
    within_class: bool = False,
) -> dict[str, float]:
    """Find k-nearest (or farthest) neighbors of a target sample in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame of samples with features.
        target_sample (str): Name of target sample.
        k (int): Number of neighbors to find.
        closest (bool, optional): Find closest neighbors if True, farthest if False. Defaults to True.
        within_class (bool, optional): Only consider samples from the same class as the target sample. Defaults to False.

    Returns:
        dict[str, float]: Dictionary of neighbors and their distances to the target sample.
    """
    if target_sample not in df["neuron_name"].values:
        raise ValueError("Target sample not found in DataFrame.")

    drop_cols = ["neuron_name"]
    if "label" in df.columns:
        drop_cols.append("label")

    if within_class:
        cls_label = df.query("neuron_name == @target_sample")["label"].tolist()[0]
        df = df.query(f"label == '{cls_label}'")

    features = df.drop(columns=drop_cols)
    dist_matrix = pd.DataFrame(
        distance_matrix(features.values, features.values),
        index=df["neuron_name"],
        columns=df["neuron_name"],
    )
    target_distances = dist_matrix[target_sample].sort_values()
    neighbors = target_distances.iloc[1 : k + 1] if closest else target_distances.iloc[-k:]

    return neighbors.to_dict()


def reduce_dimensionality(
    df: pd.DataFrame,
    method: str = "UMAP",
    n_components: int = 2,
    random_state: int | None = 42,
    **kwargs: Any,
) -> pd.DataFrame:
    """Reduce the dimensionality of a DataFrame using UMAP or t-SNE.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be reduced.
        method (str, optional): The dimensionality reduction method to use. Defaults to "UMAP".
        n_components (int, optional): The number of components in the reduced space. Defaults to 2.
        random_state (int | None, optional): The random state for reproducibility. Defaults to 42.
        **kwargs (Any): Additional keyword arguments to be passed to the dimensionality reduction method.

    Returns:
        pd.DataFrame: The reduced DataFrame with the specified number of components.
    """

    UMAP_DEFAULTS = {"n_neighbors": 15, "min_dist": 0.1}
    TSNE_DEFAULTS = {"perplexity": 30}

    if method.lower() == "umap":
        UMAP_DEFAULTS.update(kwargs)
        reducer = UMAP(
            n_components=n_components,
            random_state=random_state,
            **UMAP_DEFAULTS,
        )
    elif method.lower() == "tsne":
        TSNE_DEFAULTS.update(kwargs)
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            **TSNE_DEFAULTS,
        )
    else:
        raise ValueError("Unsupported dimensionality reduction method. Choose 'UMAP' or 'tSNE'.")

    drop_cols = ["neuron_name", "label"] if "label" in df.columns else ["neuron_name"]

    embeddings = reducer.fit_transform(df.drop(columns=drop_cols))
    df_embeds = pd.DataFrame(embeddings, columns=[f"{method} 1", f"{method} 2"])
    df_embeds.insert(0, "neuron_name", df["neuron_name"])
    if "label" in df.columns:
        df_embeds.insert(1, "label", df["label"])

    return df_embeds
