"""Create training, validation, and test splits of the dataset."""

import random
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from dgl import DGLGraph
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

from deep_neuronmorpho.utils import ProgressBar


def compute_edge_weights(G: nx.DiGraph, path_idx: int, epsilon: float = 1.0) -> nx.DiGraph:
    """Compute edge attention weights for a graph.

    Based on the method described in [Zhao et al. 2022](https://ieeexplore.ieee.org/document/9895206).

    Args:
        G (nx.DiGraph): Graph to compute edge weights for.
        path_idx (int): Index of path distance in node attributes.
        epsilon (float, optional): Small constant to prevent division by zero. Defaults to 1.0.

    Returns:
        nx.DiGraph: Graph with attention weights added as edge attributes.
    """
    for u, v in G.edges:
        G.add_edge(v, u)

    node_coeffs = {
        node: 1.0 / (ndata["nattrs"][path_idx] + epsilon) for node, ndata in G.nodes(data=True)
    }
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        local_coeffs = np.array([node_coeffs[neighbor] for neighbor in neighbors], dtype=np.float32)
        attention_coeffs = np.exp(local_coeffs) / np.sum(np.exp(local_coeffs))

        edge_weights = (
            (neighbor, coeff) for neighbor, coeff in zip(neighbors, attention_coeffs, strict=True)
        )
        for neighbor, weight in edge_weights:
            G[node][neighbor]["edge_weight"] = weight

    return G


def compute_graph_attrs(graph_attrs: list[float]) -> list[float]:
    """Compute summary statistics for a list of graph attributes.

    Args:
        graph_attrs (list[float]): Graph attribute data.

    Returns:
        list[float]: Summary statistics of graph attributes. In the following order:
         min, mean, median, max, std, num of observations

    """
    res = stats.describe(graph_attrs)
    attr_stats = [
        res.minmax[0],
        res.mean,
        np.median(graph_attrs),
        res.minmax[1],
        (res.variance**0.5),
        res.nobs,
    ]
    return attr_stats


def graph_is_broken(graph: DGLGraph) -> bool:
    """Determines if a graph is broken by checking if any node attributes are NaN.

    Args:
        graph (DGLGraph): The graph to check.

    Returns:
        bool: True if the graph is broken, False otherwise.
    """
    g_ndata = graph.ndata["nattrs"]
    nan_indices = torch.nonzero(torch.isnan(g_ndata))

    return len(nan_indices[:, 1].unique()) > 0


def add_graph_labels(label_file: str | Path, graphs: list[DGLGraph]) -> tuple[Tensor, dict]:
    """Add graph labels to the dataset.

    Note: The label file should be a CSV file with columns 'neuron_name' and 'label'. Other column names are ignored.

    Args:
        label_file (str | Path): Path to the label file.
        graphs (list[DGLGraph]): List of graphs in the dataset.

    Returns:
        tuple[torch.Tensor, dict]: A tuple containing the graph labels and a dictionary mapping
        graph labels to integers.
    """
    label_data = pd.read_csv(label_file)
    label_encoder = LabelEncoder()
    label_encoder.fit(label_data["label"])
    label_to_int = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_), strict=False)
    )
    # map neuron names to labels and then to their encoded integers
    neuron_to_label = dict(zip(label_data["neuron_name"], label_data["label"], strict=False))
    # map neuron -> labels -> encoded integers
    _labels = [label_to_int.get(neuron_to_label.get(g.id, None), -1) for g in graphs]
    labels = torch.tensor(_labels, dtype=torch.long)

    return labels, label_to_int


def parse_dataset_log(logfile: str | Path, metadata_file: str | Path) -> pd.DataFrame:
    """Parse log file assocaited with dataset to get the file name and label for each sample.

    When creating the NeuronGraphDataset, the file names are sorted in alphabetical order and
    written to the log file. This function parses the log file to get the file names, and gets
    the labels from the metadata.

    Args:
        logfile (str | Path): Path to the log file.
        metadata_file (str | Path): Path to metadata file.

    Returns:
        pd.Series: A dataframe containing the file name and label for each processed sample.
    """
    metadata_file = (
        metadata_file if Path(metadata_file).suffix == ".csv" else f"{metadata_file}.csv"
    )
    metadata = pd.read_csv(metadata_file)
    log_data = pd.read_csv(logfile, skiprows=1, header=None, names=["timestamps", "log"])
    log_data["file_name"] = log_data["log"].str.extract(r"mouse-(.*?)-resampled_\d{2}um")
    log_data["label"] = log_data["file_name"].map(metadata.set_index("neuron_name")["dataset"])

    return log_data


if __name__ == "__main__":
    from typer import Typer

    app = Typer()

    @app.command()
    def create_data_splits(
        input_dir: Path,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ) -> None:
        """Create training, validation, and test data splits from a directory containing .swc files.

        Args:
            input_dir (Path): The path to the input directory containing the .swc files.
            split_ratios (tuple[float, float, float], optional): Ratios for train, validation,
            and test splits. Defaults to (0.7, 0.2, 0.1).
            seed (int, optional): The random seed for shuffling the data. Defaults to 42.

        This function creates subdirectories named 'train', 'val', and 'test' in the parent of the
        input directory and move the .swc files into the corresponding subdirectories according
        to the specified split ratios.
        """
        input_dir = Path(input_dir)
        swc_files = list(input_dir.glob("*.swc"))

        random.seed(seed)
        random.shuffle(swc_files)

        num_files = len(swc_files)
        train_end_idx = int(num_files * split_ratios[0])
        val_end_idx = train_end_idx + int(num_files * split_ratios[1])

        train_dir = input_dir / "train"
        val_dir = input_dir / "val"
        test_dir = input_dir / "test"

        for directory in [train_dir, val_dir, test_dir]:
            directory.mkdir(exist_ok=True)

        split_mapping = {
            "train": (0, train_end_idx, train_dir),
            "val": (train_end_idx, val_end_idx, val_dir),
            "test": (val_end_idx, num_files, test_dir),
        }

        for _split, (start, end, target_dir) in split_mapping.items():
            split_files = swc_files[start:end]
            pbar = ProgressBar(split_files, desc=f"Moving {_split} files: ")
            for file in pbar:
                shutil.move(str(file), str(target_dir / file.name))

    print("Splitting dataset into train, val, and test sets...")
    app()
    print("Done splitting dataset.")
