"""Create training, validation, and test splits of the dataset."""
import random
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dgl import DGLGraph
from scipy import stats
from torch import Tensor

from ..utils import ProgressBar


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

    Args:
        label_file (str | Path): Path to the label file.
        graphs (list[DGLGraph]): List of graphs in the dataset.

    Returns:
        tuple[torch.Tensor, dict]: A tuple containing the graph labels and a dictionary mapping
        graph labels to integers.
    """
    label_data = pd.read_csv(label_file)
    unique_labels = label_data["dataset"].unique()
    glabel_dict = dict(zip(range(len(unique_labels)), unique_labels, strict=True))
    neuron_label_dict = dict(zip(label_data["neuron_name"], label_data["dataset"], strict=True))
    glabel_dict_rev = {v: k for k, v in glabel_dict.items()}
    # Extract neuron names from graph ids and assign labels
    pattern = r"[^-]+-(.*?)(?:-resampled_[^\.]+)?$"
    _labels = []
    for graph in graphs:
        match = re.search(pattern, graph.id)
        if match:
            neuron_name = match.group(1)
            neuron_label = neuron_label_dict.get(str(neuron_name))
            _labels.append(glabel_dict_rev.get(str(neuron_label), -1))
        else:
            _labels.append(-1)

    labels = torch.tensor(_labels, dtype=torch.long)

    return labels, glabel_dict


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
