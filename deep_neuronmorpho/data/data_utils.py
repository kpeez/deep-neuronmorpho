"""Create training, validation, and test splits of the dataset."""
import random
import shutil
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
from dgl import DGLGraph
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from numpy.typing import NDArray
from scipy import stats

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


def create_dataloader(
    graph_dataset: DGLDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    **kwargs: Any,
) -> GraphDataLoader:
    """Create dataloaders for training and validation datasets.

    Args:
        graph_dataset (DGLDataset): Graph dataset.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the training data. Defaults to True.
        drop_last (bool): Whether to drop the last batch if it is smaller than the batch size.
        kwargs: Additional keyword arguments to pass to the parent torch.utils.data.DataLoader
        arguments such as num_workers, pin_memory, etc.

    Returns:
        GraphDataLoader: Dataloader of graph dataset.
    """
    graph_loader = GraphDataLoader(
        graph_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **kwargs,
    )

    return graph_loader


def parse_logfile(logfile: str | Path, metadata_file: str | Path) -> NDArray:
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
        input_dir: Path, split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
    ) -> None:
        """Create training, validation, and test data splits from a directory containing .swc files.

        Args:
            input_dir (Path): The path to the input directory containing the .swc files.
            split_ratios (Tuple[float, float, float], optional): Ratios for train, validation,
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
