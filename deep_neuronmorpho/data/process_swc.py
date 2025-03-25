"""Process SWC files."""

import logging
import pickle
from datetime import datetime as dt
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from morphopy.neurontree import NeuronTree as nt
from morphopy.neurontree.utils import get_standardized_swc
from sklearn.decomposition import PCA
from typer import Argument, Option, Typer

from deep_neuronmorpho.utils import ProgressBar

app = Typer()


class SWCData:
    """Class for loading, preprocessing, and manipulating SWC data.

    This class provides methods for loading SWC data from a file, standardizing the data by
    aligning it to principal axes and centering it at the origin, removing axon nodes from the
    reconstruction, downsampling the data, plotting the raw and standardized data, and saving the
    data to a CSV file.

    Args:
        swc_file (str | Path): Path to the SWC file.
        standardize (bool, optional): Flag indicating whether to standardize the data. Defaults to True.
        align (bool, optional): Flag indicating whether to align the data to principal axes. Defaults to True.
        resample_dist (float, optional): Value to downsample the data. Default is 1.0 (no downsampling).

    Attributes:
        swc_file (Path): Path to the SWC file.
        raw_data (pd.DataFrame): Raw SWC data.
        data (pd.DataFrame): Standardized SWC data.
        ntree (nt.NeuronTree): NeuronTree object representing the SWC data.

    """

    def __init__(
        self,
        swc_file: str | Path,
        standardize: bool = True,
        align: bool = True,
        resample_dist: float | None = None,
    ):
        self.swc_file = Path(swc_file)
        self._raw_data = self.load_swc_data(self.swc_file)
        self._data = self._raw_data
        self._ntree = nt.NeuronTree(self._data)

        if resample_dist is not None:
            self.resample_distance(resample_dist)

        if standardize:
            self._data = self.standardize_swc(self._data, align=align)
            self._ntree = nt.NeuronTree(self._data)

    @property
    def raw_data(self) -> pd.DataFrame:
        """Return the raw swc data."""
        return self._raw_data

    @property
    def data(self) -> pd.DataFrame:
        """Return the (possibly) standardized swc data."""
        return self._data

    @property
    def ntree(self) -> nt.NeuronTree:
        """Return the NeuronTree object."""
        return self._ntree

    @staticmethod
    def load_swc_data(swc_file: str | Path) -> pd.DataFrame:
        """Load swc data from a file.

        Args:
            swc_file (str | Path): Path to the SWC file.

        Raises:
            AssertionError: If no dendrites are connected to the soma.

        """
        with open(swc_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        # Find the start of data
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() and not (line.startswith("#") or line.startswith("n")):
                start_idx = i
                break
        data = pd.DataFrame(
            [line.split() for line in lines[start_idx:]],
            columns=["n", "type", "x", "y", "z", "radius", "parent"],
        )
        int_cols = ["n", "type", "parent"]
        float_cols = ["x", "y", "z", "radius"]
        col_type = dict.fromkeys(int_cols, int) | dict.fromkeys(float_cols, float)
        data = data.astype(col_type)
        # only soma nodes (type=1) should have -1 as parent
        mask = data["type"] != 1
        data.loc[mask, "parent"] = data.loc[mask, "parent"].abs()
        # validate swc file
        assert not data.query("parent == 1 and (type == 3 or type == 4)").empty, (
            "Bad SWC file: No dendrites connected to soma!"
        )

        return data

    @staticmethod
    def standardize_swc(swc_data: pd.DataFrame, align: bool = True) -> pd.DataFrame:
        """Standardize swc data to single node soma, PCA aligned axes, and centered at origin.

        1. Soma is collapsed to a single node (by placing a single node at the centroid of the
        convex hull of the soma nodes).
        2. Coordinates are aligned to principal axes (PCA). The first principal component (PC1)
        is aligned to the y-axis, the second principal component (PC2) is aligned to the x-axis,
        and the third principal component (PC3) is aligned to the z-axis.
        3. Soma is centered at the origin.

        Note: Step 1 is performed by MorphoPy's `get_standardized_swc()` function.

        Args:
            swc_data (DataFrame): DataFrame of swc data.
            align (bool, optional): Align to PCA axes. Defaults to True.

        Returns:
            DataFrame: DataFrame of standardized swc data.
        """

        new_swc = get_standardized_swc(
            swc_data.copy(),
            scaling=1.0,
            soma_radius=None,
            soma_center=False,
            pca_rot=False,
        )
        if align:
            pca = PCA(random_state=42)
            xyz_pca = pca.fit_transform(new_swc[["x", "y", "z"]])
            # set PC1 = y, PC2 = x, and PC3 = z
            new_swc[["x", "y", "z"]] = xyz_pca[:, [1, 0, 2]]

        soma_coords = new_swc[["x", "y", "z"]].iloc[0]
        new_swc[["x", "y", "z"]] -= soma_coords

        return new_swc

    def remove_axon(self) -> None:
        """Use MorphoPy to remove axon nodes from the reconstruction.

        This updates both the `data` and `ntree` attributes to contain a neuron without axon nodes.
        """
        self._ntree = self.ntree.get_dendritic_tree()
        self._data = self._ntree.to_swc()
        # reset soma to origin
        soma_coords = self._data[["x", "y", "z"]].iloc[0]
        self._data[["x", "y", "z"]] -= soma_coords

    def resample_distance(self, resample_dist: float) -> None:
        """Resample the swc data to a given distance.

        Calls MorphoPy's `resample_tree()` method to resample the swc data to a given distance.

        Note: The distance is assumed to be in microns, so this should be run before data is PCA aligned.

        Args:
            resample_dist (float): Value to downsample the data.
        """
        self._ntree = self.ntree.resample_tree(resample_dist)
        self._data = self._ntree.to_swc()

    def view(self, ax: plt.Axes | None = None) -> None:
        """View the raw and standardized swc data."""
        self._ntree.draw_2D(projection="xy", ax=ax, axon_color="lightblue")

    def plot_swc(self) -> None:
        """Plot the raw and standardized swc data.

        Determine if the neuron contains axon and creates a plot of the raw and standardized swc data."""
        raw_ntree = nt.NeuronTree(self._raw_data)
        if not self._ntree:
            self._ntree = nt.NeuronTree(self._data)
        if self._ntree.get_axon_nodes().size == 0:
            raw_ntree = raw_ntree.get_dendritic_tree()

        _fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        raw_ntree.draw_2D(projection="xy", ax=axs[0], axon_color="lightblue")
        self._ntree.draw_2D(projection="xy", ax=axs[1], axon_color="lightblue")
        axs[0].set_title(f"Raw: {self.swc_file.stem}")
        axs[1].set_title(f"Standardized: {self.swc_file.stem}")
        plt.tight_layout()

        plt.show()

    def save_swc(self, file_name: str | Path, **kwargs: Any) -> None:
        """Use pandas to save data to .swc file."""
        file_path = Path(file_name)
        if file_path.suffix != ".swc":
            file_path = file_path.with_name(f"{file_path.name}.swc")
        print(f"Saving SWC data to {file_path}")
        with open(file_path, "w", encoding="utf-8") as file:
            header = " ".join(self._data.columns)
            file.write(f"# {header}\n")
        self._data.to_csv(file_path, mode="a", index=False, sep=" ", header=False, **kwargs)

    def to_graph_dict(self) -> dict[str, np.ndarray | dict[int, set[int]]]:
        """Convert SWC data to graph dictionary format using NeuronTree's network structure.

        Returns:
            dict containing:
                - 'features': Numpy array of node features [N x 3] (x, y, z coordinates only)
                - 'neighbors': dict mapping node IDs to their connected neighbors
        """
        # Get node attributes from NeuronTree
        positions = self._ntree.get_node_attributes("pos")
        types = self._ntree.get_node_attributes("type")
        # Get list of nodes
        nodes = list(self._ntree.nodes())
        nodes.sort()
        # features array w/ position (x, y, z) and node type
        features = np.zeros((len(nodes), 4))
        # use undirected graph for neighbor lookup
        adj = nx.to_numpy_array(self._ntree.get_graph().to_undirected())
        neighbors = {}
        for i, node in enumerate(nodes):
            features[i, :3] = positions[node]
            features[i, 3] = types[node]
            neighbors[i] = set(np.where(adj[i] == 1)[0])

        return {"features": features, "neighbors": neighbors}

    def save_pickle(self, file_name: str | Path) -> None:
        """Save neuron as a pickle file.

        Note: Only position and node type are included as features. The radius is typically not accurate
        and is therefore not included. The node type is used for downstream processing.

        The saved file contains:
            - features: Tensor of node position features [N x 4] (x, y, z coordinates and node type)
            - neighbors: dict mapping node IDs to their neighbors

        Args:
            file_name (str | Path): Path to save the .pt file (without extension)
        """
        file_path = Path(file_name)
        if file_path.suffix != ".pkl":
            file_path = file_path.with_name(f"{file_path.name}.pkl")

        graph_dict = self.to_graph_dict()
        with open(file_path, "wb") as f:
            pickle.dump(graph_dict, f)

    def __repr__(self) -> str:
        return f"SWCData(swc_file={self.swc_file}, standardize={self._data is not self._raw_data})"


@app.command()
def main(
    swc_folder: str = Argument(
        ...,
        help="Path to folder containing swc files.",
    ),
    standardize: bool = Option(
        True,
        help="Standardize the data by aligning to principal axes and centering at the origin. Use --no-standardize to skip.",
        is_flag=True,
    ),
    align: bool = Option(
        True,
        help="Use PCA to align the data. Default is True. Use --no-align to skip.",
        is_flag=True,
    ),
    resample_dist: float | None = Option(
        None,
        "-r",
        "--resample",
        help="Resample the data so each node is `resample_dist` apart. Default is None (no resampling).",
    ),
    drop_axon: bool = Option(
        True,
        "-d",
        "--drop-axon",
        help="Remove axon nodes from the reconstruction.",
        is_flag=True,
    ),
    export_dir: str | None = Option(
        None,
        "-e",
        "--export-dir",
        help="Path to directory to save processed SWC files. Default is `swc_folder`/output.",
    ),
    format: str = Option(
        "pkl",
        "-f",
        "--format",
        help="Output format: 'swc' for SWC files or 'pkl' for pickle files. If 'pkl', then the node features and a neighbor mapping is saved.  If 'swc', then a new .swc file is created.",
    ),
) -> None:
    """Process SWC file.

    This function takes an SWC file and processes it by loading, standardizing (optional),
    and downsampling (optional) the data. The processed data is then saved to a CSV file.

    Args:
        swc_folder (str): Path to the SWC file.
        standardize (bool, optional): Flag indicating whether to standardize the data. Defaults to True.
        resample_dist (float, optional): Value to downsample the data. Default is 1.0 (no downsampling).
    """

    # create output directories
    swc_folder_path = Path(swc_folder)
    output_dir = Path(export_dir) if export_dir else swc_folder_path.parents[0] / "output"
    output_dir.mkdir(exist_ok=True)
    cells_dir = output_dir / "cells"
    cells_dir.mkdir(exist_ok=True)
    # setup logging
    log_file = output_dir / f"{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}-swc_processing.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # Log everything to file
        ],
    )
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)

    # Log run parameters
    logging.info("PROCESSING PARAMETERS:")
    logging.info(f"Input folder: {swc_folder_path}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"File format: {format}")
    logging.info(f"Standardize data: {standardize}")
    logging.info(f"PCA alignment: {align}")
    logging.info(f"Resample distance: {resample_dist}")
    logging.info(f"Remove axon: {drop_axon}")

    swc_files_list = list(swc_folder_path.glob("*.swc"))
    if not swc_files_list:
        logging.error(f"No .swc files found in {swc_folder_path}")
        return

    logging.info(f"Found {len(swc_files_list)} SWC files to process")

    # Process files
    failed_files = []

    for swc_file in ProgressBar(swc_files_list, desc="Processing neurons: "):
        try:
            output_stem = cells_dir / swc_file.stem
            swc_data = SWCData(
                swc_file,
                standardize=standardize,
                align=align,
                resample_dist=resample_dist,
            )

            if resample_dist is not None and format.lower() == ".swc":
                output_stem = output_stem.with_name(
                    f"{output_stem.name}-resampled_{round(resample_dist)}um"
                )

            if drop_axon and format.lower() == ".swc":
                swc_data.remove_axon()
                output_stem = output_stem.with_name(f"{output_stem.name}-no_axon")

            if format.lower() in {"pickle", "pkl"}:
                swc_data.save_pickle(output_stem)
                logging.info(f"Processed: {swc_file.name} → {output_stem.name}.pkl")
            else:
                swc_data.save_swc(output_stem)
                logging.info(f"Processed: {swc_file.name} → {output_stem.name}.swc")

        except Exception as e:
            logging.error(f"Error processing {swc_file.name}: {e}")
            failed_files.append((swc_file.name, str(e)))

    # Print summary
    extension = ".pkl" if format.lower() in {"pickle", "pkl"} else ".swc"
    num_processed = len(list(cells_dir.glob(f"*{extension}")))

    summary = [
        f"\nProcessing complete: {num_processed}/{len(swc_files_list)} files processed",
        f"Cell files saved to: {cells_dir}",
        f"Log file saved to: {log_file}",
    ]

    if failed_files:
        summary.append(f"\nFailed files ({len(failed_files)}):")
        for file, error in failed_files:
            summary.append(f"- {file}: {error}")

    logging.info("\n".join(summary))
    logging.info("Processing complete")


if __name__ == "__main__":
    app()
