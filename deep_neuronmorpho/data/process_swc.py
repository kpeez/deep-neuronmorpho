"""Process SWC files."""

import logging
import os
import pickle
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime as dt
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from morphopy.neurontree import NeuronTree as nt
from morphopy.neurontree.utils import get_standardized_swc
from sklearn.decomposition import PCA
from tqdm import tqdm
from typer import Argument, Option, Typer

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
        align (bool, optional): Flag indicating whether to align the data to principal axes. Defaults to False.
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
        align: bool = False,
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
        # any nodes with parent == -1 should be type 1 (soma)
        data.loc[data["parent"] == -1, "type"] = 1
        assert (data["parent"] == -1).any(), (
            "SWC file must contain at least one root node with parent == -1"
        )

        return data

    @staticmethod
    def standardize_swc(swc_data: pd.DataFrame, align: bool = False) -> pd.DataFrame:
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
            align (bool, optional): Align to PCA axes. Defaults to False.

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

    def save_swc(self, file_path: str | Path, **kwargs: Any) -> None:
        """Use pandas to save data to .swc file."""
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
        positions = self._ntree.get_node_attributes("pos")
        types = self._ntree.get_node_attributes("type")
        nodes = list(self._ntree.nodes())
        nodes.sort()
        # features array w/ position (x, y, z) and node type
        features = np.zeros((len(nodes), 4))
        # use undirected graph for neighbor lookup
        adj = nx.to_numpy_array(self._ntree.get_graph().to_undirected())
        edge_index = np.array(np.nonzero(adj))
        neighbors = {}
        for i, node in enumerate(nodes):
            features[i, :3] = positions[node]
            features[i, 3] = types[node]
            neighbors[i] = set(edge_index[0, edge_index[1] == i])

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "neighbors": neighbors,
        }

    def save_pickle(self, file_path: str | Path) -> None:
        """Save neuron as a pickle file.

        Note: Only position and node type are included as features. The radius is typically not accurate
        and is therefore not included. The node type is used for downstream processing.

        The saved file contains:
            - features: Tensor of node position features [N x 4] (x, y, z coordinates and node type)
            - neighbors: dict mapping node IDs to their neighbors

        Args:
            file_path (str | Path): Path to save the file.
        """
        graph_dict = self.to_graph_dict()
        with open(file_path, "wb") as f:
            pickle.dump(graph_dict, f)

    def __repr__(self) -> str:
        return f"SWCData(swc_file={self.swc_file}, standardize={self._data is not self._raw_data})"


def process_swc_file(args):
    """Process a single SWC file.

    Args:
        args: Tuple containing (swc_file, standardize, align, resample_dist, drop_axon, format, cells_dir)

    Returns:
        dict containing:
            - 'filename': Name of the SWC file
            - 'status': 'success' or 'error'
            - 'output': Output message or None if successful
            - 'error': Error message or None if successful
    """
    swc_file, standardize, align, resample_dist, drop_axon, file_format, cells_dir = args
    temp_file_path = None
    try:
        output_stem = swc_file.stem
        swc_data = SWCData(
            swc_file,
            standardize=standardize,
            align=align,
            resample_dist=resample_dist,
        )

        if resample_dist is not None and file_format.lower() == "swc":
            output_stem = f"{output_stem}-resampled_{round(resample_dist)}um"

        if drop_axon:
            swc_data.remove_axon()
            if file_format.lower() == "swc":
                output_stem = f"{output_stem}-no_axon"

        if file_format.lower() in {"pickle", "pkl"}:
            final_output_path = (cells_dir / output_stem).with_suffix(".pkl")
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=cells_dir, delete=False, suffix=".pkl"
            ) as tmp:
                temp_file_path = Path(tmp.name)
                swc_data.save_pickle(temp_file_path)
        else:
            final_output_path = (cells_dir / output_stem).with_suffix(".swc")
            with tempfile.NamedTemporaryFile(
                mode="w", dir=cells_dir, delete=False, suffix=".swc", encoding="utf-8"
            ) as tmp:
                temp_file_path = Path(tmp.name)
                swc_data.save_swc(temp_file_path)

        shutil.move(temp_file_path, final_output_path)
        temp_file_path = None

        return {
            "filename": swc_file.name,
            "status": "success",
            "output": f"{swc_file.name} → {final_output_path.name}",
            "error": None,
        }
    except Exception as e:
        logging.error(f"Error processing {swc_file.name}", exc_info=True)
        return {
            "filename": swc_file.name,
            "status": "error",
            "output": None,
            "error": str(e),
        }
    finally:
        if temp_file_path and temp_file_path.exists():
            os.remove(temp_file_path)


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
        False,
        help="Use PCA to align the data. Default is False. Use --align to enable.",
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
        "--drop-axon / --no-drop-axon",
        help="Remove axon nodes (default: enabled). Use --no-drop-axon to keep them.",
    ),
    output_dir: str | None = Option(
        None,
        "-o",
        "--output-dir",
        help="Path to directory to save processed SWC files. Default is `swc_folder`/output.",
    ),
    file_format: str = Option(
        "csv",
        "-f",
        "--file-format",
        help="Output file format: 'swc' for SWC files or 'pkl' for pickle files. If 'pkl', then the node features and a neighbor mapping is saved.  If 'swc', then a new .swc file is created.",
    ),
    internal_workers: int = Option(
        None,
        "-w",
        "--internal-workers",
        help="Number of worker processes to use WITHIN this job. Default is os.cpu_count() divided by SLURM tasks per node if available, else os.cpu_count().",
    ),
    task_id: int = Option(
        None,
        "--task-id",
        help="ID of the current task (e.g., from SLURM_ARRAY_TASK_ID). Processes all files if not set.",
    ),
    num_tasks: int = Option(
        None,
        "--num-tasks",
        help="Total number of tasks (e.g., from SLURM_ARRAY_TASK_COUNT). Processes all files if not set.",
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

    swc_folder_path = Path(swc_folder)
    output_dir = Path(output_dir) if output_dir else swc_folder_path.parents[0] / "output"
    output_dir.mkdir(exist_ok=True)
    cells_dir = output_dir / "data"
    cells_dir.mkdir(exist_ok=True)
    is_array_job = task_id is not None and num_tasks is not None
    current_task_id = task_id if is_array_job else 0
    total_tasks = num_tasks if is_array_job else 1
    task_id_str = f"-{current_task_id}" if is_array_job else "-all"

    timestamp_str = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = output_dir / f"{timestamp_str}-swc_processing{task_id_str}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
        ],
    )
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console)
    if internal_workers is None:
        try:
            # Try to get SLURM environment variables to calculate workers per task
            cpus_on_node = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count()))
            # Use the determined total_tasks (could be 1 if not an array job)
            tasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", total_tasks))
            calculated_workers = max(1, cpus_on_node // tasks_per_node)
            internal_workers = calculated_workers  # Assign here after successful calculation
        except (ValueError, TypeError, KeyError):
            # Fallback if SLURM vars not set, invalid, or not found
            logging.warning(
                "Could not determine workers based on SLURM env vars, defaulting to os.cpu_count()."
            )
            internal_workers = os.cpu_count()

    processing_params = f"""
    PROCESSING PARAMETERS (Task {current_task_id}/{total_tasks}):
    Input folder: {swc_folder_path}
    Output directory: {output_dir}
    File format: {file_format}
    Standardize data: {standardize}
    PCA alignment: {align}
    Resample distance: {resample_dist}
    Remove axon: {drop_axon}
    Internal workers for this task: {internal_workers}
    Task ID: {current_task_id}
    Total Tasks: {total_tasks}
    """
    logging.info(processing_params)

    all_swc_files = sorted(swc_folder_path.glob("*.swc"))
    if not all_swc_files:
        logging.error(f"No .swc files found in {swc_folder_path}")
        return

    logging.info(f"Found {len(all_swc_files)} total SWC files")
    if is_array_job:
        if not (0 <= current_task_id < total_tasks):
            logging.error(f"Invalid Task ID {current_task_id} for total tasks {total_tasks}")
            return

        total_files = len(all_swc_files)
        files_per_task = total_files // total_tasks
        remainder = total_files % total_tasks

        start_index = current_task_id * files_per_task + min(current_task_id, remainder)
        num_files_for_this_task = files_per_task + (1 if current_task_id < remainder else 0)
        end_index = start_index + num_files_for_this_task

        swc_files_list = all_swc_files[start_index:end_index]
        logging.info(
            f"Task {current_task_id}/{total_tasks}: Processing {len(swc_files_list)} files (indices {start_index} to {end_index - 1})"
        )
    else:
        swc_files_list = all_swc_files
        logging.info(f"Processing all {len(swc_files_list)} SWC files (Task 0/1)")

    if not swc_files_list:
        logging.warning(f"Task {current_task_id}: No files assigned to this task. Exiting.")
        return

    process_args = [
        (f, standardize, align, resample_dist, drop_axon, file_format, cells_dir)
        for f in swc_files_list
    ]
    with ProcessPoolExecutor(max_workers=internal_workers) as executor:
        results = []
        with tqdm(
            total=len(swc_files_list),
            desc=f"Processing neurons (Task {current_task_id}/{total_tasks}): ",
        ) as pbar:
            for result in executor.map(process_swc_file, process_args):
                if result["status"] == "success":
                    logging.info(f"Processed: {result['output']}")
                else:
                    logging.error(
                        f"Task {current_task_id}: Error processing {result['filename']}: {result['error']}"
                    )
                results.append(result)
                pbar.update(1)

    failed_files = [
        (result["filename"], result["error"]) for result in results if result["error"] is not None
    ]
    num_processed_this_task = len([res for res in results if res["status"] == "success"])
    summary = [
        f"\nTask {current_task_id}/{total_tasks} Processing complete: {num_processed_this_task}/{len(swc_files_list)} files processed by this task.",
        f"Cell files saved to: {cells_dir}",
        f"Log file for this task: {log_file}",
    ]

    if failed_files:
        summary.append(f"\nFailed files ({len(failed_files)}) for this task:")
        for file, error in failed_files:
            summary.append(f"- {file}: {error}")

    logging.info("\n".join(summary))
    logging.info(f"Task {current_task_id}/{total_tasks} finished.")


if __name__ == "__main__":
    app()
