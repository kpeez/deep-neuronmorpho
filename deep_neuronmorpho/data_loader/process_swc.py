"""Process SWC files."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from morphopy.neurontree import NeuronTree as nt

from deep_neuronmorpho.utils.progress import ProgressBar


def set_swc_dtypes(swc_data: pd.DataFrame) -> pd.DataFrame:
    """Set dtypes for swc data. Sets ints to int32 and floats to float32.

    Args:
        swc_data (pd.DataFrame): DataFrame of swc data.

    Returns:
        pd.DataFrame: DataFrame of swc data with correct dtypes set.
    """
    int_cols = ["n", "type", "parent"]
    float_cols = ["x", "y", "z", "radius"]
    col_type = {col: np.int32 for col in int_cols} | {col: np.float32 for col in float_cols}

    return swc_data.astype(col_type)


def set_swc_soma_coords(swc_data: pd.DataFrame) -> pd.DataFrame:
    """Set coordinates of soma to (0, 0, 0).

    Args:
        swc_data (pd.DataFrame): DataFrame of swc data.

    Returns:
        pd.DataFrame: DataFrame of swc data with soma set to (0, 0, 0).
    """
    if swc_data[["x", "y", "z"]].iloc[0].all() != 0.0:
        x, y, z = swc_data[["x", "y", "z"]].iloc[0]
        swc_data[["x", "y", "z"]] = swc_data[["x", "y", "z"]] - [x, y, z]

    return swc_data


def load_swc_file(swc_file: Path | str) -> pd.DataFrame:
    """Load swc file.

    Args:
        swc_file (Path): Path to swc file.

    Returns:
        pd.DataFrame: SWC data.
    """
    swc_data = pd.read_csv(
        swc_file,
        sep=" ",
        header=None,
        names=["n", "type", "x", "y", "z", "radius", "parent"],
        low_memory=False,
    )

    # check for header
    if (swc_data.columns[1:] == swc_data.iloc[0][1:]).all():
        swc_data.drop(0, inplace=True)
        swc_data.reset_index(drop=True, inplace=True)

    swc_data = set_swc_dtypes(swc_data)
    swc_data = set_swc_soma_coords(swc_data)

    return swc_data


def swc_to_neuron_tree(swc_file: Path | str) -> nt.NeuronTree:
    """Load NeuronTree from MorphoPy.

    Args:
        swc_file (Path | str): Path to swc file.

    Returns:
        NeuronTree: NeuronTree object.
    """
    neuron_swc = load_swc_file(swc_file)
    if "id" in neuron_swc.columns:
        neuron_swc.rename(columns={"id": "n"}, inplace=True)

    return nt.NeuronTree(neuron_swc)


def downsample_swc_files(swc_files: Path, resample_dist: int | float) -> None:
    """Downsample and export swc files.

    Exported files are saved in a new folder in the parent directory to
    swc_files.

    Args:
        swc_files (Path): Path to folder containing swc files.
        resample_dist (int | float): Distance to resample neuron, in microns.
    """
    export_dir = export_dir = Path(f"{swc_files}_resampled_{resample_dist}um")
    if not export_dir.exists():
        export_dir.mkdir(exist_ok=True)

    swc_files_list = list(swc_files.glob("*.swc"))
    for swc_file in ProgressBar(swc_files_list, desc="Resampling neurons: "):
        try:
            neuron_tree = swc_to_neuron_tree(swc_file)
            neuron_tree = neuron_tree.resample_tree(resample_dist)
            new_filename = swc_file.name.replace(".swc", f"-resampled_{resample_dist}um.swc")
            neuron_tree.to_swc().to_csv(f"{export_dir}/{new_filename}", sep=" ", index=False)
        except Exception as e:
            print(f"Error processing {swc_file}. {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--swc_files",
        type=str,
        help="Path to folder containing swc files.",
    )
    parser.add_argument(
        "--resample_dist",
        type=int,
        help="Distance to resample neuron, in microns.",
    )
    args = parser.parse_args()

    swc_files = Path(args.swc_files)
    resample_dist = args.resample_dist

    downsample_swc_files(swc_files, resample_dist)
