"""Process SWC files."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from morphopy.neurontree import NeuronTree as nt
from tqdm import tqdm


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
    )

    if swc_data[["x", "y", "z"]].iloc[0].all() != 0.0:
        x, y, z = swc_data[["x", "y", "z"]].iloc[0]
        swc_data[["x", "y", "z"]] = swc_data[["x", "y", "z"]] - [x, y, z]

    return swc_data


def swc_to_neuron_tree(neuron_swc: pd.DataFrame) -> nt.NeuronTree:
    """Load NeuronTree from MorphoPy.

    Args:
        neuron_swc (pd.DataFrame): swc data.

    Returns:
        NeuronTree: NeuronTree object.
    """
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
    swc_files_list = list(swc_files.glob("*.swc"))
    num_iterations = len(swc_files_list)
    percent_increment = 5
    increment_value = int(np.ceil(num_iterations * percent_increment / 100))
    with tqdm(
        total=num_iterations,
        desc="Resampling neurons: ",
        bar_format="{desc}[{n_fmt}/{total_fmt}]{percentage:3.0f}%|{bar}"
        "{postfix} [{elapsed}<{remaining}]",
    ) as pbar:
        for n, swc_file in enumerate(swc_files_list):
            if n % increment_value == 0:
                pbar.update(increment_value)

            neuron = load_swc_file(swc_file)
            neuron_tree = swc_to_neuron_tree(neuron)
            neuron_tree = neuron_tree.resample_tree(resample_dist)

            swc_dir, old_filename = swc_file.parent, swc_file.name
            export_dir = Path(f"{swc_dir}_resampled")
            if not export_dir.exists():
                export_dir.mkdir(exist_ok=True)
            new_filename = old_filename.replace(".swc", f"-resampled_{resample_dist}um.swc")
            neuron_tree.to_swc().to_csv(f"{export_dir}/{new_filename}", sep=" ", index=False)


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
