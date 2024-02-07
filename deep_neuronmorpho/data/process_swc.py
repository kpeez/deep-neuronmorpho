"""Process SWC files."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from morphopy.neurontree import NeuronTree as nt
from morphopy.neurontree.utils import get_standardized_swc
from sklearn.decomposition import PCA

from deep_neuronmorpho.utils import ProgressBar


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
        self._data = None
        self._ntree: nt.NeuronTree = None

        if float(resample_dist) is not None:
            self.resample(resample_dist, standardize=standardize)

        if standardize:
            self._data = self.standardize_swc(self._raw_data, align=align)

    @property
    def raw_data(self) -> pd.DataFrame:
        """Return the raw swc data."""
        return self._raw_data

    @property
    def data(self) -> pd.DataFrame:
        """Return the (possibly) standardized swc data."""
        return self._data if self._data is not None else self._raw_data

    @property
    def ntree(self) -> nt.NeuronTree:
        """Return the NeuronTree object."""
        return self._ntree if self._ntree else nt.NeuronTree(self._data)

    @staticmethod
    def load_swc_data(swc_file: str | Path) -> pd.DataFrame:
        """Load swc data from a file."""
        with open(swc_file, "r") as file:
            lines = file.readlines()
        # Find the start of data
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith("#"):
                start_idx = i
                break
        data = pd.DataFrame(
            [line.split() for line in lines[start_idx:]],
            columns=["n", "type", "x", "y", "z", "radius", "parent"],
        )
        int_cols = ["n", "type", "parent"]
        float_cols = ["x", "y", "z", "radius"]
        col_type = {col: np.int32 for col in int_cols} | {col: np.float32 for col in float_cols}

        return data.astype(col_type)

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

    def remove_axon(self):
        """Use MorphoPy to remove axon nodes from the reconstruction.

        This updates both the `data` and `ntree` attributes to contain a neuron without axon nodes.
        """

        def _map_parent_id(parent_id: int, mapping: dict) -> int:
            """Map the parent ID, keeping -1 as is."""
            return mapping[parent_id] if parent_id != -1 else -1

        self._ntree = self.ntree.get_dendritic_tree()
        self._data = self._ntree.to_swc()
        sample_to_idx = dict(zip(self._data["n"], self._data.index + 1, strict=True))
        self._data["n"] = self._data["n"].map(sample_to_idx).astype(int)
        self._data["parent"] = (
            self._data["parent"].apply(lambda x: _map_parent_id(x, sample_to_idx)).astype(int)
        )
        assert len(self._data["type"].unique()) > 1, "Neuron only contained axon nodes."

    def resample(
        self,
        resample_dist: float,
        standardize: bool = True,
    ) -> None:
        """Resample the swc data to a given distance.

        Calls MorphoPy's `resample_tree()` method to resample the swc data to a given distance (assumed µm).

        Args:
            resample_dist (float): Value to downsample the data.
            standardize (bool, optional): Flag indicating whether to re-standardize the data after resampling. Defaults to True.
        """
        self._ntree = self.ntree.resample_tree(resample_dist)
        self._data = self._ntree.to_swc()

        if standardize:
            self._data = self.standardize_swc(self._data)

    def plot_swc(self, ax=None):
        """Plot the raw and standardized swc data.

        Determine if the neuron contains axon and creates a plot of the raw and standardized swc data."""
        raw_ntree = nt.NeuronTree(self._raw_data)
        if not self._ntree:
            self._ntree = nt.NeuronTree(self._data)
        if self._ntree.get_axon_nodes().size == 0:
            raw_ntree = raw_ntree.get_dendritic_tree()

        if ax is None:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        raw_ntree.draw_2D(projection="xy", ax=axs[0], axon_color="lightblue")
        self._ntree.draw_2D(projection="xy", ax=axs[1], axon_color="lightblue")
        axs[0].set_title(f"Raw: {self.swc_file.stem}")
        axs[1].set_title(f"Standardized: {self.swc_file.stem}")
        fig.tight_layout()

        return ax

    def save_swc(self, file_name: str | Path, **kwargs) -> None:
        """Use pandas to save data to .swc file."""
        file_path = Path(file_name)
        if file_path.suffix != ".swc":
            file_path = file_path.with_name(f"{file_path.name}.swc")
        print(f"Saving SWC data to {file_path}")
        self._data.to_csv(file_path, index=False, sep=" ", **kwargs)


if __name__ == "__main__":
    from typer import Argument, Option, Typer

    app = Typer()

    @app.command()
    def main(
        swc_files: str = Argument(
            ...,
            help="Path to folder containing swc files.",
        ),
        standardize: bool = Option(
            True,
            "-s",
            "--standardize",
            help="Standardize the data by aligning to principal axes and centering at the origin.",
        ),
        align: bool = Option(
            True,
            "-a",
            "--align",
            help="Align the data to principal axes.",
        ),
        resample_dist: float | None = Option(
            None,
            "-r",
            "--resample",
            help="Resample the data so each node is `resample_dist` apart. Default is 1.0 (no resampling).",
        ),
        drop_axon: bool = Option(
            True,
            "-d",
            "--drop_axon",
            help="Remove axon nodes from the reconstruction.",
        ),
    ) -> None:
        """Process SWC file.

        This function takes an SWC file and processes it by loading, standardizing (optional),
        and downsampling (optional) the data. The processed data is then saved to a CSV file.

        Args:
            swc_file (str): Path to the SWC file.
            standardize (bool, optional): Flag indicating whether to standardize the data. Defaults to True.
            resample_dist (float, optional): Value to downsample the data. Default is 1.0 (no downsampling).
        """
        swc_files = Path(swc_files)
        export_dir = swc_files.parents[0] / "interim"
        print(export_dir)
        if not export_dir.exists():
            export_dir.mkdir(exist_ok=True)
        swc_files_list = list(swc_files.glob("*.swc"))

        for swc_file in ProgressBar(swc_files_list, desc="Processing neurons: "):
            try:
                output_file = f"{export_dir}/{swc_file.stem}"
                swc_data = SWCData(
                    swc_file,
                    standardize=standardize,
                    align=align,
                    resample_dist=resample_dist,
                )
                if resample_dist is not None:
                    output_file = f"{output_file}-resampled_{resample_dist}um"

                if drop_axon:
                    swc_data.remove_axon()

                swc_data.save_swc(output_file)
                print(f"Processed: {swc_file.stem}")
            except Exception as e:
                print(f"Error processing {swc_file}. {e}")

    app()
