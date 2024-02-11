"""Process SWC files."""
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
        self._data = self._raw_data
        self._ntree = nt.NeuronTree(self._data)

        if resample_dist is not None:
            self.resample(resample_dist)

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
        col_type = {col: int for col in int_cols} | {col: float for col in float_cols}
        data.astype(col_type)

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

    def remove_axon(self) -> None:
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
    ) -> None:
        """Resample the swc data to a given distance.

        Calls MorphoPy's `resample_tree()` method to resample the swc data to a given distance.

        Note: The distance is assumed to be in microns, so this should be run before data is PCA aligned.

        Args:
            resample_dist (float): Value to downsample the data.
        """
        self._ntree = self.ntree.resample_tree(resample_dist)
        self._data = self._ntree.to_swc()

    def plot_swc(self) -> None:
        """Plot the raw and standardized swc data.

        Determine if the neuron contains axon and creates a plot of the raw and standardized swc data."""
        raw_ntree = nt.NeuronTree(self._raw_data)
        if not self._ntree:
            self._ntree = nt.NeuronTree(self._data)
        if self._ntree.get_axon_nodes().size == 0:
            raw_ntree = raw_ntree.get_dendritic_tree()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
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
        with open(file_path, "w") as file:
            header = " ".join(self._data.columns)
            file.write(f"# {header}\n")
        self._data.to_csv(file_path, mode="a", index=False, sep=" ", header=False, **kwargs)


if __name__ == "__main__":
    from typing import Optional

    from typer import Argument, Option, Typer

    app = Typer()

    @app.command()
    def main(
        swc_folder: str = Argument(
            ...,
            help="Path to folder containing swc files.",
        ),
        standardize: bool = Option(
            True,
            help="Standardize the data by aligning to principal axes and centering at the origin. Use --no-standardize to skip.",
        ),
        no_standardize: bool = Option(
            False,
            "--no-standardize",
            help="Do not standardize the data.",
            is_flag=True,
        ),
        align: bool = Option(
            True,
            help="Use PCA to align the data. Default is True. Use --no-align to skip.",
        ),
        no_align: bool = Option(
            False,
            "--no-align",
            help="Do not use PCA to align the data.",
            is_flag=True,
        ),
        resample_dist: Optional[float] = Option(
            None,
            "-r",
            "--resample",
            help="Resample the data so each node is `resample_dist` apart. Default is 1.0 (no resampling).",
        ),
        drop_axon: bool = Option(
            False,
            "-d",
            "--drop-axon",
            help="Remove axon nodes from the reconstruction.",
            is_flag=True,
        ),
        export_dir: Optional[str] = Option(
            None,
            "-e",
            "--export-dir",
            help="Path to directory to save processed SWC files. Default is `swc_folder`/interim.",
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
        swc_files = Path(swc_folder)
        output_dir = Path(export_dir) if export_dir else Path(swc_files.parents[0] / "interim")
        if not output_dir.exists():
            output_dir.mkdir(exist_ok=True)
        swc_files_list = list(swc_files.glob("*.swc"))

        for swc_file in ProgressBar(swc_files_list, desc="Processing neurons: "):
            try:
                output_file = f"{output_dir}/{swc_file.stem}"
                swc_data = SWCData(
                    swc_file,
                    standardize=standardize and not no_standardize,
                    align=align and not no_align,
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
