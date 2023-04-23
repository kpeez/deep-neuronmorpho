"""Prepare neuron graphs for conversion to DGL datasets."""
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs, save_graphs
from scipy import stats
from scipy.spatial.distance import euclidean

from deep_neuronmorpho.data_loader.process_swc import swc_to_neuron_tree
from deep_neuronmorpho.utils.progress import ProgressBar


def compute_graph_attrs(graph_attrs: list[float]) -> list[float]:
    """Compute summary statistics for a list of graph attributes.

    Args:
        graph_attrs (list[float]): Graph attribute data.

    Returns:
        dict[str, float]: Summary statistics of graph attributes. In the following order:
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


def create_neuron_graph(swc_file: str | Path) -> nx.Graph:
    """Create networkx graph of neuron from swc format.

    Args:
        swc_file (str | Path): Morphopy NeuronTree object of neuron swc data.

    Returns:
        nx.Graph: Graph of neuron.

    Notes:
        This function takes in a MorphoPy NeuronTree object and returns a networkx graph with the
        following node attributes:

            1. x: x coordinate of node.
            2. y: y coordinate of node.
            3. z: z coordinate of node.
            4. r: node radius.
            5. path_dist: path distance from soma.
            6. euclidean_dist: euclidean distance from soma.
            7.-12. angle attrs (n=6): min, mean, median, max, std, num of branch angles.
            13.-18. branch attrs (n=6): min, mean, median, max, std, num of branch lengths.
    """
    neuron_tree = swc_to_neuron_tree(swc_file)
    # get branch angles and branch lengths
    angles = list(neuron_tree.get_branch_angles().values())
    branches = list(neuron_tree.get_segment_length().values())
    angle_stats = compute_graph_attrs(angles)
    branch_stats = compute_graph_attrs(branches)
    # update graph attributes
    neuron_graph = neuron_tree.get_graph()
    soma_x, soma_y, soma_z = neuron_graph.nodes[1]["pos"]
    for node in neuron_graph.nodes():
        # expand position to x, y, z
        x, y, z = neuron_graph.nodes[node]["pos"]
        radius = neuron_graph.nodes[node]["radius"]
        node_attrs = [
            x,
            y,
            z,
            radius,
            nx.dijkstra_path_length(neuron_graph, 1, node, weight="path_length"),
            euclidean((x, y, z), (soma_x, soma_y, soma_z)),
            *angle_stats,
            *branch_stats,
        ]

        neuron_graph.nodes[node].clear()
        neuron_graph.nodes[node].update({"nattrs": [np.float32(attr) for attr in node_attrs]})
    # euclidean_dist is duplicate of path_dist for edges so remove
    # TODO: fix to compute "attention" instead
    for _, _, edge_data in neuron_graph.edges(data=True):
        del edge_data["euclidean_dist"]

    return neuron_graph


def dgl_from_swc(swc_files: list[Path]) -> list[dgl.DGLGraph]:
    """Convert a neuron swc file into a DGL graph.

    Args:
        swc_files (list[Path]): List of swc files.
        resample_dist (int, optional): Resample distance in microns. Defaults to 10.

    Returns:
        list[dgl.DGLGraph]: List of DGL graphs.
    """
    neuron_graphs = []
    for file in ProgressBar(swc_files, desc="Creating DGL graphs:"):
        neuron_graph = create_neuron_graph(file)
        neuron_graphs.append(
            dgl.from_networkx(
                neuron_graph,
                node_attrs=["nattrs"],
                edge_attrs=["path_length"],
            )
        )
    return neuron_graphs


class NeuronGraphDataset(DGLDataset):
    """A dataset consisting of DGLGraphs representing neuron morphologies.

    Args:
        graphs_path (Path): The path to the SWC file directory.
        self_loop (bool, optional): Whether to add self-loops to each graph. Defaults to False.
        data_name (str, optional): The name of the dataset. Defaults to "neuron_graph_dataset".

    Attributes:
        graphs_path (Path): The path to the SWC file directory.
        export_dir (Path): The path to the directory where the processed dataset will be saved.
        self_loop (bool): Whether to add self-loops to each graph.
        graphs (list[dgl.DGLGraph]): The list of DGLGraphs representing neuron morphologies.

    See Also:
        Documentation for DGLDataset: https://docs.dgl.ai/en/latest/api/python/dgl.data.html#dgl.data.DGLDataset
    """

    def __init__(
        self,
        graphs_path: Path,
        self_loop: bool = False,
        data_name: str = "neuron_graph_dataset",
    ):
        self.graphs_path = graphs_path
        self.export_dir = Path(self.graphs_path.parent / "processed")
        self.graphs: list = []
        self.self_loop = self_loop
        super().__init__(name=data_name, raw_dir=self.export_dir)

    def process(self) -> None:
        """Process the input data into a list of DGLGraphs."""
        self.graphs = dgl_from_swc(list(self.graphs_path.glob("*.swc")))

    def load(self) -> None:
        """Load the dataset from disk.

        If the dataset has not been cached, it will be created and cached.
        """
        self.graphs = load_graphs(str(self.cached_graphs_path))[0]

    def save(self) -> None:
        """Save the dataset to disk."""
        if not self.export_dir.exists():
            self.export_dir.mkdir(exist_ok=True)
        save_graphs(f"{self.export_dir}/{super().name}_dgl_graphs.bin", self.graphs)

    def has_cache(self) -> bool:
        """Determine whether there exists a cached dataset.

        Returns:
            bool: True if there exists a cached dataset, False otherwise.
        """
        cached_file_path = self.cached_graphs_path
        return cached_file_path.exists()

    @property
    def cached_graphs_path(self) -> Path:
        """The path to the cached graphs."""
        return Path(self.graphs_path.parent / "processed" / f"{super().name}_dgl_graphs.bin")

    def __len__(self) -> int:
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        """Get the idx-th sample."""
        return self.graphs[idx]


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def create_dataset(
        input_dir: str = typer.Argument(  # noqa: B008
            ..., help="Path to the directory containing the .swc files."
        ),
        self_loop: bool = typer.Option(  # noqa: B008
            False,
            help="Optional flag to add self-loops to each graph.",
        ),
    ) -> None:
        """Create a processed dataset of graphs from the .swc files in the specified directory.

        Args:
            input_dir (str): Path to the directory containing the .swc files.
            self_loop (bool): Optional flag to add self-loops to each graph. Defaults to False.
        """
        graphs_dir = Path(input_dir)
        NeuronGraphDataset(graphs_path=graphs_dir, self_loop=self_loop)

    app()
