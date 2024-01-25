"""Prepare neuron graphs for conversion to DGL datasets."""
from pathlib import Path
from typing import TYPE_CHECKING

import dgl
import networkx as nx
import numpy as np
import torch
from dgl.data.utils import load_graphs, load_info, save_graphs, save_info
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ..utils import EventLogger, ProgressBar
from .data_utils import add_graph_labels, compute_graph_attrs, graph_is_broken
from .process_swc import swc_to_neuron_tree

if TYPE_CHECKING:
    from dgl import DGLGraph
    from dgl.data import DGLDataset
    from networkx import DiGraph
    from torch import Tensor


def compute_edge_weights(G: DiGraph, path_idx: int, epsilon: float = 1.0) -> DiGraph:
    """Compute edge attention weights for a graph.

    Based on the method described in [Zhao et al. 2022](https://ieeexplore.ieee.org/document/9895206).

    Args:
        G (DiGraph): Graph to compute edge weights for.
        path_idx (int): Index of path distance in node attributes.
        epsilon (float, optional): Small constant to prevent division by zero. Defaults to 1.0.

    Returns:
        DiGraph: Graph with attention weights added as edge attributes.
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


def create_neuron_graph(swc_file: str | Path) -> DiGraph:
    """Create networkx graph of neuron from swc format.

    Args:
        swc_file (str | Path): Morphopy NeuronTree object of neuron swc data.

    Returns:
        DiGraph: Graph of neuron.

    Notes:
        This function takes in a MorphoPy NeuronTree object and returns a networkx graph with the
        following node attributes:

            1. x: x coordinate of node.
            2. y: y coordinate of node.
            3. z: z coordinate of node.
            4. r: node radius. (not currently in use due to unreliable radius data)
            5. path_dist: path distance from soma.
            6. euclidean_dist: euclidean distance from soma.
            7.-12. angle attrs (n=6): min, mean, median, max, std, num of branch angles.
            13.-18. branch attrs (n=6): min, mean, median, max, std, num of branch lengths.
    """
    neuron_tree = swc_to_neuron_tree(swc_file)
    angles = list(neuron_tree.get_branch_angles().values())
    branches = list(neuron_tree.get_segment_length().values())
    angle_stats = compute_graph_attrs(angles)
    branch_stats = compute_graph_attrs(branches)
    neuron_graph = neuron_tree.get_graph()
    # morphopy uses 1-indexing for nodes, dgl uses 0-indexing
    neuron_graph = nx.relabel_nodes(neuron_graph, {i: i - 1 for i in neuron_graph.nodes()})
    soma_x, soma_y, soma_z = neuron_graph.nodes[1]["pos"]
    for node in neuron_graph.nodes():
        x, y, z = neuron_graph.nodes[node]["pos"]
        node_attrs = [
            x,
            y,
            z,
            # neuron_graph.nodes[node]["radius"],  # uncomment to include radius
            euclidean((x, y, z), (soma_x, soma_y, soma_z)),
            nx.dijkstra_path_length(neuron_graph, 0, node, weight="path_length"),
            *angle_stats,
            *branch_stats,
        ]
        neuron_graph.nodes[node].clear()
        neuron_graph.nodes[node].update({"nattrs": [np.float32(attr) for attr in node_attrs]})

    for _, _, edge_data in neuron_graph.edges(data=True):
        del edge_data["euclidean_dist"]
        del edge_data["path_length"]
    # get path_idx depending on if we have radius in dataset or not
    path_idx = 5 if len(neuron_graph.nodes(data=True)[0]["nattrs"]) == 18 else 4
    neuron_graph = compute_edge_weights(neuron_graph, path_idx=path_idx)

    return neuron_graph


def create_dgl_graph(neuron_graph: nx.DiGraph) -> DGLGraph | None:
    """Create a DGLGraph object from a NetworkX DiGraph object.

    Args:
        neuron_graph (nx.DiGraph): The NetworkX DiGraph object.

    Returns:
        DGLGraph | None: The resulting DGLGraph object, or None if the graph is broken.
    """
    dgl_graph = dgl.from_networkx(neuron_graph, node_attrs=["nattrs"], edge_attrs=["edge_weight"])
    if graph_is_broken(dgl_graph):
        return None
    else:
        return dgl_graph


def dgl_from_swc(swc_files: list[Path], logger: EventLogger | None = None) -> list[DGLGraph]:
    """Convert a neuron swc file into a DGL graph.

    Args:
        swc_files (list[Path]): List of swc files.
        logger (Logger): Logger object.

    Returns:
        list[DGLGraph]: List of DGL graphs.
    """
    logger = EventLogger(Path.cwd(), "dgl_from_swc", to_file=False) if logger is None else logger

    neuron_graphs = []
    for file in ProgressBar(swc_files, desc="Creating DGLGraph:"):
        try:
            neuron_graph = create_neuron_graph(file)
            dgl_graph = create_dgl_graph(neuron_graph)
            if dgl_graph is None:
                logger.message(
                    f"Graph is broken: {file.name} contains NaN node attributes", level="error"
                )
            else:
                dgl_graph.id = file.stem
                neuron_graphs.append(dgl_graph)
                logger.message(f"Processed file: {file.name}")

        except Exception as e:
            logger.message(f"Error creating DGLGraph for {file}: {e}", level="error")

    logger.message(f"Created {len(neuron_graphs)} DGLGraphs.")

    return neuron_graphs


class GraphScaler:
    """A class used to standardize the node attributes of DGLGraphs in a dataset.

    Args:
        scale_xyz (str, optional): The type of scaling to apply to the first three node attributes
            which represent the x, y, z coordinates.
            Accepted values are 'standard', 'robust', and 'minmax'. Defaults to 'standard'.
        scale_attrs (str, optional): The type of scaling to apply to the remaining node attributes.
            Accepted values are 'standard', 'robust', and 'minmax'. Defaults to 'robust'.

    Attributes:
        scale_xyz (Scaler): The scaler object for the first three node attributes.
        scale_attrs (Scaler): The scaler object for the remaining node attributes.
        fitted (bool): Indicates whether the scalers have been fitted to a dataset.
    """

    def __init__(self, scale_xyz: str = "standard", scale_attrs: str = "robust") -> None:
        scaler_dict = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": MinMaxScaler(),
        }
        self.scale_xyz = scaler_dict[scale_xyz]
        self.scale_attrs = scaler_dict[scale_attrs]
        self.fitted = False

    def fit(self, graphs: list[DGLGraph]) -> None:
        """Fit the scalers to the node attributes of the graphs in the dataset.

        Args:
            graphs (list[DGLGraph]): The list of graphs to fit the scalers to.
        """
        graph_nattrs = [graph.ndata["nattrs"].cpu() for graph in graphs]
        nattrs = torch.cat(graph_nattrs, dim=0)
        self.scale_xyz.fit(nattrs[:, :3].numpy())
        self.scale_attrs.fit(nattrs[:, 3:].numpy())
        self.fitted = True

    def transform(self, graph: DGLGraph) -> DGLGraph:
        """Standardize the node attributes of a graph using the fitted scalers.

        Args:
            graph (DGLGraph): The graph to transform.

        Returns:
            DGLGraph: The transformed graph.

        Raises:
            RuntimeError: If the scalers have not been fitted before calling this method.
        """
        if not self.fitted:
            raise RuntimeError("GraphScaler must be fitted before transforming data")

        graph.ndata["nattrs"][:, :3] = torch.from_numpy(
            self.scale_xyz.transform(graph.ndata["nattrs"][:, :3].numpy())
        )
        graph.ndata["nattrs"][:, 3:] = torch.from_numpy(
            self.scale_attrs.transform(graph.ndata["nattrs"][:, 3:].numpy())
        )

        return graph


class NeuronGraphDataset(DGLDataset):
    """A dataset consisting of DGLGraphs representing neuron morphologies.

    Args:
        graphs_path (Path): The path to the SWC file directory.
        self_loop (bool): Whether to add self-loops to each graph. Defaults to True.
        scaler (GraphScaler): The scaler object to use to standardize the node attributes.
        dataset_name (str, optional): The name of the dataset. Defaults to "neuron_graph_dataset".
        dataset_path (Path, optional): The path where the processed dataset will be saved.
        Defaults to the parent directory of the graphs_path.

    Attributes:
        graphs_path (Path): The path to the SWC file directory.
        dataset_path (Path): The path to the directory where the processed dataset will be saved.
        self_loop (bool): Whether to add self-loops to each graph.
        graphs (list[DGLGraph]): The list of DGLGraphs representing neuron morphologies.

    See Also:
        Documentation for DGLDataset: https://docs.dgl.ai/en/latest/api/python/dgl.data.html#dgl.data.DGLDataset
    """

    def __init__(
        self,
        graphs_path: str | Path,
        self_loop: bool = True,
        scaler: GraphScaler | None = None,
        dataset_name: str = "neuron_graph_dataset",
        dataset_path: str | Path | None = None,
        label_file: str | Path | None = None,
    ):
        self.graphs_path = Path(graphs_path)
        self.dataset_path = (
            Path(dataset_path) if dataset_path else Path(self.graphs_path.parent / "dgl_datasets")
        )
        self.graphs: list[DGLGraph] = []
        self.graph_ids: list[str] = []
        self.labels: Tensor | None = None
        self.num_classes: int | None = None
        self.glabel_dict: dict[int, str] | None = None
        self.label_file = Path(label_file) if label_file else None
        self.self_loop = self_loop
        self.scaler = scaler
        self.rescaled = bool(scaler)
        self.logger: EventLogger | None = None

        super().__init__(name=dataset_name, raw_dir=self.dataset_path, verbose=False)

    def process(self) -> None:
        """Process the input data into a list of DGLGraphs."""
        self.logger = EventLogger(self.dataset_path, expt_name=self.name)
        self.logger.message(f"Creating {self.name} dataset from {self.graphs_path}")
        self.logger.message(f"Dataset {self.name} has scaler: {self.scaler}")
        self.logger.message(f"Dataset {self.name} has self-loop: {self.self_loop}")
        swc_files = sorted(self.graphs_path.glob("*.swc"))
        self.graphs = dgl_from_swc(swc_files=swc_files, logger=self.logger)
        if self.scaler is not None:
            self.scaler.fit(self.graphs)

        processed_graphs = []
        while self.graphs:
            original_graph = self.graphs.pop(0)
            graph_id = original_graph.id
            if self.self_loop:
                graph = dgl.add_self_loop(
                    dgl.remove_self_loop(original_graph),
                    edge_feat_names=["edge_weight"],
                    fill_data=1.0,
                )
            processed_graph = self.scaler.transform(graph) if self.scaler else graph
            processed_graph.id = graph_id
            processed_graphs.append(processed_graph)

        self.graphs = processed_graphs
        self.graph_ids = [graph.id for graph in self.graphs]

        if self.label_file:
            self.logger.message(f"Adding labels from {self.label_file} to graphs.")
            self.labels, self.glabel_dict = add_graph_labels(self.label_file, self.graphs)
            self.num_classes = len(self.glabel_dict)

    def save(self, filename: str | None = None) -> None:
        """Save the dataset to disk."""
        if not self.dataset_path.exists():
            self.dataset_path.mkdir(exist_ok=True)
        export_filename = filename if filename else f"{super().name}"
        label_dict = {"labels": self.labels}
        info_dict = {
            "graph_ids": self.graph_ids,
            "num_classes": self.num_classes or None,
            "glabel_dict": self.glabel_dict,
            "self_loop": self.self_loop,
            "rescaled": bool(self.scaler),
            "label_file": self.label_file,
        }
        save_graphs(
            f"{self.dataset_path}/{export_filename}.bin",
            self.graphs,
            label_dict if isinstance(self.labels, Tensor) else None,
        )
        save_info(f"{self.dataset_path}/{export_filename}.pkl", info_dict)

    def load(self) -> None:
        """Load the dataset from disk.

        If the dataset has not been cached, it will be created and cached.
        """
        graphs, label_dict = load_graphs(str(self.cached_graphs_path))
        info_dict = load_info(str(self.info_path))
        self.graph_ids = info_dict.get("graph_ids", None)
        self.labels = label_dict.get("labels", None)
        self.glabel_dict = info_dict.get("glabel_dict", None)
        self.num_classes = info_dict.get("num_classes", None)
        self.self_loop = info_dict.get("self_loop", None)
        self.rescaled = info_dict.get("rescaled", None)
        self.label_file = info_dict.get("label_file", None)
        if self.graph_ids:
            for idx, graph in enumerate(graphs):
                graph.id = info_dict.get("graph_ids")[idx]
        self.graphs = graphs

    def has_cache(self) -> bool:
        """Determine whether there exists a cached dataset.

        Returns:
            bool: True if there exists a cached dataset, False otherwise.
        """
        cached_file_path = self.cached_graphs_path
        info_file_path = self.info_path
        return cached_file_path.exists() and info_file_path.exists()

    @property
    def info_path(self) -> Path:
        """The path to the info file."""
        return Path(self.dataset_path / f"{super().name}.pkl")

    @property
    def cached_graphs_path(self) -> Path:
        """The path to the cached graphs."""
        return Path(self.dataset_path / f"{super().name}.bin")

    def __len__(self) -> int:
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx: int | list[int] | slice) -> DGLGraph:
        """Get the idx-th sample."""
        if isinstance(idx, slice):
            idx = list(range(*idx.indices(len(self.graphs))))
        if isinstance(idx, list | range):
            graphs = [self.graphs[i] for i in idx]
            labels = None if self.labels is None else self.labels[idx]
        else:
            graphs = self.graphs[idx]
            labels = None if self.labels is None else self.labels[idx]

        return (graphs, labels) if labels is not None else graphs


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def create_dataset(
        input_dir: str = typer.Argument(
            ..., help="Path to the directory containing the .swc files."
        ),
        self_loop: bool = typer.Option(
            True,
            help="Optional flag to add self-loops to each graph.",
        ),
        scale: bool = typer.Option(
            True,
            help="Optional flag to apply scaling to the dataset.",
        ),
        scale_xyz: str = typer.Option(
            "standard",
            help="The type of scaler to use for the 'xyz' features.",
        ),
        scale_attrs: str = typer.Option(
            "robust",
            help="The type of scaler to use for the 'attr' features.",
        ),
        dataset_name: str = typer.Option(
            "neuron_graph_dataset",
            help="Name of the dataset.",
        ),
        label_file: str = typer.Option(
            None,
            help="Path to the file containing the metadata (graph labels).",
        ),
    ) -> None:
        """Create a processed dataset of graphs from the .swc files in the specified directory.

        Args:
            input_dir (str): Path to the directory containing the .swc files.
            self_loop (bool): Optional flag to add self-loops to each graph. Defaults to True.
            scale (bool): Optional flag to apply scaling to the dataset. Defaults to True.
            scale_xyz (str): The type of scaler to use for the 'xyz' coordinates.
            scale_attrs (str): The type of scaler to use for the 'nattrs' features.
            dataset_name (str): Name of the dataset.
            label_file (str): Path to the file containing the metadata (graph labels).
        """
        graphs_dir = Path(input_dir)
        scaler = GraphScaler(scale_xyz=scale_xyz, scale_attrs=scale_attrs) if scale else None
        NeuronGraphDataset(
            graphs_path=graphs_dir,
            self_loop=self_loop,
            scaler=scaler,
            dataset_name=dataset_name,
            label_file=label_file,
        )

    app()
