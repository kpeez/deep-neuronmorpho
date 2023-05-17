"""Prepare neuron graphs for conversion to DGL datasets."""
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import torch
from dgl import DGLGraph
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from scipy import stats
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ..utils import ProgressBar
from .process_swc import swc_to_neuron_tree


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


def compute_edge_weights(G: nx.Graph, epsilon: float = 1.0) -> nx.Graph:
    """Compute edge attention weights for a graph.

    Based on the method described in [Zhao et al. 2022](https://ieeexplore.ieee.org/document/9895206).

    Args:
        G (nx.Graph): Graph to compute edge weights for.
        epsilon (float, optional): Small constant to prevent division by zero. Defaults to 1.0.

    Returns:
        nx.Graph: Graph with attention weights added as edge attributes.
    """
    for u, v in G.edges:
        G.add_edge(v, u)
    node_coeffs = {node: 1.0 / (ndata["nattrs"][5] + epsilon) for node, ndata in G.nodes(data=True)}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        local_coeffs = np.array([node_coeffs[neighbor] for neighbor in neighbors], dtype=np.float32)
        attention_coeffs = np.exp(local_coeffs) / np.sum(np.exp(local_coeffs))

        edge_weights = (
            (neighbor, coeff) for neighbor, coeff in zip(neighbors, attention_coeffs, strict=True)
        )

        for neighbor, weight in edge_weights:
            if G.has_edge(node, neighbor):
                G[node][neighbor]["edge_weight"] = weight
            else:
                G.add_edge(node, neighbor, edge_weight=weight)

    return G


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

        node_attrs = [
            x,
            y,
            z,
            # neuron_graph.nodes[node]["radius"], # uncomment to include radius
            nx.dijkstra_path_length(neuron_graph, 1, node, weight="path_length"),
            euclidean((x, y, z), (soma_x, soma_y, soma_z)),
            *angle_stats,
            *branch_stats,
        ]
        neuron_graph.nodes[node].clear()
        neuron_graph.nodes[node].update({"nattrs": [np.float32(attr) for attr in node_attrs]})
    for _, _, edge_data in neuron_graph.edges(data=True):
        del edge_data["euclidean_dist"]
        del edge_data["path_length"]

    neuron_graph = compute_edge_weights(neuron_graph)

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
        try:
            neuron_graph = create_neuron_graph(file)
            neuron_graphs.append(
                dgl.from_networkx(
                    neuron_graph,
                    node_attrs=["nattrs"],
                    edge_attrs=["edge_weight"],
                )
            )
        except Exception as e:
            print(f"Error creating DGL graph for {file}: {e}")

    return neuron_graphs


def create_dataloaders(
    train_dataset: DGLDataset,
    val_dataset: DGLDataset,
    batch_size: int,
    shuffle: bool = True,
) -> tuple[GraphDataLoader, GraphDataLoader]:
    """Create dataloaders for training and validation datasets.

    Args:
        train_dataset (DGLDataset): Training dataset.
        val_dataset (DGLDataset): Validation dataset.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the training data. Defaults to True.

    Returns:
        tuple[GraphDataLoader, GraphDataLoader]: _description_
    """
    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
    val_loader = GraphDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


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
        # nattrs = [graph.ndata["nattrs"].cpu() for graph in graphs]
        # nattrs = torch.cat(nattrs, dim=0)

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
        self_loop: bool = True,
        scaler: GraphScaler | None = None,
        dataset_name: str = "neuron_graph_dataset",
    ):
        self.graphs_path = graphs_path
        self.export_dir = Path(self.graphs_path.parent / "dgl_datasets")
        self.graphs: list = []
        self.self_loop = self_loop
        self.scaler = scaler

        super().__init__(name=dataset_name, raw_dir=self.export_dir)

    def process(self) -> None:
        """Process the input data into a list of DGLGraphs."""
        self.graphs = dgl_from_swc(list(self.graphs_path.glob("*.swc")))

        if self.scaler is not None:
            self.scaler.fit(self.graphs)

        processed_graphs = []
        for original_graph in self.graphs:
            if self.self_loop:
                graph = dgl.add_self_loop(
                    dgl.remove_self_loop(original_graph),
                    edge_feat_names=["edge_weight"],
                    fill_data=1.0,
                )
            processed_graphs.append(self.scaler.transform(graph) if self.scaler else graph)

        self.graphs = processed_graphs

    def load(self) -> None:
        """Load the dataset from disk.

        If the dataset has not been cached, it will be created and cached.
        """
        self.graphs = load_graphs(str(self.cached_graphs_path))[0]

    def save(self) -> None:
        """Save the dataset to disk."""
        if not self.export_dir.exists():
            self.export_dir.mkdir(exist_ok=True)
        save_graphs(f"{self.export_dir}/{super().name}.bin", self.graphs)

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
        return Path(self.export_dir / f"{super().name}.bin")

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
            True,
            help="Optional flag to add self-loops to each graph.",
        ),
        scale: bool = typer.Option(  # noqa: B008
            True,
            help="Optional flag to apply scaling to the dataset.",
        ),
        scale_xyz: str = typer.Option(  # noqa: B008
            "standard",
            help="The type of scaler to use for the 'xyz' features.",
        ),
        scale_attrs: str = typer.Option(  # noqa: B008
            "robust",
            help="The type of scaler to use for the 'attr' features.",
        ),
        dataset_name: str = typer.Option(..., help="Name of the dataset."),  # noqa: B008
    ) -> None:
        """Create a processed dataset of graphs from the .swc files in the specified directory.

        Args:
            input_dir (str): Path to the directory containing the .swc files.
            self_loop (bool): Optional flag to add self-loops to each graph. Defaults to True.
            scale (bool): Optional flag to apply scaling to the dataset. Defaults to True.
            scale_xyz (str): The type of scaler to use for the 'xyz' coordinates.
            scale_attrs (str): The type of scaler to use for the 'nattrs' features.
            dataset_name (str): Name of the dataset.
        """
        graphs_dir = Path(input_dir)
        scaler = GraphScaler(scale_xyz=scale_xyz, scale_attrs=scale_attrs) if scale else None
        NeuronGraphDataset(
            graphs_path=graphs_dir,
            self_loop=self_loop,
            scaler=scaler,
            dataset_name=dataset_name,
        )

    app()
