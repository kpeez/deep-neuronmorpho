"""Prepare neuron graphs for conversion to DGL datasets."""

from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean

from .process_swc import SWCData
from .utils import compute_edge_weights, compute_graph_attrs


def create_neuron_graph(swc_file: str | Path) -> nx.DiGraph:
    """Create networkx graph of neuron from swc format.

    Args:
        swc_file (str | Path): Morphopy NeuronTree object of neuron swc data.

    Returns:
        nx.DiGraph: Graph of neuron.

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
    neuron_tree = SWCData(swc_file=swc_file, standardize=False, align=False).ntree
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


# class NeuronGraphDataset(DGLDataset):
#     """A dataset consisting of DGLGraphs representing neuron morphologies.

#     Args:
#         name (str | Path): The name of the dataset.
#         graphs_path (str | Path, optional): The path to the SWC file directory. Defaults to the current directory.
#         dataset_path (Path, optional): The path where the processed dataset will be saved.
#         label_file (Path, optional): The path to the file containing the metadata (graph labels).
#         self_loop (bool): Whether to add self-loops to each graph. Defaults to True.
#         scaler (GraphScaler): The scaler object to use to standardize the node attributes.
#         Defaults to the parent directory of the graphs_path.
#         from_file (bool): Whether to load the dataset from a file. If True, use the full path to the
#         dataset file as the name. Defaults to False.

#     Attributes:
#         graphs_path (Path): The path to the SWC file directory.
#         dataset_path (Path): The path to the directory where the processed dataset will be saved.
#         self_loop (bool): Whether to add self-loops to each graph.
#         graphs (list[DGLGraph]): The list of DGLGraphs representing neuron morphologies.
#         graph_ids (list[str]): The list of graph IDs.
#         labels (Tensor): The tensor of graph labels.
#         num_classes (int): The number of classes in the dataset.
#         glabel_dict (dict[int, str]): The dictionary of graph labels.
#         rescaled (bool): Whether the node attributes have been rescaled.

#     See Also:
#         Documentation for DGLDataset: https://docs.dgl.ai/en/latest/api/python/dgl.data.html#dgl.data.DGLDataset
#     """

#     def __init__(
#         self,
#         name: str | Path,
#         graphs_path: str | Path | None = None,
#         dataset_path: str | Path | None = None,
#         label_file: str | Path | None = None,
#         self_loop: bool = True,
#         scaler: GraphScaler | None = None,
#         from_file: bool = False,
#     ):
#         if from_file:
#             dataset_path = Path(name).parent
#             name = Path(name).stem

#         self.graphs_path = Path(graphs_path).resolve() if graphs_path is not None else Path.cwd()
#         self.dataset_path = (
#             Path(dataset_path) if dataset_path else Path(self.graphs_path.parent / "dgl_datasets")
#         )
#         self.dataset_path.mkdir(exist_ok=True)
#         self.graphs: list[DGLGraph] = []
#         self.graph_ids: list[str] = []
#         self.labels: Tensor | None = None
#         self.num_classes: int | None = None
#         self.glabel_dict: dict[int, str] | None = None
#         self.label_file = Path(label_file) if label_file else None
#         self.self_loop = self_loop
#         self.scaler = scaler
#         self.rescaled = bool(scaler)
#         self.logger: EventLogger | None = None

#         super().__init__(
#             name=name,
#             raw_dir=self.graphs_path,
#             save_dir=self.dataset_path,
#             verbose=False,
#         )

#     def process(self) -> None:
#         """Process the input data into a list of DGLGraphs."""
#         self.logger = EventLogger(self.dataset_path, expt_name=self.name)
#         self.logger.message(
#             f"""Creating dataset: {self.name} from {self.graphs_path} \n"""
#             f"""Dataset {self.name} has scaler: {self.scaler} \n"""
#             f"""Dataset {self.name} has self-loop: {self.self_loop}"""
#         )
#         swc_files = sorted(self.graphs_path.glob("*.swc"))

#         with ProcessPoolExecutor() as executor:
#             futures = [executor.submit(swc_to_dgl, swc_file, self.logger) for swc_file in swc_files]
#             self.logger.message("Loading graphs...")
#             for future in ProgressBar(futures, desc="Loading DGLGraph:"):
#                 if future.result() is not None:
#                     self.graphs.append(future.result())
#                     self.logger.message(f"Processed: {future.result().id}")
#                 else:
#                     self.logger.message(f"Skipped: {swc_files[futures.index(future)].name}")

#         if self.scaler is not None:
#             self.scaler.fit(self.graphs)

#         self.logger.message("Processing graphs...")
#         for i in range(len(self.graphs)):
#             original_graph = self.graphs[i]
#             if self.self_loop:
#                 self.graphs[i] = dgl.add_self_loop(
#                     dgl.remove_self_loop(original_graph),
#                     edge_feat_names=["edge_weight"],
#                     fill_data=1.0,
#                 )
#             self.graphs[i] = (
#                 self.scaler.transform(self.graphs[i]) if self.scaler is not None else original_graph
#             )
#             self.graphs[i].id = original_graph.id
#         self.graph_ids = [graph.id for graph in self.graphs]

#         if self.label_file:
#             self.logger.message(f"Adding labels from {self.label_file} to graphs.")
#             self.labels, self.glabel_dict = add_graph_labels(self.label_file, self.graphs)
#             self.num_classes = len(self.glabel_dict)
#         self.logger.message(f"Processed {len(self.graphs)}/{len(swc_files)} graphs.")

#     def save(self, filename: str | None = None) -> None:
#         """Save the dataset to disk."""
#         if not self.dataset_path.exists():
#             self.dataset_path.mkdir(exist_ok=True)
#         export_filename = filename if filename else f"{self.name}"
#         label_dict = {"labels": self.labels}
#         info_dict = {
#             "graph_ids": self.graph_ids,
#             "num_classes": self.num_classes or None,
#             "glabel_dict": self.glabel_dict,
#             "self_loop": self.self_loop,
#             "rescaled": bool(self.scaler),
#             "label_file": self.label_file,
#         }
#         save_graphs(
#             f"{self.dataset_path}/{export_filename}.bin",
#             self.graphs,
#             label_dict if isinstance(self.labels, Tensor) else None,
#         )
#         save_info(f"{self.dataset_path}/{export_filename}.pkl", info_dict)

#     def load(self) -> None:
#         """Load the dataset from disk. If the dataset has not been cached,
#         it will be created and cached."""
#         graphs, label_dict = load_graphs(str(self.cached_graphs_path))
#         info_dict = load_info(str(self.info_path))
#         self.graph_ids = info_dict.get("graph_ids", None)
#         self.labels = label_dict.get("labels", None)
#         self.glabel_dict = info_dict.get("glabel_dict", None)
#         self.num_classes = info_dict.get("num_classes", None)
#         self.self_loop = info_dict.get("self_loop", None)
#         self.rescaled = info_dict.get("rescaled", None)
#         self.label_file = info_dict.get("label_file", None)
#         if self.graph_ids:
#             for idx, graph in enumerate(graphs):
#                 graph.id = info_dict.get("graph_ids")[idx]
#         self.graphs = graphs

#     def has_cache(self) -> bool:
#         """Determine whether there exists a cached dataset."""
#         return self.cached_graphs_path.exists() and self.info_path.exists()

#     @property
#     def info_path(self) -> Path:
#         """The path to the info file."""
#         return Path(self.dataset_path / f"{self.name}.pkl")

#     @property
#     def cached_graphs_path(self) -> Path:
#         """The path to the cached graphs."""
#         return Path(self.dataset_path / f"{self.name}.bin")

#     def __len__(self) -> int:
#         """Return the number of graphs in the dataset."""
#         return len(self.graphs)

#     def __getitem__(self, idx: int | list[int] | slice) -> DGLGraph:
#         """Get the idx-th sample."""
#         if isinstance(idx, slice):
#             idx = list(range(*idx.indices(len(self.graphs))))
#         if isinstance(idx, list | range):
#             graphs = [self.graphs[i] for i in idx]
#             labels = None if self.labels is None else self.labels[idx]
#         else:
#             graphs = self.graphs[idx]
#             labels = None if self.labels is None else self.labels[idx]

#         return (graphs, labels) if labels is not None else graphs


# if __name__ == "__main__":
#     from typing import Optional

#     from typer import Argument, Option, Typer

#     app = Typer()

#     @app.command()
#     def create_dataset(
#         input_dir: str = Argument(..., help="Path to the directory containing the .swc files."),
#         self_loop: bool = Option(
#             True, help="Enable self-loops by default. Use --no-self-loop to disable."
#         ),
#         no_self_loop: bool = Option(False, "--no-self-loop", help="Disable self-loops."),
#         scale: bool = Option(
#             False,
#             "--scale",
#             help="Optional flag to apply scaling to the dataset.",
#             is_flag=True,
#         ),
#         scale_xyz: str = Option(
#             "standard",
#             "--scale-xyz",
#             help="The type of scaler to use for the 'xyz' features.",
#         ),
#         scale_attrs: str = Option(
#             "robust",
#             "--scale-attrs",
#             help="The type of scaler to use for the 'attr' features.",
#         ),
#         dataset_name: str = Option(
#             "neuron_graph_dataset",
#             "--name",
#             help="Name of the dataset.",
#         ),
#         dataset_path: Optional[str] = Option(
#             None,
#             "--dataset-path",
#             help="Path to the directory where the processed dataset will be saved.",
#         ),
#         label_file: Optional[str] = Option(
#             None,
#             "--label-file",
#             help="Path to the file containing the metadata (graph labels).",
#         ),
#     ) -> None:
#         """Create a processed dataset of graphs from the .swc files in the specified directory.

#         Args:
#             input_dir (str): Path to the directory containing the .swc files.
#             self_loop (bool): Optional flag to add self-loops to each graph. Defaults to True.
#             scale (bool): Optional flag to apply scaling to the dataset. Defaults to True.
#             scale_xyz (str): The type of scaler to use for the 'xyz' coordinates.
#             scale_attrs (str): The type of scaler to use for the 'nattrs' features.
#             dataset_name (str): Name of the dataset.
#             label_file (str): Path to the file containing the metadata (graph labels).
#         """
#         graphs_dir = Path(input_dir)
#         scaler = GraphScaler(scale_xyz=scale_xyz, scale_attrs=scale_attrs) if scale else None
#         NeuronGraphDataset(
#             graphs_path=graphs_dir,
#             self_loop=self_loop and not no_self_loop,
#             scaler=scaler,
#             name=dataset_name,
#             dataset_path=dataset_path,
#             label_file=label_file,
#         )

#     app()
