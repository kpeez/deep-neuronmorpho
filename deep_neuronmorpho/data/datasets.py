"""NeuronDataset for creating PyG datasets from SWC files."""

import bisect
import os
from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.data.separate import separate
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

from deep_neuronmorpho.data.process_swc import SWCData


def swc_to_pyg(path: str, label: int | None = None) -> Data:
    swc = SWCData(path, standardize=False, align=False)
    G = swc.ntree.get_graph()
    nodes = sorted(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    coords = torch.tensor(swc.data[["x", "y", "z"]].to_numpy(), dtype=torch.float32)
    edges = torch.tensor(
        [(node_to_idx[u], node_to_idx[v]) for (u, v) in G.edges()],
        dtype=torch.long,
    ).T

    if edges.numel():
        edges, _ = remove_self_loops(edges)
        edges = to_undirected(edges, num_nodes=coords.size(0))
        edges = coalesce(edges)

    data = Data(x=coords, edge_index=edges)
    data.sample_id = Path(path).stem

    if label is not None:
        data.y = torch.tensor(label)

    return data


class NeuronDataset(Dataset):
    """
    Disk-backed, sharded PyG dataset.
      - Reads standardized SWC from `raw_dir` (e.g., datasets/interim/<dataset>)
      - Writes shards to `<root>/processed/shard_*.pt` and index to `<root>/processed/meta.pt`
      - Always returns a single `Data`; use wrappers for contrastive views.
    """

    def __init__(
        self,
        root: str,
        dataset_name: str,
        raw_dir: str | None = None,  # if None, use <root>/raw
        labels: dict[str, int] | None = None,  # basename -> label id
        shard_size: int = 2000,  # graphs per shard (tune 1-10k)
        split_list: str | None = None,  # path to txt of basenames to keep
        transform=None,
        pre_transform=None,
    ):
        self._root = root
        self._name = dataset_name
        self._raw_dir = raw_dir or str(Path(root) / "raw")
        self._labels = labels or {}
        self._shard_size = int(shard_size)

        # Optional split filter
        self._keep = None
        if split_list:
            with open(split_list, encoding="utf-8") as f:
                self._keep = {ln.strip() for ln in f if ln.strip()}

        # Enumerate raw files (stable order)
        swc_paths = sorted([str(p) for p in Path(self._raw_dir).glob("*.swc")])
        if self._keep is not None:
            swc_paths = [p for p in swc_paths if Path(p).stem in self._keep]
        self._raw_files = swc_paths
        # Lazy shard cache (keep only last loaded shard to stay tiny)
        self._cur_sid = None
        self._cur_data = None
        self._cur_slices = None

        super().__init__(root, transform=transform, pre_transform=pre_transform)

        # Load meta index (created in process)
        self._meta = torch.load(self._meta_path()) if Path(self._meta_path()).exists() else None
        if self._meta:
            self._cumsum = self._meta["cumsum"]
            self._shards = self._meta["shards"]  # list of shard filenames

    @property
    def processed_dir(self) -> str:
        return str(Path(self._root) / "processed")

    def _meta_path(self) -> str:
        return str(Path(self.processed_dir) / "meta.pt")

    @property
    def raw_file_names(self) -> list[str]:
        return [Path(p).name for p in self._raw_files]

    @property
    def processed_file_names(self) -> list[str]:
        # Only require meta.pt to exist; it indexes all shards
        return ["meta.pt"]

    def len(self) -> int:
        return 0 if self._meta is None else int(self._meta["total"])

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        data_buf: list[Data] = []
        shards, cumsum, total = [], [], 0
        sid = 0

        def flush_shard(buf: list[Data], sid: int):
            nonlocal total
            if not buf:
                return
            if self.pre_transform is not None:
                buf = [self.pre_transform(d) for d in buf]
            data, slices = InMemoryDataset.collate(buf)
            shard_name = f"{self._name}/shard_{sid:05d}.pt"
            torch.save((data, slices), str(Path(self.processed_dir) / shard_name))
            total += len(buf)
            cumsum.append(total)
            shards.append(shard_name)
            buf.clear()

        for swc_path in self._raw_files:
            stem = Path(swc_path).stem
            y = self._labels.get(stem)
            d = swc_to_pyg(swc_path, label=y)
            data_buf.append(d)
            if len(data_buf) >= self._shard_size:
                flush_shard(data_buf, sid)
                sid += 1
        flush_shard(data_buf, sid)

        meta = {"total": total, "cumsum": cumsum, "shards": shards, "shard_size": self._shard_size}
        torch.save(meta, self._meta_path())

        # Initialize in-memory index for this instance
        self._meta, self._cumsum, self._shards = meta, cumsum, shards

    # ---- item access: map idx -> shard/local_idx ----

    def get(self, idx: int) -> Data:
        sid = bisect.bisect_right(self._cumsum, idx)
        start = 0 if sid == 0 else self._cumsum[sid - 1]
        local_idx = idx - start

        if self._cur_sid != sid:
            shard_path = str(Path(self.processed_dir) / self._shards[sid])
            self._cur_data, self._cur_slices = torch.load(
                shard_path, map_location="cpu", weights_only=False
            )
            self._cur_sid = sid

        return separate(
            cls=Data,
            batch=self._cur_data,
            idx=local_idx,
            slice_dict=self._cur_slices,
            decrement=False,
        )
