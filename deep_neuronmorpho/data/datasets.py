"""NeuronDataset for creating PyG datasets from SWC files."""

import bisect
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable

import torch
import typer
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.data.separate import separate
from tqdm import tqdm

from .graph_construction import swc_df_to_pyg_data
from .graph_features import compute_neuron_node_feats
from .process_swc import SWCData

app = typer.Typer()


class NeuronDataset(Dataset):
    """
    Base dataset class for loading and processing SWC files into PyG Data objects.
      - Reads standardized SWC from `raw_dir` (e.g., datasets/interim/<dataset>)
      - Writes shards to `<root>/processed/shard_*.pt` and index to `<root>/processed/meta.pt`
      - Always returns a single `Data`; use wrappers for contrastive views.
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        raw_dir: str | None = None,
        shard_size: int = 2000,
        transform=None,
        pre_transform=None,
    ):
        self._root = root
        self._name = dataset_name or Path(root).name
        self._raw_dir = raw_dir or str(Path(root) / "raw")
        self._shard_size = int(shard_size)

        self._raw_files = sorted([str(p) for p in Path(self._raw_dir).glob("*.swc")])
        # lazy shard cache (keep only last loaded shard to stay tiny)
        self._cur_sid = None
        self._cur_data = None
        self._cur_slices = None

        super().__init__(root, transform=transform, pre_transform=pre_transform)

        self._meta = torch.load(self._meta_path()) if Path(self._meta_path()).exists() else None
        if self._meta:
            self._cumsum = self._meta["cumsum"]
            self._shards = self._meta["shards"]

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
        """Require meta.pt to exist; it indexes all shards."""
        return ["meta.pt"]

    def len(self) -> int:
        return 0 if self._meta is None else int(self._meta["total"])

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        shards, cumsum, total = [], [], 0
        shard_id = 0
        data_buf: list[Data] = []
        num_workers = cpu_count()

        def flush_shard(buf: list[Data], sid: int):
            nonlocal total
            if not buf:
                return
            if self.pre_transform is not None:
                buf = [self.pre_transform(d) for d in buf]
            data, slices = InMemoryDataset.collate(buf)
            shard_name = f"{self._name}-shard_{sid:05d}.pt"
            shard_path = Path(self.processed_dir) / shard_name
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save((data, slices), str(shard_path))
            total += len(buf)
            cumsum.append(total)
            shards.append(shard_name)
            buf.clear()

        with Pool(num_workers) as pool:
            process_iter = pool.imap(process_swc_file, self._raw_files)
            for pyg_data in tqdm(
                process_iter, total=len(self._raw_files), desc="Processing SWC files"
            ):
                data_buf.append(pyg_data)
                if len(data_buf) >= self._shard_size:
                    flush_shard(data_buf, shard_id)
                    shard_id += 1

        flush_shard(data_buf, shard_id)

        if len(shards) == 1:
            old = Path(self.processed_dir) / shards[0]
            new_name = f"{self._name}.pt"
            new = Path(self.processed_dir) / new_name
            try:
                old.replace(new)
                shards[0] = new_name
            except OSError:
                pass

        meta = {
            "total": total,
            "cumsum": cumsum,
            "shards": shards,
            "shard_size": self._shard_size,
        }
        torch.save(meta, self._meta_path())

        self._meta, self._cumsum, self._shards = meta, cumsum, shards

    def get(self, idx: int) -> Data:
        sid = bisect.bisect_right(self._cumsum, idx)
        start = 0 if sid == 0 else self._cumsum[sid - 1]
        local_idx = idx - start

        if self._cur_sid != sid:
            shard_path = str(Path(self.processed_dir) / self._shards[sid])
            self._cur_data, self._cur_slices = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=False,
            )
            self._cur_sid = sid

        return separate(
            cls=Data,
            batch=self._cur_data,
            idx=local_idx,
            slice_dict=self._cur_slices,
            decrement=False,
        )


def process_swc_file(swc_file: str) -> Data:
    swc_df = SWCData.load_swc_data(swc_file)
    pyg_data = swc_df_to_pyg_data(swc_df)
    pyg_data.sample_id = Path(swc_file).stem
    pyg_data.x = compute_neuron_node_feats(pyg_data.x, pyg_data.edge_index, pyg_data.root)

    return pyg_data


class ContrastiveNeuronDataset(Dataset):
    """
    Wrapper dataset that takes single Data object and returns two contrastive views.
    """

    # TODO: recompute all features after transform
    def __init__(self, dataset: Dataset, transform: Callable):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Data, Data]:
        data = self.dataset[idx]
        g1, g2 = self.transform(data.clone()), self.transform(data.clone())
        g1.x[:, :3], g2.x[:, :3] = g1.pos, g2.pos

        return g1, g2


@app.command()
def main(
    root_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="The root directory of the processed dataset.",
    ),
    raw_dir: Path = typer.Option(
        None,
        "-r",
        "--raw-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory containing the raw SWC files. Defaults to <root_dir>/raw.",
    ),
    dataset_name: str = typer.Option(
        None,
        "-n",
        "--dataset-name",
        help="The name of the dataset.",
    ),
    shard_size: int = typer.Option(
        4096,
        "-s",
        "--shard-size",
        help="The number of SWC files to process in each shard.",
    ),
):
    """Create a PyG dataset from SWC files."""

    raw_dir = raw_dir if raw_dir is not None else root_dir / "raw"
    dataset_name = dataset_name if dataset_name is not None else root_dir.name

    typer.echo(f"Processing dataset '{dataset_name}'...")
    typer.echo(f"  - Root directory: {root_dir}")
    typer.echo(f"  - Raw SWC files: {raw_dir}")
    typer.echo(f"  - Shard size: {shard_size}")

    NeuronDataset(
        str(root_dir),
        raw_dir=raw_dir,
        dataset_name=dataset_name,
        shard_size=shard_size,
    )

    typer.echo("\n✅ Dataset processing complete.")
    typer.echo(f"Processed shards are saved in: {root_dir / 'processed'}")


if __name__ == "__main__":
    app()
