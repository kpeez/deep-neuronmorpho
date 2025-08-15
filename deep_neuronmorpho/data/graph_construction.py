from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def swc_df_to_pyg_data(
    df: pd.DataFrame,
    *,
    include_traversal_indices: bool = True,
    assume_first_row_root: bool = False,
) -> Data:
    """
    Build a PyG Data from an SWC-like DataFrame (columns: n, type, x, y, z, radius, parent),
    using DFS pre-order as the canonical node order (soma/root becomes row 0).

    Args:
        df (pd.DataFrame): SWC-like DataFrame containing at least columns: {"n", "parent", "x", "y", "z"}.
        include_traversal_indices (bool): If True, attach per-row DFS entry/exit indices
            (data.dfs_entry, data.dfs_exit). Defaults to True.
        assume_first_row_root (bool): If True, assume the first row's id is the soma/root id.
            Otherwise, infer root via parent == -1 (fallback to type == 1 or first row).

    Returns:
        Data:
          - x: [N, 3] float32 positions in DFS pre-order
          - edge_index: [2, E] long parent->child (restricted to soma component)
          - root: int (0)
          - orig_id: [N] long original SWC 'n' in DFS order
          - dfs_entry/dfs_exit: [N] long entry/exit indices (included if requested)
    """
    required = {"n", "parent", "x", "y", "z"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"SWC DataFrame is missing required columns: {sorted(missing)}")

    n = df["n"].to_numpy(np.int64)
    parent = df["parent"].to_numpy(np.int64)
    xyz = df[["x", "y", "z"]].to_numpy(np.float32)
    soma_id = int(n[0]) if assume_first_row_root else infer_soma_id(n, parent, df)
    neigh = build_swc_undirected_adj(n, parent)
    dfs_ids, first_id, last_id = dfs_preorder_with_entry_exit(neigh, soma_id)
    # map ids -> original file row for fast vectorized gather
    id2row = {int(_n): i for i, _n in enumerate(n)}
    idx_orig = np.fromiter((id2row[i] for i in dfs_ids), dtype=np.int64, count=len(dfs_ids))
    pos = torch.from_numpy(xyz[idx_orig])  # [N,3], float32
    orig_id = torch.as_tensor(dfs_ids, dtype=torch.long)  # [N], original SWC ids
    id_in_comp = set(dfs_ids)
    new_index = {nid: i for i, nid in enumerate(dfs_ids)}

    mask = parent != -1
    child_ids = n[mask]
    parent_ids = parent[mask]

    edge_par, edge_ch = [], []
    for cid_raw, pid_raw in zip(child_ids, parent_ids, strict=True):
        cid = int(cid_raw)
        pid = int(pid_raw)
        if cid in id_in_comp and pid in id_in_comp:
            edge_par.append(new_index[pid])
            edge_ch.append(new_index[cid])

    edge_index = torch.tensor([edge_par, edge_ch], dtype=torch.long)
    data = Data(x=pos, edge_index=edge_index)
    data.root = 0
    data.orig_id = orig_id
    if include_traversal_indices:
        dfs_entry = torch.empty(len(dfs_ids), dtype=torch.long)
        dfs_exit = torch.empty(len(dfs_ids), dtype=torch.long)
        for nid, row_idx in new_index.items():
            dfs_entry[row_idx] = first_id[nid]
            dfs_exit[row_idx] = last_id[nid]
        data.dfs_entry = dfs_entry
        data.dfs_exit = dfs_exit
    # alias coordinates to PyG position
    data.pos = data.x

    return data


def infer_soma_id(n: np.ndarray, parent: np.ndarray, df: pd.DataFrame) -> int:
    """
    Determine the soma/root SWC id from arrays and dataframe.

    Preference order:
      1) parent == -1 (SWC root definition)
      2) if none and 'type' column exists: type == 1 (soma type)
      3) fallback: first row's id
    """
    soma_mask = parent == -1
    if soma_mask.sum() == 0 and "type" in df.columns:
        soma_mask = df["type"].to_numpy(np.int64) == 1
    if soma_mask.sum() == 0:
        return int(n[0])
    return int(n[soma_mask][0])


def build_swc_undirected_adj(n: np.ndarray, parent: np.ndarray) -> dict[int, list[int]]:
    """
    Build an undirected adjacency mapping on SWC ids.
    Guards against stray parent references not present in `n`.
    """
    neighbors: dict[int, list[int]] = defaultdict(list)
    id_set = {int(_n) for _n in n}
    for cid_raw, pid_raw in zip(n, parent, strict=False):
        cid = int(cid_raw)
        pid = int(pid_raw)
        if pid == -1:
            continue
        if pid in id_set:
            neighbors[pid].append(cid)
            neighbors[cid].append(pid)
    # ensure keys for all ids (even isolated)
    for _n in n:
        neighbors.setdefault(int(_n), [])
    return dict(neighbors)


def dfs_preorder_with_entry_exit(
    neighbors: dict[int, list[int]], root_id: int
) -> tuple[list[int], dict[int, int], dict[int, int]]:
    """
    Iterative DFS pre-order over ids starting at `root_id`.

    Returns:
      - dfs_ids: list of node ids in DFS pre-order (root first)
      - entry_idx: id -> DFS entry index
      - exit_idx:  id -> DFS exit  index (inclusive)
    Only visits the root-connected component.
    """
    dfs_ids: list[int] = []
    entry_idx: dict[int, int] = {}
    exit_idx: dict[int, int] = {}
    stack: list[tuple[int, int, int]] = [(root_id, root_id, 0)]
    seen = {root_id}
    while stack:
        u, p, state = stack.pop()
        if state == 0:
            entry_idx[u] = len(dfs_ids)
            dfs_ids.append(u)
            stack.append((u, p, 1))
            # push children in reverse for natural order; skip parent edge
            for v in reversed(neighbors.get(u, [])):
                if v == p or v in seen:
                    continue
                seen.add(v)
                stack.append((v, u, 0))
        else:
            exit_idx[u] = len(dfs_ids) - 1

    return dfs_ids, entry_idx, exit_idx
