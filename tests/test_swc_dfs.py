"""
Tests for building a PyG Data from an SWC DataFrame.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from deep_neuronmorpho.data import SWCData, convert_swc_to_pyg


def test_synthetic_swc_dfs_invariants():
    # Simple tree:
    # 1 (-1)
    # ├─ 2 (1)
    # └─ 3 (1)
    #     └─ 4 (3)
    df = pd.DataFrame(
        {
            "n": [1, 2, 3, 4],
            "type": [1, 3, 3, 3],
            "x": [0.0, 1.0, 1.0, 2.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "radius": [1.0, 0.5, 0.5, 0.5],
            "parent": [-1, 1, 1, 3],
        }
    )

    data = convert_swc_to_pyg(df)

    # N = 4 nodes; E = 3 directed edges
    assert data.x.shape == (4, 3)
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] == 3

    # Root is first in DFS order and should correspond to id 1
    assert data.root == 0
    assert int(data.orig_id[0]) == 1

    # Positions alias
    assert torch.equal(data.pos, data.x)

    # Traversal indices attached by default
    assert hasattr(data, "dfs_first") and hasattr(data, "dfs_last")
    assert data.dfs_first.shape == (4,)
    assert data.dfs_last.shape == (4,)
    assert torch.all(data.dfs_first <= data.dfs_last)


def test_real_swc_file_order_roundtrip_if_available():
    # Optional integration-like test: requires a local SWC file. Skip if missing
    swc_path = Path("../data/1164438028_191812_5676-X8072-Y26215_reg.swc")
    if not swc_path.exists():
        pytest.skip("external SWC file not available; skipping integration test")

    df = SWCData.load_swc_data(swc_path)
    data = convert_swc_to_pyg(df)

    # Reconstruct positions in original file order using orig_id mapping
    pos_file = torch.tensor(df[["x", "y", "z"]].to_numpy(), dtype=torch.float32)
    id2row = {int(n): i for i, n in enumerate(df["n"].to_numpy(np.int64))}
    perm = torch.tensor([id2row[int(i)] for i in data.orig_id.tolist()], dtype=torch.long)
    pos_new_in_file_order = data.x[perm]

    torch.testing.assert_close(pos_new_in_file_order, pos_file, rtol=1e-6, atol=1e-6)
