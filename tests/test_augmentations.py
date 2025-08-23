"""Tests for custom PyG transforms in deep_neuronmorpho.data.augmentations."""

import torch
from torch_geometric.data import Data

from deep_neuronmorpho.data.augmentations import DropRandomBranches, RandomTranslate


def test_drop_random_branches_drops_x_when_not_recomputed(pyg_data):
    transform = DropRandomBranches(drop_fraction=0.25, min_keep_nodes=1, recompute_features=False)
    out = transform(pyg_data)
    print(out)
    # by design: x should be absent unless recompute_features=True
    assert out.x is None
    assert hasattr(out, "pos") and hasattr(out, "edge_index") and hasattr(out, "root")
    assert out.pos.ndim == 2 and out.pos.size(1) == 3
    assert out.edge_index.ndim == 2 and out.edge_index.size(0) == 2


def test_random_translate_constant_shift_and_edge_distance_invariance(pyg_data):
    data = pyg_data.clone()
    orig_pos = data.pos.clone()
    orig_ei = data.edge_index.clone()

    torch.manual_seed(0)
    t = RandomTranslate(translate=0.5)
    out = t(data)

    assert torch.equal(out.edge_index, orig_ei)
    delta = out.pos - orig_pos
    delta_var = torch.var(delta, dim=0)
    assert torch.allclose(delta_var, torch.zeros_like(delta_var), atol=1e-10)

    idx0 = orig_ei[0]
    idx1 = orig_ei[1]
    dist_before = (orig_pos[idx0] - orig_pos[idx1]).norm(dim=1)
    dist_after = (out.pos[idx0] - out.pos[idx1]).norm(dim=1)
    assert torch.allclose(dist_before, dist_after, atol=1e-5, rtol=1e-5)


def test_random_translate_pass_through_when_no_pos():
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    data = Data(edge_index=edge_index)

    t = RandomTranslate(translate=1.0)
    out = t(data)

    assert out is data
    assert getattr(out, "pos", None) is None
    assert torch.equal(out.edge_index, edge_index)
