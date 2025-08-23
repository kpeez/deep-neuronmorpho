"""Tests for custom PyG transforms in deep_neuronmorpho.data.augmentations."""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from deep_neuronmorpho.data import compute_neuron_node_feats
from deep_neuronmorpho.data.augmentations import (
    DropRandomBranches,
    RandomTranslate,
    RecomputeNodeFeatures,
)


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


def test_drop_random_branches_input_validation():
    """Test that DropRandomBranches validates input parameters correctly."""
    with pytest.raises(ValueError, match="drop_fraction must be in"):
        DropRandomBranches(drop_fraction=-0.1)

    with pytest.raises(ValueError, match="drop_fraction must be in"):
        DropRandomBranches(drop_fraction=1.5)

    with pytest.raises(ValueError, match="min_keep_nodes must be >= 1"):
        DropRandomBranches(min_keep_nodes=0)


def test_drop_random_branches_structural_invariants(pyg_data):
    """Test structural invariants of DropRandomBranches transform."""
    N = pyg_data.pos.size(0)
    for drop_frac, min_keep in [(0.0, 1), (0.2, 50), (0.5, 100)]:
        torch.manual_seed(42)
        transform = DropRandomBranches(
            drop_fraction=drop_frac, min_keep_nodes=min_keep, recompute_features=False
        )
        out = transform(pyg_data)
        N_out = out.pos.size(0)
        # root should be preserved
        assert out.root == 0
        expected_min_keep = max(1, min(N - 1, max(min_keep, int((1 - drop_frac) * N))))
        assert N_out >= expected_min_keep
        # Valid edge indices and shape (no out-of-bounds, no -1)
        assert out.edge_index.min() >= 0
        assert out.edge_index.max() < N_out
        assert out.edge_index.size(0) == 2
        # connectivity check
        undirected_ei = to_undirected(out.edge_index, num_nodes=N_out)
        reachable = _check_connectivity_from_root(undirected_ei, out.root, N_out)
        assert reachable.sum() == N_out, "Graph should remain connected from root"


def test_drop_random_branches_with_recompute_features(pyg_data):
    """Test DropRandomBranches with recompute_features=True."""
    torch.manual_seed(42)
    transform = DropRandomBranches(drop_fraction=0.3, min_keep_nodes=50, recompute_features=True)
    out = transform(pyg_data)
    assert hasattr(out, "x") and out.x is not None
    assert out.x.size() == (out.pos.size(0), 8)
    expected_x = compute_neuron_node_feats(out.pos, out.edge_index, out.root)
    assert torch.allclose(out.x, expected_x, atol=1e-6, rtol=1e-6)


def test_drop_random_branches_boundary_cases(pyg_data):
    """Test boundary parameter values."""
    N = pyg_data.pos.size(0)
    # drop_fraction = 0.0 should keep almost everything (may still drop x if not recomputed)
    torch.manual_seed(42)
    transform = DropRandomBranches(drop_fraction=0.0, min_keep_nodes=1, recompute_features=True)
    out = transform(pyg_data)
    # should keep all or nearly all nodes
    assert out.pos.size(0) >= N - 1
    # small graph with large min_keep_nodes should drop nothing significant
    torch.manual_seed(42)
    transform = DropRandomBranches(
        drop_fraction=0.8, min_keep_nodes=N - 10, recompute_features=True
    )
    out = transform(pyg_data)
    assert out.pos.size(0) >= N - 10


def _check_connectivity_from_root(edge_index, root, num_nodes):
    """Helper: Check which nodes are reachable from root via BFS."""

    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        adj[u].append(v)
        adj[v].append(u)

    visited = torch.zeros(num_nodes, dtype=torch.bool)
    queue = [int(root)]
    visited[root] = True

    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)

    return visited


def test_recompute_node_features_basic_functionality(pyg_data):
    """Test RecomputeNodeFeatures produces correct shape and values."""
    data = pyg_data.clone()

    transform = RecomputeNodeFeatures()
    out = transform(data)
    assert hasattr(out, "x") and out.x is not None
    assert out.x.size() == (data.pos.size(0), 8)
    expected_x = compute_neuron_node_feats(data.pos, data.edge_index, data.root)
    assert torch.allclose(out.x, expected_x, atol=1e-6, rtol=1e-6)


def test_recompute_node_features_translation_invariance(pyg_data):
    """Test that higher-order features are invariant under translation."""
    data1 = pyg_data.clone()
    data2 = pyg_data.clone()
    torch.manual_seed(0)
    translate_transform = RandomTranslate(translate=1.0)
    translate_transform(data2)
    original_features = RecomputeNodeFeatures()(data1).x
    translated_features = RecomputeNodeFeatures()(data2).x
    assert not torch.allclose(original_features[:, :3], translated_features[:, :3], atol=1e-3)
    assert torch.allclose(
        original_features[:, 3:], translated_features[:, 3:], atol=1e-6, rtol=1e-6
    )


def test_recompute_node_features_idempotent(pyg_data):
    """Test that applying RecomputeNodeFeatures twice gives same result."""
    data = pyg_data.clone()

    transform = RecomputeNodeFeatures()
    out1 = transform(data)
    out2 = transform(out1)

    # Should be identical
    assert torch.allclose(out1.x, out2.x, atol=1e-8, rtol=1e-8)


def test_integration_chain_transform_pipeline(pyg_data):
    """Test integration of RandomTranslate → DropRandomBranches → RecomputeNodeFeatures."""
    data = pyg_data.clone()
    N = data.pos.size(0)
    torch.manual_seed(42)
    translate = RandomTranslate(translate=0.8)
    drop_branches = DropRandomBranches(
        drop_fraction=0.3, min_keep_nodes=50, recompute_features=True
    )
    recompute = RecomputeNodeFeatures()
    step1 = translate(data)
    step2 = drop_branches(step1)
    final = recompute(step2)
    # final result should have all required attributes
    assert hasattr(final, "pos") and final.pos is not None
    assert hasattr(final, "edge_index") and final.edge_index is not None
    assert hasattr(final, "root") and final.root == 0
    assert hasattr(final, "x") and final.x is not None
    # shape consistency
    N_final = final.pos.size(0)
    assert final.x.size() == (N_final, 8)
    assert final.edge_index.size(0) == 2
    assert final.edge_index.max() < N_final
    assert final.edge_index.min() >= 0
    # some nodes should have been dropped
    assert N_final < N
    assert N_final >= 50  # Respects min_keep_nodes
    undirected_ei = to_undirected(final.edge_index, num_nodes=N_final)
    reachable = _check_connectivity_from_root(undirected_ei, final.root, N_final)
    assert reachable.sum() == N_final
    # Features should match direct computation (idempotent RecomputeNodeFeatures)
    expected_x = compute_neuron_node_feats(final.pos, final.edge_index, final.root)
    assert torch.allclose(final.x, expected_x, atol=1e-6, rtol=1e-6)
