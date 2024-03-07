"""Tests for data_utils.py."""
import dgl
import numpy as np
import torch

from deep_neuronmorpho.data.data_utils import compute_graph_attrs, graph_is_broken


def test_compute_graph_attrs() -> None:
    graph_attrs = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected_result = [1.0, 3.0, 3.0, 5.0, 1.581138, 5]
    result = compute_graph_attrs(graph_attrs)

    np.testing.assert_allclose(result, expected_result, rtol=1e-4)


def test_graph_is_broken() -> None:
    g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 0])))
    g.ndata["nattrs"] = torch.tensor(
        [
            [1.0, 2.0, float("nan"), 4.0, 5.0],
            [0.5, 1.5, 2.5, 3.5, 4.5],
        ]
    )
    assert graph_is_broken(g)
