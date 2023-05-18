import pytest
import mentpy as mp


@pytest.mark.parametrize("n_wires", [1, 2, 3, 4])
def test_lie_algebra_grid(n_wires):
    gs = mp.templates.grid_cluster(n_wires, n_wires + 3)
    lieAlg = mp.utils.calculate_lie_algebra(gs, max_iter=100000)
    assert len(lieAlg) - 1 == mp.utils.dim_su(2**n_wires)
