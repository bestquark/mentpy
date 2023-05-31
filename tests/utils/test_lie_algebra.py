import pytest
import mentpy as mp


@pytest.mark.parametrize("n_wires", [2, 3, 4])
def test_lie_algebra_grid(n_wires):
    gs = mp.templates.grid_cluster(n_wires, n_wires + 3)
    lieAlg = mp.utils.calculate_lie_algebra(gs, max_iter=100000)
    assert len(lieAlg) == mp.utils.dim_su(2**n_wires) + 1


# @pytest.mark.parametrize("n_wires", [4, 5])
# def test_lie_algebra_cylinder(n_wires):
#     gs = mp.templates.grid_cluster(n_wires, n_wires + 3, periodic=True)
#     lieAlg = mp.utils.calculate_lie_algebra(gs, max_iter=100000)
#     assert len(lieAlg) == mp.utils.dim_so(2**n_wires) + 1
