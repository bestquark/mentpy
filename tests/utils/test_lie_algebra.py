import pytest
import mentpy as mp


@pytest.mark.parametrize("n_wires", [2, 3, 4])
def test_lie_algebra_grid(n_wires):
    gs = mp.templates.grid_cluster(n_wires, n_wires + 3)
    lieAlg = mp.utils.calculate_lie_algebra(gs, max_iter=100000)
    assert (
        len(lieAlg) == mp.utils.dim_su(2**n_wires) + 1
    ), f"Grid cluster of size ({n_wires}, {n_wires + 3}) failed"


@pytest.mark.parametrize("n_wires", [4])
def test_lie_algebra_cylinder(n_wires):
    gs = mp.templates.grid_cluster(n_wires, n_wires + 3, periodic=True)
    extralegs = mp.templates.many_wires([2] * n_wires)
    gs = mp.hstack((extralegs, gs, extralegs))
    lieAlg = mp.utils.calculate_lie_algebra(gs, max_iter=100000)
    assert (
        len(lieAlg) == mp.utils.dim_so(2**n_wires) + 1
    ), f"Cylinder cluster of size ({n_wires}, {n_wires + 2}) failed"
