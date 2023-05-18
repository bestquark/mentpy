import pytest
import numpy as np
import mentpy as mp

all_backends = ["numpy-dm", "numpy-sv", "pennylane"]


@pytest.mark.parametrize("backend", all_backends)
class TestBackends(object):
    def test_teleportation(cls, backend):
        """Test teleportation in one wire."""
        for i in range(1, 5):
            gs = mp.templates.linear_cluster(2 * i + 1)
            sts = mp.utils.generate_haar_random_states(1, 5)
            ps = mp.simulators.PatternSimulator(gs, backend=backend)
            for st in sts:
                ps.reset(input_state=st)
                assert (
                    len(ps.mbqcircuit.trainable_nodes) == 2 * i
                ), f"{ps.mbqcircuit.trainable_nodes} != {2*i}, {backend} failed"
                output_dm = ps([0] * (2 * i))
                expected_dm = np.outer(st, st.conj().T)
                assert np.allclose(
                    output_dm, expected_dm, atol=1e-3
                ), f"{backend} failed, {2*i+1} qubits"


param_templates = [
    lambda i: mp.templates.grid_cluster(2, 3 + i),
    lambda i: mp.templates.linear_cluster(3 + i),
    lambda i: mp.templates.muta(2, 1),
]


@pytest.mark.parametrize("templ", param_templates)
def test_random_measurements_equal(templ):
    """Test random measurements."""
    for i in range(1, 3):
        gs = templ(i)
        if np.random.rand() < 0.5:
            indx = np.random.choice(gs.outputc, 2, replace=False)
            gs[indx[0]] = mp.Ment("X")
            gs[indx[1]] = mp.Ment("X")

        # random measurements
        angles = 2 * np.pi * np.random.rand(len(gs.trainable_nodes))
        results = []
        for backend in all_backends:
            ps = mp.simulators.PatternSimulator(
                gs, backend=backend, window_size=min(5, len(gs.trainable_nodes))
            )
            ps.reset()
            results.append(ps.run(angles))

        # check that the results are equal
        for j in range(len(results) - 1):
            assert np.allclose(
                results[j], results[j + 1]
            ), f"results not equal for {all_backends[j]} and {all_backends[j+1]}"
