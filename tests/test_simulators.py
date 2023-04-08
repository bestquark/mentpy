import pytest
import numpy as np
import mentpy as mp



# test numpy and pennylane simulators
@pytest.mark.parametrize("simulator", ["numpy", "pennylane"])
class TestSimulators(object):

    def test_teleportation(cls, simulator):
        """Test teleportation in one wire."""
        for i in range(1,5):
            gs = mp.templates.linear_cluster(2*i+1)
            sts = mp.utils.generate_haar_random_states(1, 5)
            ps = mp.simulators.PatternSimulator(gs, simulator=simulator)
            for st in sts:
                ps.reset(input_state=st)
                output_dm = ps([0]*(2*i))
                expected_dm = np.outer(st, st.conj().T)
                assert np.allclose(output_dm, expected_dm, atol=1e-3)