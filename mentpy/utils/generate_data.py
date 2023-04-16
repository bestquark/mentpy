import numpy as np
import scipy
import cirq

from mentpy import MBQCircuit


def _generate_haar_random_state(n_qubits: int) -> np.ndarray:
    r"""Makes one Haar random state over n_qubits"""

    zero_list = n_qubits * [cirq.KET_ZERO.state_vector()]
    ket_zeros = cirq.kron(*zero_list)
    haar_random_u = cirq.testing.random_special_unitary(dim=int(2**n_qubits))
    return (haar_random_u @ ket_zeros.T).T[0]


def generate_haar_random_states(n_qubits: int, n_samples: int = 1) -> np.ndarray:
    r"""Makes one Haar random state over n_qubits"""

    if n_samples == 1:
        return _generate_haar_random_state(n_qubits)
    else:
        return [_generate_haar_random_state(n_qubits) for _ in range(n_samples)]


def random_special_unitary(n_qubits: int):
    """Returns a random special unitary in ``n_qubits`` sampled from the Haar distribution."""
    return cirq.testing.random_special_unitary(dim=int(2**n_qubits))


def random_train_test_states_unitary(
    unitary: np.ndarray, n_samples: int, test_size: float = 0.3
) -> tuple:
    r"Return random training and test data (input, target) for a given unitary gate ``unitary``."
    n_qubits = int(np.log2(unitary.shape[0]))
    random_inputs = generate_haar_random_states(n_qubits, n_samples=n_samples)
    random_targets = [(unitary @ st.T).T for st in random_inputs]
    return train_test_split(random_inputs, random_targets, test_size=test_size)


def train_test_split(inputs, targets, test_size: float = 0.3, randomize=False) -> tuple:
    r"Split the data into training and test sets."
    n_samples = len(inputs)
    n_test_samples = int(n_samples * test_size)
    n_train_samples = n_samples - n_test_samples

    if randomize:
        shuffled_indices = np.random.permutation(n_samples)
        inputs = inputs[shuffled_indices]
        targets = targets[shuffled_indices]

    return (inputs[:n_train_samples], targets[:n_train_samples]), (
        inputs[n_train_samples:],
        targets[n_train_samples:],
    )


def brownian_circuit(dim, n, dt):
    """Returns a random unitary matrix close to the identity matrix"""
    u = np.eye(dim)
    for j in range(n):
        re = np.random.normal(size=(dim, dim))
        im = 1j * np.random.normal(size=(dim, dim))
        c = re + im
        h = (c + np.conj(c.T)) / 4
        u = u @ scipy.linalg.expm(1j * h * dt)
    return u


def randomUnitary_closetoid(dim, t, n):
    """Returns a random unitary matrix close to the identity matrix"""
    return brownian_circuit(dim, n, np.sqrt(1 / (n * dim)) * 2 * np.pi * t)


def random_train_test_states_unitary_noise(
    unitary: np.ndarray,
    n_samples: int,
    noise_level: float = 0.0,
    noise_type="brownian",
    test_size: float = 0.3,
) -> tuple:
    r"""Return random training data (input, target) for a given unitary gate ``unitary`` with
    brownian noise parametrized by ``noise_level``.

    Args:
        unitary (np.ndarray): unitary gate
        n_samples (int): number of samples
        noise_level (float): noise level
        noise_type (str): type of noise. Either 'brownian' or 'bitflip'
        test_size (float): percentage of test data
    """
    if noise_level > 1 or noise_level < 0:
        raise ValueError("noise_level must be between 0 and 1")

    n_qubits = int(np.log2(unitary.shape[0]))
    random_inputs = generate_haar_random_states(n_qubits, n_samples=n_samples)
    n_samples = len(random_inputs)
    n_test_samples = int(n_samples * test_size)
    n_train_samples = n_samples - n_test_samples

    random_targets_train = []

    for curr_st in range(n_train_samples):
        if noise_type == "brownian":
            noise_u = randomUnitary_closetoid(2**n_qubits, noise_level, 20)
            random_targets_train += [
                (noise_u @ unitary @ np.conj(noise_u.T) @ random_inputs[curr_st].T).T
            ]

        elif noise_type == "bitflip":
            noise_u = 1
            for i in range(n_qubits):
                if np.random.rand() < noise_level:
                    noise_u = np.kron(noise_u, np.array([[0, 1], [1, 0]]))
                else:
                    noise_u = np.kron(noise_u, np.eye(2))
            random_targets_train += [(noise_u @ unitary @ random_inputs[curr_st].T).T]

    random_targets_test = [(unitary @ st.T).T for st in random_inputs[n_train_samples:]]

    return (random_inputs[:n_train_samples], random_targets_train), (
        random_inputs[n_train_samples:],
        random_targets_test,
    )
