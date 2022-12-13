import numpy as np


CNOT = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]])

SWAP = np.array([[1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])

CS = np.array([[1, 0, 0, 0],    
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1j]])

H = np.array([[1, 1],
            [1, -1]]) / np.sqrt(2)

PauliX = np.array([[0, 1],
                    [1, 0]])

PauliY = np.array([[0, -1j], 
                    [1j, 0]])

PauliZ = np.array([[1, 0], 
                    [0, -1]])

T = np.array([[1, 0],
                [0, np.exp(1j*np.pi/4)]])