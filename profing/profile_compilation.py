import networkx as nx
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath("../src"))
from mentpy import *
import cirq
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.stats import unitary_group
from scipy.special import binom
import itertools as it

from datetime import datetime

startTime = datetime.now()

g = nx.Graph()
g.add_edges_from([(0,1), (1,2), (2,3), (3,4)])
gs = MBQCGraph(g, input_nodes = [0], output_nodes = [4])
gp = mtp.PatternSimulator(gs, trace_in_middle = False)

def average_cost(pattern, n_samples=100):
    gate = cirq.H._unitary_()
    x_train, y_train = random_training_data_for_unitary(gs, n_samples=n_samples, unitary = gate)
    fids = []
    for s in range(n_samples):
        gp.reset()
        outcomes, output_state = gp.measure_pattern(pattern, input_state=x_train[s], correct_for_outcome=True)

        fids.append(cirq.fidelity(output_state, y_train[s]))
        
    return 1 - np.mean(fids)

theta = np.random.rand(4)

average_cost(theta, n_samples=1)

ans = optimize.minimize(average_cost, 
                        theta, 
                        method='trust-constr', 
                        tol = 1e-5,
                        bounds = 4*[(-np.pi, np.pi)]
                       )

print(datetime.now() - startTime)