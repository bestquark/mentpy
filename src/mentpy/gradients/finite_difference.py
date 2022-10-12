import numpy as np
from mentpy import PatternSimulator
from typing import Optional, Union

#TODO!!

def finite_difference(circuit : PatternSimulator, indices: Optional[Union[list, np.ndarray]] = None , h=1e-5, samples_per_param = 1000, type="central"):
    """Returns a function that calculates the finite difference gradient of the given circuit.
    
    Args
    ----
        circuit: Circuit we want to calculate the finite difference from
        indices: List or  np.ndarray with integer entries. Indices where we will calculate the 
                 gradient. If None, then it calculates the gradient to all entries. 
        h: float equal to the step size
        samples_per_param: number of samples taken from the circuit per parameter to estimate gradient.
        type: str equal to "central", "forward", or "backward" 

    """

    if type not in ["central", "forward", "backward"]:
        raise UserWarning(f"Expected type to be 'central', 'forward', or 'backward' but {type} was given")
    

    if indices is None:
        indices = np.arange(len(circuit.state.outputc))
    
    def fin_diff(args: np.ndarray):
        grad = args.copy()
        for ind in indices:

            pass
    
    return fin_diff


def _finite_diff_ind(circuit: PatternSimulator, index: int, h=1e-5, samples_per_param = 1000, type="central"):

    def fin_diff(arg):

        pass

    return fin_diff



    
    