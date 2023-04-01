from typing import Union, List, Tuple, Optional
import numpy as np
import math

from mentpy.states.mbqcstate import MBQCState
from mentpy.simulators.base_simulator import BaseSimulator


class NumpySimulator(BaseSimulator):
    def __init__(self, mbqcstate: MBQCState, input_state: np.ndarray = None) -> None:
        super().__init__(mbqcstate, input_state)

        self.n_qubits = n_qubits
        self.width = width
        
        self.graph = graph
        self.flow = flow
        
        self.unitary = unitary
        self.noise = noise
        self.noise_type = noise_type
        self.test_fidelity = test_fidelity
        self.init_state_random = init_state_random
        self.init_state = init_state # PF
        self.init_noise = init_noise # PF
        self.init_noise_type = init_noise_type # PF
        
        self.state = -4*np.ones(self.n_qubits)
        self.total_measurements = self.n_qubits
        self.measurements_left = self.n_qubits
        
        q_zeros = 1
        for i in range(self.width):
            q_zeros = np.kron(q_zeros,H@q_zero) # PF: H@q_zero already defined as qubit_plus
        
        if self.init_state_random:
            st = unitary_group.rvs(2**self.width)@q_zeros
        elif not self.init_state_random:
            if self.init_state is None: # PF
                st = np.eye(2**self.width)@q_zeros # PF: what's the use of eye?
            else:
                st = self.init_state

        self.final_qstate_test = self.pure2density(self.unitary@st)

        if input_state_indxs == [-1]:
            self.input_state_indices = list(range(self.width+1))
        else:
            self.input_state_indices = input_state_indxs
           
        if output_state_indxs ==[-1]:
            self.output_state_indices = list(range(n_qubits, n_qubits+width))
        else:
            self.output_state_indices = output_state_indxs
            
        assert len(self.input_state_indices)==self.width+1, "ERROR WITH INPUT STATE INDICES"

        subgr = self.graph.subgraph(self.input_state_indices).copy()
        mapping = {nod:idx for idx,nod in enumerate(self.input_state_indices)}
        subgr = nx.relabel_nodes(subgr, mapping)

        self.current_simulated_nodes = np.array(self.input_state_indices)
        self.qstate = self.pure2density(self.graph_with_multiple_inputs(subgr, inputstates=st, width=self.width))


    def measure(self, angle: float, plane: str = "XY", **kwargs):
        raise NotImplementedError

    def measure_pattern(
        self, angles: List[float], planes: Union[List[str], str] = "XY", **kwargs
    ) -> Tuple[List[int], np.ndarray]:
        raise NotImplementedError

    def reset(self, input_state: np.ndarray = None):
        raise NotImplementedError

    def step(self, action):
        """
        Step function with the convention of the gym library
        It measures the current qubit with an angle of (action)
        """
        
        current_measurement = np.min(self.current_simulated_nodes)
        self.state[current_measurement] = action[0]
        self.measurements_left -= 1
        
        qubit_to_measure = np.argmin(self.current_simulated_nodes)
        self.qstate, outcome = self.measure_angle(action[0] , qubit_to_measure) # PF: adjusted to changed definition of measure_angle

        # if outcome == 1:
        #     fi = self.flow(current_measurement)
        #     assert fi in self.current_simulated_nodes, "ERROR WITH FLOW"
        #     modified_qubit  = np.where(np.array(self.current_simulated_nodes)==fi)[0][0]
        #     self.qstate = self.arbitrary_qubit_gate(sx,modified_qubit,self.width+1)@self.qstate@np.conj(self.arbitrary_qubit_gate(sx,modified_qubit,self.width+1).T)
            
        #     for ne in self.graph.neighbors(fi):
        #         if ne in self.current_simulated_nodes and ne!=current_measurement:
        #             modified_qubit2  = np.where(np.array(self.current_simulated_nodes)==ne)[0][0]
        #             self.qstate = self.arbitrary_qubit_gate(sz,modified_qubit2,self.width+1)@self.qstate@np.conj(self.arbitrary_qubit_gate(sz,modified_qubit2,self.width+1).T)
        
        self.qstate = self.partial_trace(self.qstate, [qubit_to_measure])
        self.current_simulated_nodes = np.delete(self.current_simulated_nodes, np.where(self.current_simulated_nodes==current_measurement))        
        
        err_temp = False
        # if (qubit_to_measure not in self.output_state_indices):
        # if (np.min(self.current_simulated_nodes) not in self.output_state_indices):
        if self.measurements_left!=0:
            new_qubit_indx = self.flow(np.min(self.current_simulated_nodes))
            if new_qubit_indx in self.current_simulated_nodes:
                err_temp = True
            elif new_qubit_indx in list(self.graph.nodes()):
                self.current_simulated_nodes = np.append(self.current_simulated_nodes, [new_qubit_indx])

        # if self.measurements_left!=0:
            if err_temp:
                print("ERROR, CHECK FLOW?")
            self.qstate = np.kron(self.qstate, self.pure2density(qubit_plus))
            for ne in self.graph.neighbors(new_qubit_indx):
                if ne in self.current_simulated_nodes:
                    q1 = np.where(self.current_simulated_nodes==ne)[0][0]
                    q2 = np.where(self.current_simulated_nodes==new_qubit_indx)[0][0]
                    cgate=self.controlled_z(q1,q2, self.width+1)
                    self.qstate = cgate@self.qstate@np.conj(cgate.T)      
        
        reward = 0 #fidelity
        
        if self.measurements_left == 0:  
            sorted_nodes = self.current_simulated_nodes.copy()
            sorted_nodes.sort()
            if (self.current_simulated_nodes==sorted_nodes).all():
                pass
            else:
                sim_nodes = self.current_simulated_nodes.copy()
                for n_iteret in range(1,len(sim_nodes)):
                    ll = np.argmin(sim_nodes[:-n_iteret])
                    sim_nodes[n_iteret], sim_nodes[ll+n_iteret] =sim_nodes[ll+n_iteret], sim_nodes[n_iteret]  
                    swapgate = self.swap_ij(ll+n_iteret-1,n_iteret,len(sim_nodes))
                    self.qstate =swapgate@self.qstate@np.conj(swapgate.T)
            
            if not self.test_fidelity:
                reward = self.fidelity(self.final_qstate_train, self.qstate)
            elif self.test_fidelity:
                reward = self.fidelity(self.final_qstate_test, self.qstate)
            done = True
        else:
            done = False
        
        info = {}

        return self.state, reward, done, info
    

    def reset(self):
        """
        Resets MDP.
        """
        self.state = -4*np.ones(self.n_qubits)
        self.total_measurements = self.n_qubits
        self.measurements_left = self.n_qubits
        
        q_zeros = 1
        for i in range(self.width):
            q_zeros = np.kron(q_zeros, H@q_zero)
        
        if self.init_state_random:
            st = unitary_group.rvs(2**self.width)@q_zeros
        elif not self.init_state_random:
            if self.init_state is None: # PF
                st = np.eye(2**self.width)@q_zeros # PF: what's the use of eye?
            else:
                st = self.init_state
                
            
        self.final_qstate_test = self.pure2density(self.unitary@st)
        

        subgr = self.graph.subgraph(self.input_state_indices).copy()
        mapping = {nod:idx for idx,nod in enumerate(self.input_state_indices)}
        subgr = nx.relabel_nodes(subgr, mapping)

        self.current_simulated_nodes = np.array(self.input_state_indices)

        self.qstate = self.pure2density(self.graph_with_multiple_inputs(subgr, inputstates=st, width=self.width))
        
        return self.state

    def arbitrary_qubit_gate(self,u,i,n):
        """
        Single qubit gate u acting on qubit i
        n is the number of qubits
        """
        op = 1
        for k in range(n):
            if k==i:
                op = np.kron(op, u)
            else:
                op = np.kron(op, np.eye(2))
        return op
    
    def swap_ij(self,i,j,n):
        """
        Swaps qubit i with qubit j
        """
        assert i<n and j<n
        op1,op2,op3,op4 = np.ones(4)
        for k in range(n):
            if k==i or k==j:
                op1 = np.kron(op1,np.kron(np.array([[1],[0]]).T, np.array([[1],[0]])))
                op4 = np.kron(op4,np.kron(np.array([[0],[1]]).T, np.array([[0],[1]])))
            else:
                op1 = np.kron(op1, np.eye(2))
                op4 = np.kron(op4, np.eye(2))

            if k == i:
                op2 = np.kron(op2,np.kron(np.array([[1],[0]]).T, np.array([[0],[1]])))
                op3 = np.kron(op3,np.kron(np.array([[0],[1]]).T, np.array([[1],[0]])))
            elif k==j:
                op2 = np.kron(op2,np.kron(np.array([[0],[1]]).T, np.array([[1],[0]])))
                op3 = np.kron(op3,np.kron(np.array([[1],[0]]).T, np.array([[0],[1]])))
            else:
                op2 = np.kron(op2, np.eye(2))
                op3 = np.kron(op3, np.eye(2))
        return op1+op2+op3+op4
    
    def partial_trace(self, rho, indices):
        """
        Partial trace of state rho over some indices 
        """
        x,y = rho.shape
        n = int(math.log(x,2))
        r = len(indices)
        sigma = np.zeros((int(x/(2**r)), int(y/(2**r))))
        for m in range(0, 2**r):
            qubits = format(m,'0'+f'{r}'+'b')
            ptrace = 1
            for k in range(0,n):
                if k in indices:
                    idx = indices.index(k)
                    if qubits[idx]=='0':
                        ptrace = np.kron(ptrace, np.array([[1],[0]]))
                    elif qubits[idx]=='1':
                        ptrace = np.kron(ptrace, np.array([[0],[1]]))
                else:
                    ptrace = np.kron(ptrace, np.eye(2))
            sigma = sigma + np.conjugate(ptrace.T)@rho@(ptrace)
        return sigma

    def measure_angle(self, angle, i, rho=None): # PF: made rho optional argument
        """
        Measures qubit i of state rho with an angle 
        """
        if rho is None: # PF: allow to get rho from self
            rho = self.qstate
        n = int(np.log2(np.shape(rho)[0])) # PF: get n from rho
        pi0 = 1
        pi1 = 1
        pi0op = np.array([[1, np.exp(-angle*1j)],[np.exp(angle*1j), 1]])/2
        pi1op = np.array([[1,-np.exp(-angle*1j)],[-np.exp(angle*1j), 1]])/2
        for k in range(0,n):
            if k == i:
                pi0 = np.kron(pi0, pi0op)
                pi1 = np.kron(pi1, pi1op)
            else:
                pi0 = np.kron(pi0, np.eye(2))
                pi1 = np.kron(pi1, np.eye(2))
        prob0, prob1 = np.around(np.real(np.trace(rho@pi0)),10), np.around(np.real(np.trace(rho@pi1)),10) # PF: round to deal with deterministic outcomes (0 and 1 can be numerically outside of [0,1])
        measurement = np.random.choice([0,1], p=[prob0,prob1]/(prob0+prob1))
        
        if measurement==0:
            rho = pi0@rho@pi0/prob0
        elif measurement==1:
            rho = pi1@rho@pi1/prob1
            
        return rho, measurement

    def controlled_z(self, i, j , n):
        """
        Controlled z gate between qubits i and j. 
        n is the total number of qubits
        """
        assert i<n and j<n
        op1, op2 = 1, 2
        for k in range(0,n):
            op1 = np.kron(op1, np.eye(2))
            if k in [i,j]:
                op2 = np.kron(op2, np.kron(np.conjugate(np.array([[0],[1]]).T), np.array([[0],[1]])))
            else:
                op2 = np.kron(op2, np.eye(2))
        return op1-op2
    
    def cnot_ij(self, i,j, n):
        """
        CNOT gate with 
        i: control qubit
        j: target qubit
        n: number of qubits
        """
        op1,op2,op3,op4 = np.ones(4)
        for k in range(1,n+1):
            if k==i or k==j:
                op1 = np.kron(op1,np.kron(np.array([[1],[0]]).T, np.array([[1],[0]])))
            else:
                op1 = np.kron(op1, np.eye(2))        
            if k == i:
                op2 = np.kron(op2,np.kron(np.array([[1],[0]]).T, np.array([[1],[0]])))
                op3 = np.kron(op3,np.kron(np.array([[0],[1]]).T, np.array([[0],[1]])))
                op4 = np.kron(op4,np.kron(np.array([[0],[1]]).T, np.array([[0],[1]])))
            elif k==j:
                op2 = np.kron(op2,np.kron(np.array([[0],[1]]).T, np.array([[0],[1]])))
                op3 = np.kron(op3,np.kron(np.array([[1],[0]]).T, np.array([[0],[1]])))
                op4 = np.kron(op4,np.kron(np.array([[0],[1]]).T, np.array([[1],[0]])))
            else:
                op2 = np.kron(op2, np.eye(2))
                op3 = np.kron(op3, np.eye(2))
                op4 = np.kron(op4, np.eye(2))

        return op1+op2+op3+op4