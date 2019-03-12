"""
Since 2013, there have been more algorithms appearing that are
focused on getting an advantage from imperfect quantum computers

Basic idea includes running a short sequence of gates where some gates are parametrized, then
read out the results, make adjustments to the parameters on a classical computer
and repeat calculations with new parameters on the Quantum hardware. 

As a result we get classical-quantum hybrid algorithms
"""


# Quantum Approximate Optimization Algorithm #########################################

"""
QAOA is a shallow circuit variational algorithm for gate model quantum computers that
was inspired by quantum annealing

The adiabatic patheway is discretized into p steps, where p influences precision
Each discrete time step i has two parameters - beta_i, gamma_i 

The classical variational algorithms does an optimization over these parameters based on 
the observed energy at the end of a hardware run

Hamiltonian to be discretized:
H(t) = (1 - t)*H_0 + t * H_1
under adiabatic conditions. 

One can achieve this by trotterizing the unitary. 
For time step t0, the unitary can be split as
U(t0) = U(H0, beta0)U(H1, gamma0), this can be done for subsequent time steps, eventually splitting up the
evolution to p such chunks

U = U(H0, beta0)U(H1, gamma0)...U(H0, betaP)U(H1, gammaP)

At the end of optimizing the parameters, the discretized evolution will approximate the adiabatic pathway

The hamiltonian H0 - driving or mixing Hamiltonian
The hamiltonian H1 - cost Hamiltonian

The simplest mixing hamiltonian is H0 = - sum_i sigma_i^X

the same as the initial Hamiltonian in quantum annealing

By alternating between the two hamiltonian, the mixing Hamiltonian drives the state towards and equal superposition
, whereas the cost Hamiltonian tries to seek its own ground state
"""

import numpy as np 
from functools import partial 
from pyquil import Program, api
from pyquil.paulis import PauliSum, PauliTerm, exponential_map, sZ
from pyquil.gates import *
from scipy.optimize import minimize
from forest_tools import *
np.set_printoptions(precision=3, suppress=True)
qvm_server, quilc_server, fc = init_qvm_and_quilc()

n_qubits = 2

# Define mixing hamiltonian on some qubits
# Define an IZ operator to express I tensorprod sigma_i^Z, where sigma_i^Z operated only on qubit 1
# Now we do the same thing but using Pauli X operator
# Coefficient here means the strength of the transverse field at the given qubir
# The operator will act trivially on all qubits, except the given one. 
# Define mixing Hamiltonian over two qubits

Hm = [PauliTerm("X", i, 1.0) for i in range(n_qubits)]
# Minimize the ising problem defined by the cost hamiltonian
# Hc = -sigma_1^Z tensorprod sigma_2^Z
# minimum is reach whenever sigma_1^Z = sigma_2^Z for the states |-1,-1>, |1, 1> or any superposition of bothg

# Weight matrix of the ising model, only the coefficient (0, 1) is non-zero
J =  np.array([[0, 1], [0, 0]])

Hc = []
for i in range(n_qubits):
    for j in range(n_qubits):
        Hc.append(PauliTerm("Z", i, -J[i, j]) * PauliTerm("Z", j, 1.0))

#print("Cost Hamiltonian = \n", Hc)

# During iterative procedure, only need to compute exp(-i*beta*Hc) and exp(-i*gamma*Hm)
# Use exponential_map of PyQuil, we can build two functions
# that take beta and gamma respectively and return the above

exp_Hm = []
exp_Hc = []
for term in Hm:
    exp_Hm.append(exponential_map(term))
for term in Hc:
    exp_Hc.append(exponential_map(term))

# Set p = 2, and initialize beta_i and gamma_i params
n_iter = 10  # num of iterations of optimization procedure
p = 1
beta = np.random.uniform(0, np.pi*2, p)
gamma = np.random.uniform(0, np.pi*2, p)

# The initial state is a uniform superposition of all states 
# |q1, ...., qn> , it can be created using hadamard gates on all qubits |0> on a new program

initial_state = Program()
for i in range(n_qubits):
    initial_state += H(i)

# Create circuit, compose the different unitary matrix given by evolve
def create_circuit(beta, gamma):
    circuit = Program()
    circuit += initial_state
    for i in range(p):
        for term_exp_Hc in exp_Hc:
            circuit += term_exp_Hc(-beta[i])
        for term_exp_Hm in exp_Hm:
            circuit += term_exp_Hm(-gamma[i])
    return circuit

# Now create a function evaluate_circuit that takes a single vector beta_gamma (concatenation)
# and returns <Hc> = <psi|Hc|psi> where psi is defined by the circuit created with the function above

def evaluate_circuit(beta_gamma):
    beta = beta_gamma[:p]
    gamma = beta_gamma[p:]
    circuit = create_circuit(beta, gamma)
    return qvm.pauli_expectation(circuit, sum(Hc))

# Optimize the angles

qvm = api.QVMConnection(endpoint=fc.sync_endpoint, compiler_endppint=fc.compiler_endpoint)

result = minimize(evaluate_circuit, np.concatenate([beta, gamma]), method='L-BFGS-B')
print(result)

"""
Output:
/usr/local/lib/python3.6/site-packages/pyquil/paulis.py:703: UserWarning: The term Z1Z0 will be combined with Z0Z1, but they have different orders of operations. This doesn't matter for QVM or wavefunction simulation but may be important when running on an actual device.
  .format(t.id(sort_ops=False), first_term.id(sort_ops=False)))
/usr/local/lib/python3.6/site-packages/scipy/optimize/optimize.py:663: ComplexWarning: Casting complex values to real discards the imaginary part
  grad[k] = (f(*((xk + d,) + args)) - f0) / d[k]
/usr/local/lib/python3.6/site-packages/scipy/optimize/lbfgsb.py:328: ComplexWarning: Casting complex values to real discards the imaginary part
  isave, dsave, maxls)
      fun: (-0.9999999999996858+0j)
 hess_inv: <2x2 LbfgsInvHessProduct with dtype=float64>
      jac: array([-0.,  0.])
  message: b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
     nfev: 30
      nit: 6
   status: 0
  success: True
        x: array([0.785, 2.749])
"""

# Analysis of results

# Create a circuit with optimal params
circuit = create_circuit(result['x'][:p], result['x'][p:])
# Use statevector_simulator backend to display state created by the circuit
wf_sim = api.WavefunctionSimulator(connection=fc)
state = wf_sim.wavefunction(circuit)
print(state)
# prints (0.5-0.5j)|00> + (2.731e-07+6.08e-08j)|01> + (2.731e-07+6.08e-08j)|10> + (0.5-0.5j)|11>

# The state is approximately
# (0.5 - 0.5i)(|00> + |11>) = exp(i*t) 1/sqrt(2) * (|00> + |11>) where t is the phase factor
# that doesnt change the probabilities
# it corresponds to a uniform superposition of the two solutions to the classical problem
# (sigma1 = 1, sigma2= 1) and (sigma1=-1, sigma2=-1)

#Evaluate the operators sigma_1^Z and sigma_2^Z independently
print(qvm.pauli_expectation(circuit, PauliSum([sZ(0)])))
print(qvm.pauli_expectation(circuit, PauliSum([sZ(1)])))
# We see that both are approximately equal to 0
# Its expected given the state we found above, -1 and 1 half the time
# Typical quantum behaviour of E[sigma1z, sigma2z] =/= E[sigma1z]E[sigma2z]
